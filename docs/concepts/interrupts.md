# Interrupts

`interrupt()` pauses a node until a person answers: the thread is stored as paused, and a later call with `resume=Resume(...)` continues it.

```python
from nodestep import (
    END,
    START,
    BaseState,
    Graph,
    InMemoryStateStore,
    Resume,
    interrupt,
    node,
)


class Refund(BaseState):
    amount: float = 0.0
    approved: bool = False


@node
def review(state: Refund) -> dict:
    answer = interrupt(f"Approve a refund of {state.amount}?", id="approve")
    return {"approved": answer == "yes"}


graph = Graph(Refund, state_store=InMemoryStateStore()).flow(
    START >> review,
    review >> END,
)

paused = graph.invoke({"amount": 80}, thread_id="refund-1")
print(paused.status)
for key, pending in paused.interrupts.items():
    print(key, pending.payload)

done = graph.invoke(None, thread_id="refund-1", resume=Resume("yes"))
print(done.status, done.state.approved)
```

```text
interrupted
review:approve Approve a refund of 80.0?
completed True
```

The first call stops in `review` and returns the pending interrupt with its payload. The second call answers it. `review` runs again, and this time `interrupt()` returns the answer.

A paused thread must be stored, so interrupts need a [state store](persistence.md#state-stores) and a `thread_id`.

## Keys and answers

Each interrupt has an `id` that you choose. Its key in the result is the task and the id, as in `"review:approve"`.

`Resume(value)` answers the only pending interrupt. When several are pending, `Resume(answers={key: value, ...})` answers each of them by key:

```python
from typing import Annotated

from pydantic import Field

from nodestep import (
    END,
    START,
    BaseState,
    Command,
    Graph,
    InMemoryStateStore,
    Resume,
    add,
    interrupt,
    node,
)


class Release(BaseState):
    sign_offs: Annotated[list[str], add] = Field(default_factory=list)


@node
def legal(state: Release) -> dict:
    answer = interrupt("Legal: approve the release?", id="sign_off")
    return {"sign_offs": [f"legal: {answer}"]}


@node
def security(state: Release) -> dict:
    answer = interrupt("Security: approve the release?", id="sign_off")
    return {"sign_offs": [f"security: {answer}"]}


@node(goto=[legal, security])
def request_sign_offs(state: Release) -> Command:
    return Command(goto=[legal, security])


graph = Graph(Release, state_store=InMemoryStateStore()).flow(
    START >> request_sign_offs,
    legal >> END,
    security >> END,
)

paused = graph.invoke({}, thread_id="release-1")
print(paused.status, sorted(paused.interrupts))

answers = {"legal:sign_off": "approved", "security:sign_off": "approved"}
done = graph.invoke(None, thread_id="release-1", resume=Resume(answers=answers))
print(done.status, done.state.sign_offs)
```

```text
interrupted ['legal:sign_off', 'security:sign_off']
completed ['legal: approved', 'security: approved']
```

Both nodes use the id `sign_off`, and their keys still differ, because each key starts with its task. Each pending `Interrupt` has its `key`, `id`, `node`, `task_id` and `payload`.

## What happens on resume

The resumed node runs again from the top. Code before `interrupt()` therefore runs twice. Put side effects after the interrupt, or keep expensive results in `ctx.cache`, which survives the pause:

```python
from nodestep import (
    END,
    START,
    BaseState,
    Graph,
    InMemoryStateStore,
    NodeContext,
    Resume,
    interrupt,
    node,
)


class Quote(BaseState):
    price: float = 0.0
    accepted: bool = False


@node
def offer(state: Quote, ctx: NodeContext) -> dict:
    print("offer runs")
    if "price" not in ctx.cache:
        print("pricing the order")
        ctx.cache["price"] = 42.0
    answer = interrupt(f"Accept a price of {ctx.cache['price']}?", id="accept")
    return {"price": ctx.cache["price"], "accepted": answer == "yes"}


graph = Graph(Quote, state_store=InMemoryStateStore()).flow(
    START >> offer,
    offer >> END,
)

graph.invoke({}, thread_id="quote-1")
result = graph.invoke(None, thread_id="quote-1", resume=Resume("yes"))
print(result.state)
```

```text
offer runs
pricing the order
offer runs
price=42.0 accepted=True
```

`offer` runs twice, and the price is computed once. While a thread is paused, it takes only answers. To start it over, fork it at an earlier event; see [Time travel](persistence.md#time-travel).

## Answering in code

A middleware can answer an interrupt, so the run does not pause. `interrupt()` asks the `on_interrupt` hook of each middleware in order, and the first answer wins:

```python
from nodestep import END, START, BaseState, Graph, Middleware, interrupt, node
from nodestep.middleware import NodeMiddlewareContext


class AutoApproveSmallAmounts(Middleware):
    def on_interrupt(self, ctx: NodeMiddlewareContext, payload: object) -> object:
        if isinstance(payload, dict) and payload.get("amount", 0) < 50:
            return "yes"
        return None


class Refund(BaseState):
    amount: float = 0.0
    approved: bool = False


@node
def review(state: Refund) -> dict:
    answer = interrupt({"amount": state.amount}, id="approve")
    return {"approved": answer == "yes"}


graph = Graph(Refund, middleware=[AutoApproveSmallAmounts()]).flow(
    START >> review,
    review >> END,
)

print(graph.invoke({"amount": 20}).state.approved)
```

```text
True
```

The hook answers the small refund, so this graph needs no state store. A larger amount would pause the run, and that needs one.

## Tool approvals

`ToolInterruptMiddleware` pauses an agent before a tool call runs, so a person can approve, edit or deny it. Each `InterruptRule` names a tool, or `"*"` for every tool, and may add a predicate on the arguments:

```python
from nodestep.middleware import InterruptRule, gt, should_interrupt

rules = [InterruptRule(tool="refund", match=gt("amount", 100))]

print(should_interrupt(rules, "refund", {"amount": 120.0}))
print(should_interrupt(rules, "refund", {"amount": 20.0}))
print(should_interrupt(rules, "refund", {"amount": "a lot"}))
print(should_interrupt(rules, "lookup_order", {"order_id": "A-1001"}))
```

```text
True
False
True
False
```

`should_interrupt` is the check the middleware makes before each call. The third call asks too: `gt` cannot compare a string with a number, and a predicate that fails counts as a match. Pass the rules as `ToolInterruptMiddleware(rules=rules)`.

The paused call arrives as a `ToolInterrupt` payload. Answer it with `True`, or with a `ToolDecision` to approve, edit or deny the call. [Pause for approval](../guides/approval.md) shows a complete agent.

## Rules and defaults

| Rule or setting | What happens |
|---|---|
| State store | Interrupts need a state store, and so a `thread_id` |
| Interrupt `id` | Required. Unique within one run of its node, and made of letters, digits, `_`, `.` and `-` |
| Interrupt key | `"<task_id>:<id>"`, for example `"review:approve"` |
| `Resume(value)` | Answers the only pending interrupt |
| `Resume(answers={key: value, ...})` | Answers several pending interrupts; it must answer all of them |
| `resume=` | It takes only a `Resume`. `Command` is a node's return value, not an answer |
| Pending `Interrupt` | Has `key`, `id`, `node`, `task_id` and `payload` |
| Checking the answers | Answers are checked before anything runs. A key that is not pending, a pending interrupt without an answer, a single `Resume(value)` while several are pending, or `resume=` with nothing pending raises `ResumeError` |
| Resumed node | It runs again from the top, so code before `interrupt()` runs twice. Put side effects after it, or keep expensive results in `ctx.cache` |
| Answer types | Answers are normalized to JSON values: a pydantic model arrives as a dict, a `date` as a string |
| `context=` | The context of the paused run is not stored. Pass it again with `resume=` |
| `GraphInterrupt` | `interrupt()` raises `GraphInterrupt`, which derives from `BaseException`, so `except Exception` in the node does not catch it |
| Result of a paused run | It shows the state before the paused superstep. Writes of tasks that finished in that superstep are applied on resume |
| Input on a paused thread | New input raises `ResumeError`. Answer the interrupts, or fork the thread at an earlier event |
| `on_interrupt` hooks | Asked in middleware order; the first hook that returns something other than `None` gives the answer, and the node continues |
| `on_interrupt` signature | It must be synchronous |
| No hook answers | The run pauses, which needs a state store |
| `InterruptRule(tool=...)` | A tool name, or `"*"` for every tool |
| `InterruptRule(match=None)` | Without a predicate, every call of the tool is asked for |
| Predicates | `gt`, `ge`, `lt`, `le`, `eq`, `matches` (a regular expression on a string argument), `all_fields` and `any_field` (patterns for several arguments) |
| A predicate that raises | Counts as a match, for example on a missing argument or one of another type: the call is asked for, not run |
| Approval interrupt id | The id is `"approve_"` plus the tool call id. Characters other than letters, digits, `_` and `-` are written as `.` and two hex digits per UTF-8 byte |
| Approval payload | A `ToolInterrupt` with `tool_name`, `tool_call_id` and `arguments` |
| Approve | `True` or `ToolDecision(action="approve")` runs the call |
| Edit | `ToolDecision(action="edit", arguments={...})` runs the call with changed arguments. An edit is merged into the call's arguments: its keys replace those of the call, and the others keep their values, so an edit cannot remove an argument |
| Deny | Any other answer denies the call. A denied call is not run, and the model gets a tool message with the reason |
| Decision as a dict | A dict with the fields of `ToolDecision` works too, since answers arrive as JSON |
| History after an approval | The model's original call stays in the history. The tool message records the arguments the tool ran with in `arguments` |
