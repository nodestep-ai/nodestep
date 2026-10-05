# Middleware

Middleware is a class whose hooks run your code around a run: around the graph, each node, each model call and each tool call.

```python
from nodestep import (
    END,
    START,
    BaseState,
    Graph,
    Middleware,
    RunOutcome,
    node,
)
from nodestep.middleware import GraphMiddlewareContext, NodeMiddlewareContext


class PrintSteps(Middleware):
    def before_node(self, ctx: NodeMiddlewareContext) -> None:
        print(f"step {ctx.step}: {ctx.node_name} starts")

    def after_node(self, ctx: NodeMiddlewareContext) -> None:
        print(f"step {ctx.step}: {ctx.node_name} returned {ctx.state}")

    def on_run_end(self, ctx: GraphMiddlewareContext, outcome: RunOutcome) -> None:
        print(f"run of {ctx.graph_name} ended: {outcome.status}")


class Order(BaseState):
    quantity: int = 0
    total: float = 0.0


@node
def price(state: Order) -> dict:
    if state.quantity < 0:
        raise ValueError("negative quantity")
    return {"total": state.quantity * 2.5}


graph = Graph(Order, middleware=[PrintSteps()]).flow(START >> price, price >> END)

graph.invoke({"quantity": 4})
try:
    graph.invoke({"quantity": -1})
except ValueError as error:
    print("raised:", error)
```

```text
step 0: price starts
step 0: price returned {'total': 10.0}
run of graph ended: completed
step 0: price starts
run of graph ended: failed
raised: negative quantity
```

The second run fails inside `price`, so `after_node` is not called, and `on_run_end` reports the failure.

## Add middleware

Subclass `Middleware`, override the hooks you need, and pass instances to `Graph(middleware=[...])` or `build_react_agent(middleware=[...])`. Typical uses are logging, tracing, approvals, limits and prompt changes. The [API reference](../reference/middleware/index.md) documents the base class, the hooks and their contexts.

## Hooks

| Hook | Context | Called | Can replace |
|---|---|---|---|
| `before_graph(ctx)` | `GraphMiddlewareContext` | before the first superstep of a run, also on a resume | no |
| `after_graph(ctx)` | `GraphMiddlewareContext` | after the last superstep, only when the run completes | no |
| `on_run_end(ctx, outcome)` | `GraphMiddlewareContext`, `RunOutcome` | once at every end of a run | no |
| `before_node(ctx)` | `NodeMiddlewareContext` | before a node runs, with a copy of its input | the input, for this call only |
| `after_node(ctx)` | `NodeMiddlewareContext` | after a node returns, with its update | the update |
| `on_error(ctx, error)` | `NodeMiddlewareContext` | when a node raises | a result to use instead |
| `on_interrupt(ctx, payload)` | `NodeMiddlewareContext` | inside `interrupt()` | returns an answer, see [Interrupts](interrupts.md#answering-in-code) |
| `before_model(ctx)` | `ModelMiddlewareContext` | before every call of a `model_node` | the `ChatRequest` |
| `after_model(ctx)` | `ModelMiddlewareContext` | after every call of a `model_node` | the `ChatResponse` |
| `before_tool(ctx)` | `ToolMiddlewareContext` | with the validated arguments of a tool call | the arguments; may raise `ToolDeniedError` |
| `after_tool(ctx)` | `ToolMiddlewareContext` | with the validated result | the result |
| `on_tool_error(ctx, error)` | `ToolMiddlewareContext` | when a tool raises | no |

Graph hooks are read-only. Node hooks see copies of the state, so `before_node` changes what the node receives, never what is stored.

## Hook order and return values

- `before_*` hooks run in list order.
- `after_*` hooks and the observer hooks, `on_tool_error` and `on_run_end`, run in reverse order, so the first middleware wraps the others.
- A hook may be `def` or `async def`; only `on_interrupt` must be synchronous.
- A hook that can change something returns `ctx.replace(value)`, or `None` to leave it as it is.
- A method named like a hook that is not one, such as a misspelled `after_tools`, is refused when the class is defined.

## on_run_end

`on_run_end(ctx, outcome)` is called once when a run ends, however it ends. Use it for work that every run needs at its end, such as closing a trace span.

`outcome` is a `RunOutcome`:

| Attribute | Meaning |
|---|---|
| `status` | `"completed"`, `"paused"`, `"failed"` or `"cancelled"` |
| `interrupts` | The pending interrupts by key; empty unless the run paused |
| `error` | What the run raised; set only when `status` is `"failed"` |

Only middleware whose `before_graph` was called gets `on_run_end`. The table at the end of this page lists the order and what happens when the hook raises.

## on_tool_error

`on_tool_error(ctx, error)` is called when a tool raises. It runs before the `tool_errors` policy decides whether the error fails the run or goes to the model. It only observes: it cannot replace the error.

```python
import asyncio

from nodestep import ScriptedChat, build_react_agent, run_agent, tool
from nodestep.chat import ChatResponse, ToolCall
from nodestep.middleware import Middleware, ToolMiddlewareContext


class LogToolErrors(Middleware):
    def on_tool_error(self, ctx: ToolMiddlewareContext, error: Exception) -> None:
        print(f"{ctx.tool_name} failed in task {ctx.task_id}: {error}")


@tool
def weather(city: str) -> str:
    """Return the weather for a city."""
    raise TimeoutError("weather service did not answer")


chat = ScriptedChat(
    [
        ChatResponse(
            tool_calls=[
                ToolCall(id="call_1", name="weather", arguments={"city": "Oslo"})
            ]
        ),
        "I cannot get the weather right now.",
    ]
)
agent = build_react_agent(
    chat, tools=[weather], middleware=[LogToolErrors()], tool_errors="return"
)

print(asyncio.run(run_agent(agent, "Weather in Oslo?")))
print(chat.requests[1].messages[-1].content)
```

```text
weather failed in task act: weather service did not answer
I cannot get the weather right now.
Tool error: TimeoutError: weather service did not answer
```

With `tool_errors="return"`, the model gets the error's text as the tool result, and the run goes on.

`ToolMiddlewareContext` carries `tool_name`, the arguments in `value`, `tool_call_id`, and the `task_id` of the node task that runs the tool.

The hook is not called when the call never reaches the tool, for example when a `before_tool` hook denies it. A tracer that opens a span in `before_tool` must then close it elsewhere, for example in `after_node` or `on_run_end`.

## Model hooks

`before_model` and `after_model` run around every model call of a `model_node`, including those in `build_react_agent`. `ModelMiddlewareContext` has the `request`, the `response` (in `after_model`) and `ctx.model`, the `model` name of the chat model. It also names the graph, node, run, thread, step and task.

```python
import asyncio

from nodestep import ScriptedChat, build_react_agent, run_agent, tool
from nodestep.chat import ChatResponse, SystemMessage, ToolCall
from nodestep.middleware import Middleware, ModelMiddlewareContext, Replacement


class AddDate(Middleware):
    def before_model(self, ctx: ModelMiddlewareContext) -> Replacement:
        note = SystemMessage(content="Today is 2026-10-01.")
        request = ctx.request.model_copy(
            update={"messages": [note, *ctx.request.messages]}
        )
        return ctx.replace(request)


class Usage(Middleware):
    def after_model(self, ctx: ModelMiddlewareContext) -> None:
        print(ctx.model, ctx.node_name, ctx.response.usage)


@tool
def weather(city: str) -> str:
    """Return the weather for a city."""
    return "Sunny"


chat = ScriptedChat(
    [
        ChatResponse(
            tool_calls=[
                ToolCall(id="call_1", name="weather", arguments={"city": "Oslo"})
            ],
            usage={"input_tokens": 40, "output_tokens": 8},
        ),
        ChatResponse(
            content="It is sunny in Oslo.",
            usage={"input_tokens": 70, "output_tokens": 9},
        ),
    ]
)
agent = build_react_agent(chat, tools=[weather], middleware=[AddDate(), Usage()])

print(asyncio.run(run_agent(agent, "Weather in Oslo?")))
print(chat.requests[0].messages[0].content)
```

```text
scripted think {'input_tokens': 40, 'output_tokens': 8}
scripted think {'input_tokens': 70, 'output_tokens': 9}
It is sunny in Oslo.
Today is 2026-10-01.
```

`AddDate` puts a system message in front of every request, and `Usage` prints the token usage of each call. The last line shows that the model received the date.

## Middleware tools

Some middleware offers tools through `tools()`. `build_react_agent` adds the tools of the middleware passed to it. A node of your own gets only the tools in its `tools=`, so list them yourself, as in `tools=[*todo.tools(), search]`.

## Built-in middleware

All of these are in `nodestep.middleware`.

| Middleware | What it does |
|---|---|
| [`ToolInterruptMiddleware`](../reference/middleware/interrupt.md) | Asks a person before matching tool calls run; see [Tool approvals](interrupts.md#tool-approvals) |
| [`ToolLimitMiddleware`](../reference/middleware/tool_limit.md) | Denies tool calls over a budget, for all tools with `max_calls=` and per tool with `per_tool={...}` |
| [`SummarizationMiddleware`](../reference/middleware/summarization.md) | Replaces older messages with a summary before a model call once a trigger fires |
| [`TodoListMiddleware`](../reference/middleware/todolist.md) | Offers `read_todos` and `write_todos` tools with one list per thread |
| [`FilesystemSkills`](../reference/middleware/skills.md) | Lists the skills found in `SKILL.md` files in the requests of the named model nodes and offers a `load_skill` tool |
| [`FilesystemMemory`](../reference/middleware/memory.md) | Long-term memory tools; see [Memory](workspace.md#memory) |

## Rules and defaults

| Rule or setting | What happens |
|---|---|
| Order of `before_*` hooks | List order |
| Order of `after_*` hooks, `on_tool_error` and `on_run_end` | Reverse order, so the first middleware wraps the others |
| `def` or `async def` | Every hook may be either, except `on_interrupt`, which must be synchronous |
| A misspelled hook name | A method named `before_*`, `after_*` or `on_*` that is not a hook, such as a misspelled `after_tools`, raises `TypeError` when the class is defined |
| Return value of a hook that can replace | `ctx.replace(value)` or `None`; any other return value raises `TypeError` |
| Graph hooks | Read-only: their `ctx.state` is a copy, and returning a replacement raises `GraphExecutionError` |
| Node hooks | See copies of the state; `before_node` changes what the node receives, never what is stored |
| `before_graph` | Called before the first superstep of a run, also on a resume |
| `after_graph` | Called only when the run completes |
| When `on_run_end` is called | Once per run, however it ends: completed, paused, failed, cancelled, closed early by the consumer, timed out or over `max_steps` |
| Who gets `on_run_end` | Only middleware whose `before_graph` was called; if a `before_graph` raises, that middleware and the ones before it get `on_run_end`, the ones after it do not |
| Order of `on_run_end` | Reverse order, after `after_graph` when the run completes |
| A call refused before the run starts | Calls no hook, for example new input on a paused thread |
| `ctx.state` in `on_run_end` | A copy of the last merged state |
| An `on_run_end` hook raises | Every hook still runs; if the run failed, the hook's error becomes a note on the run's error, otherwise the hook's error propagates |
| `on_run_end` returns a value | An error: `TypeError`, or `GraphExecutionError` for a `Replacement` |
| `asyncio.CancelledError` from `on_run_end` | Raised after the other hooks, ahead of the run's error |
| `on_tool_error` raises or returns a value | It never replaces the tool's error; it is added to the tool's error as a note |
| `on_tool_error` with `tool_errors="return"` | The model gets only the error's text, without the notes |
| A call that never reaches the tool | `on_tool_error` is not called: a `before_tool` hook raised, such as `ToolDeniedError` from `ToolLimitMiddleware`, or arguments replaced by a `before_tool` hook failed the tool's input model |
| `before_tool` hooks of a call that never reached the tool | The ones that already ran get neither `after_tool` nor `on_tool_error` |
| `ctx.model` | The `model` name of the chat model, such as `"scripted"` for `ScriptedChat` |
| A request changed in `before_model` | Sent to the model but not stored: the state keeps the messages as they were |
| Middleware tools on a node | A node's `tools=` is its complete tool list; middleware adds nothing to it, so you list the tools yourself |
| Middleware tools on `build_react_agent` | Added to the agent's tools; a tool listed twice raises `GraphConfigError` |
| `ToolLimitMiddleware(max_calls=50)` | The budget covers one run and its resumes; calls over it are denied |
| `ToolLimitMiddleware` budgets | The budgets live in the instance's memory: a resume in another process starts a new budget, and the budget of a run that failed or was never resumed stays in memory |
| `SummarizationMiddleware` triggers and policies | A trigger such as `MessageCountTrigger(max_messages=40)` or `TokenLimitTrigger(max_input_tokens=...)` decides when to summarize; `RecentMessagesPolicy` or `RecentTokensPolicy` decides which recent messages stay |
| `TodoListMiddleware` lists | One list per thread, kept in this process's memory |
| `FilesystemSkills(nodes={...})` | The names in `nodes` are not checked against the graph: a misspelled name means that node never gets the list |
