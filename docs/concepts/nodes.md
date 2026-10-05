# Nodes and context

A node is a function, `async def` or plain `def`, decorated with `@node`: it receives the state and returns an update.

```python
from nodestep import END, START, BaseState, Graph, NodeContext, node


class Note(BaseState):
    note: str = ""


@node
def stamp(state: Note, ctx: NodeContext) -> dict:
    return {"note": f"{ctx.node} ran in step {ctx.step}"}


graph = Graph(Note).flow(START >> stamp, stamp >> END)

print(graph.invoke({}).state.note)
```

```text
stamp ran in step 0
```

`stamp` reads the run information from its second parameter and returns a dict with the field it changes.

## What a node receives

The state comes first, as the declared type. A parameter annotated `NodeContext`, under any name, gets the run information:

| Attribute | Meaning |
|---|---|
| `ctx.graph_name`, `ctx.node` | Names of the graph and of the node |
| `ctx.run_id`, `ctx.root_run_id` | Id of this run, and of the outermost run when the graph runs inside another one |
| `ctx.thread_id`, `ctx.branch_id` | Thread and history branch of the run |
| `ctx.step`, `ctx.task_id` | Superstep, counted from 0, and the id of this task |
| `ctx.context` | The `context=` object of the run, see [Run context](#run-context) |
| `ctx.emit(data)` | Sends a `"custom"` stream event, see [Streaming](streaming.md) |
| `ctx.cache` | A dict that survives a pause of this task, see [What happens on resume](interrupts.md#what-happens-on-resume) |
| `ctx.spawn`, `ctx.gather`, `ctx.submit` | Start sub-agents and wait for them, see [Sub-agents](sub-agents.md) |

## What a node returns

A node returns an update or a routing decision:

| Return value | Meaning |
|---|---|
| a dict | An update: the fields it names are merged into the state through their reducers |
| the state object it received | The node changed the state in place; nodestep stores the difference |
| `Command(update=..., goto=...)` | An update and the next targets |
| `Send(node, payload)` | Run one node with its own input |
| `END` | End the run after this node |
| `None` | No update |

Both styles end up as an update:

```python
from nodestep import END, START, BaseState, Graph, node


class Cart(BaseState):
    items: list[str] = []
    count: int = 0


@node
def add_item(state: Cart) -> Cart:
    state.items.append("book")
    return state


@node
def count_items(state: Cart) -> dict:
    return {"count": len(state.items)}


graph = Graph(Cart).flow(
    START >> add_item,
    add_item >> count_items,
    count_items >> END,
)

print(graph.invoke({"items": ["pen"]}).state)
```

```text
items=['pen', 'book'] count=2
```

`add_item` changes the state in place, and `count_items` returns a dict. Pick one style per node: changing the state and returning a dict raises an error.

## Run context

The run context holds what nodes need but the state must not keep, such as settings, API clients and database handles. You pass it as `context=` on `invoke`, `ainvoke`, `stream` or `astream`. Nodes read it as `ctx.context`, and tools as `ToolContext.context`.

```python
from dataclasses import dataclass

from nodestep import (
    END,
    START,
    BaseState,
    ContextNotProvidedError,
    Graph,
    NodeContext,
    node,
)


@dataclass
class Pricing:
    tax_rate: float


class Invoice(BaseState):
    net: float = 0.0
    gross: float = 0.0


@node
def add_tax(state: Invoice, ctx: NodeContext) -> dict:
    pricing: Pricing = ctx.context
    return {"gross": round(state.net * (1 + pricing.tax_rate), 2)}


graph = Graph(Invoice).flow(START >> add_tax, add_tax >> END)

print(graph.invoke({"net": 100}, context=Pricing(tax_rate=0.2)).state.gross)
try:
    graph.invoke({"net": 100})
except ContextNotProvidedError as error:
    print(type(error).__name__)
```

```text
120.0
ContextNotProvidedError
```

The second run passes no context, so reading `ctx.context` raises `ContextNotProvidedError`. The context is not part of the thread: it is never stored, so pass it again on resume.

## Command and Send

`Command(update=..., goto=...)` returns an update and the next targets. `Send(node, payload)` runs a node with the payload laid over the state, for that task only. Several `Send`s in one `Command` fan out: one task per payload, in the next superstep.

```python
from typing import Annotated

from pydantic import Field

from nodestep import END, START, BaseState, Command, Graph, Send, add, node


class Research(BaseState):
    topics: list[str] = Field(default_factory=list)
    topic: str = ""
    notes: Annotated[list[str], add] = Field(default_factory=list)


@node
def research(state: Research) -> dict:
    return {"notes": [f"notes on {state.topic}"]}


@node(goto=[research])
def fan_out(state: Research) -> Command:
    return Command(goto=[Send(research, {"topic": topic}) for topic in state.topics])


graph = Graph(Research).flow(
    START >> fan_out,
    research >> END,
)

result = graph.invoke({"topics": ["solar", "wind"]})
print(result.state.notes)
print(repr(result.state.topic))
```

```text
['notes on solar', 'notes on wind']
''
```

Each `research` task saw its own `topic`. The payload is not stored, so `topic` is empty after the run. [Fan out with Send](../guides/fan-out.md) shows the pattern step by step.

## Timeouts

`@node(timeout=seconds)` limits one run of the node. When the time is up, the node fails with `NodeTimeoutError`.

```python
import asyncio

from nodestep import END, START, BaseState, Graph, NodeTimeoutError, node


class Lookup(BaseState):
    answer: str = ""


@node(timeout=0.1)
async def slow_lookup(state: Lookup) -> dict:
    await asyncio.sleep(5)
    return {"answer": "found"}


graph = Graph(Lookup).flow(START >> slow_lookup, slow_lookup >> END)

try:
    graph.invoke({})
except NodeTimeoutError as error:
    print(error)
```

```text
Node 'slow_lookup' timed out after 0.1s
```

## Rules and defaults

| Rule or setting | What happens |
|---|---|
| Plain `def` node | Runs in a worker thread |
| Bound method | Works as a node |
| Callable instances, classes and `functools.partial` objects | `TypeError` |
| `NodeContext` parameter | Found by its annotation, under any name |
| `ctx.step` | Counted from 0 |
| `ctx.task_id` | The node name, then `name[1]`, `name[2]` for further tasks of the node in the same superstep |
| Return value not in the table | Any other value raises `TypeError` |
| Changing the input and returning a dict, `Command`, `Send`, `END` or `None` | `GraphExecutionError` |
| In-place change of an `add` or `add_messages` field | The update holds the appended items |
| In-place change of a `merge_dict` field | The update holds the added or changed keys |
| Any other in-place change of those fields | Replaces the whole field |
| In-place change of a field with a custom reducer | `GraphExecutionError` |
| State copies | Each task gets its own copy of the state |
| `context=` | Passed on `invoke`, `ainvoke`, `stream` or `astream`; read as `ctx.context` in nodes and `ToolContext.context` in tools |
| `ctx.context` when no context was passed | `ContextNotProvidedError` |
| Storage of the run context | It is never stored: pass `context=` again on resume |
| Several `Send`s in one `Command` | One task per payload in the next superstep |
| `Send` targets | Every target is declared in `@node(goto=[...])` |
| `Send` payload | Laid over the state for that task only; it skips the reducers and is not stored |
| `Send` payload keys | Must be state fields with valid values |
| A pydantic model as `Send` payload | Contributes only the fields set on it |
| `@node(timeout=...)` | `None`: no limit. A run of the node that takes longer fails with `NodeTimeoutError` |
| Async node at the timeout | Cancelled at the limit |
| Sync node at the timeout | Its worker thread cannot be stopped, so its error is raised when the function returns |
| Resumed node | Runs again from the top |
| `ctx.cache` | Keeps what the node stored before the pause. It is saved with the paused task, normalized to JSON, and empty in every other task and in later runs of the node |
