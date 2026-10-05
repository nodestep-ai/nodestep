# Running

A run is one call of `invoke()`, `stream()` or their async forms, and it proceeds in supersteps: every ready node runs, their updates merge, then the next nodes start.

```python
from nodestep import END, START, BaseState, Graph, node


class Counter(BaseState):
    count: int = 0


@node
def increment(state: Counter) -> dict:
    return {"count": state.count + 1}


graph = Graph(Counter).flow(START >> increment, increment >> END)

result = graph.invoke({"count": 1})
print(result.status, result.step, result.data)
```

```text
completed 0 {'count': 2}
```

`invoke()` runs the graph until it completes or pauses, and returns a `GraphResult` with the status, the superstep it ended in, and the final state.

## Invoke and stream

A `GraphResult` has these fields:

| Field | Meaning |
|---|---|
| `status` | `"completed"` or `"interrupted"` |
| `state`, `data` | The final state, as the declared type and as a dict |
| `step` | The superstep in which the run finished or paused, counted from 0 |
| `thread_id`, `checkpoint_id` | The thread the run used, and the id of its last history event |
| `interrupts` | The pending interrupts, by key |

`stream()` runs the same graph and yields a `StreamEvent` for each thing that happens; [Streaming](streaming.md) explains the events.

Both take these keyword arguments; the [API reference](../reference/core/graphs.md#nodestep.core.Graph) has the full signatures:

- `thread_id` and `branch_id` choose the thread and its branch, see [Persistence and time travel](persistence.md).
- `resume` answers a pause, see [Interrupts](interrupts.md).
- `state_store` sets the store for this call, see [Persistence and time travel](persistence.md#state-stores).
- `context` passes the run context, see [Nodes and context](nodes.md#run-context).

## Sync and async

`ainvoke()` and `astream()` are the async entry points. `invoke()` and `stream()` run them on a private event loop per call. Inside a running event loop they raise `RuntimeError`, so in async code you call the async forms.

```python
import asyncio

from nodestep import END, START, BaseState, Graph, node


class Counter(BaseState):
    count: int = 0


@node
async def increment(state: Counter) -> dict:
    return {"count": state.count + 1}


graph = Graph(Counter).flow(START >> increment, increment >> END)


async def main() -> None:
    result = await graph.ainvoke({"count": 1})
    print(result.state.count)
    async for event in graph.astream({"count": 5}, stream_mode="updates"):
        if event.mode == "updates":
            print(event.node, event.data)


asyncio.run(main())
```

```text
2
increment {'count': 6}
```

## Supersteps

A superstep is one round of the run. Every ready task runs concurrently, on its own copy of the state. The run goes through these steps:

1. The input is validated and merged into the state.
2. Superstep 0 runs the node after `START`.
3. After each superstep, the updates merge in node-name order. Then the routes of every task are resolved: edges, branches, `Command(goto=...)` and `Send`.
4. A target named by several tasks of one superstep runs once in the next one. Each `Send` is its own task.

Routers see the merged state, including the writes of the other tasks. The run ends when a superstep produces no new tasks.

Here `plan` sends the run to `flights` and `hotels`, which both lead to `book`:

```python
import asyncio
from typing import Annotated

from pydantic import Field

from nodestep import END, START, BaseState, Command, Graph, add, node


class Trip(BaseState):
    log: Annotated[list[str], add] = Field(default_factory=list)


@node
async def flights(state: Trip) -> dict:
    await asyncio.sleep(0.05)
    return {"log": ["flights"]}


@node
async def hotels(state: Trip) -> dict:
    return {"log": ["hotels"]}


@node
def book(state: Trip) -> dict:
    return {"log": [f"book after {len(state.log)} searches"]}


@node(goto=[flights, hotels])
def plan(state: Trip) -> Command:
    return Command(goto=[flights, hotels])


graph = Graph(Trip).flow(
    START >> plan,
    flights >> book,
    hotels >> book,
    book >> END,
)

for event in graph.stream({}, stream_mode="updates"):
    if event.mode == "updates":
        print(event.step, event.node, event.data)
```

```text
0 plan None
1 flights {'log': ['flights']}
1 hotels {'log': ['hotels']}
2 book {'log': ['book after 2 searches']}
```

`flights` and `hotels` run concurrently in superstep 1. `hotels` finishes first, but the updates merge in node-name order, so the result does not depend on timing. Both route to `book`, which runs once, in superstep 2.

## No join barrier

nodestep has no join barrier: a node runs in the superstep after any task routes to it. It does not wait for other paths. Here the hotel path has one more node, so `book` runs twice.

```python
from typing import Annotated

from pydantic import Field

from nodestep import END, START, BaseState, Command, Graph, add, node


class Trip(BaseState):
    log: Annotated[list[str], add] = Field(default_factory=list)


@node
def flights(state: Trip) -> dict:
    return {"log": ["flights"]}


@node
def hotels(state: Trip) -> dict:
    return {"log": ["hotels"]}


@node
def hotel_details(state: Trip) -> dict:
    return {"log": ["hotel details"]}


@node
def book(state: Trip) -> dict:
    return {"log": [f"book after {len(state.log)} entries"]}


@node(goto=[flights, hotels])
def plan(state: Trip) -> Command:
    return Command(goto=[flights, hotels])


graph = Graph(Trip).flow(
    START >> plan,
    flights >> book,
    hotels >> hotel_details,
    hotel_details >> book,
    book >> END,
)

for event in graph.stream({}, stream_mode="updates"):
    if event.mode == "updates":
        print(event.step, event.node)
```

```text
0 plan
1 flights
1 hotels
2 book
2 hotel_details
3 book
```

To act once both paths are done, do one of these:

- Give the paths the same length.
- Check in the node that everything it needs is in the state, and return no update until it is.

## Parallel writes

Tasks of one superstep may write the same field when its reducer merges, like `add` in the [Supersteps](#supersteps) example. A write to a field without a merging reducer conflicts, and the superstep fails with `InvalidUpdateError`.

## Failures

When tasks raise, the run raises the first failure. The writes of the failed superstep are not merged. With a state store, you can continue later: `invoke(None, thread_id=...)` runs the tasks that did not finish, see [Persistence and time travel](persistence.md#continue-an-unfinished-superstep).

## Limits and timeouts

Three settings stop a run that takes too long. `Graph(max_steps=...)` limits the supersteps of one call. `Graph(timeout=...)` limits the seconds one call spends inside supersteps. `@node(timeout=...)` limits one run of a node, see [Nodes and context](nodes.md#timeouts).

## Rules and defaults

| Rule or setting | What happens |
|---|---|
| Input | Validated and merged into the state before superstep 0 |
| First superstep | Superstep 0 runs the node after `START` |
| Tasks of one superstep | Run concurrently, each on its own copy of the state |
| Merge order | The updates merge in node-name order, so the result does not depend on timing |
| Routes | Resolved after the merge, for every task: edges, branches, `Command(goto=...)` and `Send` |
| What routers see | Routers see the merged state, including the writes of the other tasks of the superstep |
| A target named by several tasks | Runs once in the next superstep |
| `Send` | Each `Send` is its own task |
| End of the run | When a superstep produces no new tasks |
| Join barrier | There is no join barrier: a node runs in the superstep after any task routes to it |
| Parallel writes with a merging reducer | Allowed |
| Parallel writes when one has no merging reducer (the default `replace`) or is a `Replace` | The superstep fails with `InvalidUpdateError` |
| `Send` tasks of the same node writing a `replace` field | No conflict: the last one wins |
| Several failing tasks | The run raises the first failure in scheduling order; the others are not reported |
| Writes of a failed superstep | Not merged. With a state store, `invoke(None, thread_id=...)` later runs the tasks that did not finish |
| `KeyboardInterrupt` and `SystemExit` raised in node code | Not recorded as node failures. They propagate as in plain Python and stop the event loop that runs the graph, including the sandbox's |
| `Graph(max_steps=25)` | A call that needs more supersteps raises `RunLimitExceededError`. The count is per call: a resume or continuation starts a new count |
| `Graph(timeout=None)` | No limit. With a limit, a call that spends longer inside supersteps raises `GraphTimeoutError` |
| Time your stream loop spends on an event | Not counted on `"updates"` and `"values"` events, between supersteps. Counted on `"custom"` and `"tokens"` events, during a superstep. See [Streaming](streaming.md#closing-a-stream-early) |
| `@node(timeout=None)` | No limit. With a limit, a node run that takes longer raises `NodeTimeoutError`, see [Nodes and context](nodes.md#timeouts) |
| `invoke()` and `stream()` | Run `ainvoke()` and `astream()` on a private event loop per call |
| `invoke()` or `stream()` inside a running event loop | `RuntimeError`; call `ainvoke()` or `astream()` there |
| Plain `def` nodes and tools | Run in a worker thread through `asyncio.to_thread` |
