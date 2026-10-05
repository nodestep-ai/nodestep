# Persistence and time travel

A state store keeps the history of every thread, so a run can continue where the last one stopped, be resumed from another process, or start again from an earlier step.

```python
from typing import Annotated

from pydantic import Field

from nodestep import END, START, BaseState, Graph, InMemoryStateStore, add, node


class Poll(BaseState):
    votes: Annotated[list[str], add] = Field(default_factory=list)
    count: int = 0


@node
def tally(state: Poll) -> dict:
    return {"count": len(state.votes)}


graph = Graph(Poll, state_store=InMemoryStateStore()).flow(
    START >> tally,
    tally >> END,
)

graph.invoke({"votes": ["yes"]}, thread_id="poll-1")
print(graph.invoke({"votes": ["no"]}, thread_id="poll-1").state)
print(graph.invoke({"votes": ["no"]}, thread_id="poll-2").state)
```

```text
votes=['yes', 'no'] count=2
votes=['no'] count=1
```

The second run on `poll-1` starts from the state the first run left, so its vote is added to `yes`. `poll-2` is a new thread and starts from the defaults.

## State stores

A state store records each step of a run as history events. nodestep has two:

| Store | Where the history lives |
|---|---|
| `InMemoryStateStore()` | In the memory of this process; gone when it exits |
| `FilesystemStateStore(path)` | In one JSON Lines file at `path`, which several processes can share |

Use `InMemoryStateStore` in tests and short scripts. Use `FilesystemStateStore` when the history must outlive the process. Pass a store to `Graph(state_store=...)`, or to a single call as `state_store=`.

With `FilesystemStateStore`, the history stays in the file, and a new graph on the same file reads it:

```python
import asyncio
import tempfile
from pathlib import Path
from typing import Annotated

from pydantic import Field

from nodestep import END, START, BaseState, FilesystemStateStore, Graph, add, node


class Poll(BaseState):
    votes: Annotated[list[str], add] = Field(default_factory=list)
    count: int = 0


@node
def tally(state: Poll) -> dict:
    return {"count": len(state.votes)}


async def main(directory: Path) -> None:
    store = FilesystemStateStore(directory / "threads" / "polls.jsonl")
    graph = Graph(Poll, state_store=store).flow(START >> tally, tally >> END)

    await graph.ainvoke({"votes": ["yes"]}, thread_id="poll-1")
    result = await graph.ainvoke({"votes": ["no"]}, thread_id="poll-1")
    print(result.state)

    reopened = Graph(Poll, state_store=FilesystemStateStore(store.path))
    reopened.flow(START >> tally, tally >> END)
    print((await reopened.get_state("poll-1")).value)


with tempfile.TemporaryDirectory() as directory:
    asyncio.run(main(Path(directory)))
```

```text
votes=['yes', 'no'] count=2
votes=['yes', 'no'] count=2
```

The second line comes from `reopened`, a new graph that reads the same file.

## Threads

A thread is one line of history, named by its `thread_id`. With a store, every run belongs to a thread, so every call needs a `thread_id`.

The next run on a thread starts from the state the last run left. Its input is a patch: it merges into that state through the reducers, as the votes in the opening example do.

## Read and change a thread

These methods read and change a thread outside a run. All of them are async.

| Method | Returns |
|---|---|
| `await graph.exists(thread_id)` | Whether the store has history for the thread |
| `await graph.history(thread_id)` | A `History` with every recorded event and checkpoint |
| `await graph.load(thread_id, at=None)` | The state as a dict, at the latest event or at event `at` |
| `await graph.get_state(thread_id, at=None)` | A `StateSnapshot` with the state as the declared type (`value`) and the event's `sequence` |
| `await graph.update_state(thread_id, values)` | Applies an update as a node would, and returns the event id |
| `await graph.fork(thread_id, from_=event_id, name=None)` | A new branch that starts at an event, as a `BranchRecord` |
| `await graph.branches(thread_id)` | The branches forked from the thread |

Each method also takes `branch_id=` and `state_store=`. See [`Graph`](../reference/core/graphs.md#nodestep.core.Graph) for the full signatures.

## Time travel

Every history event has an id. `load(..., at=...)` shows the state at an event, and `fork(...)` starts a new branch there. A run with `branch_id=` continues the branch, and the original branch stays as it was.

```python
import asyncio
from typing import Annotated

from pydantic import Field

from nodestep import END, START, BaseState, Graph, InMemoryStateStore, add, node


class Poll(BaseState):
    votes: Annotated[list[str], add] = Field(default_factory=list)
    count: int = 0


@node
def tally(state: Poll) -> dict:
    return {"count": len(state.votes)}


graph = Graph(Poll, state_store=InMemoryStateStore()).flow(
    START >> tally,
    tally >> END,
)


async def main() -> None:
    await graph.ainvoke({"votes": ["yes"]}, thread_id="poll-1")
    await graph.ainvoke({"votes": ["no"]}, thread_id="poll-1")

    history = await graph.history("poll-1")
    first_turn = next(
        event for event in history.events if event.type == "run_completed"
    )
    print(await graph.load("poll-1", at=first_turn.id))

    branch = await graph.fork("poll-1", from_=first_turn.id, name="recount")
    result = await graph.ainvoke(
        {"votes": ["maybe"]}, thread_id="poll-1", branch_id=branch.id
    )
    print(branch.id, result.state.votes)
    print("main", (await graph.get_state("poll-1")).value.votes)


asyncio.run(main())
```

```text
{'votes': ['yes'], 'count': 1}
recount ['yes', 'maybe']
main ['yes', 'no']
```

The fork starts after the first turn, so the new vote follows `yes` alone. The `main` branch still has both votes.

## Continue an unfinished superstep

A superstep stays unfinished when a node fails, when a limit or a timeout stops the run, or when the caller closes a stream early. `invoke(None, thread_id=...)` continues it. Tasks that finished keep their updates and do not run again.

```python
from nodestep import END, START, BaseState, Command, Graph, InMemoryStateStore, node


class Trip(BaseState):
    flight: str = ""
    hotel: str = ""


hotel_service = {"up": False}


@node
async def book_flight(state: Trip) -> dict:
    print("book_flight runs")
    return {"flight": "LH 1234"}


@node
async def book_hotel(state: Trip) -> dict:
    print("book_hotel runs")
    if not hotel_service["up"]:
        raise ConnectionError("the hotel service is down")
    return {"hotel": "Hotel Adler"}


@node(goto=[book_flight, book_hotel])
def plan(state: Trip) -> Command:
    return Command(goto=[book_flight, book_hotel])


graph = Graph(Trip, state_store=InMemoryStateStore()).flow(
    START >> plan,
    book_flight >> END,
    book_hotel >> END,
)

try:
    graph.invoke({}, thread_id="trip-1")
except ConnectionError as error:
    print("failed:", error)

hotel_service["up"] = True
result = graph.invoke(None, thread_id="trip-1")
print(result.status, result.state)
```

```text
book_flight runs
book_hotel runs
failed: the hotel service is down
book_hotel runs
completed flight='LH 1234' hotel='Hotel Adler'
```

`book_flight` finished before the failure, so only `book_hotel` runs again.

## Edit a paused thread

`update_state` also works on a thread that is paused or unfinished. The resumed node then reads the edited value.

```python
import asyncio

from nodestep import (
    END,
    START,
    BaseState,
    Command,
    Graph,
    InMemoryStateStore,
    InvalidUpdateError,
    Resume,
    interrupt,
    node,
)


class Offer(BaseState):
    price: int = 0
    note: str = ""


@node
def quote(state: Offer) -> dict:
    return {"price": 100}


@node
def review(state: Offer) -> dict:
    answer = interrupt({"note": state.note}, id="review")
    return {"note": f"{answer}, after reading {state.note!r}"}


@node(goto=[quote, review])
def start(state: Offer) -> Command:
    return Command(goto=[quote, review])


graph = Graph(Offer, state_store=InMemoryStateStore()).flow(
    START >> start,
    quote >> END,
    review >> END,
)


async def main() -> None:
    await graph.ainvoke({}, thread_id="offer-1")
    try:
        await graph.update_state("offer-1", {"price": 90})
    except InvalidUpdateError:
        print("price: refused, quote already wrote it")
    await graph.update_state("offer-1", {"note": "check the discount"})
    result = await graph.ainvoke(None, thread_id="offer-1", resume=Resume("approved"))
    print(result.state)


asyncio.run(main())
```

```text
price: refused, quote already wrote it
price=100 note="approved, after reading 'check the discount'"
```

`quote` finished in the paused superstep, so its write to `price` cannot be changed. The edit of `note` is applied, and `review` reads it on resume.

## What is stored

The store keeps the input and every update, in the JSON form of their field types. Loading turns them back into the same types. A node is stored by its name, so a node's name is part of the thread's history.

## Rules and defaults

| Rule or setting | What happens |
|---|---|
| Run without a state store | Starts from the schema defaults and keeps nothing |
| Run without `thread_id` and without a store | Gets a generated thread id, which is never reused |
| Run with a store | Needs a `thread_id`; leaving it out raises `GraphConfigError` |
| `Graph(state_store=None)` | No store by default. A store passed as `state_store=` to `ainvoke` or to a thread method wins over the graph's |
| Input on an existing thread | A patch, merged into the thread's state through the reducers |
| `InMemoryStateStore()` | Keeps the history in the memory of this process; it is gone when the process exits |
| `FilesystemStateStore(path)` | Keeps the history in one JSON Lines file. Missing parent directories are created when the store is created |
| File header | `FilesystemStateStore` writes a header record and refuses files without one |
| File locks | The file is locked while writing, with `fcntl` on Linux and macOS and `msvcrt` on a `<path>.lock` file on Windows, so processes can share it |
| Broken last line | Left by an interrupted write; it is cut off, and its size is reported in `store.repaired_bytes` |
| A complete record that does not parse | `StateStoreError` |
| `branch_id=` | Default `"main"`. Every run and every thread method takes it |
| Missing thread | Reading it raises `UnknownThreadError`, and so does `update_state`, unless you pass `create=True` |
| `history()` of a thread that never ran | Returns an empty history |
| `update_state(thread_id, values)` | Applies the update through the reducers, as a node would, and returns the event id |
| `fork(..., name=None)` | The new branch's id is `name`, or a random id |
| Edit of a paused thread | Changing a field that a finished task of the paused superstep already wrote, without a merging reducer, raises `InvalidUpdateError` |
| Other edits of a paused thread | They are applied: the resumed node reads the edited value, and its own write to the field wins |
| `load(at=...)` and `fork()` | `load` shows the state at an event. `fork` starts a new branch there and leaves the original branch unchanged |
| What a fork continues | A fork records where the run stood at `from_`. `ainvoke(None, branch_id=...)` continues an unfinished superstep there, `resume=` answers a pause, and new input starts a new turn from that state |
| When a superstep stays unfinished | A node fails, `max_steps` or a timeout stops the run, or the caller closes a stream early |
| `invoke(None, thread_id=...)` | Continues an unfinished superstep: finished tasks keep their recorded updates and do not run again. It never starts a new turn |
| `invoke(None)` on a paused thread | Raises `ResumeError`; answer with `resume=` |
| `invoke(None)` on a completed thread | Raises `ResumeError`; pass input to start a new turn |
| `invoke(None)` after a failed superstep whose writes cannot be merged | Raises `ResumeError`, for example after a parallel write conflict |
| `invoke(None)` on a missing thread | Raises `UnknownThreadError` |
| New input on a paused thread | Raises `ResumeError`. Answer the interrupts, or fork the thread at an earlier event |
| Stored form | Input and updates are stored in the JSON form of their field types (pydantic models through `model_dump(mode="json")`) and come back as the same types |
| Values that do not survive JSON | Raise `StateStoreError` when they are written |
| Checkpoints | A full checkpoint is written when a thread starts and after every 50 stored updates, so loading does not replay the whole history |
| History records | Each gets a `uuid4` id and a `time.time()` timestamp, and the store numbers them in sequence |
| JSON key order | `nodestep.state.dump_json_object` sorts keys by default, and `sort_keys=False` keeps the order. Stored state payloads are not sorted |
| `context=` | The context of a run is never stored |
| Node names | A node's stored identity is its name. Renaming a node, or removing it, means a paused or unfinished thread that waits on it cannot continue |
