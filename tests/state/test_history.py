from typing import Annotated

import pytest
from pydantic import BaseModel, Field, SecretStr

from nodestep.core.command import END, START, Command, Send, interrupt
from nodestep.core.graph import Graph
from nodestep.core.node import node
from nodestep.exceptions import StateUpdateError
from nodestep.state import (
    BranchRecord,
    CheckpointRecord,
    History,
    HistoryEvent,
)
from nodestep.state.integrations import FilesystemStateStore, InMemoryStateStore
from nodestep.utils.reducers import add


def test_history_public_models_import_from_top_level_package() -> None:
    assert History is not None
    assert HistoryEvent is not None
    assert CheckpointRecord is not None
    assert BranchRecord is not None


class State(BaseModel):
    counter: int = 0
    log: Annotated[list[str], add] = Field(default_factory=list)


@node
def step_one(state: State) -> dict:
    return {"counter": state.counter + 1, "log": ["one"]}


@node
def step_two(state: State) -> dict:
    return {"counter": state.counter + 1, "log": ["two"]}


@node(name="approval_interrupt")
def approval_interrupt(_state: State) -> dict:
    interrupt({"prompt": "confirm?"}, id="confirm")
    return {"counter": 1}


async def test_history_records_run_lifecycle() -> None:
    store = InMemoryStateStore()
    graph = Graph(State, state_store=store).flow(
        START >> step_one, step_one >> step_two, step_two >> END
    )

    result = await graph.ainvoke({}, thread_id="thread-1")

    types = [event.type for event in store.events if event.thread_id == "thread-1"]
    assert "run_started" in types
    assert "node_started" in types
    assert "node_completed" in types
    assert "run_completed" in types
    assert result.state.counter == 2


async def test_load_replays_state_deltas_to_event() -> None:
    store = InMemoryStateStore()
    graph = Graph(State, state_store=store).flow(
        START >> step_one, step_one >> step_two, step_two >> END
    )

    await graph.ainvoke({}, thread_id="thread-2")

    history = await graph.history("thread-2")
    first_state_delta = next(
        event
        for event in history.events
        if event.type == "state_delta" and event.node == "step_one"
    )
    state_at_first = await graph.load("thread-2", at=first_state_delta.id)
    assert state_at_first["counter"] == 1


async def test_json_state_store_persists_to_disk(tmp_path) -> None:
    store_path = tmp_path / "history.json"
    store = FilesystemStateStore(store_path)
    graph = Graph(State, state_store=store).flow(
        START >> step_one, step_one >> step_two, step_two >> END
    )

    await graph.ainvoke({}, thread_id="thread-3")
    assert store_path.exists()
    text = store_path.read_text(encoding="utf-8")
    assert "run_started" in text
    assert ", " not in text


async def test_filesystem_store_concurrent_appends_are_consistent(tmp_path) -> None:
    import asyncio
    import json as _json

    store_path = tmp_path / "concurrent.json"
    store = FilesystemStateStore(store_path)
    events = [
        HistoryEvent(thread_id="t", sequence=i, type="state_delta") for i in range(50)
    ]

    await asyncio.gather(*[store.append_event(event) for event in events])

    lines = [
        _json.loads(line)
        for line in store_path.read_text(encoding="utf-8").splitlines()
    ]
    assert sum(1 for line in lines if line["kind"] == "event") == 50

    reloaded = FilesystemStateStore(store_path)
    history = await reloaded.get_history("t")
    assert len(history.events) == 50


async def test_fork_creates_branch_without_mutating_main() -> None:
    store = InMemoryStateStore()
    graph = Graph(State, state_store=store).flow(
        START >> step_one, step_one >> step_two, step_two >> END
    )

    await graph.ainvoke({}, thread_id="thread-4")
    history = await graph.history("thread-4")
    first_state_delta = next(
        event for event in history.events if event.type == "state_delta"
    )
    branch = await graph.fork("thread-4", from_=first_state_delta.id)

    main_history = await graph.history("thread-4")
    assert all(event.branch_id == "main" for event in main_history.events)

    branch_checkpoints = [
        checkpoint
        for checkpoint in store.checkpoints
        if checkpoint.branch_id == branch.id
    ]
    assert len(branch_checkpoints) == 1


@node
def boom(state: State) -> dict:
    raise RuntimeError("explode")


async def test_node_exception_records_error_event() -> None:
    store = InMemoryStateStore()
    graph = Graph(State, state_store=store).flow(START >> boom, boom >> END)

    with pytest.raises(RuntimeError):
        await graph.ainvoke({}, thread_id="thread-err")

    error_events = [event for event in store.events if event.type == "error"]
    assert len(error_events) == 1
    assert error_events[0].error_type == "RuntimeError"


async def test_interrupt_records_event_and_returns_interrupted_status() -> None:
    store = InMemoryStateStore()

    graph = Graph(State, state_store=store).flow(
        START >> approval_interrupt, approval_interrupt >> END
    )

    result = await graph.ainvoke({}, thread_id="thread-int")

    assert result.status == "interrupted"
    interrupted_events = [
        event for event in store.events if event.type == "interrupted"
    ]
    assert len(interrupted_events) == 1


async def test_load_replays_from_nearest_sparse_checkpoint() -> None:
    store = InMemoryStateStore()
    graph = Graph(State, state_store=store).flow(
        START >> step_one,
        step_one >> step_two,
        step_two >> END,
    )

    await graph.ainvoke({}, thread_id="thread-checkpoint")

    history = await graph.history("thread-checkpoint")
    last_delta = [event for event in history.events if event.type == "state_delta"][-1]

    loaded = await graph.load("thread-checkpoint", at=last_delta.id)

    assert loaded["counter"] == 2
    assert loaded["log"] == ["one", "two"]


async def test_fork_loads_original_state_at_branch_point() -> None:
    store = InMemoryStateStore()
    graph = Graph(State, state_store=store).flow(
        START >> step_one, step_one >> step_two, step_two >> END
    )

    await graph.ainvoke({}, thread_id="thread-fork-load")
    history = await graph.history("thread-fork-load")
    first_delta = next(
        event
        for event in history.events
        if event.type == "state_delta" and event.node == "step_one"
    )
    branch = await graph.fork("thread-fork-load", from_=first_delta.id)

    loaded = await graph.load("thread-fork-load", branch_id=branch.id)

    assert loaded["counter"] == 1
    assert loaded["log"] == ["one"]


class RuntimeState(BaseModel):
    prompt: str = ""
    password: SecretStr | None = None


@node
def uses_prompt(state: RuntimeState) -> dict:
    return {"prompt": "used"}


async def test_a_secret_in_the_input_is_refused_before_anything_is_stored() -> None:
    store = InMemoryStateStore()
    graph = Graph(RuntimeState, state_store=store).flow(
        START >> uses_prompt, uses_prompt >> END
    )

    with pytest.raises(StateUpdateError, match=r"'password'.*context="):
        await graph.ainvoke({"password": SecretStr("db-password")}, thread_id="s")

    assert store.events == []
    assert store.checkpoints == []


def test_filesystem_store_rejects_directory_path(tmp_path) -> None:
    from nodestep.exceptions import StateStoreError

    with pytest.raises(StateStoreError, match="not a file"):
        FilesystemStateStore(tmp_path)


def test_filesystem_store_corrupt_file_raises_clear_error(tmp_path) -> None:
    from nodestep.exceptions import StateStoreError

    store_path = tmp_path / "corrupt.json"
    store_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(StateStoreError, match=r"corrupt\.json"):
        FilesystemStateStore(store_path)


async def test_inmemory_store_concurrent_appends_are_consistent() -> None:
    import asyncio

    store = InMemoryStateStore()
    events = [
        HistoryEvent(thread_id="t", sequence=i, type="state_delta") for i in range(50)
    ]
    await asyncio.gather(*[store.append_event(event) for event in events])
    history = await store.get_history("t")
    assert len(history.events) == 50


async def test_load_unknown_at_id_raises_clear_error() -> None:
    from nodestep.exceptions import StateUpdateError

    store = InMemoryStateStore()
    graph = Graph(State, state_store=store).flow(
        START >> step_one, step_one >> step_two, step_two >> END
    )
    await graph.ainvoke({}, thread_id="thread-unknown-at")
    with pytest.raises(StateUpdateError, match="does-not-exist"):
        await graph.load("thread-unknown-at", at="does-not-exist")


async def test_load_corrupt_state_delta_raises_clear_error() -> None:
    from nodestep.exceptions import StateUpdateError
    from nodestep.state.replay import load_history_state
    from nodestep.state.schema import StateSchema

    store = InMemoryStateStore()
    schema = StateSchema.from_type(State)
    await store.append_event(
        HistoryEvent(
            id="evt-corrupt",
            thread_id="t-corrupt",
            sequence=1,
            type="state_delta",
            data_json="{not json",
        )
    )
    with pytest.raises(StateUpdateError, match="evt-corrupt"):
        await load_history_state(store, schema, "t-corrupt")


class ConflictState(BaseModel):
    x: int = 0


@node
def write_one(state: ConflictState) -> dict:
    return {"x": 1}


@node
def write_two(state: ConflictState) -> dict:
    return {"x": 2}


@node(goto=[write_one, write_two])
def fan_conflict(state: ConflictState) -> Command:
    return Command(goto=[Send(write_one), Send(write_two)])


async def test_rejected_superstep_is_not_persisted() -> None:
    from nodestep.exceptions import InvalidUpdateError

    graph = Graph(ConflictState, state_store=InMemoryStateStore()).flow(
        START >> fan_conflict,
        write_one >> END,
        write_two >> END,
    )

    with pytest.raises(InvalidUpdateError):
        await graph.ainvoke({"x": 0}, thread_id="conflict")

    assert (await graph.load("conflict"))["x"] == 0


async def test_two_store_instances_on_one_file_keep_both_threads(tmp_path) -> None:
    path = tmp_path / "shared.jsonl"
    first = Graph(State, state_store=FilesystemStateStore(path)).flow(
        START >> step_one, step_one >> END
    )
    second = Graph(State, state_store=FilesystemStateStore(path)).flow(
        START >> step_one, step_one >> END
    )

    await first.ainvoke({}, thread_id="A")
    await second.ainvoke({}, thread_id="B")
    await first.ainvoke({}, thread_id="A")

    fresh = FilesystemStateStore(path)
    assert sorted({event.thread_id for event in fresh.events}) == ["A", "B"]
    reloaded = Graph(State, state_store=fresh).flow(START >> step_one, step_one >> END)
    assert (await reloaded.load("A"))["counter"] == 2
    sequences = [event.sequence for event in (await fresh.get_history("A")).events]
    assert sequences == sorted(set(sequences))


async def test_many_runs_append_without_rewriting_the_file(tmp_path) -> None:
    import time

    path = tmp_path / "long.jsonl"
    graph = Graph(State, state_store=FilesystemStateStore(path)).flow(
        START >> step_one, step_one >> END
    )
    await graph.ainvoke({}, thread_id="chat")
    prefix = path.read_bytes()
    started = time.monotonic()

    for _ in range(300):
        await graph.ainvoke({}, thread_id="chat")

    assert time.monotonic() - started < 10
    assert path.read_bytes().startswith(prefix)
    assert (await graph.load("chat"))["counter"] == 301


async def test_truncated_last_line_is_ignored_and_repaired(tmp_path) -> None:
    path = tmp_path / "crash.jsonl"
    graph = Graph(State, state_store=FilesystemStateStore(path)).flow(
        START >> step_one, step_one >> END
    )
    await graph.ainvoke({}, thread_id="t")
    with path.open("ab") as handle:
        handle.write(b'{"kind":"event","data":{"thread_id":"t","seq')

    reopened = Graph(State, state_store=FilesystemStateStore(path)).flow(
        START >> step_one, step_one >> END
    )
    assert (await reopened.load("t"))["counter"] == 1
    await reopened.ainvoke({}, thread_id="t")

    final = Graph(State, state_store=FilesystemStateStore(path)).flow(
        START >> step_one, step_one >> END
    )
    assert (await final.load("t"))["counter"] == 2


_CROSS_LOOP_SCRIPT = """
import asyncio
import faulthandler
import sys
import tempfile
from pathlib import Path
from typing import Annotated, TypedDict

from nodestep import END, START, Command, FilesystemStateStore, Graph, InMemoryStateStore, Send, add, node

faulthandler.dump_traceback_later(30, exit=True)


class S(TypedDict, total=False):
    log: Annotated[list[str], add]


def build(store):
    @node
    def child_step(state: S) -> dict:
        return {"log": ["child"]}

    child = Graph(S, name="child", state_store=store).flow(START >> child_step, child_step >> END)

    @node
    def sync_caller(state: S) -> dict:
        for index in range(10):
            child.invoke({"log": []}, thread_id=f"child-{index}")
        return {"log": ["sync_caller"]}

    @node
    async def chatty(state: S) -> dict:
        await asyncio.sleep(0.01)
        return {"log": ["chatty"]}

    @node(goto=[sync_caller, chatty])
    def fan(state: S) -> Command:
        return Command(goto=[Send(sync_caller, {}), *[Send(chatty, {}) for _ in range(20)]])

    return Graph(S, state_store=store).flow(START >> fan, fan >> END, sync_caller >> END, chatty >> END)


async def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        for store in (InMemoryStateStore(), FilesystemStateStore(Path(directory) / "h.jsonl")):
            result = await build(store).ainvoke({"log": []}, thread_id="parent")
            assert result.status == "completed", result.status
            assert len(result.data["log"]) == 21, result.data
    print("ok")


asyncio.run(main())
"""


def test_stores_are_safe_to_share_across_event_loops() -> None:
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-c", _CROSS_LOOP_SCRIPT],
        capture_output=True,
        text=True,
        timeout=90,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip().endswith("ok")


async def test_long_thread_with_growing_messages_stays_small(tmp_path) -> None:
    import time

    from nodestep.chat import AIMessage, HumanMessage, Message
    from nodestep.utils.reducers import add_messages

    class Chat(BaseModel):
        messages: Annotated[list[Message], add_messages] = Field(default_factory=list)

    @node
    def reply(state: Chat) -> dict:
        return {"messages": [AIMessage(content="r" * 200)]}

    path = tmp_path / "chat.jsonl"
    graph = Graph(Chat, state_store=FilesystemStateStore(path)).flow(
        START >> reply, reply >> END
    )
    started = time.monotonic()

    for turn in range(150):
        await graph.ainvoke(
            {"messages": [HumanMessage(content=f"q{turn}" * 50)]}, thread_id="t"
        )

    reopened = Graph(Chat, state_store=FilesystemStateStore(path)).flow(
        START >> reply, reply >> END
    )
    assert len((await reopened.load("t"))["messages"]) == 300
    assert path.stat().st_size < 1_500_000
    assert time.monotonic() - started < 10


async def test_torn_first_record_does_not_brick_the_store(tmp_path) -> None:
    path = tmp_path / "torn.jsonl"
    path.write_bytes(
        b'{"kind":"nodestep-state-store","version":1}\n'
        b'{"kind":"event","data":{"thread_id":"t","seq'
    )

    graph = Graph(State, state_store=FilesystemStateStore(path)).flow(
        START >> step_one, step_one >> END
    )
    await graph.ainvoke({}, thread_id="t")

    reopened = Graph(State, state_store=FilesystemStateStore(path)).flow(
        START >> step_one, step_one >> END
    )
    assert (await reopened.load("t"))["counter"] == 1
