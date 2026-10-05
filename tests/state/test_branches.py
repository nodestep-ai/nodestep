from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from nodestep import (
    END,
    START,
    Command,
    Graph,
    InMemoryStateStore,
    NodeContext,
    Resume,
    Send,
    add,
    interrupt,
    node,
    subgraph,
)
from nodestep.exceptions import (
    ResumeError,
    StateStoreError,
    StateUpdateError,
    UnknownThreadError,
)
from nodestep.state import BranchRecord, HistoryEvent
from nodestep.state.integrations import FilesystemStateStore

calls: list[str] = []


class Log(BaseModel):
    log: Annotated[list[str], add] = Field(default_factory=list)
    item: str = ""


@node
def a(state: Log) -> dict:
    calls.append("a")
    return {"log": ["a"]}


@node
def b(state: Log) -> dict:
    calls.append("b")
    return {"log": ["b"]}


@node
def c(state: Log) -> dict:
    calls.append("c")
    return {"log": ["c"]}


@node
def ask(state: Log) -> dict:
    calls.append("ask")
    return {"log": [f"answer={interrupt('ok?', id='ok')}"]}


@node
def quotes(state: Log, ctx: NodeContext) -> dict:
    if "quote" not in ctx.cache:
        calls.append("quote")
        ctx.cache["quote"] = 42
    answer = interrupt("buy?", id="buy")
    return {"log": [f"{answer}@{ctx.cache['quote']}"]}


@node
def work(state: Log) -> dict:
    calls.append(f"work:{state.item}")
    return {"log": [f"work:{state.item}"]}


@node(goto=[work])
def spread(state: Log) -> Command:
    calls.append("spread")
    return Command(goto=[Send(work, {"item": "x"}), Send(work, {"item": "y"})])


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    calls.clear()


def _linear(store: InMemoryStateStore | None = None) -> Graph:
    return Graph(Log, state_store=store or InMemoryStateStore()).flow(
        START >> a, a >> b, b >> c, c >> END
    )


async def _event(
    graph: Graph,
    thread_id: str,
    event_type: str,
    node_name: str | None = None,
    *,
    branch_id: str = "main",
) -> HistoryEvent:
    history = await graph.history(thread_id, branch_id=branch_id)
    return next(
        event
        for event in history.events
        if event.type == event_type and (node_name is None or event.node == node_name)
    )


async def test_a_fork_continues_from_the_event_it_was_made_at() -> None:
    graph = _linear()
    await graph.ainvoke({}, thread_id="t")
    after_a = await _event(graph, "t", "state_delta", "a")
    fork = await graph.fork("t", from_=after_a.id)
    calls.clear()

    result = await graph.ainvoke(None, thread_id="t", branch_id=fork.id)

    assert result.data["log"] == ["a", "b", "c"]
    assert calls == ["b", "c"]
    assert (await graph.load("t"))["log"] == ["a", "b", "c"]


async def test_a_fork_before_a_write_was_committed_keeps_the_recorded_update() -> None:
    graph = _linear()
    await graph.ainvoke({}, thread_id="t")
    completed_a = await _event(graph, "t", "node_completed", "a")
    fork = await graph.fork("t", from_=completed_a.id)
    calls.clear()

    assert (await graph.load("t", branch_id=fork.id))["log"] == []
    result = await graph.ainvoke(None, thread_id="t", branch_id=fork.id)

    assert result.data["log"] == ["a", "b", "c"]
    assert calls == ["b", "c"]


async def test_a_fork_at_a_step_boundary_runs_the_next_step() -> None:
    graph = _linear()
    await graph.ainvoke({}, thread_id="t")
    boundary = await _event(graph, "t", "superstep_completed")
    fork = await graph.fork("t", from_=boundary.id)
    calls.clear()

    result = await graph.ainvoke(None, thread_id="t", branch_id=fork.id)

    assert result.data["log"] == ["a", "b", "c"]
    assert calls == ["b", "c"]


async def test_a_fork_inside_a_send_fan_out_keeps_the_payloads() -> None:
    graph = Graph(Log, state_store=InMemoryStateStore()).flow(
        START >> spread, work >> END
    )
    await graph.ainvoke({}, thread_id="t")
    first = await _event(graph, "t", "node_completed", "work")
    fork = await graph.fork("t", from_=first.id)
    calls.clear()

    result = await graph.ainvoke(None, thread_id="t", branch_id=fork.id)

    assert sorted(result.data["log"]) == ["work:x", "work:y"]
    assert len(calls) == 1
    assert calls[0] in ("work:x", "work:y")


async def test_a_fork_of_a_finished_run_has_nothing_to_continue() -> None:
    graph = _linear()
    await graph.ainvoke({}, thread_id="t")
    finished = await _event(graph, "t", "run_completed")
    fork = await graph.fork("t", from_=finished.id)

    with pytest.raises(ResumeError, match="no unfinished run"):
        await graph.ainvoke(None, thread_id="t", branch_id=fork.id)
    result = await graph.ainvoke({}, thread_id="t", branch_id=fork.id)

    assert result.data["log"] == ["a", "b", "c", "a", "b", "c"]


async def test_a_fork_of_a_paused_thread_is_resumed_on_its_own() -> None:
    graph = Graph(Log, state_store=InMemoryStateStore()).flow(
        START >> a, a >> ask, ask >> END
    )
    await graph.ainvoke({}, thread_id="t")
    pause = await _event(graph, "t", "superstep_pending")
    fork = await graph.fork("t", from_=pause.id)
    calls.clear()

    with pytest.raises(ResumeError, match=r"paused on \['ask:ok'\]"):
        await graph.ainvoke(None, thread_id="t", branch_id=fork.id)
    on_fork = await graph.ainvoke(
        None, thread_id="t", branch_id=fork.id, resume=Resume("no")
    )
    on_main = await graph.ainvoke(None, thread_id="t", resume=Resume("yes"))

    assert on_fork.data["log"] == ["a", "answer=no"]
    assert on_main.data["log"] == ["a", "answer=yes"]
    assert calls == ["ask", "ask"]


async def test_a_fork_of_a_fork_continues_from_its_own_event() -> None:
    graph = _linear()
    await graph.ainvoke({}, thread_id="t")
    after_a = await _event(graph, "t", "state_delta", "a")
    first = await graph.fork("t", from_=after_a.id, name="first")
    await graph.ainvoke(None, thread_id="t", branch_id="first")
    after_b = await _event(graph, "t", "state_delta", "b", branch_id="first")
    second = await graph.fork("t", from_=after_b.id, branch_id="first", name="second")
    calls.clear()

    result = await graph.ainvoke(None, thread_id="t", branch_id="second")

    assert result.data["log"] == ["a", "b", "c"]
    assert calls == ["c"]
    assert second.parent_branch_id == first.id
    assert second.from_event_id == after_b.id


async def test_the_start_of_a_fork_can_be_forked_again() -> None:
    graph = _linear()
    await graph.ainvoke({}, thread_id="t")
    after_a = await _event(graph, "t", "state_delta", "a")
    await graph.fork("t", from_=after_a.id, name="first")
    start = (await graph.history("t", branch_id="first")).events[0]
    await graph.fork("t", from_=start.id, branch_id="first", name="second")
    calls.clear()

    result = await graph.ainvoke(None, thread_id="t", branch_id="second")

    assert result.data["log"] == ["a", "b", "c"]
    assert calls == ["b", "c"]


async def test_a_fork_takes_a_name() -> None:
    graph = _linear()
    await graph.ainvoke({}, thread_id="t")
    after_a = await _event(graph, "t", "state_delta", "a")

    fork = await graph.fork("t", from_=after_a.id, name="draft")

    assert fork.id == "draft"
    assert (await graph.load("t", branch_id="draft"))["log"] == ["a"]


@pytest.mark.parametrize("name", ["draft", "main"])
async def test_a_taken_branch_name_raises_and_writes_nothing(name: str) -> None:
    store = InMemoryStateStore()
    graph = _linear(store)
    await graph.ainvoke({}, thread_id="t")
    after_a = await _event(graph, "t", "state_delta", "a")
    await graph.fork("t", from_=after_a.id, name="draft")
    stored = (len(store.events), len(store.checkpoints))

    with pytest.raises(StateStoreError, match=f"Branch '{name}' of thread 't'"):
        await graph.fork("t", from_=after_a.id, name=name)

    assert (len(store.events), len(store.checkpoints)) == stored
    assert [branch.id for branch in await graph.branches("t")] == ["draft"]


async def test_a_blank_branch_name_raises() -> None:
    graph = _linear()
    await graph.ainvoke({}, thread_id="t")
    after_a = await _event(graph, "t", "state_delta", "a")

    with pytest.raises(ValueError, match="must not be blank"):
        await graph.fork("t", from_=after_a.id, name="  ")

    assert await graph.branches("t") == []


async def test_fork_checks_the_event_before_writing() -> None:
    store = InMemoryStateStore()
    graph = _linear(store)
    await graph.ainvoke({}, thread_id="t")
    stored = (len(store.events), len(store.checkpoints))

    with pytest.raises(StateUpdateError, match="'bogus'"):
        await graph.fork("t", from_="bogus")

    assert await graph.branches("t") == []
    assert (len(store.events), len(store.checkpoints)) == stored


async def test_fork_of_an_unknown_thread_or_branch_raises() -> None:
    graph = _linear()
    await graph.ainvoke({}, thread_id="t")
    after_a = await _event(graph, "t", "state_delta", "a")

    with pytest.raises(UnknownThreadError, match="Thread 'missing'"):
        await graph.fork("missing", from_=after_a.id)
    with pytest.raises(UnknownThreadError, match="Branch 'typo' of thread 't'"):
        await graph.fork("t", from_=after_a.id, branch_id="typo")


async def test_branches_lists_the_forks_of_a_thread() -> None:
    graph = _linear()
    await graph.ainvoke({}, thread_id="t")
    await graph.ainvoke({}, thread_id="other")
    after_a = await _event(graph, "t", "state_delta", "a")
    first = await graph.fork("t", from_=after_a.id, name="first")
    second = await graph.fork("t", from_=after_a.id)

    branches = await graph.branches("t")

    assert branches == [first, second]
    assert all(isinstance(branch, BranchRecord) for branch in branches)
    assert [(branch.parent_branch_id, branch.from_event_id) for branch in branches] == [
        ("main", after_a.id),
        ("main", after_a.id),
    ]
    assert await graph.branches("other") == []
    assert await graph.branches("missing") == []


async def test_an_unknown_branch_raises_and_writes_nothing() -> None:
    store = InMemoryStateStore()
    graph = _linear(store)
    await graph.ainvoke({}, thread_id="t")
    stored = len(store.events)
    calls.clear()

    for call in (
        graph.load("t", branch_id="typo"),
        graph.get_state("t", branch_id="typo"),
        graph.history("t", branch_id="typo"),
        graph.update_state("t", {"item": "x"}, branch_id="typo", create=True),
        graph.ainvoke({}, thread_id="t", branch_id="typo"),
    ):
        with pytest.raises(UnknownThreadError, match="Branch 'typo' of thread 't'"):
            await call

    assert calls == []
    assert len(store.events) == stored
    assert not await graph.exists("t", branch_id="typo")


async def test_a_fork_exists() -> None:
    graph = _linear()
    await graph.ainvoke({}, thread_id="t")
    after_a = await _event(graph, "t", "state_delta", "a")

    await graph.fork("t", from_=after_a.id, name="draft")

    assert await graph.exists("t", branch_id="draft")


async def test_a_filesystem_store_keeps_named_branches(tmp_path) -> None:
    path = tmp_path / "threads.jsonl"
    graph = _linear(FilesystemStateStore(path))
    reader = FilesystemStateStore(path)
    other_store = FilesystemStateStore(path)
    other = _linear(other_store)
    await graph.ainvoke({}, thread_id="t")
    after_a = await _event(graph, "t", "state_delta", "a")
    fork = await graph.fork("t", from_=after_a.id, name="draft")

    assert await reader.list_branches("t") == [fork]
    with pytest.raises(StateStoreError, match="Branch 'draft' of thread 't'"):
        await other_store.fork("t", from_=after_a.id, name="draft")
    with pytest.raises(StateStoreError, match="Branch 'draft' of thread 't'"):
        await other.fork("t", from_=after_a.id, name="draft")
    assert await other.branches("t") == [fork]
    calls.clear()
    result = await other.ainvoke(None, thread_id="t", branch_id="draft")

    assert result.data["log"] == ["a", "b", "c"]
    assert calls == ["b", "c"]


async def test_a_fork_of_a_paused_thread_keeps_the_node_cache() -> None:
    graph = Graph(Log, state_store=InMemoryStateStore()).flow(
        START >> quotes, quotes >> END
    )
    await graph.ainvoke({}, thread_id="t")
    pause = await _event(graph, "t", "superstep_pending")
    fork = await graph.fork("t", from_=pause.id)

    result = await graph.ainvoke(
        None, thread_id="t", branch_id=fork.id, resume=Resume("yes")
    )

    assert result.data["log"] == ["yes@42"]
    assert calls == ["quote"]


async def test_a_subgraph_runs_on_a_forked_branch() -> None:
    child = Graph(Log, name="child").flow(START >> c, c >> END)
    subgraph_node = subgraph("sub", child, share=["log"], child_thread="fresh")
    graph = Graph(Log, state_store=InMemoryStateStore()).flow(
        START >> a, a >> subgraph_node, subgraph_node >> END
    )
    await graph.ainvoke({}, thread_id="t")
    after_a = await _event(graph, "t", "state_delta", "a")
    fork = await graph.fork("t", from_=after_a.id)
    calls.clear()

    result = await graph.ainvoke(None, thread_id="t", branch_id=fork.id)

    assert result.data["log"] == ["a", "c"]
    assert calls == ["c"]


@node
def after_b(state: Log) -> dict:
    calls.append("after_b")
    return {"log": ["after_b"]}


@node
def after_ask(state: Log) -> dict:
    return {"log": ["after_ask"]}


@node(goto=[b, ask])
def fans_b_ask(state: Log) -> Command:
    return Command(goto=[b, ask])


async def test_a_fork_of_a_paused_parallel_step_keeps_the_finished_sibling() -> None:
    graph = Graph(Log, state_store=InMemoryStateStore()).flow(
        START >> fans_b_ask,
        b >> after_b,
        ask >> after_ask,
        after_b >> END,
        after_ask >> END,
    )
    await graph.ainvoke({}, thread_id="t")
    pause = await _event(graph, "t", "superstep_pending")
    fork = await graph.fork("t", from_=pause.id)
    calls.clear()

    on_fork = await graph.ainvoke(
        None, thread_id="t", branch_id=fork.id, resume=Resume("no")
    )
    on_main = await graph.ainvoke(None, thread_id="t", resume=Resume("yes"))

    assert on_fork.data["log"] == ["answer=no", "b", "after_ask", "after_b"]
    assert on_main.data["log"] == ["answer=yes", "b", "after_ask", "after_b"]
    assert calls == ["ask", "after_b", "ask", "after_b"]


async def test_a_fork_paused_inside_a_subgraph_resumes_its_own_child_run() -> None:
    child = Graph(Log, name="child").flow(START >> ask, ask >> END)
    subgraph_node = subgraph("sub", child, share=["log"], child_thread="fresh")
    graph = Graph(Log, state_store=InMemoryStateStore()).flow(
        START >> a, a >> subgraph_node, subgraph_node >> END
    )
    paused = await graph.ainvoke({}, thread_id="t")
    pause = await _event(graph, "t", "superstep_pending")
    fork = await graph.fork("t", from_=pause.id)

    on_fork = await graph.ainvoke(
        None, thread_id="t", branch_id=fork.id, resume=Resume("no")
    )
    on_main = await graph.ainvoke(None, thread_id="t", resume=Resume("yes"))

    assert list(paused.interrupts) == ["sub/ask:ok"]
    assert on_fork.data["log"] == ["a", "answer=no"]
    assert on_main.data["log"] == ["a", "answer=yes"]


async def test_a_fork_of_a_fork_paused_inside_a_subgraph_resumes_its_own_child() -> (
    None
):
    child = Graph(Log, name="child").flow(START >> ask, ask >> END)
    subgraph_node = subgraph("sub", child, share=["log"], child_thread="fresh")
    graph = Graph(Log, state_store=InMemoryStateStore()).flow(
        START >> a, a >> subgraph_node, subgraph_node >> END
    )
    await graph.ainvoke({}, thread_id="t")
    pause = await _event(graph, "t", "superstep_pending")
    first = await graph.fork("t", from_=pause.id, name="first")
    start = (await graph.history("t", branch_id="first")).events[0]
    second = await graph.fork("t", from_=start.id, branch_id="first", name="second")

    on_second = await graph.ainvoke(
        None, thread_id="t", branch_id=second.id, resume=Resume("two")
    )
    on_first = await graph.ainvoke(
        None, thread_id="t", branch_id=first.id, resume=Resume("one")
    )
    on_main = await graph.ainvoke(None, thread_id="t", resume=Resume("main"))

    assert on_second.data["log"] == ["a", "answer=two"]
    assert on_first.data["log"] == ["a", "answer=one"]
    assert on_main.data["log"] == ["a", "answer=main"]


@pytest.mark.parametrize("kind", ["memory", "file"])
async def test_a_store_refuses_main_as_a_branch_name(kind: str, tmp_path) -> None:
    store = (
        InMemoryStateStore()
        if kind == "memory"
        else FilesystemStateStore(tmp_path / "threads.jsonl")
    )

    with pytest.raises(StateStoreError, match="Branch 'main' of thread 't'"):
        await store.fork("t", from_="e", name="main")

    assert await store.list_branches("t") == []
