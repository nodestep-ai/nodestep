import asyncio
import datetime
from typing import Annotated, Any
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field

from nodestep import (
    END,
    START,
    Command,
    Graph,
    InMemoryStateStore,
    Replace,
    Resume,
    add,
    interrupt,
    node,
)
from nodestep.core.graph import GraphResult
from nodestep.exceptions import (
    GraphConfigError,
    InvalidUpdateError,
    ResumeError,
    RunLimitExceededError,
    StateStoreError,
    StateUpdateError,
    UnknownThreadError,
)
from nodestep.state import History, HistoryEvent, StateSnapshot


class Request(BaseModel):
    query: str
    count: int = 0
    run_tag: str = Field(default_factory=lambda: uuid4().hex)
    log: Annotated[list[str], add] = Field(default_factory=list)


@node
def echo(state: Request) -> dict:
    return {"log": [state.query]}


def _requests(store: InMemoryStateStore | None = None) -> Graph:
    return Graph(Request, state_store=store or InMemoryStateStore()).flow(
        START >> echo, echo >> END
    )


async def test_load_and_get_state_of_an_unknown_thread_raise() -> None:
    graph = _requests()

    with pytest.raises(UnknownThreadError, match="Thread 'missing'") as loaded:
        await graph.load("missing")
    with pytest.raises(UnknownThreadError, match="Thread 'missing'"):
        await graph.get_state("missing")

    assert loaded.value.thread_id == "missing"


async def test_history_of_an_unknown_thread_is_empty() -> None:
    history = await _requests().history("missing")

    assert isinstance(history, History)
    assert history.events == []
    assert history.checkpoints == []


async def test_exists_tells_whether_a_thread_has_history() -> None:
    graph = _requests()

    assert not await graph.exists("t")
    await graph.ainvoke({"query": "q"}, thread_id="t")

    assert await graph.exists("t")
    assert not await graph.exists("t", branch_id="draft")
    assert not await graph.exists("other")


async def test_update_state_on_an_unknown_thread_raises_and_writes_nothing() -> None:
    store = InMemoryStateStore()
    graph = _requests(store)

    with pytest.raises(UnknownThreadError, match="Thread 'never-ran'"):
        await graph.update_state("never-ran", {"log": ["manual"]})

    assert store.events == []
    assert store.checkpoints == []


async def test_update_state_creates_a_thread_when_asked() -> None:
    graph = _requests()

    await graph.update_state(
        "new", {"query": "q", "count": "7", "log": ["manual"]}, create=True
    )
    first = await graph.load("new")
    second = await graph.load("new")

    assert first == second
    assert first["count"] == 7
    assert first["log"] == ["manual"]
    result = await graph.ainvoke({"query": "again"}, thread_id="new")
    assert result.data["log"] == ["manual", "again"]
    assert result.data["run_tag"] == first["run_tag"]


async def test_a_created_thread_stores_the_coerced_value() -> None:
    store = InMemoryStateStore()
    graph = _requests(store)

    await graph.update_state("new", {"query": "q", "count": "7"}, create=True)

    deltas = [event for event in store.events if event.type == "state_delta"]
    assert len(deltas) == 1
    assert deltas[0].data_json is not None
    assert '"count":7' in deltas[0].data_json


async def test_a_created_thread_has_nothing_to_continue() -> None:
    graph = _requests()
    await graph.update_state("new", {"query": "q"}, create=True)

    with pytest.raises(ResumeError, match="no unfinished run"):
        await graph.ainvoke(None, thread_id="new")


async def test_creating_a_thread_checks_it_like_the_input_of_a_new_thread() -> None:
    store = InMemoryStateStore()
    graph = _requests(store)

    with pytest.raises(StateUpdateError, match=r"from update_state.*'query'"):
        await graph.update_state("new", {"log": ["manual"]}, create=True)

    assert store.events == []
    assert not await graph.exists("new")


class Loose(BaseModel):
    when: Any = None


@node
def keeps(state: Loose) -> None:
    return None


async def test_creating_a_thread_names_update_state_in_its_errors() -> None:
    graph = Graph(Loose, state_store=InMemoryStateStore()).flow(
        START >> keeps, keeps >> END
    )

    with pytest.raises(
        StateStoreError, match="State field 'when' got a value from update_state "
    ):
        await graph.update_state(
            "new", {"when": datetime.datetime(2020, 1, 1)}, create=True
        )


async def test_creating_a_thread_refuses_an_empty_thread_id() -> None:
    store = InMemoryStateStore()
    graph = _requests(store)

    with pytest.raises(ValueError, match="thread_id must not be empty"):
        await graph.update_state("", {"query": "q"}, create=True)

    assert store.events == []


async def test_create_on_an_existing_thread_is_an_ordinary_update() -> None:
    graph = _requests()
    await graph.ainvoke({"query": "q"}, thread_id="t")

    await graph.update_state("t", {"log": ["manual"]}, create=True)

    assert (await graph.load("t"))["log"] == ["q", "manual"]


async def test_get_state_reports_the_sequence_of_its_event() -> None:
    graph = _requests()
    await graph.ainvoke({"query": "q"}, thread_id="t")
    events = (await graph.history("t")).events
    delta = next(event for event in events if event.type == "state_delta")

    latest = await graph.get_state("t")
    at_delta = await graph.get_state("t", at=delta.id)

    assert isinstance(latest, StateSnapshot)
    assert latest.sequence == events[-1].sequence
    assert at_delta.sequence == delta.sequence
    assert at_delta.value.log == []
    assert set(StateSnapshot.model_fields) == {"value", "sequence"}


async def test_get_state_at_an_unknown_event_raises() -> None:
    graph = _requests()
    await graph.ainvoke({"query": "q"}, thread_id="t")

    with pytest.raises(StateUpdateError, match="Event id 'bogus' not found"):
        await graph.get_state("t", at="bogus")


class _GrowsAfterRead(InMemoryStateStore):
    def __init__(self) -> None:
        super().__init__()
        self.late: HistoryEvent | None = None

    async def get_history(self, thread_id: str, branch_id: str = "main") -> History:
        history = await super().get_history(thread_id, branch_id)
        if self.late is not None:
            late, self.late = self.late, None
            await self.append_next_event(late)
        return history


async def test_get_state_reads_its_value_and_sequence_at_one_event() -> None:
    store = _GrowsAfterRead()
    graph = _requests(store)
    await graph.ainvoke({"query": "q"}, thread_id="t")
    before = (await graph.history("t")).events[-1]
    store.late = HistoryEvent(
        thread_id="t",
        sequence=0,
        type="state_delta",
        data_json='{"update":{"log":["late"]}}',
    )

    snapshot = await graph.get_state("t")

    assert snapshot.sequence == before.sequence
    assert snapshot.value.log == ["q"]
    assert (await graph.load("t"))["log"] == ["q", "late"]


def test_a_graph_result_has_no_snapshot() -> None:
    assert not hasattr(GraphResult, "snapshot")


async def test_every_read_and_write_takes_a_per_call_store() -> None:
    graph_store, run_store = InMemoryStateStore(), InMemoryStateStore()
    graph = _requests(graph_store)
    await graph.ainvoke({"query": "q"}, thread_id="t", state_store=run_store)
    delta = next(event for event in run_store.events if event.type == "state_delta")

    assert await graph.exists("t", state_store=run_store)
    assert (await graph.load("t", state_store=run_store))["log"] == ["q"]
    assert (await graph.get_state("t", state_store=run_store)).value.log == ["q"]
    assert (await graph.history("t", state_store=run_store)).events
    await graph.update_state("t", {"log": ["manual"]}, state_store=run_store)
    fork = await graph.fork("t", from_=delta.id, state_store=run_store)
    assert await graph.branches("t", state_store=run_store) == [fork]
    assert (await graph.load("t", state_store=run_store))["log"] == ["q", "manual"]

    assert graph_store.events == []
    assert graph_store.checkpoints == []
    assert not await graph.exists("t")


async def test_the_read_and_write_api_needs_a_store() -> None:
    graph = Graph(Request).flow(START >> echo, echo >> END)

    for call in (
        graph.exists("t"),
        graph.history("t"),
        graph.branches("t"),
        graph.load("t"),
        graph.get_state("t"),
        graph.update_state("t", {"log": []}, create=True),
        graph.fork("t", from_="e"),
    ):
        with pytest.raises(GraphConfigError, match="no state_store"):
            await call


class Parent(BaseModel):
    x: int = 0
    note: str = ""
    log: Annotated[list[str], add] = Field(default_factory=list)


flaky = {"fail": False}


@pytest.fixture(autouse=True)
def _reset_flaky() -> None:
    flaky["fail"] = False


@node
def writer(state: Parent) -> dict:
    return {"x": 1, "log": ["writer"]}


@node
def asker(state: Parent) -> dict:
    answer = interrupt("?", id="q")
    return {"log": [f"asker saw x={state.x} answer={answer}"]}


@node
def claimer(state: Parent) -> dict:
    answer = interrupt("?", id="q")
    return {"x": 5, "log": [f"claimer {answer}"]}


@node
def fragile(state: Parent) -> dict:
    if flaky["fail"]:
        flaky["fail"] = False
        raise RuntimeError("fragile failed")
    return {"log": ["fragile"]}


@node
def fragile_noter(state: Parent) -> dict:
    if flaky["fail"]:
        flaky["fail"] = False
        raise RuntimeError("fragile failed")
    return {"note": f"{state.note}!"}


@node
def fragile_claimer(state: Parent) -> dict:
    answer = interrupt("?", id="q")
    if flaky["fail"]:
        flaky["fail"] = False
        raise RuntimeError("fragile failed")
    return {"x": 5, "log": [f"claimer {answer}"]}


@node
def resetter(state: Parent) -> dict:
    return {"log": Replace(["reset"])}


@node
def join(state: Parent) -> dict:
    return {"log": [f"join saw x={state.x}"]}


def _fan_graph(first, second) -> Graph:
    @node(name="fan", goto=[first, second])
    def fan(state: Parent) -> Command:
        return Command(goto=[first, second])

    return Graph(Parent, state_store=InMemoryStateStore()).flow(
        START >> fan, first >> join, second >> join, join >> END
    )


async def test_a_conflicting_manual_update_while_paused_is_refused() -> None:
    graph = _fan_graph(writer, asker)
    await graph.ainvoke({}, thread_id="p")

    with pytest.raises(InvalidUpdateError, match="'x'") as info:
        await graph.update_state("p", {"x": 42})

    assert info.value.nodes == ["writer"]
    assert info.value.superstep == "paused"
    assert (await graph.load("p"))["x"] == 0
    result = await graph.ainvoke(None, thread_id="p", resume=Resume("ok"))
    assert result.data["x"] == 1


async def test_a_manual_update_conflict_names_update_state_and_the_node() -> None:
    graph = _fan_graph(writer, asker)
    await graph.ainvoke({}, thread_id="p")

    with pytest.raises(
        InvalidUpdateError,
        match=(
            r"^update_state conflicts with the write of node 'writer' to field 'x' "
            r"in the paused superstep: the field has no merging reducer "
            r"\(replace/default\), so that write would silently overwrite the "
            r"update\. Update the field after the superstep has finished\.$"
        ),
    ):
        await graph.update_state("p", {"x": 42})


async def test_a_manual_append_conflicts_with_a_finished_sibling_replace() -> None:
    graph = _fan_graph(resetter, asker)
    await graph.ainvoke({}, thread_id="p")

    with pytest.raises(InvalidUpdateError, match="Replace") as info:
        await graph.update_state("p", {"log": ["human note"]})

    assert info.value.field == "log"
    assert info.value.nodes == ["resetter"]
    assert info.value.replacement
    assert (await graph.load("p"))["log"] == []


async def test_a_manual_replace_conflicts_with_a_finished_sibling_append() -> None:
    graph = _fan_graph(writer, asker)
    await graph.ainvoke({}, thread_id="p")

    with pytest.raises(InvalidUpdateError, match="Replace") as info:
        await graph.update_state("p", {"log": Replace(["human note"])})

    assert info.value.field == "log"
    assert info.value.nodes == ["writer"]


@node
def edits_x(state: Parent) -> dict:
    answer = interrupt("?", id="q")
    return {"x": state.x + 1, "log": [f"edits saw x={state.x} answer={answer}"]}


async def test_the_resumed_task_reads_a_manual_update_made_while_paused() -> None:
    graph = Graph(Parent, state_store=InMemoryStateStore()).flow(
        START >> edits_x, edits_x >> END
    )
    await graph.ainvoke({}, thread_id="p")
    await graph.update_state("p", {"x": 41})

    result = await graph.ainvoke(None, thread_id="p", resume=Resume("ok"))

    assert result.status == "completed"
    assert result.data["x"] == 42
    assert result.data["log"] == ["edits saw x=41 answer=ok"]


async def test_the_resumed_task_write_wins_over_a_manual_update_while_paused() -> None:
    graph = Graph(Parent, state_store=InMemoryStateStore()).flow(
        START >> claimer, claimer >> END
    )
    await graph.ainvoke({}, thread_id="p")
    await graph.update_state("p", {"x": 42, "log": ["manual"]})

    result = await graph.ainvoke(None, thread_id="p", resume=Resume("ok"))

    assert result.status == "completed"
    assert result.data["x"] == 5
    assert result.data["log"] == ["manual", "claimer ok"]
    assert (await graph.load("p"))["x"] == 5


async def test_a_fork_after_a_manual_update_resumes_with_the_task_write() -> None:
    graph = Graph(Parent, state_store=InMemoryStateStore()).flow(
        START >> claimer, claimer >> END
    )
    await graph.ainvoke({}, thread_id="p")
    manual = await graph.update_state("p", {"x": 42})
    fork = await graph.fork("p", from_=manual)

    result = await graph.ainvoke(
        None, thread_id="p", branch_id=fork.id, resume=Resume("ok")
    )

    assert result.data["x"] == 5
    assert (await graph.load("p"))["x"] == 42


async def test_a_manual_append_while_paused_is_merged() -> None:
    graph = _fan_graph(writer, asker)
    await graph.ainvoke({}, thread_id="p")
    await graph.update_state("p", {"log": ["manual"]})

    result = await graph.ainvoke(None, thread_id="p", resume=Resume("ok"))

    assert result.data["x"] == 1
    assert result.data["log"] == [
        "manual",
        "asker saw x=0 answer=ok",
        "writer",
        "join saw x=1",
    ]


@node(name="writer")
async def quick_writer(state: Parent) -> dict:
    return {"x": 1, "log": ["writer"]}


@node(name="fragile")
async def late_fragile(state: Parent) -> dict:
    await asyncio.sleep(0.05)
    raise RuntimeError("fragile failed")


async def test_a_conflicting_manual_update_on_an_unfinished_step_is_refused() -> None:
    graph = _fan_graph(quick_writer, late_fragile)
    with pytest.raises(RuntimeError, match="fragile failed"):
        await graph.ainvoke({}, thread_id="u")

    with pytest.raises(InvalidUpdateError, match="'x'") as info:
        await graph.update_state("u", {"x": 42})

    assert info.value.nodes == ["writer"]
    assert info.value.superstep == "unfinished"


async def test_an_unfinished_task_reads_a_manual_update_of_its_field() -> None:
    graph = _fan_graph(writer, fragile_noter)
    flaky["fail"] = True
    with pytest.raises(RuntimeError, match="fragile failed"):
        await graph.ainvoke({}, thread_id="u")
    await graph.update_state("u", {"note": "manual"})

    result = await graph.ainvoke(None, thread_id="u")

    assert result.data["note"] == "manual!"
    assert result.data["x"] == 1


@node
def starts(state: Parent) -> dict:
    return {"log": ["starts"]}


@node
def bumps(state: Parent) -> dict:
    return {"x": state.x + 1}


async def test_a_manual_update_before_a_step_runs_is_read_by_it() -> None:
    graph = Graph(Parent, state_store=InMemoryStateStore(), max_steps=1).flow(
        START >> starts, starts >> bumps, bumps >> END
    )
    with pytest.raises(RunLimitExceededError):
        await graph.ainvoke({}, thread_id="u")
    await graph.update_state("u", {"x": 41})

    result = await graph.ainvoke(None, thread_id="u")

    assert result.data["x"] == 42
    assert (await graph.load("u"))["x"] == 42


async def test_a_continued_resumed_step_writes_over_a_manual_update() -> None:
    graph = Graph(Parent, state_store=InMemoryStateStore()).flow(
        START >> fragile_claimer, fragile_claimer >> END
    )
    await graph.ainvoke({}, thread_id="p")
    await graph.update_state("p", {"x": 42})
    flaky["fail"] = True
    with pytest.raises(RuntimeError, match="fragile failed"):
        await graph.ainvoke(None, thread_id="p", resume=Resume("ok"))

    result = await graph.ainvoke(None, thread_id="p")

    assert result.data["x"] == 5
    assert result.data["log"] == ["claimer ok"]


async def test_a_manual_update_between_runs_is_not_a_writer_of_the_next_run() -> None:
    graph = _fan_graph(writer, asker)
    await graph.ainvoke({}, thread_id="p")
    await graph.ainvoke(None, thread_id="p", resume=Resume("ok"))
    await graph.update_state("p", {"x": 42})

    result = await graph.ainvoke({"log": []}, thread_id="p")

    assert result.status == "interrupted"
    assert result.data["x"] == 42


@node
def asks_only(state: Parent) -> dict:
    answer = interrupt("?", id="q")
    return {"log": [f"asked {answer}"]}


@node
def sets_x(state: Parent) -> dict:
    return {"x": 7}


async def test_a_manual_update_counts_only_in_the_pending_superstep() -> None:
    graph = Graph(Parent, state_store=InMemoryStateStore()).flow(
        START >> asks_only, asks_only >> sets_x, sets_x >> END
    )
    await graph.ainvoke({}, thread_id="p")
    await graph.update_state("p", {"x": 42})

    result = await graph.ainvoke(None, thread_id="p", resume=Resume("ok"))

    assert result.data["x"] == 7
    assert result.data["log"] == ["asked ok"]


@node
def resets_log(state: Parent) -> dict:
    answer = interrupt("?", id="q")
    return {"log": Replace([f"reset {answer}"])}


async def test_a_resumed_task_replace_wins_over_a_manual_replace() -> None:
    graph = Graph(Parent, state_store=InMemoryStateStore()).flow(
        START >> resets_log, resets_log >> END
    )
    await graph.ainvoke({}, thread_id="p")
    await graph.update_state("p", {"log": Replace(["manual"])})

    result = await graph.ainvoke(None, thread_id="p", resume=Resume("ok"))

    assert result.data["log"] == ["reset ok"]
