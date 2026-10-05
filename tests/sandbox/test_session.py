import asyncio
import json
import threading
from collections.abc import AsyncIterator
from typing import Any, TypedDict

import pytest

from nodestep import END, START, Graph, InMemoryStateStore, StreamEvent, node
from nodestep.core.stream import FinalEventData
from nodestep_sandbox.errors import RunStateError
from nodestep_sandbox.runs import Run
from nodestep_sandbox.session import STREAM_MODES, Sandbox


async def finish(run: Run) -> Run:
    async for _ in run.follow():
        pass
    return run


def loads(text: str | None) -> Any:
    assert text is not None
    return json.loads(text)


def test_runs_ask_for_updates_custom_and_debug_events() -> None:
    assert STREAM_MODES == ("updates", "custom", "debug")


async def test_a_run_records_every_event_as_a_row(graphs: Any) -> None:
    run = await finish(Sandbox(graphs.echo()).start({"text": "hi"}))
    assert run.status == "completed"
    assert [(row.kind, row.title) for row in run.rows] == [
        ("debug", "node_input"),
        ("custom", "custom"),
        ("debug", "node_output"),
        ("debug", "routes"),
        ("debug", "state_merge"),
        ("update", "update"),
    ]
    assert loads(run.rows[1].data) == {"heard": "hi"}
    assert run.rows[3].summary == "next: END"
    update = run.rows[-1]
    assert (update.node, update.step, update.task) == ("shout", 0, 0)
    assert loads(update.data) == {"text": "HI", "log": ["shout"]}
    assert loads(update.state_after) == {"text": "HI", "log": ["shout"]}
    assert loads(run.final_state) == {"text": "HI", "log": ["shout"]}
    assert run.finished_at is not None


async def test_a_task_record_keeps_state_before_update_after_and_timing(
    graphs: Any,
) -> None:
    run = await finish(Sandbox(graphs.echo()).start({"text": "hi"}))
    [task] = run.tasks
    assert (task.node, task.step, task.task_id) == ("shout", 0, "shout")
    assert loads(task.state_before)["text"] == "hi"
    assert loads(task.update) == {"text": "HI", "log": ["shout"]}
    assert task.state_after == run.rows[-1].state_after
    assert task.elapsed is not None
    assert task.elapsed >= 0


async def test_follow_yields_rows_while_the_run_is_still_going() -> None:
    gate = asyncio.Event()

    async def events() -> AsyncIterator[StreamEvent]:
        yield StreamEvent(mode="custom", data={"n": 1}, node="a", step=0)
        await gate.wait()
        yield StreamEvent(
            mode="final", data=FinalEventData(state={"n": 1}, thread_id="t")
        )

    run = Run(run_id="r", thread_id="t", kind="start", input_json="{}")
    consumer = asyncio.create_task(run.consume(events()))
    rows = run.follow()
    first = await anext(rows)
    assert first.kind == "custom"
    assert run.finished is False
    gate.set()
    assert [row async for row in rows] == []
    await consumer
    assert run.status == "completed"


async def test_a_late_follower_gets_every_row_and_stops(graphs: Any) -> None:
    run = await finish(Sandbox(graphs.echo()).start({"text": "hi"}))
    assert [row async for row in run.follow()] == run.rows


async def test_two_followers_see_the_same_rows(graphs: Any) -> None:
    run = Sandbox(graphs.echo()).start({"text": "hi"})

    async def collect() -> list[Any]:
        return [row async for row in run.follow()]

    first, second = await asyncio.gather(collect(), collect())
    assert first == second == run.rows


async def test_a_failing_node_marks_the_run_failed(graphs: Any) -> None:
    run = await finish(Sandbox(graphs.failing()).start({"text": "hi"}))
    assert run.status == "error"
    assert "the node failed on purpose" in (run.error or "")
    assert "RuntimeError" in (run.traceback or "")


async def test_invalid_input_marks_the_run_failed_with_the_graph_error(
    graphs: Any,
) -> None:
    run = await finish(Sandbox(graphs.echo()).start({"nope": 1}))
    assert run.status == "error"
    assert "StateUpdateError" in (run.error or "")
    assert "nope" in (run.error or "")


async def test_the_sandbox_runs_a_copy_and_never_changes_the_graph(
    graphs: Any,
) -> None:
    graph = graphs.echo()
    before = dict(vars(graph))
    sandbox = Sandbox(graph)
    await finish(sandbox.start({"text": "hi"}))
    assert sandbox.source is graph
    assert sandbox.graph is not graph
    assert graph.state_store is None
    assert isinstance(sandbox.store, InMemoryStateStore)
    assert sandbox.graph.state_store is sandbox.store
    assert sandbox.in_memory_store is True
    assert vars(graph) == before


async def test_a_graph_with_its_own_store_keeps_that_store(graphs: Any) -> None:
    store = InMemoryStateStore()
    sandbox = Sandbox(graphs.echo(state_store=store))
    run = await finish(sandbox.start({"text": "hi"}))
    assert sandbox.store is store
    assert sandbox.in_memory_store is False
    assert (await store.get_history(run.thread_id)).events


async def test_every_run_gets_its_own_sandbox_thread(graphs: Any) -> None:
    sandbox = Sandbox(graphs.echo())
    first = await finish(sandbox.start({"text": "a"}))
    second = await finish(sandbox.start({"text": "b"}))
    assert first.thread_id != second.thread_id
    assert first.thread_id.startswith("sandbox-")
    assert [summary.id for summary in sandbox.thread_summaries()] == [
        second.thread_id,
        first.thread_id,
    ]
    assert sandbox.thread_summaries()[0].status == "completed"


async def test_resume_answers_every_pending_interrupt_by_key(graphs: Any) -> None:
    sandbox = Sandbox(graphs.review())
    run = await finish(sandbox.start({"draft": "notes"}))
    assert run.status == "interrupted"
    keys = {item.id: item.key for item in run.pending}
    assert set(keys) == {"title", "approve_publish"}
    resumed = await finish(
        sandbox.resume(
            run,
            {keys["title"]: "Hello", keys["approve_publish"]: {"action": "approve"}},
        )
    )
    assert resumed.status == "completed"
    state = loads(resumed.final_state)
    assert state["title"] == "Hello"
    assert json.loads(state["body"]) == {"action": "approve"}
    assert (resumed.kind, resumed.previous, resumed.thread_id) == (
        "resume",
        run.id,
        run.thread_id,
    )
    assert sandbox.next_run(run) is resumed
    assert sandbox.latest_run(run.thread_id) is resumed
    assert sandbox.thread_runs(run.thread_id) == [run, resumed]


async def test_a_pause_can_be_answered_once(graphs: Any) -> None:
    sandbox = Sandbox(graphs.review())
    run = await finish(sandbox.start({"draft": "notes"}))
    answers = {
        item.key: {"action": "approve"} if item.approval else "Hello"
        for item in run.pending
    }
    resumed = sandbox.resume(run, answers)
    with pytest.raises(RunStateError, match=f"already answered; see run {resumed.id}"):
        sandbox.resume(run, answers)
    await finish(resumed)


async def test_a_finished_run_cannot_be_resumed(graphs: Any) -> None:
    sandbox = Sandbox(graphs.echo())
    run = await finish(sandbox.start({"text": "hi"}))
    with pytest.raises(
        RunStateError, match="is completed, so there is nothing to answer"
    ):
        sandbox.resume(run, {})


class WaitState(TypedDict):
    text: str


async def test_stop_ends_the_runs_that_are_still_going() -> None:
    entered = threading.Event()
    release = threading.Event()

    @node
    def wait_in_thread(state: WaitState) -> dict[str, Any]:
        entered.set()
        release.wait(timeout=10)
        return {"text": "late"}

    @node
    async def wait_forever(state: WaitState) -> dict[str, Any]:
        entered.set()
        await asyncio.Event().wait()
        return {"text": "never"}

    threaded = Sandbox(
        Graph(WaitState, name="threaded").flow(
            START >> wait_in_thread, wait_in_thread >> END
        )
    )
    awaiting = Sandbox(
        Graph(WaitState, name="awaiting").flow(
            START >> wait_forever, wait_forever >> END
        )
    )
    try:
        for sandbox in (threaded, awaiting):
            entered.clear()
            run = sandbox.start({"text": "hi"})
            assert await asyncio.to_thread(entered.wait, 5)
            async with asyncio.timeout(1):
                assert await sandbox.stop() == [run]
            assert run.finished
            assert run.status == "error"
            assert run.error == "The sandbox stopped before the run finished"
            assert [row async for row in run.follow()] == run.rows
    finally:
        release.set()


async def test_stop_leaves_finished_runs_alone(graphs: Any) -> None:
    sandbox = Sandbox(graphs.echo())
    run = await finish(sandbox.start({"text": "hi"}))
    assert await sandbox.stop() == []
    assert run.status == "completed"


async def test_history_comes_from_the_sandbox_store(graphs: Any) -> None:
    sandbox = Sandbox(graphs.echo())
    run = await finish(sandbox.start({"text": "hi"}))
    history = await sandbox.history(run.thread_id)
    types = [event.type for event in history.events]
    assert types[0] == "run_started"
    assert types[-1] == "run_completed"
    assert "node_completed" in types
