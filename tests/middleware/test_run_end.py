import asyncio
import dataclasses
import gc
import warnings
from collections.abc import Generator
from contextlib import aclosing
from typing import Annotated, Any, TypedDict

import pytest

import nodestep
import nodestep.core
from nodestep import (
    END,
    START,
    AgentTask,
    Graph,
    InMemoryStateStore,
    Middleware,
    NodeContext,
    Resume,
    RunOutcome,
    add,
    interrupt,
    node,
)
from nodestep.exceptions import (
    GraphExecutionError,
    GraphTimeoutError,
    NodeTimeoutError,
    ResumeError,
    RunLimitExceededError,
)
from nodestep.middleware import GraphMiddlewareContext, Replacement


class Log(TypedDict, total=False):
    log: Annotated[list[str], add]


class Recorder(Middleware):
    def __init__(self, label: str = "recorder", calls: list[Any] | None = None) -> None:
        self.label = label
        self.calls: list[Any] = [] if calls is None else calls
        self.outcomes: list[RunOutcome] = []
        self.contexts: list[GraphMiddlewareContext] = []

    def before_graph(self, ctx):
        self.calls.append((self.label, "before_graph"))

    def after_graph(self, ctx):
        self.calls.append((self.label, "after_graph"))

    def on_run_end(self, ctx, outcome):
        self.calls.append((self.label, "on_run_end", outcome.status))
        self.contexts.append(ctx)
        self.outcomes.append(outcome)


@node
def writes(state: Log) -> dict:
    return {"log": ["writes"]}


@node
def asks(state: Log) -> dict:
    answer = interrupt("approve?", id="approve")
    return {"log": [f"answer={answer}"]}


@node
def breaks(state: Log) -> dict:
    raise ValueError("node broke")


def simple_graph(*middleware: Middleware, **options: Any) -> Graph:
    return Graph(Log, middleware=list(middleware), **options).flow(
        START >> writes, writes >> END
    )


def test_run_outcome_is_a_frozen_slotted_dataclass_exported_from_both_packages() -> (
    None
):
    assert nodestep.RunOutcome is nodestep.core.RunOutcome
    assert "RunOutcome" in nodestep.__all__
    assert "RunOutcome" in nodestep.core.__all__
    assert dataclasses.is_dataclass(RunOutcome)
    assert hasattr(RunOutcome, "__slots__")
    outcome = RunOutcome(status="completed", interrupts={}, error=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.status = "failed"  # ty: ignore[invalid-assignment]


def test_core_exports_the_terminal_event_data() -> None:
    assert {"FinalEventData", "InterruptEventData"} <= set(nodestep.core.__all__)


async def test_a_completed_run_reports_completed_once_with_the_final_state() -> None:
    recorder = Recorder()
    graph = simple_graph(recorder)

    result = await graph.ainvoke({"log": ["input"]})

    assert recorder.calls == [
        ("recorder", "before_graph"),
        ("recorder", "after_graph"),
        ("recorder", "on_run_end", "completed"),
    ]
    [outcome] = recorder.outcomes
    assert outcome == RunOutcome(status="completed", interrupts={}, error=None)
    [ctx] = recorder.contexts
    assert ctx.state == result.data
    assert ctx.graph_name == graph.name
    assert ctx.thread_id == result.thread_id


async def test_a_paused_run_reports_its_interrupts() -> None:
    recorder = Recorder()
    graph = Graph(Log, state_store=InMemoryStateStore(), middleware=[recorder]).flow(
        START >> asks, asks >> END
    )

    result = await graph.ainvoke({"log": []}, thread_id="t")

    [outcome] = recorder.outcomes
    assert outcome.status == "paused"
    assert outcome.error is None
    assert dict(outcome.interrupts) == result.interrupts
    assert list(outcome.interrupts) == ["asks:approve"]
    assert ("recorder", "after_graph") not in recorder.calls


async def test_a_resumed_run_reports_its_own_end() -> None:
    recorder = Recorder()
    graph = Graph(Log, state_store=InMemoryStateStore(), middleware=[recorder]).flow(
        START >> asks, asks >> END
    )
    await graph.ainvoke({"log": []}, thread_id="t")

    await graph.ainvoke(thread_id="t", resume=Resume("yes"))

    assert [outcome.status for outcome in recorder.outcomes] == [
        "paused",
        "completed",
    ]
    assert recorder.contexts[1].resuming is True
    assert recorder.contexts[1].state == {"log": ["answer=yes"]}


async def test_a_failed_run_reports_the_error_and_raises_it() -> None:
    recorder = Recorder()
    graph = Graph(Log, middleware=[recorder]).flow(START >> breaks, breaks >> END)

    with pytest.raises(ValueError, match="node broke") as info:
        await graph.ainvoke({"log": []})

    [outcome] = recorder.outcomes
    assert outcome.status == "failed"
    assert outcome.error is info.value
    assert outcome.interrupts == {}
    assert ("recorder", "after_graph") not in recorder.calls


async def test_a_failure_the_on_error_hook_recovers_from_completes() -> None:
    class Recovers(Middleware):
        def on_error(self, ctx, error):
            return {"log": ["recovered"]}

    recorder = Recorder()
    graph = Graph(Log, middleware=[Recovers(), recorder]).flow(
        START >> breaks, breaks >> END
    )

    await graph.ainvoke({"log": []})

    assert [outcome.status for outcome in recorder.outcomes] == ["completed"]


async def test_the_state_of_a_failed_run_is_the_last_merged_state() -> None:
    recorder = Recorder()
    graph = Graph(Log, middleware=[recorder]).flow(
        START >> writes, writes >> breaks, breaks >> END
    )

    with pytest.raises(ValueError, match="node broke"):
        await graph.ainvoke({"log": ["input"]})

    assert recorder.contexts[0].state == {"log": ["input", "writes"]}


async def test_a_graph_timeout_reports_failed() -> None:
    @node
    async def sleeps(state: Log) -> dict:
        await asyncio.sleep(5)
        return {}

    recorder = Recorder()
    graph = Graph(Log, middleware=[recorder], timeout=0.05).flow(
        START >> sleeps, sleeps >> END
    )

    with pytest.raises(GraphTimeoutError):
        await graph.ainvoke({"log": []})

    [outcome] = recorder.outcomes
    assert outcome.status == "failed"
    assert isinstance(outcome.error, GraphTimeoutError)


async def test_a_node_timeout_reports_failed() -> None:
    @node(timeout=0.05)
    async def sleeps(state: Log) -> dict:
        await asyncio.sleep(5)
        return {}

    recorder = Recorder()
    graph = Graph(Log, middleware=[recorder]).flow(START >> sleeps, sleeps >> END)

    with pytest.raises(NodeTimeoutError):
        await graph.ainvoke({"log": []})

    assert isinstance(recorder.outcomes[0].error, NodeTimeoutError)


async def test_a_run_limit_reports_failed() -> None:
    @node(goto=["loops"])
    def loops(state: Log) -> dict:
        return {"log": ["loop"]}

    recorder = Recorder()
    graph = Graph(Log, middleware=[recorder], max_steps=2).flow(
        START >> loops, loops >> loops
    )

    with pytest.raises(RunLimitExceededError):
        await graph.ainvoke({"log": []})

    [outcome] = recorder.outcomes
    assert outcome.status == "failed"
    assert isinstance(outcome.error, RunLimitExceededError)


async def test_ungathered_sub_agents_report_failed() -> None:
    @node
    async def sleeps(state: Log) -> dict:
        await asyncio.sleep(5)
        return {}

    slow = Graph(Log, name="slow").flow(START >> sleeps, sleeps >> END)

    @node
    async def dispatch(state: Log, ctx: NodeContext) -> dict:
        await ctx.spawn([AgentTask(graph=slow, input={}, name="researcher")])
        return {}

    recorder = Recorder()
    graph = Graph(Log, middleware=[recorder]).flow(START >> dispatch, dispatch >> END)

    with pytest.raises(GraphExecutionError, match="never gathered") as info:
        await graph.ainvoke({"log": []})

    [outcome] = recorder.outcomes
    assert outcome.status == "failed"
    assert outcome.error is info.value


async def test_a_cancelled_run_reports_cancelled_and_stays_cancelled() -> None:
    started = asyncio.Event()

    @node
    async def waits(state: Log) -> dict:
        started.set()
        await asyncio.sleep(5)
        return {}

    recorder = Recorder()
    graph = Graph(Log, middleware=[recorder]).flow(START >> waits, waits >> END)
    run = asyncio.ensure_future(graph.ainvoke({"log": []}))
    await started.wait()

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert recorder.outcomes == [
        RunOutcome(status="cancelled", interrupts={}, error=None)
    ]


async def test_a_caller_timeout_reports_cancelled() -> None:
    @node
    async def sleeps(state: Log) -> dict:
        await asyncio.sleep(5)
        return {}

    recorder = Recorder()
    graph = Graph(Log, middleware=[recorder]).flow(START >> sleeps, sleeps >> END)

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await graph.ainvoke({"log": []})

    assert [outcome.status for outcome in recorder.outcomes] == ["cancelled"]


async def test_a_stream_closed_before_the_terminal_event_reports_cancelled() -> None:
    @node
    def second(state: Log) -> dict:
        return {"log": ["second"]}

    recorder = Recorder()
    graph = Graph(Log, middleware=[recorder]).flow(
        START >> writes, writes >> second, second >> END
    )

    async with aclosing(graph.astream({"log": []}, stream_mode="values")) as events:
        async for _ in events:
            break

    assert recorder.outcomes == [
        RunOutcome(status="cancelled", interrupts={}, error=None)
    ]


async def test_a_stream_closed_after_the_final_event_reports_completed() -> None:
    recorder = Recorder()
    graph = simple_graph(recorder)

    async with aclosing(graph.astream({"log": []}, stream_mode=[])) as events:
        async for event in events:
            assert event.mode == "final"
            break

    assert [outcome.status for outcome in recorder.outcomes] == ["completed"]


async def test_a_stream_closed_after_the_interrupt_event_reports_paused() -> None:
    recorder = Recorder()
    graph = Graph(Log, state_store=InMemoryStateStore(), middleware=[recorder]).flow(
        START >> asks, asks >> END
    )

    async with aclosing(
        graph.astream({"log": []}, stream_mode=[], thread_id="t")
    ) as events:
        async for _ in events:
            break

    assert [outcome.status for outcome in recorder.outcomes] == ["paused"]


def test_a_sync_stream_the_caller_stops_reports_cancelled() -> None:
    @node
    def second(state: Log) -> dict:
        return {"log": ["second"]}

    recorder = Recorder()
    graph = Graph(Log, middleware=[recorder]).flow(
        START >> writes, writes >> second, second >> END
    )

    events = graph.stream({"log": []}, stream_mode="values")
    assert isinstance(events, Generator)
    next(events)
    events.close()

    assert [outcome.status for outcome in recorder.outcomes] == ["cancelled"]


def test_a_sync_stream_closes_its_loop_when_a_hook_raises_on_close() -> None:
    @node
    def second(state: Log) -> dict:
        return {"log": ["second"]}

    graph = Graph(Log, middleware=[Raises()]).flow(
        START >> writes, writes >> second, second >> END
    )
    events = graph.stream({"log": []}, stream_mode="values")
    assert isinstance(events, Generator)
    next(events)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="observer broke on cancelled"):
            events.close()
        del events
        gc.collect()

    assert [str(warning.message) for warning in caught] == []


def test_a_sync_invoke_reports_completed() -> None:
    recorder = Recorder()

    simple_graph(recorder).invoke({"log": []})

    assert [outcome.status for outcome in recorder.outcomes] == ["completed"]


async def test_hooks_run_in_reverse_middleware_order() -> None:
    calls: list[Any] = []
    graph = simple_graph(Recorder("first", calls), Recorder("second", calls))

    await graph.ainvoke({"log": []})

    assert [call for call in calls if call[1] == "on_run_end"] == [
        ("second", "on_run_end", "completed"),
        ("first", "on_run_end", "completed"),
    ]


async def test_async_hooks_are_awaited() -> None:
    seen: list[str] = []

    class Async(Middleware):
        async def on_run_end(self, ctx, outcome):
            await asyncio.sleep(0)
            seen.append(outcome.status)

    await simple_graph(Async()).ainvoke({"log": []})

    assert seen == ["completed"]


async def test_hooks_get_a_copy_of_the_state() -> None:
    class Mutates(Middleware):
        def on_run_end(self, ctx, outcome):
            ctx.state["log"].append("changed")

    recorder = Recorder()
    graph = Graph(
        Log, state_store=InMemoryStateStore(), middleware=[recorder, Mutates()]
    ).flow(START >> writes, writes >> END)

    result = await graph.ainvoke({"log": []}, thread_id="t")

    assert result.data == {"log": ["writes"]}
    assert recorder.contexts[0].state == {"log": ["writes"]}
    assert await graph.load("t") == {"log": ["writes"]}


async def test_the_interrupts_cannot_be_changed() -> None:
    class Clears(Middleware):
        def on_run_end(self, ctx, outcome):
            outcome.interrupts.clear()

    graph = Graph(Log, state_store=InMemoryStateStore(), middleware=[Clears()]).flow(
        START >> asks, asks >> END
    )

    with pytest.raises(AttributeError):
        await graph.ainvoke({"log": []}, thread_id="t")


async def test_a_replacement_from_the_hook_raises() -> None:
    class Replaces(Middleware):
        def on_run_end(self, ctx, outcome):
            return Replacement({})

    with pytest.raises(GraphExecutionError, match=r"on_run_end.*read-only"):
        await simple_graph(Replaces()).ainvoke({"log": []})


async def test_any_other_return_value_raises() -> None:
    class ReturnsValue(Middleware):
        def on_run_end(self, ctx, outcome):
            return "done"

    with pytest.raises(TypeError, match="on_run_end returned str"):
        await simple_graph(ReturnsValue()).ainvoke({"log": []})


class Raises(Middleware):
    def on_run_end(self, ctx, outcome):
        raise RuntimeError(f"observer broke on {outcome.status}")


async def test_a_hook_error_after_a_completed_run_propagates() -> None:
    recorder = Recorder()
    graph = simple_graph(recorder, Raises())

    with pytest.raises(RuntimeError, match="observer broke on completed"):
        await graph.ainvoke({"log": []})

    assert [outcome.status for outcome in recorder.outcomes] == ["completed"]


async def test_a_hook_error_after_a_failed_run_is_a_note_on_the_run_error() -> None:
    recorder = Recorder()
    graph = Graph(Log, middleware=[recorder, Raises()]).flow(
        START >> breaks, breaks >> END
    )

    with pytest.raises(ValueError, match="node broke") as info:
        await graph.ainvoke({"log": []})

    assert info.value.__notes__ == [
        "Raises.on_run_end raised RuntimeError: observer broke on failed"
    ]
    assert [outcome.status for outcome in recorder.outcomes] == ["failed"]


async def test_a_second_hook_error_is_a_note_on_the_first() -> None:
    class AlsoRaises(Middleware):
        def on_run_end(self, ctx, outcome):
            raise KeyError("second")

    graph = simple_graph(AlsoRaises(), Raises())

    with pytest.raises(RuntimeError, match="observer broke") as info:
        await graph.ainvoke({"log": []})

    assert info.value.__notes__ == ["AlsoRaises.on_run_end raised KeyError: 'second'"]


async def test_a_hook_error_after_a_cancelled_run_propagates() -> None:
    started = asyncio.Event()

    @node
    async def waits(state: Log) -> dict:
        started.set()
        await asyncio.sleep(5)
        return {}

    graph = Graph(Log, middleware=[Raises()]).flow(START >> waits, waits >> END)
    run = asyncio.ensure_future(graph.ainvoke({"log": []}))
    await started.wait()

    run.cancel()
    with pytest.raises(RuntimeError, match="observer broke on cancelled"):
        await run


class RaisesCancelled(Middleware):
    def on_run_end(self, ctx, outcome):
        raise asyncio.CancelledError("observer cancelled")


async def test_a_hook_cancellation_does_not_skip_the_later_hooks() -> None:
    recorder = Recorder()
    graph = simple_graph(recorder, RaisesCancelled())

    with pytest.raises(asyncio.CancelledError, match="observer cancelled"):
        await graph.ainvoke({"log": []})

    assert [outcome.status for outcome in recorder.outcomes] == ["completed"]


async def test_a_hook_cancellation_after_a_failed_run_keeps_the_run_error() -> None:
    recorder = Recorder()
    graph = Graph(Log, middleware=[recorder, RaisesCancelled(), Raises()]).flow(
        START >> breaks, breaks >> END
    )

    with pytest.raises(asyncio.CancelledError) as info:
        await graph.ainvoke({"log": []})

    assert isinstance(info.value.__context__, ValueError)
    assert info.value.__context__.__notes__ == [
        "Raises.on_run_end raised RuntimeError: observer broke on failed"
    ]
    assert [outcome.status for outcome in recorder.outcomes] == ["failed"]


async def test_a_before_graph_error_ends_the_run_for_the_middleware_it_reached() -> (
    None
):
    class FailsBefore(Middleware):
        def before_graph(self, ctx):
            raise RuntimeError("before failed")

    calls: list[Any] = []
    first = Recorder("first", calls)
    last = Recorder("last", calls)
    failing = FailsBefore()
    graph = simple_graph(first, failing, last)

    with pytest.raises(RuntimeError, match="before failed"):
        await graph.ainvoke({"log": []})

    assert calls == [
        ("first", "before_graph"),
        ("first", "on_run_end", "failed"),
    ]
    assert isinstance(first.outcomes[0].error, RuntimeError)
    assert last.outcomes == []


async def test_a_run_refused_before_it_starts_calls_no_hook() -> None:
    recorder = Recorder()
    graph = Graph(Log, state_store=InMemoryStateStore(), middleware=[recorder]).flow(
        START >> asks, asks >> END
    )
    await graph.ainvoke({"log": []}, thread_id="t")
    recorder.calls.clear()

    with pytest.raises(ResumeError):
        await graph.ainvoke({"log": []}, thread_id="t")

    assert recorder.calls == []


async def test_a_subgraph_run_ends_for_its_own_middleware() -> None:
    child_recorder = Recorder("child")
    child = Graph(Log, name="child", middleware=[child_recorder]).flow(
        START >> writes, writes >> END
    )
    parent_recorder = Recorder("parent")
    inner = nodestep.subgraph("inner", child, share=("log",), child_thread="fresh")
    parent = Graph(Log, middleware=[parent_recorder]).flow(START >> inner, inner >> END)

    await parent.ainvoke({"log": []})

    assert [outcome.status for outcome in child_recorder.outcomes] == ["completed"]
    assert [outcome.status for outcome in parent_recorder.outcomes] == ["completed"]


def test_the_hook_is_known_to_the_middleware_base() -> None:
    class Observer(Middleware):
        def on_run_end(self, ctx, outcome):
            return None

    assert Observer().on_run_end is not None
