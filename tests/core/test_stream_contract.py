import asyncio
import dataclasses
import inspect
from typing import Annotated, Any, TypedDict, cast

import pytest

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
)
from nodestep.core.graph import GraphResult
from nodestep.core.stream import FinalEventData, InterruptEventData, StreamEvent


class Fan(TypedDict, total=False):
    log: Annotated[list[str], add]
    picked: str
    items: list[str]


@node
async def left(state: Fan) -> dict:
    return {"log": ["left"]}


@node
async def right(state: Fan) -> dict:
    await asyncio.sleep(0.05)
    return {"picked": "right"}


@node
def quiet(state: Fan) -> None:
    return None


@node(goto=[right, left, quiet])
def fan(state: Fan) -> Command:
    return Command(goto=[right, left, quiet])


def _fan_graph(**options: Any) -> Graph:
    return Graph(Fan, **options).flow(
        START >> fan, left >> END, right >> END, quiet >> END
    )


def _summary(events: list[StreamEvent], mode: str) -> list[tuple[Any, ...]]:
    return [
        (event.node, event.step, event.task_id, event.data)
        for event in events
        if event.mode == mode
    ]


async def _collect(graph: Graph, *args: Any, **kwargs: Any) -> list[StreamEvent]:
    return [event async for event in graph.astream(*args, **kwargs)]


def _data(event: StreamEvent) -> Any:
    return event.data


async def test_astream_requires_an_explicit_stream_mode() -> None:
    graph = _fan_graph()

    with pytest.raises(TypeError, match="stream_mode"):
        cast(Any, graph).astream({})
    with pytest.raises(TypeError, match="stream_mode"):
        await _collect(graph, {}, stream_mode=None)


def test_stream_requires_an_explicit_stream_mode() -> None:
    graph = _fan_graph()

    with pytest.raises(TypeError, match="stream_mode"):
        cast(Any, graph).stream({})


@pytest.mark.parametrize("mode", ["interrupt", "final", "custm", ["updates", "nope"]])
async def test_unknown_or_terminal_modes_are_rejected(mode: Any) -> None:
    with pytest.raises(ValueError, match="stream_mode"):
        await _collect(_fan_graph(), {}, stream_mode=mode)


async def test_empty_stream_mode_yields_only_the_terminal_event() -> None:
    events = await _collect(_fan_graph(), {}, stream_mode=[])

    assert [event.mode for event in events] == ["final"]
    assert events[0].step == 1


async def test_updates_are_one_event_per_task_after_the_step_merges() -> None:
    store = InMemoryStateStore()
    graph = _fan_graph(state_store=store)

    events = await _collect(graph, {}, thread_id="t", stream_mode="updates")

    assert _summary(events, "updates") == [
        ("fan", 0, "fan", None),
        ("right", 1, "right", {"picked": "right"}),
        ("left", 1, "left", {"log": ["left"]}),
        ("quiet", 1, "quiet", None),
    ]
    updates = [event for event in events if event.mode == "updates"]
    step_one = {event.checkpoint_id for event in updates if event.step == 1}
    assert len(step_one) == 1
    checkpoint = step_one.pop()
    history = await graph.history("t")
    assert next(event for event in history.events if event.id == checkpoint).type == (
        "superstep_completed"
    )
    assert await graph.load("t", at=checkpoint) == {"log": ["left"], "picked": "right"}
    assert {event.run_id for event in events} == {events[-1].run_id}


async def test_updates_without_a_store_have_no_checkpoint_id() -> None:
    events = await _collect(_fan_graph(), {}, stream_mode="updates")

    assert {event.checkpoint_id for event in events} == {None}


async def test_values_carry_the_merged_state_step_and_checkpoint() -> None:
    graph = _fan_graph(state_store=InMemoryStateStore())

    events = await _collect(graph, {}, thread_id="v", stream_mode="values")

    values = [event for event in events if event.mode == "values"]
    assert [(event.step, event.data) for event in values] == [
        (0, {}),
        (1, {"log": ["left"], "picked": "right"}),
    ]
    for event in values:
        assert event.checkpoint_id is not None
        assert await graph.load("v", at=event.checkpoint_id) == event.data


@node
def worker(state: Fan) -> dict:
    return {"log": [f"worker:{state['picked']}"]}


@node(goto=[worker, END])
def sender(state: Fan) -> Command:
    return Command(goto=[Send(worker, {"picked": "x"}), END])


async def test_debug_events_carry_node_step_task_and_structured_routes() -> None:
    graph = Graph(Fan).flow(START >> sender, worker >> END)

    events = await _collect(graph, {}, stream_mode="debug")

    debug = [
        (_data(event)["type"], event.node, event.step, event.task_id)
        for event in events
        if event.mode == "debug"
    ]
    assert debug == [
        ("node_input", "sender", 0, "sender"),
        ("node_output", "sender", 0, "sender"),
        ("routes", "sender", 0, "sender"),
        ("state_merge", None, 0, None),
        ("node_input", "worker", 1, "worker"),
        ("node_output", "worker", 1, "worker"),
        ("routes", "worker", 1, "worker"),
        ("state_merge", None, 1, None),
    ]
    routes = [
        _data(event)["targets"]
        for event in events
        if event.mode == "debug" and _data(event)["type"] == "routes"
    ]
    assert routes == [
        [
            {"kind": "send", "node": "worker", "payload": {"picked": "x"}},
            {"kind": "end", "node": None, "payload": None},
        ],
        [{"kind": "end", "node": None, "payload": None}],
    ]


@node
def pre(state: Fan) -> dict:
    return {"log": ["pre"]}


@node
def asks(state: Fan) -> dict:
    return {"log": [f"answer={interrupt('ok?', id='ok')}"]}


async def test_terminal_events_carry_the_step() -> None:
    graph = Graph(Fan, state_store=InMemoryStateStore()).flow(
        START >> pre, pre >> asks, asks >> END
    )

    paused = await _collect(graph, {}, thread_id="p", stream_mode=[])
    resumed = await _collect(
        graph, None, thread_id="p", resume=Resume("yes"), stream_mode=[]
    )

    assert [(event.mode, event.step) for event in paused] == [("interrupt", 1)]
    assert isinstance(paused[0].data, InterruptEventData)
    assert [(event.mode, event.step) for event in resumed] == [("final", 0)]
    assert isinstance(resumed[0].data, FinalEventData)


@node
def emitter(state: Fan, ctx: NodeContext) -> dict:
    ctx.emit({"progress": 1})
    return {"log": ["emitter"]}


async def test_emit_always_sends_custom_events_with_the_task() -> None:
    graph = Graph(Fan).flow(START >> pre, pre >> emitter, emitter >> END)

    events = await _collect(graph, {}, stream_mode=["custom"])

    custom = [event for event in events if event.mode == "custom"]
    assert [
        (event.node, event.step, event.task_id, event.data) for event in custom
    ] == [("emitter", 1, "emitter", {"progress": 1})]
    assert custom[0].run_id == events[-1].run_id


def test_emit_has_no_mode_parameter() -> None:
    assert list(inspect.signature(NodeContext.emit).parameters) == ["self", "data"]


@node
def fakes_an_update(state: Fan, ctx: NodeContext) -> dict:
    cast(Any, ctx).emit({"fake": "update"}, mode="updates")
    return {}


async def test_emit_with_a_mode_fails_the_node() -> None:
    graph = Graph(Fan).flow(START >> fakes_an_update, fakes_an_update >> END)

    with pytest.raises(TypeError, match="mode"):
        await _collect(graph, {}, stream_mode=["custom", "updates"])


@node
def a(state: Fan) -> dict:
    return {"log": ["a"]}


@node
def b(state: Fan) -> dict:
    return {"log": [f"b saw items={state['items']!r}"]}


async def test_consumers_cannot_change_the_state_through_events() -> None:
    graph = Graph(Fan).flow(START >> a, a >> b, b >> END)

    final: FinalEventData | None = None
    async for event in graph.astream(
        {"items": [], "log": []}, stream_mode=["values", "updates", "debug"]
    ):
        data = _data(event)
        if event.mode == "values":
            data["items"].append("from values")
        if event.mode == "updates" and data:
            data["log"].append("from updates")
        if event.mode == "debug" and data["type"] in ("node_input", "state_merge"):
            data["state"]["items"].append("from debug")
        if event.mode == "debug" and data["type"] == "node_output":
            data["updates"]["log"].append("from debug output")
        if isinstance(data, FinalEventData):
            final = data

    assert final is not None
    assert final.state == {"items": [], "log": ["a", "b saw items=[]"]}


def test_stream_event_and_node_context_have_no_namespace() -> None:
    assert "namespace" not in {field.name for field in dataclasses.fields(StreamEvent)}
    assert "namespace" not in {field.name for field in dataclasses.fields(NodeContext)}
    assert {"step", "task_id"} <= {
        field.name for field in dataclasses.fields(StreamEvent)
    }


async def test_graph_result_reports_the_real_step() -> None:
    graph = Graph(Fan).flow(START >> a, a >> b, b >> END)

    result = await graph.ainvoke({"items": []})

    assert "output" not in {field.name for field in dataclasses.fields(GraphResult)}
    assert result.step == 1


@node
def sibling(state: Fan) -> dict:
    return {"log": ["sibling"]}


@node(goto=[sibling, asks])
def fan_and_ask(state: Fan) -> Command:
    return Command(goto=[sibling, asks])


async def test_a_resumed_step_reports_the_siblings_that_finished_before_the_pause() -> (
    None
):
    graph = Graph(Fan, state_store=InMemoryStateStore()).flow(
        START >> fan_and_ask, sibling >> END, asks >> END
    )

    paused = await _collect(graph, {}, thread_id="r", stream_mode=["updates", "debug"])
    resumed = await _collect(
        graph,
        None,
        thread_id="r",
        resume=Resume("yes"),
        stream_mode=["updates", "debug"],
    )

    assert _summary(paused, "updates") == [("fan_and_ask", 0, "fan_and_ask", None)]
    assert _summary(resumed, "updates") == [
        ("sibling", 0, "sibling", {"log": ["sibling"]}),
        ("asks", 0, "asks", {"log": ["answer=yes"]}),
    ]
    routes = [
        (event.node, event.task_id, _data(event)["targets"])
        for event in resumed
        if event.mode == "debug" and _data(event)["type"] == "routes"
    ]
    end = [{"kind": "end", "node": None, "payload": None}]
    assert routes == [("sibling", "sibling", end), ("asks", "asks", end)]


@node(goto=[asks, sibling])
def ask_then_sibling(state: Fan) -> Command:
    return Command(goto=[asks, sibling])


async def test_a_resumed_step_reports_its_tasks_in_scheduling_order() -> None:
    graph = Graph(Fan, state_store=InMemoryStateStore()).flow(
        START >> ask_then_sibling, sibling >> END, asks >> END
    )
    await _collect(graph, {}, thread_id="o", stream_mode=[])

    resumed = await _collect(
        graph,
        None,
        thread_id="o",
        resume=Resume("yes"),
        stream_mode=["updates", "debug"],
    )

    assert [event.task_id for event in resumed if event.mode == "updates"] == [
        "asks",
        "sibling",
    ]
    assert [
        event.task_id
        for event in resumed
        if event.mode == "debug" and _data(event)["type"] == "routes"
    ] == ["asks", "sibling"]
