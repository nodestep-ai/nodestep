from typing import Any, cast

import pytest

from nodestep.core.command import END, START
from nodestep.core.flow import FlowBranchEdge, FlowEdge, branch, when
from nodestep.core.graph import Graph
from nodestep.core.node import node
from nodestep.exceptions import GraphConfigError


@node
def start(state: dict) -> dict:
    return state


@node
def finish(state: dict) -> dict:
    return state


def route_state(state: dict) -> str:
    return "finish" if state.get("done") else "end"


def test_shift_builds_flow_edge() -> None:
    spec = start >> finish

    assert isinstance(spec, FlowEdge)
    assert spec.source is start
    assert spec.target is finish


def test_shift_accepts_end() -> None:
    spec = start >> END

    assert isinstance(spec, FlowEdge)
    assert spec.source is start
    assert spec.target is END


def test_shift_builds_branch_edge() -> None:
    spec = start >> branch(route_state, {"finish": finish, "end": END})

    assert isinstance(spec, FlowBranchEdge)
    assert spec.source is start
    assert spec.branch.router is route_state
    assert spec.branch.mapping["finish"] is finish


def test_flow_rejects_string_targets() -> None:
    bad_target = cast(Any, "finish")
    with pytest.raises(GraphConfigError, match="Unsupported flow target"):
        Graph(dict).flow(START >> start, start >> bad_target)


def test_graph_serializes_lightweight_spec() -> None:
    graph = Graph(dict, name="serial").flow(
        START >> start, start >> finish, finish >> END
    )

    spec = graph.to_spec()

    assert spec.name == "serial"
    assert spec.start == "start"
    assert [edge.model_dump() for edge in spec.edges] == [
        {"source": "start", "target": "finish"},
        {"source": "finish", "target": "END"},
    ]


def test_graph_serialization_rejects_unreferenced_nodes() -> None:
    dynamic = node(lambda state: state, name="dynamic")
    graph = Graph(dict).flow(START >> dynamic, dynamic >> END)

    with pytest.raises(GraphConfigError, match="not serializable"):
        graph.to_spec()


async def test_invoke_raises_in_async_context() -> None:
    graph = Graph(dict).flow(START >> start, start >> finish, finish >> END)
    with pytest.raises(RuntimeError, match=r"Cannot call graph\.invoke"):
        graph.invoke({"x": 1})


def is_done(state: dict) -> bool:
    return bool(state.get("done"))


def test_when_requires_otherwise() -> None:
    untyped_when = cast(Any, when)
    with pytest.raises(TypeError, match="otherwise"):
        untyped_when(is_done, finish)


def test_when_maps_true_and_false() -> None:
    spec = when(is_done, finish, otherwise=END)

    assert dict(spec.mapping) == {True: finish, False: END}


def route_one(state: dict) -> int:
    return 1


def test_to_spec_rejects_branch_keys_that_serialize_the_same() -> None:
    graph = Graph(dict).flow(
        START >> start,
        start >> branch(route_one, {1: finish, "1": END}),
        finish >> END,
    )

    with pytest.raises(GraphConfigError, match="both serialize to '1'"):
        graph.to_spec()


@node(goto=[finish, END])
def dispatch(state: dict) -> dict:
    return {}


def test_to_spec_lists_declared_goto_targets() -> None:
    graph = Graph(dict).flow(START >> dispatch, finish >> END)

    spec = graph.to_spec()

    assert [(item.name, item.goto) for item in spec.nodes] == [
        ("dispatch", ["finish", "END"]),
        ("finish", None),
    ]
    assert '"goto":["finish","END"]' in graph.to_json()


@node(goto=["repeat", "finish", END])
def repeat(state: dict) -> dict:
    return {}


def test_to_spec_lists_goto_names() -> None:
    graph = Graph(dict).flow(START >> repeat, repeat >> finish, finish >> END)

    spec = graph.to_spec()

    assert spec.nodes[0].goto == ["repeat", "finish", "END"]
