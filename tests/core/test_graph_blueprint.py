import pytest

from nodestep.core.command import END, START, Command
from nodestep.core.graph import Graph
from nodestep.core.node import node
from nodestep.exceptions import GraphConfigError, GraphExecutionError


@node
def first(state: dict) -> dict:
    return {"value": 1}


@node
def second(state: dict) -> dict:
    return {"value": 2}


def test_graph_accepts_blueprint() -> None:
    graph = Graph(dict).flow(START >> first, first >> second, second >> END)

    assert graph.start_node == "first"
    assert set(graph.nodes) == {"first", "second"}


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_graph_name_is_rejected(blank: str) -> None:
    with pytest.raises(GraphConfigError, match="Graph name must not be blank"):
        Graph(dict, name=blank)


def test_graph_name_is_kept_verbatim() -> None:
    assert Graph(dict, name=" support ").name == " support "


def test_duplicate_nodes_rejected() -> None:
    with pytest.raises(GraphConfigError, match="already has an outgoing flow"):
        Graph(dict).flow(START >> first, first >> second, first >> END)


@node(name="first")
def duplicate_name(state: dict) -> dict:
    return {"duplicate": True}


def test_duplicate_node_names_rejected() -> None:
    with pytest.raises(GraphConfigError, match="already registered"):
        Graph(dict).flow(START >> first, first >> second, second >> duplicate_name)


def test_duplicate_outgoing_flow_rejected() -> None:
    with pytest.raises(GraphConfigError, match="already has an outgoing flow"):
        Graph(dict).flow(START >> first, first >> second, first >> END)


@node
def third(state: dict) -> dict:
    return {"value": 3}


def test_flow_can_be_defined_only_once() -> None:
    graph = Graph(dict).flow(START >> first, first >> END)

    with pytest.raises(
        GraphConfigError, match="flow already defined for graph 'graph'"
    ):
        graph.flow(START >> second, second >> END)

    assert set(graph.nodes) == {"first"}


async def test_failed_flow_leaves_the_graph_without_a_flow() -> None:
    graph = Graph(dict)

    with pytest.raises(GraphConfigError, match="already has an outgoing flow"):
        graph.flow(START >> first, first >> second, first >> third)

    assert graph.start_node is None
    assert graph.nodes == {}
    assert graph.edges == {}
    assert graph.branch_edges == {}
    with pytest.raises(GraphConfigError, match="has no flow"):
        await graph.ainvoke({})


def test_flow_can_be_defined_after_a_failed_attempt() -> None:
    graph = Graph(dict)
    with pytest.raises(GraphConfigError):
        graph.flow(START >> first, first >> second)

    graph.flow(START >> first, first >> END)

    assert set(graph.nodes) == {"first"}


def test_node_without_outgoing_edge_raises() -> None:
    graph = Graph(dict)

    with pytest.raises(GraphConfigError, match=r"'second' has no outgoing edge"):
        graph.flow(START >> first, first >> second)

    assert graph.start_node is None


def test_flow_error_lists_every_problem() -> None:
    with pytest.raises(GraphConfigError) as info:
        Graph(dict).flow(START >> first, third >> second)

    message = str(info.value)
    assert "Node 'first' has no outgoing edge" in message
    assert "Node 'second' has no outgoing edge" in message
    assert "Node 'second' is not reachable from START" in message
    assert "Node 'third' is not reachable from START" in message


def test_graph_has_no_warnings_attribute() -> None:
    graph = Graph(dict).flow(START >> first, first >> END)

    assert not hasattr(graph, "warnings")


def test_graph_has_no_validate_method() -> None:
    assert not hasattr(Graph(dict), "validate")


@node
def refund(state: dict) -> dict:
    return {"decision": "refund"}


@node
def deny(state: dict) -> dict:
    return {"decision": "deny"}


@node(goto=[refund, deny])
def decide(state: dict) -> dict:
    return {}


def test_declared_goto_targets_count_as_outgoing_and_reachable() -> None:
    graph = Graph(dict).flow(START >> decide, refund >> END, deny >> END)

    assert set(graph.nodes) == {"decide", "refund", "deny"}


def test_declared_goto_target_joins_the_flow_and_needs_its_own_edge() -> None:
    with pytest.raises(
        GraphConfigError,
        match="Node 'deny' has no outgoing edge, branch or goto",
    ):
        Graph(dict).flow(START >> decide, refund >> END)


@node
def settle(state: dict) -> dict:
    return {"path": state["path"] + "c"}


@node(goto=[settle, END])
def review(state: dict) -> Command:
    return Command(update={"path": "b"}, goto=settle)


@node(goto=[review])
def triage(state: dict) -> Command:
    return Command(goto=review)


def test_goto_targets_join_the_flow_through_each_other() -> None:
    graph = Graph(dict).flow(START >> triage, settle >> END)

    assert set(graph.nodes) == {"triage", "review", "settle"}
    assert graph.nodes["review"].handler is review
    assert graph.edges.keys() == {"settle"}
    assert graph.invoke({}).data == {"path": "bc"}


@node(name="refund")
def other_refund(state: dict) -> dict:
    return {"decision": "other"}


@node(goto=[other_refund])
def decide_other(state: dict) -> dict:
    return {}


def test_declared_goto_target_must_be_the_registered_node() -> None:
    with pytest.raises(GraphConfigError, match="different node"):
        Graph(dict).flow(START >> decide_other, decide_other >> refund, refund >> END)


@node(goto=["countdown", END])
def countdown(state: dict) -> Command:
    remaining = state["n"] - 1
    return Command(update={"n": remaining}, goto=countdown if remaining else END)


def test_goto_name_declares_a_self_loop() -> None:
    graph = Graph(dict).flow(START >> countdown)

    assert graph.nodes["countdown"].goto == (countdown, END)
    assert graph.invoke({"n": 3}).data == {"n": 0}


@node(goto=["draft", END])
def check(state: dict) -> Command:
    return Command(goto=END if len(state["text"]) >= 2 else draft)


@node(goto=[check])
def draft(state: dict) -> Command:
    return Command(update={"text": state.get("text", "") + "d"}, goto=check)


def test_goto_name_declares_a_node_defined_later() -> None:
    graph = Graph(dict).flow(START >> draft)

    assert set(graph.nodes) == {"draft", "check"}
    assert graph.nodes["check"].goto == (draft, END)
    assert graph.invoke({}).data == {"text": "dd"}


@node(goto=["missing", "refund"])
def points_nowhere(state: dict) -> dict:
    return {}


def test_unknown_goto_name_raises_at_flow_time() -> None:
    graph = Graph(dict)

    with pytest.raises(GraphConfigError) as info:
        graph.flow(START >> points_nowhere, points_nowhere >> END)

    message = str(info.value)
    assert (
        "Node 'points_nowhere' declares goto target 'missing', but the flow has no "
        "node named 'missing'"
    ) in message
    assert "'refund'" in message
    assert graph.start_node is None


@node(goto=["refund"])
def routes_to_impostor(state: dict) -> Command:
    return Command(goto=other_refund)


def test_goto_name_resolves_to_the_registered_node_only() -> None:
    graph = Graph(dict).flow(
        START >> routes_to_impostor, routes_to_impostor >> refund, refund >> END
    )

    assert graph.nodes["routes_to_impostor"].goto == (refund,)
    with pytest.raises(GraphExecutionError, match="different node named 'refund'"):
        graph.invoke({})
