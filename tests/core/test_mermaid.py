import re

from nodestep.core.command import END, START
from nodestep.core.flow import branch
from nodestep.core.graph import Graph
from nodestep.core.node import node
from nodestep.core.render import mermaid_id


@node
def first(state: dict) -> dict:
    return {}


@node
def second(state: dict) -> dict:
    return {}


def next_step(state: dict) -> str:
    return "more" if state.get("more") else "done"


def _node_id(diagram: str, label: str) -> str:
    line = next(
        candidate for candidate in diagram.splitlines() if f'"{label}"' in candidate
    )
    return line.strip().split("[", 1)[0]


def test_mermaid_renders_linear_route() -> None:
    graph = Graph(dict).flow(START >> first, first >> second, second >> END)
    diagram = graph.to_mermaid()
    assert "graph TD" in diagram
    first_id = _node_id(diagram, "first")
    second_id = _node_id(diagram, "second")
    assert f"{first_id} {first_id}-{second_id}@--> {second_id}" in diagram
    assert f"{second_id} {second_id}-END@--> END" in diagram


def test_mermaid_renders_branch_label() -> None:
    graph = Graph(
        dict,
    ).flow(
        START >> first,
        first >> branch(next_step, {"more": second, "done": END}),
        second >> END,
    )
    diagram = graph.to_mermaid()
    first_id = _node_id(diagram, "first")
    second_id = _node_id(diagram, "second")
    assert f'{first_id} {first_id}-{second_id}@-->|"more"| {second_id}' in diagram


def test_mermaid_renders_end() -> None:
    graph = Graph(dict).flow(START >> first, first >> END)
    diagram = graph.to_mermaid()
    first_id = _node_id(diagram, "first")
    assert f"{first_id} {first_id}-END@--> END" in diagram


@node(name="pay|out")
def piped(state: dict) -> dict:
    return {}


def test_mermaid_escapes_pipe_in_node_name() -> None:
    graph = Graph(dict).flow(START >> piped, piped >> END)
    diagram = graph.to_mermaid()
    assert "pay|out" not in diagram
    assert '["pay#124;out"]' in diagram


def test_mermaid_escapes_characters_mermaid_reads_in_node_names() -> None:
    @node(name='say "hi" <b>#1; & a\\b\nnext')
    def odd(state: dict) -> dict:
        return {}

    diagram = Graph(dict).flow(START >> odd, odd >> END).to_mermaid()

    assert (
        '  say__hi___b__1____a_b_next["say #34;hi#34; #60;b#62;#35;1; #38; a\\b'
        '#10;next"]'
    ) in diagram.splitlines()


def test_mermaid_quotes_and_escapes_branch_labels() -> None:
    def pick(state: dict) -> str:
        return "a"

    graph = Graph(dict).flow(
        START >> first,
        first >> branch(pick, {'yes|no "x"': second, "<i>": END}),
        second >> END,
    )

    lines = graph.to_mermaid().splitlines()

    assert '  first first-second@-->|"yes#124;no #34;x#34;"| second' in lines
    assert '  first first-END@-->|"#60;i#62;"| END([END])' in lines


def test_mermaid_escaped_names_are_referenced_consistently() -> None:
    graph = Graph(dict).flow(START >> piped, piped >> END)
    diagram = graph.to_mermaid()
    node_id = _node_id(diagram, "pay#124;out")
    assert node_id.isidentifier()
    assert f"START START-{node_id}@--> {node_id}" in diagram
    assert f"{node_id} {node_id}-END@--> END" in diagram


def test_mermaid_id_keeps_safe_characters_and_replaces_the_rest() -> None:
    assert mermaid_id("first") == "first"
    assert mermaid_id("Step_2") == "Step_2"
    assert mermaid_id("pay|out") == "pay_out"
    assert mermaid_id("a b-c.d") == "a_b_c_d"
    assert mermaid_id("café") == "caf_"


def test_mermaid_node_ids_derive_from_names() -> None:
    graph = Graph(dict).flow(START >> first, first >> second, second >> END)

    diagram = graph.to_mermaid()

    assert diagram.splitlines() == [
        "graph TD",
        "  START([START])",
        '  first["first"]',
        "  START START-first@--> first",
        '  second["second"]',
        "  first first-second@--> second",
        "  second second-END@--> END([END])",
    ]


@node(name="a b")
def spaced(state: dict) -> dict:
    return {}


@node(name="a_b")
def underscored(state: dict) -> dict:
    return {}


@node(name="a-b")
def dashed(state: dict) -> dict:
    return {}


def test_mermaid_id_collisions_get_numbered_suffixes() -> None:
    graph = Graph(dict).flow(
        START >> spaced, spaced >> underscored, underscored >> dashed, dashed >> END
    )

    diagram = graph.to_mermaid()

    assert _node_id(diagram, "a b") == "a_b"
    assert _node_id(diagram, "a_b") == "a_b_2"
    assert _node_id(diagram, "a-b") == "a_b_3"
    assert "  a_b a_b-a_b_2@--> a_b_2" in diagram
    assert "  a_b_3 a_b_3-END@--> END([END])" in diagram


@node(name="START")
def named_start(state: dict) -> dict:
    return {}


@node(name="end")
def named_end(state: dict) -> dict:
    return {}


def test_mermaid_reserved_ids_are_not_reused_by_nodes() -> None:
    graph = Graph(dict).flow(
        START >> named_start, named_start >> named_end, named_end >> END
    )

    diagram = graph.to_mermaid()

    assert _node_id(diagram, "START") == "START_2"
    assert _node_id(diagram, "end") == "end_2"
    assert "  START START-START_2@--> START_2" in diagram
    assert "  START_2 START_2-end_2@--> end_2" in diagram


@node
def refund(state: dict) -> dict:
    return {}


@node(goto=[refund, END])
def decide(state: dict) -> dict:
    return {}


def test_mermaid_draws_declared_goto_targets_as_dashed_edges() -> None:
    graph = Graph(dict).flow(START >> decide, refund >> END)

    diagram = graph.to_mermaid()

    assert "  decide decide-refund@-.-> refund" in diagram
    assert "  decide decide-END@-.-> END([END])" in diagram
    assert "  decide decide-END@--> END([END])" not in diagram


@node
def settle(state: dict) -> dict:
    return {}


@node(goto=[settle, END])
def review(state: dict) -> dict:
    return {}


@node(goto=[review])
def triage(state: dict) -> dict:
    return {}


def test_mermaid_draws_a_goto_only_target_without_a_solid_edge() -> None:
    graph = Graph(dict).flow(START >> triage, settle >> END)

    diagram = graph.to_mermaid()

    assert '  review["review"]' in diagram
    assert "  triage triage-review@-.-> review" in diagram
    assert "  review review-settle@-.-> settle" in diagram
    assert "  review review-END@-.-> END([END])" in diagram
    assert not re.search(r"^  review \S+@-->", diagram, re.M)


@node(goto=["loop", END])
def loop(state: dict) -> dict:
    return {}


@node(goto=["later"])
def early(state: dict) -> dict:
    return {}


@node(goto=[early, END])
def later(state: dict) -> dict:
    return {}


def test_mermaid_draws_goto_names_including_self_loops() -> None:
    assert "  loop loop-loop@-.-> loop" in Graph(dict).flow(START >> loop).to_mermaid()

    diagram = Graph(dict).flow(START >> later).to_mermaid()

    assert "  later later-early@-.-> early" in diagram
    assert "  early early-later@-.-> later" in diagram


@node(name="a")
def joined_a(state: dict) -> dict:
    return {}


@node(name="b c")
def joined_b_c(state: dict) -> dict:
    return {}


@node(name="a b")
def joined_a_b(state: dict) -> dict:
    return {}


@node(name="c")
def joined_c(state: dict) -> dict:
    return {}


def test_mermaid_edge_ids_of_ids_that_join_alike_do_not_collide() -> None:
    def pick(state: dict) -> str:
        return "one"

    graph = Graph(dict).flow(
        START >> first,
        first >> branch(pick, {"one": joined_a, "two": joined_a_b}),
        joined_a >> joined_b_c,
        joined_a_b >> joined_c,
        joined_b_c >> END,
        joined_c >> END,
    )

    lines = graph.to_mermaid().splitlines()

    assert "  a a-b_c@--> b_c" in lines
    assert "  a_b a_b-c@--> c" in lines


def test_mermaid_edge_ids_of_a_repeated_pair_are_numbered() -> None:
    def pick(state: dict) -> str:
        return "a"

    graph = Graph(dict).flow(
        START >> first,
        first >> branch(pick, {"a": second, "b": second, "c": END, "d": END}),
        second >> END,
    )

    lines = graph.to_mermaid().splitlines()

    assert '  first first-second@-->|"a"| second' in lines
    assert '  first first-second-2@-->|"b"| second' in lines
    assert '  first first-END@-->|"c"| END([END])' in lines
    assert '  first first-END-2@-->|"d"| END([END])' in lines


def test_mermaid_edge_ids_are_unique_and_unlike_node_ids() -> None:
    graph = Graph(dict).flow(START >> later)

    diagram = graph.to_mermaid()
    edge_ids = re.findall(r"^  \w+ (\S+)@", diagram, re.M)
    node_ids = re.findall(r"^  (\w+)\[", diagram, re.M)

    assert edge_ids == ["START-later", "later-early", "later-END", "early-later"]
    assert len(set(edge_ids)) == len(edge_ids)
    assert not set(edge_ids) & {*node_ids, "START", "END"}
