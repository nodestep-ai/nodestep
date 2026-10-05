from typing import Annotated, Any, Literal, TypedDict, cast

import pytest

from nodestep import (
    END,
    START,
    Command,
    Graph,
    InMemoryStateStore,
    Interrupt,
    Middleware,
    NodeContext,
    Resume,
    add,
    interrupt,
    node,
    subgraph,
)
from nodestep.exceptions import GraphConfigError, GraphExecutionError


@node
def inner_start(state: dict) -> dict:
    return {"report": f"analyzed:{state.get('query', '')}"}


@node
def outer_start(state: dict) -> dict:
    return {"task": state["task"]}


def test_subgraph_runs_inner_graph_with_state_mapping() -> None:
    inner = Graph(dict).flow(START >> inner_start, inner_start >> END)
    analyst = subgraph(
        "data_analyst",
        inner,
        state_in=lambda state: {"query": state["task"]},
        state_out=lambda child_state: {"findings": child_state["report"]},
        child_thread="fresh",
    )

    outer = Graph(dict).flow(
        START >> outer_start, outer_start >> analyst, analyst >> END
    )

    result = outer.invoke({"task": "research X"})

    assert result.data["findings"] == "analyzed:research X"


def test_subgraph_share_passes_the_named_fields_both_ways() -> None:
    inner = Graph(dict).flow(START >> inner_start, inner_start >> END)
    analyst = subgraph(
        "data_analyst", inner, share=["query", "report"], child_thread="fresh"
    )

    outer = Graph(dict).flow(START >> analyst, analyst >> END)

    result = outer.invoke({"query": "hello", "other": 1})

    assert result.data == {"query": "hello", "other": 1, "report": "analyzed:hello"}


def test_subgraph_propagates_thread_id() -> None:
    seen: dict[str, str | None] = {}

    @node
    def inner_node(state: dict, ctx: NodeContext) -> dict:
        seen["thread_id"] = ctx.thread_id
        return {}

    inner = Graph(dict).flow(START >> inner_node, inner_node >> END)
    child = subgraph("child", inner, share=["x"], child_thread="fresh")
    outer = Graph(dict).flow(START >> child, child >> END)

    outer.invoke({}, thread_id="parent")

    assert str(seen["thread_id"]).startswith("parent:child:")


def test_subgraph_isolates_state() -> None:
    inner_seen: list[dict] = []

    @node
    def inner_recorder(state: dict) -> dict:
        inner_seen.append(dict(state))
        return {"echoed": state.get("payload", "missing")}

    inner = Graph(dict).flow(START >> inner_recorder, inner_recorder >> END)
    forward = subgraph(
        "forward",
        inner,
        state_in=lambda state: {"payload": state["payload"]},
        state_out=lambda child_state: {"echoed": child_state["echoed"]},
        child_thread="fresh",
    )

    outer = Graph(dict).flow(START >> forward, forward >> END)

    outer.invoke({"payload": "ping", "secret": "do-not-leak"})

    assert inner_seen == [{"payload": "ping"}]


class Shared(TypedDict, total=False):
    log: Annotated[list[str], add]


class ParentOnly(TypedDict, total=False):
    log: Annotated[list[str], add]


class ChildWithExtra(TypedDict, total=False):
    log: Annotated[list[str], add]
    scratch: str


@node
def gate(state: Shared) -> dict:
    return {"log": [f"gate={interrupt('approve?', id='approve')}"]}


@node
def two_gates(state: Shared) -> dict:
    first = interrupt("first?", id="first")
    second = interrupt("second?", id="second")
    return {"log": [f"gates={first}{second}"]}


@node
def child_step(state: Shared) -> dict:
    return {"log": ["child"]}


@node
def parent_first(state: Shared) -> dict:
    return {"log": ["parent"]}


@node
def child_with_scratch(state: ChildWithExtra) -> dict:
    return {"log": ["child"], "scratch": "internal"}


async def test_child_interrupt_pauses_the_parent() -> None:
    inner = Graph(Shared).flow(START >> gate, gate >> END)
    subgraph_node = subgraph("inner", inner, share=["log"], child_thread="fresh")
    outer = Graph(Shared, state_store=InMemoryStateStore()).flow(
        START >> subgraph_node, subgraph_node >> END
    )

    result = await outer.ainvoke({}, thread_id="p")

    assert result.status == "interrupted"
    assert result.interrupts == {
        "inner/gate:approve": Interrupt(
            key="inner/gate:approve",
            id="approve",
            node="inner",
            task_id="inner",
            payload="approve?",
        )
    }


async def test_nested_subgraphs_prefix_every_level() -> None:
    inner = Graph(Shared).flow(START >> gate, gate >> END)
    inner_node = subgraph("inner", inner, share=["log"], child_thread="fresh")
    middle = Graph(Shared).flow(START >> inner_node, inner_node >> END)
    middle_node = subgraph("middle", middle, share=["log"], child_thread="fresh")
    outer = Graph(Shared, state_store=InMemoryStateStore()).flow(
        START >> middle_node, middle_node >> END
    )

    paused = await outer.ainvoke({}, thread_id="p")
    final = await outer.ainvoke(
        resume=Resume(answers={"middle/inner/gate:approve": "yes"}), thread_id="p"
    )

    assert list(paused.interrupts) == ["middle/inner/gate:approve"]
    assert paused.interrupts["middle/inner/gate:approve"].id == "approve"
    assert final.data["log"] == ["gate=yes"]


async def test_parent_resume_resumes_the_child() -> None:
    inner = Graph(Shared).flow(START >> gate, gate >> END)
    subgraph_node = subgraph("inner", inner, share=["log"], child_thread="fresh")
    outer = Graph(Shared, state_store=InMemoryStateStore()).flow(
        START >> subgraph_node, subgraph_node >> END
    )

    await outer.ainvoke({}, thread_id="p")
    final = await outer.ainvoke(resume=Resume("yes"), thread_id="p")

    assert final.status == "completed"
    assert final.data["log"] == ["gate=yes"]


async def test_child_asking_twice_needs_two_parent_resumes() -> None:
    inner = Graph(Shared).flow(START >> two_gates, two_gates >> END)
    subgraph_node = subgraph("inner", inner, share=["log"], child_thread="fresh")
    outer = Graph(Shared, state_store=InMemoryStateStore()).flow(
        START >> subgraph_node, subgraph_node >> END
    )

    await outer.ainvoke({}, thread_id="p")
    middle = await outer.ainvoke(resume=Resume("1"), thread_id="p")
    final = await outer.ainvoke(resume=Resume("2"), thread_id="p")

    assert list(middle.interrupts) == ["inner/two_gates:second"]
    assert final.data["log"] == ["gates=12"]


async def test_passthrough_adds_only_the_child_changes() -> None:
    inner = Graph(Shared).flow(START >> child_step, child_step >> END)
    subgraph_node = subgraph("inner", inner, share=["log"], child_thread="fresh")
    outer = Graph(Shared).flow(
        START >> parent_first, parent_first >> subgraph_node, subgraph_node >> END
    )

    result = await outer.ainvoke({})

    assert result.data["log"] == ["parent", "child"]


async def test_passthrough_drops_child_only_fields() -> None:
    inner = Graph(ChildWithExtra).flow(
        START >> child_with_scratch, child_with_scratch >> END
    )
    subgraph_node = subgraph("inner", inner, share=["log"], child_thread="fresh")
    outer = Graph(ParentOnly).flow(START >> subgraph_node, subgraph_node >> END)

    result = await outer.ainvoke({"log": ["seed"]})

    assert result.data == {"log": ["seed", "child"]}


async def test_each_invocation_starts_a_fresh_child() -> None:
    inner = Graph(Shared).flow(START >> child_step, child_step >> END)
    subgraph_node = subgraph("inner", inner, share=["log"], child_thread="fresh")
    store = InMemoryStateStore()
    outer = Graph(Shared, state_store=store).flow(
        START >> subgraph_node, subgraph_node >> END
    )

    await outer.ainvoke({}, thread_id="p")
    second = await outer.ainvoke({}, thread_id="p")

    assert second.data["log"] == ["child", "child"]


async def test_parent_and_child_interrupts_are_answered_together() -> None:
    from nodestep import Send

    @node
    def ask_parent(state: Shared) -> dict:
        return {"log": [f"parent={interrupt('parent?', id='parent')}"]}

    inner = Graph(Shared).flow(START >> gate, gate >> END)
    subgraph_node = subgraph("inner", inner, share=["log"], child_thread="fresh")

    @node(goto=[ask_parent, subgraph_node])
    def split(state: Shared) -> Command:
        return Command(goto=[Send(ask_parent, {}), Send(subgraph_node, {})])

    outer = Graph(Shared, state_store=InMemoryStateStore()).flow(
        START >> split, ask_parent >> END, subgraph_node >> END
    )

    paused = await outer.ainvoke({}, thread_id="p")
    final = await outer.ainvoke(
        resume=Resume(answers={"ask_parent:parent": "ok", "inner/gate:approve": "yes"}),
        thread_id="p",
    )

    assert sorted(paused.interrupts) == ["ask_parent:parent", "inner/gate:approve"]
    assert paused.interrupts["inner/gate:approve"].payload == "approve?"
    assert sorted(final.data["log"]) == ["gate=yes", "parent=ok"]


async def test_fanned_out_subgraphs_get_distinct_interrupt_ids() -> None:
    from nodestep import Send

    class Job(TypedDict, total=False):
        log: Annotated[list[str], add]
        who: str

    @node
    def approve_job(state: Job) -> dict:
        return {"log": [f"{state['who']}={interrupt(state['who'], id='job')}"]}

    inner = Graph(Job).flow(START >> approve_job, approve_job >> END)
    subgraph_node = subgraph("sub", inner, share=["log", "who"], child_thread="fresh")

    @node(goto=[subgraph_node])
    def split(state: Job) -> Command:
        return Command(
            goto=[Send(subgraph_node, {"who": "x"}), Send(subgraph_node, {"who": "y"})]
        )

    outer = Graph(Job, state_store=InMemoryStateStore()).flow(
        START >> split, subgraph_node >> END
    )

    first = await outer.ainvoke({}, thread_id="p")
    by_payload = {item.payload: key for key, item in first.interrupts.items()}
    final = await outer.ainvoke(
        resume=Resume(answers={by_payload["x"]: "ok-x", by_payload["y"]: "ok-y"}),
        thread_id="p",
    )

    assert sorted(first.interrupts) == ["sub/approve_job:job", "sub[1]/approve_job:job"]
    assert sorted(final.data["log"]) == ["x=ok-x", "y=ok-y"]


async def test_share_adds_only_the_child_messages_to_the_parent() -> None:
    from nodestep.chat import AIMessage, HumanMessage, Message
    from nodestep.utils.reducers import add_messages

    class Chat(TypedDict, total=False):
        messages: Annotated[list[Message], add_messages]

    @node
    def answer(state: Chat) -> dict:
        return {"messages": [AIMessage(content="child answer")]}

    inner = Graph(Chat).flow(START >> answer, answer >> END)
    subgraph_node = subgraph("helper", inner, share=["messages"], child_thread="fresh")
    outer = Graph(Chat).flow(START >> subgraph_node, subgraph_node >> END)

    result = await outer.ainvoke(
        {
            "messages": [
                HumanMessage(content="earlier turn 1"),
                AIMessage(content="earlier reply"),
            ]
        }
    )

    assert [message.content for message in result.data["messages"]] == [
        "earlier turn 1",
        "earlier reply",
        "child answer",
    ]


async def test_share_with_a_store_keeps_the_parent_history() -> None:
    from nodestep.chat import AIMessage, HumanMessage, Message
    from nodestep.utils.reducers import add_messages

    class Loose(TypedDict, total=False):
        messages: Annotated[list[Message], add_messages]
        question: str

    @node
    def answer(state: Loose) -> dict:
        return {"messages": [AIMessage(content="child answer")]}

    inner = Graph(Loose).flow(START >> answer, answer >> END)
    subgraph_node = subgraph("helper", inner, share=["messages"], child_thread="fresh")
    outer = Graph(Loose, state_store=InMemoryStateStore()).flow(
        START >> subgraph_node, subgraph_node >> END
    )

    result = await outer.ainvoke(
        {
            "messages": [
                HumanMessage(content="earlier turn 1"),
                AIMessage(content="earlier reply"),
            ],
            "question": "q?",
        },
        thread_id="m",
    )

    assert [message.content for message in result.data["messages"]] == [
        "earlier turn 1",
        "earlier reply",
        "child answer",
    ]


class AnswersAsParent(Middleware):
    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    def on_interrupt(self, ctx, payload):
        self.seen.append((ctx.node_name, payload))
        return "parent-auto"


class AnswersAsChild(Middleware):
    def on_interrupt(self, ctx, payload):
        return "child-auto"


async def test_parent_on_interrupt_answers_a_child_interrupt() -> None:
    inner = Graph(Shared).flow(START >> gate, gate >> END)
    subgraph_node = subgraph("inner", inner, share=["log"], child_thread="fresh")
    parent_middleware = AnswersAsParent()
    outer = Graph(Shared, middleware=[parent_middleware]).flow(
        START >> subgraph_node, subgraph_node >> END
    )

    result = await outer.ainvoke({})

    assert result.status == "completed"
    assert result.data["log"] == ["gate=parent-auto"]
    assert parent_middleware.seen == [("inner", "approve?")]


async def test_the_child_on_interrupt_answers_before_the_parent() -> None:
    inner = Graph(Shared, middleware=[AnswersAsChild()]).flow(
        START >> gate, gate >> END
    )
    subgraph_node = subgraph("inner", inner, share=["log"], child_thread="fresh")
    parent_middleware = AnswersAsParent()
    outer = Graph(Shared, middleware=[parent_middleware]).flow(
        START >> subgraph_node, subgraph_node >> END
    )

    result = await outer.ainvoke({})

    assert result.data["log"] == ["gate=child-auto"]
    assert parent_middleware.seen == []


async def test_the_outermost_on_interrupt_answers_through_nested_subgraphs() -> None:
    inner = Graph(Shared).flow(START >> gate, gate >> END)
    inner_node = subgraph("inner", inner, share=["log"], child_thread="fresh")
    middle = Graph(Shared).flow(START >> inner_node, inner_node >> END)
    middle_node = subgraph("middle", middle, share=["log"], child_thread="fresh")
    parent_middleware = AnswersAsParent()
    outer = Graph(
        Shared, state_store=InMemoryStateStore(), middleware=[parent_middleware]
    ).flow(START >> middle_node, middle_node >> END)

    result = await outer.ainvoke({}, thread_id="n")

    assert result.status == "completed"
    assert result.data["log"] == ["gate=parent-auto"]
    assert parent_middleware.seen == [("middle", "approve?")]


def concat(current: list[str] | None, update: list[str]) -> list[str]:
    return [*(current or []), *update]


class Logged(TypedDict):
    log: Annotated[list[str], concat]


@node
def child_log(state: Logged) -> dict:
    return {"log": ["child"]}


async def test_a_shared_custom_reducer_field_needs_state_in_and_state_out() -> None:
    child = Graph(Logged, name="child").flow(START >> child_log, child_log >> END)
    subgraph_node = subgraph("sub", child, share=["log"], child_thread="fresh")
    parent = Graph(Logged, name="parent").flow(
        START >> subgraph_node, subgraph_node >> END
    )

    with pytest.raises(
        GraphExecutionError,
        match=r"^Subgraph node 'sub' cannot derive the parent update from child "
        r"graph 'child': field 'log' has a custom reducer; pass state_in= and "
        r"state_out= instead of share=$",
    ):
        await parent.ainvoke({"log": ["parent"]})

    explicit = subgraph(
        "sub",
        child,
        state_in=lambda state: {"log": state["log"]},
        state_out=lambda data: {"log": data["log"][1:]},
        child_thread="fresh",
    )
    fixed = Graph(Logged, name="parent").flow(START >> explicit, explicit >> END)

    assert (await fixed.ainvoke({"log": ["parent"]})).data == {
        "log": ["parent", "child"]
    }


def _child() -> Graph:
    return Graph(Shared, name="child").flow(START >> child_step, child_step >> END)


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"state_in": dict},
        {"state_out": dict},
        {"share": ["log"], "state_in": dict, "state_out": dict},
        {"share": ["log"], "state_out": dict},
        {"share": []},
        {"share": "log"},
    ],
)
def test_subgraph_needs_either_both_mappings_or_share(arguments: dict) -> None:
    with pytest.raises(TypeError, match="state_in= and state_out=, or share="):
        subgraph("sub", _child(), child_thread="fresh", **arguments)


def test_subgraph_refuses_an_unknown_child_thread_policy() -> None:
    with pytest.raises(ValueError, match="child_thread must be 'fresh' or 'stable'"):
        subgraph("sub", _child(), share=["log"], child_thread=cast(Any, "sticky"))


def test_subgraph_refuses_a_shared_field_the_child_does_not_declare() -> None:
    with pytest.raises(GraphConfigError, match="'missing'"):
        subgraph("sub", _child(), share=["log", "missing"], child_thread="fresh")


def test_subgraph_refuses_a_child_graph_with_its_own_store() -> None:
    child = Graph(Shared, name="child", state_store=InMemoryStateStore()).flow(
        START >> child_step, child_step >> END
    )

    with pytest.raises(GraphConfigError, match="child graph 'child' has its own"):
        subgraph("sub", child, share=["log"], child_thread="fresh")


async def test_a_shared_field_the_parent_does_not_declare_raises() -> None:
    class Other(TypedDict, total=False):
        text: str

    subgraph_node = subgraph("sub", _child(), share=["log"], child_thread="fresh")
    parent = Graph(Other, name="parent").flow(
        START >> subgraph_node, subgraph_node >> END
    )

    with pytest.raises(GraphConfigError, match=r"parent graph 'parent'.*'log'"):
        await parent.ainvoke({})


async def test_state_out_must_return_a_dict() -> None:
    subgraph_node = subgraph(
        "sub",
        _child(),
        state_in=lambda state: {},
        state_out=lambda data: data["log"],
        child_thread="fresh",
    )
    parent = Graph(Shared, name="parent").flow(
        START >> subgraph_node, subgraph_node >> END
    )

    with pytest.raises(TypeError, match=r"state_out of subgraph node 'sub'.*list"):
        await parent.ainvoke({})


class Counter(TypedDict, total=False):
    log: Annotated[list[str], add]
    turns: int


@node
def count_turn(state: Counter) -> dict:
    turns = state.get("turns", 0) + 1
    return {"turns": turns, "log": [f"turn {turns}"]}


def _counting_child() -> Graph:
    return Graph(Counter, name="counter").flow(START >> count_turn, count_turn >> END)


def _mapped(child_thread: Literal["fresh", "stable"]) -> Graph:
    subgraph_node = subgraph(
        "sub",
        _counting_child(),
        state_in=lambda state: {},
        state_out=lambda data: {"log": data["log"][-1:]},
        child_thread=child_thread,
    )
    return Graph(Shared, name="parent", state_store=InMemoryStateStore()).flow(
        START >> subgraph_node, subgraph_node >> END
    )


async def test_a_fresh_child_forgets_between_parent_turns() -> None:
    parent = _mapped("fresh")

    await parent.ainvoke({}, thread_id="p")
    second = await parent.ainvoke({}, thread_id="p")

    assert second.data["log"] == ["turn 1", "turn 1"]


async def test_a_stable_child_remembers_between_parent_turns() -> None:
    parent = _mapped("stable")

    await parent.ainvoke({}, thread_id="p")
    second = await parent.ainvoke({}, thread_id="p")
    child = await _counting_child().get_state("p:sub", state_store=parent.state_store)

    assert second.data["log"] == ["turn 1", "turn 2"]
    assert child.value["turns"] == 2


async def test_a_stable_child_with_share_adds_only_its_changes() -> None:
    subgraph_node = subgraph(
        "sub", _counting_child(), share=["log"], child_thread="stable"
    )
    parent = Graph(Shared, name="parent", state_store=InMemoryStateStore()).flow(
        START >> subgraph_node, subgraph_node >> END
    )

    await parent.ainvoke({}, thread_id="p")
    second = await parent.ainvoke({}, thread_id="p")

    assert second.data["log"] == ["turn 1", "turn 2"]


async def test_a_stable_child_sees_the_parent_value_of_shared_fields() -> None:
    subgraph_node = subgraph(
        "sub", _counting_child(), share=["log"], child_thread="stable"
    )
    store = InMemoryStateStore()
    parent = Graph(Shared, name="parent", state_store=store).flow(
        START >> subgraph_node, subgraph_node >> END
    )

    for _ in range(3):
        result = await parent.ainvoke({}, thread_id="p")
    child = await _counting_child().get_state("p:sub", state_store=store)

    assert result.data["log"] == ["turn 1", "turn 2", "turn 3"]
    assert child.value["log"] == ["turn 1", "turn 2", "turn 3"]
    assert child.value["turns"] == 3


async def test_share_resumes_a_child_holding_an_infinite_float() -> None:
    from pydantic import BaseModel, ConfigDict, Field

    class Wide(BaseModel):
        model_config = ConfigDict(ser_json_inf_nan="constants")
        x: float = 0.0
        log: Annotated[list[str], add] = Field(default_factory=list)

    @node
    def ask(state: Wide) -> dict:
        interrupt("ok?", id="q")
        return {"log": ["asked"]}

    @node
    def set_x(state: Wide) -> dict:
        return {"x": float("inf")}

    child = Graph(Wide, name="child").flow(START >> ask, ask >> END)
    subgraph_node = subgraph("sub", child, share=["x", "log"], child_thread="fresh")
    parent = Graph(Wide, name="parent", state_store=InMemoryStateStore()).flow(
        START >> set_x, set_x >> subgraph_node, subgraph_node >> END
    )

    await parent.ainvoke({}, thread_id="t")
    result = await parent.ainvoke(None, thread_id="t", resume=Resume(True))

    assert result.status == "completed"
    assert result.data["x"] == float("inf")
    assert result.data["log"] == ["asked"]


async def test_a_stable_child_needs_a_state_store() -> None:
    subgraph_node = subgraph(
        "sub", _counting_child(), share=["log"], child_thread="stable"
    )
    parent = Graph(Shared, name="parent").flow(
        START >> subgraph_node, subgraph_node >> END
    )

    with pytest.raises(GraphConfigError, match=r"child_thread='stable'.*state store"):
        await parent.ainvoke({})


async def test_a_stable_child_refuses_to_run_twice_at_once() -> None:
    from nodestep import Send

    subgraph_node = subgraph(
        "sub", _counting_child(), share=["log"], child_thread="stable"
    )

    @node(goto=[subgraph_node])
    def split(state: Shared) -> Command:
        return Command(goto=[Send(subgraph_node, {}), Send(subgraph_node, {})])

    parent = Graph(Shared, name="parent", state_store=InMemoryStateStore()).flow(
        START >> split, subgraph_node >> END
    )

    with pytest.raises(GraphExecutionError, match="'p:sub' is already running"):
        await parent.ainvoke({}, thread_id="p")


async def test_a_stable_child_refuses_a_forked_parent_branch() -> None:
    parent = _mapped("stable")
    await parent.ainvoke({}, thread_id="p")
    start = (await parent.history("p")).events[0]
    fork = await parent.fork("p", from_=start.id)

    with pytest.raises(GraphExecutionError, match="branch"):
        await parent.ainvoke({}, thread_id="p", branch_id=fork.id)


async def test_a_fork_paused_inside_a_stable_child_resumes_its_own_child_run() -> None:
    inner = Graph(Shared, name="child").flow(START >> gate, gate >> END)
    subgraph_node = subgraph("sub", inner, share=["log"], child_thread="stable")
    parent = Graph(Shared, name="parent", state_store=InMemoryStateStore()).flow(
        START >> subgraph_node, subgraph_node >> END
    )
    await parent.ainvoke({}, thread_id="p")
    pause = next(
        event
        for event in (await parent.history("p")).events
        if event.type == "superstep_pending"
    )
    fork = await parent.fork("p", from_=pause.id)

    on_fork = await parent.ainvoke(
        None, thread_id="p", branch_id=fork.id, resume=Resume("no")
    )
    on_main = await parent.ainvoke(None, thread_id="p", resume=Resume("yes"))

    assert on_fork.data["log"] == ["gate=no"]
    assert on_main.data["log"] == ["gate=yes"]
