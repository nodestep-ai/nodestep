from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field

from nodestep import (
    END,
    START,
    BaseState,
    Command,
    Graph,
    Send,
    add,
    branch,
    node,
)
from nodestep.chat.messages import BaseMessage, HumanMessage
from nodestep.middleware.base import GraphMiddlewareContext, ToolMiddlewareContext
from nodestep.middleware.tool_limit import ToolLimitMiddleware
from nodestep.utils.reducers import add_messages


class CounterState(BaseState):
    count: int = 0


@node
def increment_mutating(state: CounterState) -> CounterState:
    state.count = state.count + 1
    return state


@node
def increment_dict(state: CounterState) -> dict:
    return {"count": state.count + 1}


async def test_basestate_inplace_mutation_produces_update() -> None:
    graph = Graph(CounterState).flow(
        START >> increment_mutating, increment_mutating >> END
    )
    result = await graph.ainvoke({"count": 0})
    assert result.state.count == 1


async def test_basestate_inplace_mutation_chained() -> None:
    @node
    def double_mutating(state: CounterState) -> CounterState:
        state.count = state.count * 2
        return state

    graph = Graph(CounterState).flow(
        START >> increment_mutating,
        increment_mutating >> double_mutating,
        double_mutating >> END,
    )
    result = await graph.ainvoke({"count": 0})
    assert result.state.count == 2


async def test_basestate_dict_return_still_works() -> None:
    graph = Graph(CounterState).flow(START >> increment_dict, increment_dict >> END)
    result = await graph.ainvoke({"count": 5})
    assert result.state.count == 6


class SendPayload(BaseModel):
    value: str = ""


class SendState(BaseModel):
    results: Annotated[list[str], add] = Field(default_factory=list)
    value: str = ""


@node
def collector(state: SendState) -> dict:
    return {}


@node
def worker(state: SendState) -> dict:
    return {"results": [f"done:{state.value}"]}


async def test_send_with_basemodel_payload() -> None:
    @node(goto=[worker])
    def dispatch(state: SendState) -> Command:
        return Command(goto=[Send(node=worker, payload=SendPayload(value="x"))])

    graph = Graph(SendState).flow(START >> dispatch, worker >> END)
    result = await graph.ainvoke({})
    assert result.state.results == ["done:x"]


async def test_send_with_dict_payload() -> None:
    class PayloadState(BaseModel):
        tag: str = ""

    @node
    def tagged(state: PayloadState) -> dict:
        return {"tag": state.tag}

    @node(goto=[tagged])
    def sender(state: PayloadState) -> Command:
        return Command(goto=[Send(node=tagged, payload={"tag": "hello"})])

    graph = Graph(PayloadState).flow(START >> sender, tagged >> END)
    result = await graph.ainvoke({})
    assert result.state.tag == "hello"


async def test_send_with_none_payload() -> None:
    @node
    def noop(state: SendState) -> dict:
        return {"results": ["ok"]}

    @node(goto=[noop])
    def sender(state: SendState) -> Command:
        return Command(goto=[Send(node=noop, payload=None)])

    graph = Graph(SendState).flow(START >> sender, noop >> END)
    result = await graph.ainvoke({})
    assert result.state.results == ["ok"]


async def test_send_with_unsupported_payload_raises() -> None:
    @node
    def target(state: SendState) -> dict:
        return {}

    @node(goto=[target])
    def sender(state: SendState) -> Command:
        return Command(goto=[Send(node=target, payload="bad")])

    graph = Graph(SendState).flow(START >> sender, target >> END)
    with pytest.raises(Exception, match="Send payload must be dict or BaseModel"):
        await graph.ainvoke({})


def _graph_ctx(root_run_id: str) -> GraphMiddlewareContext:
    return GraphMiddlewareContext(
        graph_name="g",
        state={},
        run_id=root_run_id,
        thread_id="t",
        branch_id="main",
        root_run_id=root_run_id,
    )


def _tool_ctx(root_run_id: str, tool_name: str = "t") -> ToolMiddlewareContext:
    return ToolMiddlewareContext(
        graph_name="g",
        node_name="n",
        tool_name=tool_name,
        value=None,
        root_run_id=root_run_id,
    )


def test_tool_limit_resets_between_runs() -> None:
    from nodestep.exceptions import ToolDeniedError

    limit = ToolLimitMiddleware(max_calls=2)
    limit.before_graph(_graph_ctx("r1"))
    limit.before_tool(_tool_ctx("r1"))
    limit.before_tool(_tool_ctx("r1"))
    with pytest.raises(ToolDeniedError):
        limit.before_tool(_tool_ctx("r1"))

    limit.before_graph(_graph_ctx("r2"))

    assert limit.before_tool(_tool_ctx("r2")) is None


def test_tool_limit_per_tool_resets() -> None:
    from nodestep.exceptions import ToolDeniedError

    limit = ToolLimitMiddleware(max_calls=100, per_tool={"write": 1})
    limit.before_graph(_graph_ctx("r1"))
    limit.before_tool(_tool_ctx("r1", "write"))
    with pytest.raises(ToolDeniedError):
        limit.before_tool(_tool_ctx("r1", "write"))

    limit.before_graph(_graph_ctx("r1"))

    assert limit.before_tool(_tool_ctx("r1", "write")) is None


def test_tool_limit_refuses_a_call_outside_a_graph_run() -> None:
    from nodestep.exceptions import GraphConfigError

    limit = ToolLimitMiddleware(max_calls=2)

    for root_run_id in ("", "never-started"):
        with pytest.raises(GraphConfigError, match="graph run"):
            limit.before_tool(_tool_ctx(root_run_id))
    assert limit._budgets == {}


def test_tool_limit_gives_a_resume_it_did_not_see_start_a_budget() -> None:
    from dataclasses import replace

    from nodestep.exceptions import ToolDeniedError

    limit = ToolLimitMiddleware(max_calls=1)
    limit.before_graph(replace(_graph_ctx("r1"), resuming=True))

    assert limit.before_tool(_tool_ctx("r1")) is None
    with pytest.raises(ToolDeniedError):
        limit.before_tool(_tool_ctx("r1"))
    limit.before_graph(replace(_graph_ctx("r1"), resuming=True))
    with pytest.raises(ToolDeniedError):
        limit.before_tool(_tool_ctx("r1"))


def test_end_sentinel_is_truthy() -> None:
    assert bool(END) is True


def test_graph_stream_sync_yields_events() -> None:
    graph = Graph(CounterState).flow(START >> increment_dict, increment_dict >> END)
    events = list(graph.stream({"count": 0}, stream_mode="updates"))
    assert any(event.mode == "final" for event in events)


def test_add_messages_replaces_by_id() -> None:
    first_message = HumanMessage(id="a", content="first")
    second_message = HumanMessage(id="b", content="second")
    updated_message = HumanMessage(id="a", content="updated")

    result = add_messages([first_message, second_message], [updated_message])
    assert len(result) == 2
    assert result[0].content == "updated"
    assert result[1].content == "second"


def test_add_messages_without_id_appends() -> None:
    first_message = HumanMessage(content="first")
    second_message = HumanMessage(content="second")

    result = add_messages([first_message], [second_message])
    assert len(result) == 2


def test_base_message_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseMessage(content="should fail")  # ty: ignore[call-non-callable]


async def test_basestate_mutation_with_branch() -> None:
    class RouterState(BaseState):
        text: str = ""
        intent: str = ""
        answer: str = ""

    @node
    def classify(state: RouterState) -> RouterState:
        state.intent = "support" if "help" in state.text.lower() else "sales"
        return state

    @node
    def support(state: RouterState) -> RouterState:
        state.answer = "turning it off and on"
        return state

    @node
    def sales(state: RouterState) -> RouterState:
        state.answer = "new pricing"
        return state

    def router(state: RouterState) -> str:
        return state.intent

    graph = Graph(RouterState).flow(
        START >> classify,
        classify >> branch(router, {"support": support, "sales": sales}),
        support >> END,
        sales >> END,
    )

    result = await graph.ainvoke({"text": "I need help"})
    assert result.state.intent == "support"
    assert result.state.answer == "turning it off and on"

    result2 = await graph.ainvoke({"text": "pricing info"})
    assert result2.state.intent == "sales"
    assert result2.state.answer == "new pricing"


async def test_ainvoke_rejects_removed_checkpoint_id_parameter() -> None:
    from nodestep import END, START, Graph, node

    @node
    def noop(state: dict) -> dict:
        return {}

    graph = Graph(dict).flow(START >> noop, noop >> END)

    removed: dict[str, Any] = {"checkpoint_id": "anything"}
    with pytest.raises(TypeError):
        await graph.ainvoke({}, **removed)
