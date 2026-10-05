from typing import Annotated, Any, TypedDict, cast

import pytest
from pydantic import BaseModel

from nodestep import END, START, Graph, NodeContext, Resume, node
from nodestep.chat import Message, ToolCall
from nodestep.core import tool_runner
from nodestep.core.command import (
    GraphInterrupt,
    GraphInterruptGroup,
    InterruptFrame,
    bind_interrupt_frame,
    interrupt,
    unbind_interrupt_frame,
)
from nodestep.core.tool import ToolContext, call_tool, tool
from nodestep.exceptions import ResumeError
from nodestep.middleware.base import Middleware, NodeMiddlewareContext
from nodestep.middleware.interrupt import (
    InterruptRule,
    ToolDecision,
    ToolInterrupt,
    ToolInterruptMiddleware,
)
from nodestep.state.integrations import InMemoryStateStore
from nodestep.utils.reducers import add_messages


class ToolState(TypedDict, total=False):
    messages: Annotated[list[Message], add_messages]
    tool_calls: list[ToolCall]


class NumberResult(BaseModel):
    value: int


@tool(name="adder")
async def adder(value: int) -> NumberResult:
    """Adder."""
    return NumberResult(value=value + 1)


def _make_runner_graph(middleware: ToolInterruptMiddleware, *, state_store=None):
    @node
    def request(_: dict) -> dict:
        return {
            "tool_calls": [
                ToolCall(id="call_1", name="adder", arguments={"value": 1}),
            ]
        }

    runner = tool_runner("tools", tools=[adder])
    return Graph(
        ToolState,
        middleware=[middleware],
        state_store=state_store,
    ).flow(START >> request, request >> runner, runner >> END)


def test_no_rules_lets_tool_execute() -> None:
    middleware = ToolInterruptMiddleware(rules=[])
    graph = _make_runner_graph(middleware)

    result = graph.invoke({"messages": []})

    assert result.data["messages"][0].content == '{"value":2}'


def test_matching_rule_interrupts() -> None:
    middleware = ToolInterruptMiddleware(rules=[InterruptRule(tool="adder")])
    graph = _make_runner_graph(middleware)

    result = graph.invoke({"messages": []}, thread_id="ask")

    assert result.status == "interrupted"
    assert list(result.interrupts) == ["tools:approve_call_1"]
    pending = result.interrupts["tools:approve_call_1"]
    assert pending.id == "approve_call_1"
    assert pending.payload.kind == "tool_interrupt"
    assert pending.payload.tool_name == "adder"


def test_resume_truthy_executes_tool() -> None:
    store = InMemoryStateStore()
    middleware = ToolInterruptMiddleware(rules=[InterruptRule(tool="adder")])
    graph = _make_runner_graph(middleware, state_store=store)

    graph.invoke({"messages": []}, thread_id="approve")
    final = graph.invoke(
        None,
        thread_id="approve",
        resume=Resume(True),
    )

    assert final.data["messages"][0].content == '{"value":2}'


def test_resume_falsy_denies_tool() -> None:
    store = InMemoryStateStore()
    middleware = ToolInterruptMiddleware(rules=[InterruptRule(tool="adder")])
    graph = _make_runner_graph(middleware, state_store=store)

    graph.invoke({"messages": []}, thread_id="deny")
    final = graph.invoke(
        None,
        thread_id="deny",
        resume=Resume(False),
    )

    assert "denied" in final.data["messages"][0].content
    assert final.data["messages"][0].tool_call_id == "call_1"


def test_match_predicate_filters() -> None:
    middleware = ToolInterruptMiddleware(
        rules=[
            InterruptRule(
                tool="adder",
                match=lambda arguments: arguments.get("value", 0) > 10,
            ),
        ]
    )
    graph = _make_runner_graph(middleware)

    result = graph.invoke({"messages": []})
    assert result.data["messages"][0].content == '{"value":2}'


def test_unmatched_tool_runs_without_interrupt() -> None:
    middleware = ToolInterruptMiddleware(rules=[InterruptRule(tool="other_tool")])
    graph = _make_runner_graph(middleware)

    result = graph.invoke({"messages": []})
    assert result.data["messages"][0].content == '{"value":2}'


class AutoApproveMiddleware(Middleware):
    def on_interrupt(
        self, ctx: NodeMiddlewareContext, payload: object
    ) -> object | None:
        if isinstance(payload, ToolInterrupt):
            return True
        return None


def test_on_interrupt_auto_resolves() -> None:
    request_node = _make_request_node()
    runner = _make_runner()
    graph = Graph(
        ToolState,
        middleware=[
            ToolInterruptMiddleware(rules=[InterruptRule(tool="adder")]),
            AutoApproveMiddleware(),
        ],
    ).flow(START >> request_node, request_node >> runner, runner >> END)

    result = graph.invoke({"messages": []})

    assert result.status == "completed"
    assert result.data["messages"][0].content == '{"value":2}'


def test_on_interrupt_fallback_pauses() -> None:
    request_node = _make_request_node()
    runner = _make_runner()
    graph = Graph(
        ToolState,
        middleware=[
            ToolInterruptMiddleware(rules=[InterruptRule(tool="adder")]),
        ],
    ).flow(START >> request_node, request_node >> runner, runner >> END)

    result = graph.invoke({"messages": []}, thread_id="pause")

    assert result.status == "interrupted"


def test_on_interrupt_multiple_interrupts() -> None:
    @node
    def two_interrupts(_: dict) -> dict:
        first = interrupt({"kind": "first"}, id="first")
        second = interrupt({"kind": "second"}, id="second")
        return {"results": [first, second]}

    class ResolveAll(Middleware):
        def on_interrupt(
            self, ctx: NodeMiddlewareContext, payload: object
        ) -> object | None:
            if isinstance(payload, dict):
                kind = cast(dict[str, Any], payload).get("kind")
                if kind is not None:
                    return f"resolved-{kind}"
            return None

    graph = Graph(dict, middleware=[ResolveAll()]).flow(
        START >> two_interrupts, two_interrupts >> END
    )
    result = graph.invoke({})

    assert result.status == "completed"
    assert result.data["results"] == ["resolved-first", "resolved-second"]


class RecoverMiddleware(Middleware):
    def on_error(self, ctx: NodeMiddlewareContext, error: Exception) -> object | None:
        if isinstance(error, ValueError):
            return {"recovered": True}
        return None


def test_on_error_recovers() -> None:
    @node
    def failing(_: dict) -> dict:
        raise ValueError("boom")

    graph = Graph(dict, middleware=[RecoverMiddleware()]).flow(
        START >> failing, failing >> END
    )
    result = graph.invoke({})

    assert result.status == "completed"
    assert result.data["recovered"] is True


def test_on_error_fallback_reraises() -> None:
    @node
    def failing(_: dict) -> dict:
        raise ValueError("boom")

    graph = Graph(dict).flow(START >> failing, failing >> END)

    with pytest.raises(ValueError, match="boom"):
        graph.invoke({})


def test_resume_answers_target_the_interrupt_key() -> None:
    store = InMemoryStateStore()

    @node
    def ask(_: dict) -> dict:
        answer = interrupt({"q": "?"}, id="q")
        return {"answer": answer}

    graph = Graph(dict, state_store=store).flow(START >> ask, ask >> END)

    first = graph.invoke({}, thread_id="rm")
    assert first.status == "interrupted"
    assert list(first.interrupts) == ["ask:q"]

    final = graph.invoke(
        None, thread_id="rm", resume=Resume(answers={"ask:q": "mapped"})
    )
    assert final.data["answer"] == "mapped"


def _make_request_node():
    @node
    def request(_: dict) -> dict:
        return {
            "tool_calls": [
                ToolCall(id="call_1", name="adder", arguments={"value": 1}),
            ]
        }

    return request


def _make_runner():
    return tool_runner("tools", tools=[adder])


def test_interrupt_key_joins_the_task_id_and_the_id() -> None:
    frame = InterruptFrame(task_id="n", resume_values={"n:answered": "yes"})
    token = bind_interrupt_frame(frame)
    try:
        answered = interrupt({"p": 0}, id="answered")
        with pytest.raises(GraphInterrupt) as raised:
            interrupt({"p": 1}, id="tool_a")
    finally:
        unbind_interrupt_frame(token)

    assert answered == "yes"
    assert raised.value.key == "n:tool_a"
    assert raised.value.id == "tool_a"
    assert raised.value.payload == {"p": 1}


def test_graph_interrupt_group_collects_all() -> None:
    first_interrupt = GraphInterrupt("a", id="t1", key="n:t1")
    second_interrupt = GraphInterrupt("b", id="t2", key="n:t2")
    group = GraphInterruptGroup([first_interrupt, second_interrupt])
    assert len(group.interrupts) == 2
    assert "n:t1" in str(group)
    assert "n:t2" in str(group)


class MultResult(BaseModel):
    value: int


@tool(name="mult")
async def mult(value: int) -> MultResult:
    """Mult."""
    return MultResult(value=value * 2)


def test_parallel_tool_interrupts_collected() -> None:
    store = InMemoryStateStore()
    middleware = ToolInterruptMiddleware(rules=[InterruptRule(tool="*")])

    @node
    def request_two(_: dict) -> dict:
        return {
            "tool_calls": [
                ToolCall(id="c1", name="adder", arguments={"value": 1}),
                ToolCall(id="c2", name="mult", arguments={"value": 3}),
            ]
        }

    runner = tool_runner("tools", tools=[adder, mult])
    graph = Graph(ToolState, middleware=[middleware], state_store=store).flow(
        START >> request_two, request_two >> runner, runner >> END
    )

    first = graph.invoke({"messages": []}, thread_id="par-int")
    assert first.status == "interrupted"
    assert sorted(first.interrupts) == ["tools:approve_c1", "tools:approve_c2"]

    final = graph.invoke(
        None,
        thread_id="par-int",
        resume=Resume(answers={"tools:approve_c1": True, "tools:approve_c2": True}),
    )
    assert final.status == "completed"
    messages = final.data["messages"]
    contents = sorted(message.content for message in messages)
    assert contents == ['{"value":2}', '{"value":6}']


sent_mail: list[dict[str, str]] = []


@tool(name="send_mail")
async def send_mail(to: str, body: str) -> str:
    """Send mail."""
    sent_mail.append({"to": to, "body": body})
    return "sent"


def _mail_agent(chat_calls: list[ToolCall]):
    from nodestep import ScriptedChat, build_react_agent
    from nodestep.chat import ChatResponse

    chat = ScriptedChat(
        [ChatResponse(tool_calls=chat_calls), ChatResponse(content="done")]
    )
    agent = build_react_agent(
        chat,
        tools=[send_mail],
        middleware=[ToolInterruptMiddleware(rules=[InterruptRule(tool="send_mail")])],
        state_store=InMemoryStateStore(),
    )
    return agent


def _mail_call(call_id: str = "m1", to: str = "boss@x.com") -> ToolCall:
    return ToolCall(id=call_id, name="send_mail", arguments={"to": to, "body": "hi"})


@pytest.mark.parametrize("response", [None, "deny", 0, {"unexpected": True}, False])
async def test_anything_but_an_approval_denies(response: Any) -> None:
    from nodestep.chat import HumanMessage, ToolMessage

    sent_mail.clear()
    agent = _mail_agent([_mail_call()])
    await agent.ainvoke({"messages": [HumanMessage(content="mail")]}, thread_id="d")

    final = await agent.ainvoke(resume=Resume(response), thread_id="d")

    assert sent_mail == []
    reply = next(
        message
        for message in final.data["messages"]
        if isinstance(message, ToolMessage)
    )
    assert "denied" in (reply.content or "")


@pytest.mark.parametrize(
    "response", [True, ToolDecision(action="approve"), {"action": "approve"}]
)
async def test_explicit_approvals_run_the_tool(response: Any) -> None:
    from nodestep.chat import HumanMessage

    sent_mail.clear()
    agent = _mail_agent([_mail_call()])
    await agent.ainvoke({"messages": [HumanMessage(content="mail")]}, thread_id="a")

    await agent.ainvoke(resume=Resume(response), thread_id="a")

    assert sent_mail == [{"to": "boss@x.com", "body": "hi"}]


async def test_payload_shows_the_call_and_its_arguments() -> None:
    from nodestep.chat import HumanMessage

    agent = _mail_agent([_mail_call()])

    paused = await agent.ainvoke(
        {"messages": [HumanMessage(content="mail")]}, thread_id="p"
    )

    payload = paused.interrupts["act:approve_m1"].payload
    assert payload.tool_name == "send_mail"
    assert payload.tool_call_id == "m1"
    assert payload.arguments == {"to": "boss@x.com", "body": "hi"}


async def test_partial_edit_merges_and_history_keeps_the_models_call() -> None:
    from nodestep.chat import AIMessage, HumanMessage, ToolMessage

    sent_mail.clear()
    agent = _mail_agent([_mail_call()])
    await agent.ainvoke({"messages": [HumanMessage(content="mail")]}, thread_id="e")

    final = await agent.ainvoke(
        resume=Resume(ToolDecision(action="edit", arguments={"to": "team@x.com"})),
        thread_id="e",
    )

    assert sent_mail == [{"to": "team@x.com", "body": "hi"}]
    calls = [
        call
        for message in final.data["messages"]
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]
    assert calls[0].arguments == {"to": "boss@x.com", "body": "hi"}
    [reply] = [
        message
        for message in final.data["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert reply.arguments == {"to": "team@x.com", "body": "hi"}


async def test_every_pending_approval_must_be_answered() -> None:
    from nodestep.chat import HumanMessage

    sent_mail.clear()
    agent = _mail_agent([_mail_call("m1", "a@x.com"), _mail_call("m2", "b@x.com")])
    await agent.ainvoke({"messages": [HumanMessage(content="mail")]}, thread_id="u")

    with pytest.raises(ResumeError, match="act:approve_m2"):
        await agent.ainvoke(
            resume=Resume(answers={"act:approve_m1": True}), thread_id="u"
        )
    final = await agent.ainvoke(
        resume=Resume(answers={"act:approve_m1": True, "act:approve_m2": False}),
        thread_id="u",
    )

    assert final.status == "completed"
    assert sent_mail == [{"to": "a@x.com", "body": "hi"}]


def test_match_predicates_receive_argument_dicts() -> None:
    middleware = ToolInterruptMiddleware(
        rules=[
            InterruptRule(tool="adder", match=lambda args: args.get("value", 0) >= 1)
        ]
    )
    graph = _make_runner_graph(middleware)

    result = graph.invoke({"messages": []}, thread_id="m")

    assert result.status == "interrupted"


def test_tool_call_ids_with_other_characters_are_encoded() -> None:
    store = InMemoryStateStore()
    middleware = ToolInterruptMiddleware(rules=[InterruptRule(tool="adder")])

    @node
    def request_odd_ids(_: dict) -> dict:
        return {
            "tool_calls": [
                ToolCall(id="functions.adder:0", name="adder", arguments={"value": 1}),
                ToolCall(id="functions_adder_0", name="adder", arguments={"value": 2}),
                ToolCall(id="fc:1|x", name="adder", arguments={"value": 3}),
            ]
        }

    runner = tool_runner("tools", tools=[adder])
    graph = Graph(ToolState, middleware=[middleware], state_store=store).flow(
        START >> request_odd_ids, request_odd_ids >> runner, runner >> END
    )

    paused = graph.invoke({"messages": []}, thread_id="odd")
    final = graph.invoke(
        None,
        thread_id="odd",
        resume=Resume(answers=dict.fromkeys(paused.interrupts, True)),
    )

    assert sorted(paused.interrupts) == [
        "tools:approve_fc.3a1.7cx",
        "tools:approve_functions.2eadder.3a0",
        "tools:approve_functions_adder_0",
    ]
    payload = paused.interrupts["tools:approve_fc.3a1.7cx"].payload
    assert payload.tool_call_id == "fc:1|x"
    assert sorted(message.content for message in final.data["messages"]) == [
        '{"value":2}',
        '{"value":3}',
        '{"value":4}',
    ]


async def test_a_gated_call_without_a_tool_call_id_raises() -> None:
    middleware = ToolInterruptMiddleware(rules=[InterruptRule(tool="adder")])

    @node
    async def calls_directly(state: dict, ctx: NodeContext) -> dict:
        await call_tool(adder, {"value": 1}, ToolContext.from_node_context(ctx, state))
        return {}

    graph = Graph(dict, middleware=[middleware]).flow(
        START >> calls_directly, calls_directly >> END
    )

    with pytest.raises(ValueError, match="tool call id"):
        await graph.ainvoke({})


async def test_an_edit_answer_with_the_old_args_key_denies_the_call() -> None:
    from nodestep.chat import HumanMessage, ToolMessage

    sent_mail.clear()
    agent = _mail_agent([_mail_call()])
    await agent.ainvoke({"messages": [HumanMessage(content="mail")]}, thread_id="o")

    final = await agent.ainvoke(
        resume=Resume({"action": "edit", "args": {"to": "team@x.com"}}),
        thread_id="o",
    )

    assert sent_mail == []
    [reply] = [
        message
        for message in final.data["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert "invalid approval response" in (reply.content or "")


async def test_a_dict_edit_answer_uses_arguments() -> None:
    from nodestep.chat import HumanMessage

    sent_mail.clear()
    agent = _mail_agent([_mail_call()])
    await agent.ainvoke({"messages": [HumanMessage(content="mail")]}, thread_id="k")

    await agent.ainvoke(
        resume=Resume({"action": "edit", "arguments": {"to": "team@x.com"}}),
        thread_id="k",
    )

    assert sent_mail == [{"to": "team@x.com", "body": "hi"}]
