from typing import Annotated, Any, TypedDict

import pytest
from pydantic import BaseModel, Field

from nodestep import InMemoryStateStore, Resume, ScriptedChat, build_react_agent
from nodestep.chat import (
    AIMessage,
    ChatResponse,
    HumanMessage,
    Message,
    ToolCall,
    ToolMessage,
)
from nodestep.core import tool_runner
from nodestep.core.command import END, START
from nodestep.core.graph import Graph
from nodestep.core.tool import ToolContext, tool
from nodestep.exceptions import (
    ContextNotProvidedError,
    GraphConfigError,
    ToolExecutionError,
    ToolGroupExecutionError,
)
from nodestep.middleware import InterruptRule, Middleware, ToolInterruptMiddleware
from nodestep.utils.reducers import add_messages


@tool
def echo(value: str) -> str:
    """Echo."""
    return value


class State(BaseModel):
    messages: Annotated[list[Message], add_messages] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)


@pytest.mark.asyncio
async def test_tool_runner_executes_known_tool() -> None:
    runner = tool_runner("act", tools=[echo])
    graph = Graph(State).flow(START >> runner, runner >> END)
    call = ToolCall(id="1", name="echo", arguments={"value": "hi"})

    result = await graph.ainvoke({"tool_calls": [call]})

    assert result.state.tool_calls == []
    assert isinstance(result.state.messages[-1], ToolMessage)
    assert "hi" in (result.state.messages[-1].content or "")


@pytest.mark.asyncio
async def test_tool_runner_unknown_tool_raises_by_default() -> None:
    runner = tool_runner("act", tools=[echo])
    graph = Graph(State).flow(START >> runner, runner >> END)
    call = ToolCall(id="1", name="missing", arguments={})

    with pytest.raises(ToolExecutionError):
        await graph.ainvoke({"tool_calls": [call]})


@pytest.mark.asyncio
async def test_tool_runner_returns_unknown_tool_when_asked() -> None:
    runner = tool_runner("act", tools=[echo], tool_errors="return")
    graph = Graph(State).flow(START >> runner, runner >> END)
    call = ToolCall(id="1", name="missing", arguments={})

    result = await graph.ainvoke({"tool_calls": [call]})

    assert result.state.messages[-1].content == (
        "Tool error: ToolExecutionError: Tool 'missing' execution failed: Unknown tool"
    )


@pytest.mark.asyncio
async def test_tool_runner_clears_tool_calls() -> None:
    runner = tool_runner("act", tools=[echo])
    graph = Graph(State).flow(START >> runner, runner >> END)
    call = ToolCall(id="1", name="echo", arguments={"value": "x"})

    result = await graph.ainvoke({"tool_calls": [call]})

    assert result.state.tool_calls == []


@tool
def boom(value: str) -> str:
    """Boom."""
    raise ValueError(f"boom: {value}")


@pytest.mark.asyncio
async def test_tool_runner_raises_tool_errors_by_default() -> None:
    runner = tool_runner("act", tools=[boom])
    graph = Graph(State).flow(START >> runner, runner >> END)
    call = ToolCall(id="1", name="boom", arguments={"value": "x"})

    with pytest.raises(ValueError, match="boom: x"):
        await graph.ainvoke({"tool_calls": [call]})


@pytest.mark.asyncio
async def test_tool_runner_returns_tool_errors_when_asked() -> None:
    runner = tool_runner("act", tools=[boom], tool_errors="return")
    graph = Graph(State).flow(START >> runner, runner >> END)
    call = ToolCall(id="1", name="boom", arguments={"value": "x"})

    result = await graph.ainvoke({"tool_calls": [call]})

    assert isinstance(result.state.messages[-1], ToolMessage)
    assert result.state.messages[-1].content == "Tool error: ValueError: boom: x"
    assert result.state.messages[-1].tool_call_id == "1"


@pytest.mark.asyncio
async def test_tool_runner_runs_calls_concurrently() -> None:
    import asyncio

    order: list[str] = []

    @tool(name="slow")
    async def slow(tag: str, delay: float) -> str:
        """Slow."""
        order.append(f"{tag}_start")
        await asyncio.sleep(delay)
        order.append(f"{tag}_end")
        return tag

    runner = tool_runner("act", tools=[slow])
    graph = Graph(State).flow(START >> runner, runner >> END)
    calls = [
        ToolCall(id="1", name="slow", arguments={"tag": "a", "delay": 0.05}),
        ToolCall(id="2", name="slow", arguments={"tag": "b", "delay": 0.05}),
    ]

    result = await graph.ainvoke({"tool_calls": calls})

    assert len(result.state.messages) == 2
    assert order[0].endswith("_start")
    assert order[1].endswith("_start")


@tool(name="boom_a")
def boom_a(value: str) -> str:
    """Boom a."""
    raise ValueError(f"boom_a: {value}")


@tool(name="boom_b")
def boom_b(value: str) -> str:
    """Boom b."""
    raise RuntimeError(f"boom_b: {value}")


@pytest.mark.asyncio
async def test_tool_runner_raise_aggregates_all_failures() -> None:
    runner = tool_runner("act", tools=[boom_a, boom_b])
    graph = Graph(State).flow(START >> runner, runner >> END)
    calls = [
        ToolCall(id="1", name="boom_a", arguments={"value": "x"}),
        ToolCall(id="2", name="boom_b", arguments={"value": "y"}),
    ]

    with pytest.raises(ToolGroupExecutionError) as exc_info:
        await graph.ainvoke({"tool_calls": calls})

    error = exc_info.value
    assert len(error.errors) == 2
    messages = str(error)
    assert "boom_a" in messages
    assert "boom_b" in messages


@pytest.mark.asyncio
async def test_tool_runner_raise_single_failure_reraises_original() -> None:
    runner = tool_runner("act", tools=[boom_a])
    graph = Graph(State).flow(START >> runner, runner >> END)
    calls = [ToolCall(id="1", name="boom_a", arguments={"value": "x"})]

    with pytest.raises(ValueError, match="boom_a: x"):
        await graph.ainvoke({"tool_calls": calls})


@pytest.mark.asyncio
async def test_tool_runner_return_surfaces_every_failure_as_message() -> None:
    runner = tool_runner("act", tools=[boom_a, boom_b], tool_errors="return")
    graph = Graph(State).flow(START >> runner, runner >> END)
    calls = [
        ToolCall(id="1", name="boom_a", arguments={"value": "x"}),
        ToolCall(id="2", name="boom_b", arguments={"value": "y"}),
    ]

    result = await graph.ainvoke({"tool_calls": calls})

    contents = [
        message.content or ""
        for message in result.state.messages
        if isinstance(message, ToolMessage)
    ]
    assert len(contents) == 2
    assert any("boom_a: x" in content for content in contents)
    assert any("boom_b: y" in content for content in contents)


charges: list[int] = []
emails: list[str] = []


@tool
def charge_card(amount: int) -> str:
    """Charge card."""
    charges.append(amount)
    return "charged"


@tool
def send_email(to: str) -> str:
    """Send email."""
    emails.append(to)
    return "sent"


def _pay_and_mail_chat() -> ScriptedChat:
    return ScriptedChat(
        [
            ChatResponse(
                tool_calls=[
                    ToolCall(id="c1", name="charge_card", arguments={"amount": 100}),
                    ToolCall(id="c2", name="send_email", arguments={"to": "cfo@x.com"}),
                ]
            ),
            ChatResponse(content="done"),
        ]
    )


def _gated_agent(chat: ScriptedChat, *extra: Middleware):
    return build_react_agent(
        chat,
        tools=[charge_card, send_email],
        middleware=[
            ToolInterruptMiddleware(rules=[InterruptRule(tool="send_email")]),
            *extra,
        ],
        state_store=InMemoryStateStore(),
    )


async def test_ungated_sibling_runs_once_across_interrupt_and_resume() -> None:
    charges.clear()
    emails.clear()
    agent = _gated_agent(_pay_and_mail_chat())

    paused = await agent.ainvoke(
        {"messages": [HumanMessage(content="pay")]}, thread_id="t"
    )
    final = await agent.ainvoke(resume=Resume(True), thread_id="t")

    assert paused.status == "interrupted"
    assert charges == [100]
    assert emails == ["cfo@x.com"]
    assert final.data["final_text"] == "done"


async def test_ungated_sibling_runs_once_when_middleware_auto_approves() -> None:
    charges.clear()
    emails.clear()

    class AutoApprove(Middleware):
        def on_interrupt(self, ctx, payload):
            return True

    agent = _gated_agent(_pay_and_mail_chat(), AutoApprove())

    final = await agent.ainvoke(
        {"messages": [HumanMessage(content="pay")]}, thread_id="t"
    )

    assert final.status == "completed"
    assert charges == [100]
    assert emails == ["cfo@x.com"]


async def test_denied_call_becomes_a_tool_message_and_the_model_continues() -> None:
    charges.clear()
    emails.clear()
    chat = _pay_and_mail_chat()
    agent = _gated_agent(chat)

    await agent.ainvoke({"messages": [HumanMessage(content="pay")]}, thread_id="t")
    final = await agent.ainvoke(resume=Resume(False), thread_id="t")

    assert emails == []
    assert final.data["final_text"] == "done"
    replies = {
        message.tool_call_id: message.content
        for message in final.data["messages"]
        if isinstance(message, ToolMessage)
    }
    assert "denied" in replies["c2"]
    assert len(chat.requests) == 2


async def test_tool_limit_hit_becomes_a_tool_message() -> None:
    chat = ScriptedChat(
        [
            ChatResponse(
                tool_calls=[
                    ToolCall(id="a", name="echo", arguments={"value": "1"}),
                    ToolCall(id="b", name="echo", arguments={"value": "2"}),
                ]
            ),
            ChatResponse(content="ok"),
        ]
    )
    agent = build_react_agent(chat, tools=[echo], max_tool_calls=1)

    final = await agent.ainvoke({"messages": [HumanMessage(content="go")]})

    replies = [
        message.content
        for message in final.data["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert final.data["final_text"] == "ok"
    assert sum("limit" in reply for reply in replies) == 1


class Opaque:
    pass


@tool
def opaque() -> object:
    """Opaque."""
    return Opaque()


async def test_unserializable_result_becomes_a_tool_error() -> None:
    runner = tool_runner("act", tools=[opaque], tool_errors="return")
    graph = Graph(State).flow(START >> runner, runner >> END)

    result = await graph.ainvoke({"tool_calls": [ToolCall(id="o", name="opaque")]})

    message = result.state.messages[-1]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "o"
    assert (message.content or "").startswith("Tool error:")


class RawCalls(TypedDict, total=False):
    messages: Annotated[list[Message], add_messages]
    tool_calls: list[dict[str, Any]]


async def test_invalid_tool_call_entry_keeps_its_id() -> None:
    runner = tool_runner("act", tools=[echo], tool_errors="return")
    graph = Graph(RawCalls).flow(START >> runner, runner >> END)

    result = await graph.ainvoke(
        {"messages": [], "tool_calls": [{"id": "bad1", "arguments": {}}]}
    )

    assert result.data["messages"][-1].tool_call_id == "bad1"


@pytest.mark.parametrize("entry", [{"arguments": {}}, {"id": 7, "arguments": {}}])
async def test_an_invalid_tool_call_entry_without_an_id_raises(
    entry: dict[str, Any],
) -> None:
    from pydantic import ValidationError

    runner = tool_runner("act", tools=[echo], tool_errors="return")
    graph = Graph(RawCalls).flow(START >> runner, runner >> END)

    with pytest.raises(ValidationError, match="ToolCall"):
        await graph.ainvoke({"messages": [], "tool_calls": [entry]})


async def test_invalid_arguments_are_fed_back_and_the_model_retries() -> None:
    chat = ScriptedChat(
        [
            ChatResponse(
                tool_calls=[
                    ToolCall(
                        id="bad",
                        name="echo",
                        arguments_error='Arguments are not valid JSON: {"value": ',
                    )
                ]
            ),
            ChatResponse(
                tool_calls=[ToolCall(id="good", name="echo", arguments={"value": "x"})]
            ),
            ChatResponse(content="fixed"),
        ]
    )
    agent = build_react_agent(chat, tools=[echo], tool_errors="return")

    final = await agent.ainvoke({"messages": [HumanMessage(content="go")]})

    replies = {
        message.tool_call_id: message.content
        for message in final.data["messages"]
        if isinstance(message, ToolMessage)
    }
    assert "not valid JSON" in replies["bad"]
    assert "x" in replies["good"]
    assert final.data["final_text"] == "fixed"


async def test_tool_errors_callable_decides_the_message() -> None:
    seen: list[Exception] = []

    def describe(error: Exception) -> str:
        seen.append(error)
        return "the tool failed"

    runner = tool_runner("act", tools=[boom], tool_errors=describe)
    graph = Graph(State).flow(START >> runner, runner >> END)

    result = await graph.ainvoke(
        {"tool_calls": [ToolCall(id="1", name="boom", arguments={"value": "x"})]}
    )

    assert result.state.messages[-1].content == "the tool failed"
    assert [type(error) for error in seen] == [ValueError]


async def test_tool_errors_callable_must_return_a_string() -> None:
    runner = tool_runner("act", tools=[boom], tool_errors=lambda error: 42)  # ty: ignore[invalid-argument-type]
    graph = Graph(State).flow(START >> runner, runner >> END)

    with pytest.raises(TypeError, match="tool_errors"):
        await graph.ainvoke(
            {"tool_calls": [ToolCall(id="1", name="boom", arguments={"value": "x"})]}
        )


@tool
def read_context(value: str, ctx: ToolContext) -> str:
    """Read the run context."""
    return f"{ctx.context}{value}"


@tool
def misconfigured(value: str) -> str:
    """Raise a configuration error."""
    raise GraphConfigError(f"bad setup for {value}")


@pytest.mark.parametrize("tool_errors", ["return", lambda error: "the tool failed"])
async def test_a_missing_context_propagates_whatever_tool_errors_says(
    tool_errors: Any,
) -> None:
    runner = tool_runner("act", tools=[read_context], tool_errors=tool_errors)
    graph = Graph(State).flow(START >> runner, runner >> END)
    call = ToolCall(id="1", name="read_context", arguments={"value": "x"})

    with pytest.raises(ContextNotProvidedError):
        await graph.ainvoke({"tool_calls": [call]})


@pytest.mark.parametrize("tool_errors", ["return", lambda error: "the tool failed"])
async def test_a_graph_config_error_from_a_tool_propagates_whatever_tool_errors_says(
    tool_errors: Any,
) -> None:
    runner = tool_runner("act", tools=[misconfigured], tool_errors=tool_errors)
    graph = Graph(State).flow(START >> runner, runner >> END)
    call = ToolCall(id="1", name="misconfigured", arguments={"value": "x"})

    with pytest.raises(GraphConfigError, match="bad setup for x"):
        await graph.ainvoke({"tool_calls": [call]})


async def test_the_react_agent_propagates_a_missing_context_with_return() -> None:
    chat = ScriptedChat(
        [
            ChatResponse(
                tool_calls=[
                    ToolCall(id="1", name="read_context", arguments={"value": "x"})
                ]
            ),
            ChatResponse(content="done"),
        ]
    )
    agent = build_react_agent(chat, tools=[read_context], tool_errors="return")

    with pytest.raises(ContextNotProvidedError):
        await agent.ainvoke({"messages": [HumanMessage(content="go")]})


def test_tool_errors_must_be_raise_return_or_a_callable() -> None:
    with pytest.raises(ValueError, match="tool_errors"):
        tool_runner("act", tools=[echo], tool_errors="ignore")  # ty: ignore[invalid-argument-type]


async def test_invalid_arguments_follow_tool_errors() -> None:
    runner = tool_runner("act", tools=[echo])
    graph = Graph(State).flow(START >> runner, runner >> END)
    call = ToolCall(id="bad", name="echo", arguments_error="Arguments are not valid")

    with pytest.raises(ToolExecutionError, match="Arguments are not valid"):
        await graph.ainvoke({"tool_calls": [call]})


async def test_the_react_agent_raises_tool_errors_by_default() -> None:
    chat = ScriptedChat(
        [
            ChatResponse(
                tool_calls=[ToolCall(id="1", name="boom", arguments={"value": "x"})]
            ),
            ChatResponse(content="unreachable"),
        ]
    )
    agent = build_react_agent(chat, tools=[boom])

    with pytest.raises(ValueError, match="boom: x"):
        await agent.ainvoke({"messages": [HumanMessage(content="go")]})

    assert len(chat.requests) == 1


async def test_the_react_agent_returns_tool_errors_when_asked() -> None:
    chat = ScriptedChat(
        [
            ChatResponse(
                tool_calls=[ToolCall(id="1", name="boom", arguments={"value": "x"})]
            ),
            ChatResponse(content="recovered"),
        ]
    )
    agent = build_react_agent(chat, tools=[boom], tool_errors="return")

    final = await agent.ainvoke({"messages": [HumanMessage(content="go")]})

    assert final.data["final_text"] == "recovered"
    reply = chat.requests[1].messages[-1]
    assert reply.content == "Tool error: ValueError: boom: x"


async def test_middleware_tools_are_not_run_implicitly() -> None:
    class Provider(Middleware):
        def tools(self):
            return [boom]

    runner = tool_runner("act", tools=[echo])
    graph = Graph(State, middleware=[Provider()]).flow(START >> runner, runner >> END)

    with pytest.raises(ToolExecutionError, match="Unknown tool"):
        await graph.ainvoke(
            {"tool_calls": [ToolCall(id="1", name="boom", arguments={"value": "x"})]}
        )


async def test_the_react_agent_splices_the_tools_of_its_middleware() -> None:
    class Provider(Middleware):
        def tools(self):
            return [boom]

    chat = ScriptedChat(
        [
            ChatResponse(
                tool_calls=[ToolCall(id="1", name="boom", arguments={"value": "x"})]
            ),
            ChatResponse(content="done"),
        ]
    )
    agent = build_react_agent(
        chat, tools=[echo], middleware=[Provider()], tool_errors="return"
    )

    await agent.ainvoke({"messages": [HumanMessage(content="go")]})

    assert [tool.name for tool in chat.requests[0].tools] == ["echo", "boom"]
    assert chat.requests[1].messages[-1].content == "Tool error: ValueError: boom: x"


def test_the_react_agent_requires_tools() -> None:
    with pytest.raises(TypeError, match="tools"):
        build_react_agent(ScriptedChat())  # ty: ignore[missing-argument]


class PlainMessages(BaseModel):
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)


async def test_messages_field_must_use_add_messages_before_a_tool_runs() -> None:
    charges.clear()
    runner = tool_runner("act", tools=[charge_card])
    graph = Graph(PlainMessages).flow(START >> runner, runner >> END)
    call = ToolCall(id="1", name="charge_card", arguments={"amount": 5})

    with pytest.raises(GraphConfigError, match=r"'messages'.*add_messages"):
        await graph.ainvoke({"tool_calls": [call]})

    assert charges == []


class Upper(Middleware):
    def before_tool(self, ctx):
        return ctx.replace(
            ctx.value.model_copy(update={"value": ctx.value.value.upper()})
        )


async def test_the_models_call_is_kept_and_effective_arguments_are_recorded() -> None:
    chat = ScriptedChat(
        [
            ChatResponse(
                tool_calls=[
                    ToolCall(id="1", name="echo", arguments={"value": "debit"}),
                    ToolCall(id="2", name="echo", arguments={"value": "SAME"}),
                ]
            ),
            ChatResponse(content="done"),
        ]
    )
    agent = build_react_agent(chat, tools=[echo], middleware=[Upper()])

    final = await agent.ainvoke({"messages": [HumanMessage(content="go")]})

    ai_message = final.data["messages"][1]
    assert [call.arguments for call in ai_message.tool_calls] == [
        {"value": "debit"},
        {"value": "SAME"},
    ]
    replies = {
        message.tool_call_id: message
        for message in final.data["messages"]
        if isinstance(message, ToolMessage)
    }
    assert replies["1"].arguments == {"value": "DEBIT"}
    assert replies["2"].arguments is None
    sent_call = chat.requests[1].messages[1]
    assert isinstance(sent_call, AIMessage)
    assert sent_call.tool_calls[0].arguments == {"value": "debit"}
