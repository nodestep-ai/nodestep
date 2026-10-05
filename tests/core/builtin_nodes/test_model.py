import json
from typing import Annotated, Any, TypedDict

import pytest
from pydantic import BaseModel, Field

from nodestep import ScriptedChat
from nodestep.chat import (
    AIMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    HumanMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from nodestep.core import model_node
from nodestep.core.builtin_nodes.agent import AgentState
from nodestep.core.builtin_nodes.model import StructuredOutputEvent
from nodestep.core.command import END, START
from nodestep.core.graph import Graph
from nodestep.core.tool import tool
from nodestep.exceptions import (
    ChatHistoryError,
    GraphConfigError,
    StructuredOutputError,
)
from nodestep.middleware import Middleware, ModelMiddlewareContext
from nodestep.state import to_json_value
from nodestep.utils.reducers import add_messages


class FakeChat:
    model = "fake-model"

    def __init__(self, response: ChatResponse) -> None:
        self.response = response
        self.last_request: ChatRequest | None = None
        self.calls: list[str] = []

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls.append("complete")
        self.last_request = request
        return self.response

    async def stream(self, request: ChatRequest):  # type: ignore[override]
        self.calls.append("stream")
        self.last_request = request
        yield ChatStreamChunk(content_delta=self.response.content)
        yield ChatStreamChunk(finish_reason="stop")


@tool
def example(value: int) -> int:
    """Example."""
    return value


class State(BaseModel):
    messages: Annotated[list[Message], add_messages] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_text: str | None = None


def _one_node_graph(think: Any, state: Any = State, **kwargs: Any) -> Graph:
    return Graph(state, **kwargs).flow(START >> think, think >> END)


def _hi() -> dict:
    return {"messages": [HumanMessage(content="hi")]}


async def test_model_node_records_text_when_no_tool_calls() -> None:
    chat = FakeChat(ChatResponse(content="hello"))
    think = model_node(
        "think", chat=chat, system_prompt="you are friendly", tools=[example]
    )

    result = await _one_node_graph(think).ainvoke(_hi())

    assert chat.last_request is not None
    assert chat.last_request.messages[0].content == "you are friendly"
    assert chat.last_request.tools[0].name == "example"
    assert result.state.final_text == "hello"
    assert isinstance(result.state.messages[-1], AIMessage)


async def test_model_node_records_tool_calls() -> None:
    call = ToolCall(id="call-1", name="example", arguments={"value": 1})
    chat = FakeChat(ChatResponse(content=None, tool_calls=[call]))
    think = model_node("think", chat=chat, tools=[example])

    result = await _one_node_graph(think).ainvoke(_hi())

    assert result.state.tool_calls == [call]
    assert result.state.final_text is None


async def test_middleware_tools_are_not_offered_implicitly() -> None:
    @tool(name="middleware_tool")
    async def middleware_tool() -> int:
        """Middleware tool."""
        return 7

    class ToolProvider(Middleware):
        def tools(self):
            return [middleware_tool]

    chat = FakeChat(ChatResponse(content="ok"))
    think = model_node("think", chat=chat, tools=[example])

    await _one_node_graph(think, middleware=[ToolProvider()]).ainvoke(_hi())

    assert chat.last_request is not None
    assert [tool.name for tool in chat.last_request.tools] == ["example"]


async def test_middleware_tools_are_offered_when_spliced_explicitly() -> None:
    @tool(name="middleware_tool")
    async def middleware_tool() -> int:
        """Middleware tool."""
        return 7

    class ToolProvider(Middleware):
        def tools(self):
            return [middleware_tool]

    provider = ToolProvider()
    chat = FakeChat(ChatResponse(content="ok"))
    think = model_node("think", chat=chat, tools=[*provider.tools(), example])

    await _one_node_graph(think, middleware=[provider]).ainvoke(_hi())

    assert chat.last_request is not None
    assert [tool.name for tool in chat.last_request.tools] == [
        "middleware_tool",
        "example",
    ]


def test_graph_takes_no_tools() -> None:
    with pytest.raises(TypeError, match="tools"):
        Graph(State, tools=[example])  # ty: ignore[unknown-argument]


def test_two_different_tools_with_one_name_are_rejected() -> None:
    @tool(name="example")
    def impostor(value: int) -> int:
        """Impostor."""
        return -value

    with pytest.raises(GraphConfigError, match="example"):
        model_node(
            "think", chat=FakeChat(ChatResponse(content="x")), tools=[example, impostor]
        )


async def test_tool_choice_is_forwarded() -> None:
    chat = ScriptedChat(["ok"])
    think = model_node("think", chat=chat, tools=[example], tool_choice="required")

    await _one_node_graph(think).ainvoke(_hi())

    assert chat.requests[0].tool_choice == "required"


class PlainMessages(BaseModel):
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_text: str | None = None


class NoToolCalls(TypedDict, total=False):
    messages: Annotated[list[Message], add_messages]
    final_text: str | None


class NoFinalText(TypedDict, total=False):
    messages: Annotated[list[Message], add_messages]
    tool_calls: list[ToolCall]


@pytest.mark.parametrize(
    ("state", "match"),
    [
        (PlainMessages, r"'messages'.*add_messages"),
        (dict, r"'messages'.*add_messages"),
        (NoToolCalls, r"'tool_calls'"),
        (NoFinalText, r"'final_text'"),
    ],
)
async def test_fields_are_checked_before_the_model_is_called(
    state: Any, match: str
) -> None:
    chat = ScriptedChat(["hi"])
    think = model_node("think", chat=chat)

    with pytest.raises(GraphConfigError, match=match):
        await _one_node_graph(think, state).ainvoke({"messages": []})

    assert chat.requests == []


async def test_a_misnamed_messages_field_raises_before_the_model_is_called() -> None:
    chat = ScriptedChat(["hi"])
    think = model_node("think", chat=chat, messages_field="chat_history")

    with pytest.raises(GraphConfigError, match="'chat_history'"):
        await _one_node_graph(think).ainvoke(_hi())

    assert chat.requests == []


class Answer(BaseModel):
    x: int


async def test_the_output_field_is_checked_before_the_model_is_called() -> None:
    chat = ScriptedChat(['{"x": 1}'])
    think = model_node("think", chat=chat, output_schema=Answer)

    with pytest.raises(GraphConfigError, match="'final_output'"):
        await _one_node_graph(think).ainvoke(_hi())

    assert chat.requests == []


class AnswerState(AgentState):
    final_output: Answer | None = None


async def test_model_node_with_output_schema() -> None:
    chat = ScriptedChat([json.dumps({"x": 3})])
    think = model_node("think", chat=chat, output_schema=Answer)

    result = await _one_node_graph(think, AnswerState).ainvoke(_hi())

    assert chat.requests[0].output_schema is not None
    assert result.data["final_output"] == Answer(x=3)


async def test_invalid_structured_output_raises_by_default() -> None:
    think = model_node(
        "think", chat=ScriptedChat(["this is not json"]), output_schema=Answer
    )

    with pytest.raises(StructuredOutputError) as error:
        await _one_node_graph(think, AnswerState).ainvoke(_hi())

    assert error.value.schema_name == "Answer"
    assert error.value.raw_content == "this is not json"


async def test_invalid_structured_output_can_be_emitted_instead() -> None:
    think = model_node(
        "think",
        chat=ScriptedChat(["this is not json"]),
        output_schema=Answer,
        on_invalid="emit",
    )
    graph = _one_node_graph(think, AnswerState)

    events = [
        event
        async for event in graph.astream(
            {"messages": [HumanMessage(content="test")], "final_output": Answer(x=1)},
            stream_mode=["custom", "values"],
        )
    ]

    [custom] = [event.data for event in events if event.mode == "custom"]
    assert isinstance(custom, StructuredOutputEvent)
    assert custom.schema_name == "Answer"
    assert custom.raw_content == "this is not json"
    assert to_json_value(custom)["kind"] == "structured_output_error"
    [*_, final] = [event.data for event in events if event.mode == "values"]
    assert isinstance(final, dict)
    assert final["final_output"] is None
    assert final["final_text"] == "this is not json"


def test_on_invalid_must_be_raise_or_emit() -> None:
    with pytest.raises(ValueError, match="on_invalid"):
        model_node(
            "think",
            chat=ScriptedChat(),
            output_schema=Answer,
            on_invalid="ignore",  # ty: ignore[invalid-argument-type]
        )


async def test_refusal_is_kept_on_the_message_and_is_not_the_final_text() -> None:
    think = model_node(
        "think", chat=ScriptedChat([ChatResponse(refusal="I can't help with that.")])
    )

    result = await _one_node_graph(think).ainvoke(_hi())

    assert result.data["final_text"] is None
    assert result.data["messages"][-1].refusal == "I can't help with that."


async def test_refusal_with_content_is_not_the_final_text() -> None:
    think = model_node(
        "think", chat=ScriptedChat([ChatResponse(content="partial", refusal="no")])
    )

    result = await _one_node_graph(think).ainvoke(_hi())

    assert result.data["final_text"] is None
    assert result.data["messages"][-1].content == "partial"


async def test_final_text_is_reset_on_tool_call_turns() -> None:
    call = ToolCall(id="c1", name="example", arguments={"value": 1})
    think = model_node(
        "think",
        chat=ScriptedChat([ChatResponse(content="Checking.", tool_calls=[call])]),
        tools=[example],
    )

    result = await _one_node_graph(think).ainvoke(
        {**_hi(), "final_text": "Your balance is $10."}
    )

    assert result.data["final_text"] is None


async def test_final_output_is_reset_on_tool_call_turns() -> None:
    call = ToolCall(id="c1", name="example", arguments={"value": 1})
    think = model_node(
        "think",
        chat=ScriptedChat([ChatResponse(tool_calls=[call])]),
        tools=[example],
        output_schema=Answer,
    )

    result = await _one_node_graph(think, AnswerState).ainvoke(
        {**_hi(), "final_output": Answer(x=1)}
    )

    assert result.data["final_output"] is None


async def test_finish_reason_and_usage_are_kept_on_the_message() -> None:
    response = ChatResponse(
        content="The capital of Fr",
        finish_reason="length",
        usage={"completion_tokens": 5},
    )
    think = model_node("think", chat=FakeChat(response))

    result = await _one_node_graph(think).ainvoke(_hi())

    message = result.data["messages"][-1]
    assert message.finish_reason == "length"
    assert message.usage == {"completion_tokens": 5}


def _calls(*ids: str) -> AIMessage:
    return AIMessage(
        tool_calls=[
            ToolCall(id=id, name="example", arguments={"value": 1}) for id in ids
        ]
    )


INCONSISTENT = [
    pytest.param(
        [ToolMessage(content="stray", tool_call_id="zzz"), HumanMessage(content="hi")],
        r"message 0 \(ToolMessage 'zzz'\) answers no tool call",
        id="orphan",
    ),
    pytest.param(
        [
            HumanMessage(content="deploy"),
            _calls("a", "b"),
            ToolMessage(content="1", tool_call_id="a"),
            HumanMessage(content="and now?"),
        ],
        r"message 1 \(AIMessage\) has tool calls without results: 'b'",
        id="unanswered",
    ),
    pytest.param(
        [
            HumanMessage(content="pay"),
            _calls("c1"),
            HumanMessage(content="interjection"),
            ToolMessage(content="late", tool_call_id="c1"),
        ],
        r"message 1 \(AIMessage\) has tool calls without results: 'c1'.*"
        r"message 3 \(ToolMessage 'c1'\) answers no tool call",
        id="late",
    ),
    pytest.param(
        [
            HumanMessage(content="pay"),
            _calls("c1"),
            ToolMessage(content="1", tool_call_id="c1"),
            ToolMessage(content="again", tool_call_id="c1"),
        ],
        r"message 3 \(ToolMessage 'c1'\) answers no tool call",
        id="duplicate",
    ),
    pytest.param(
        [HumanMessage(content="pay"), _calls("c1")],
        r"message 1 \(AIMessage\) has tool calls without results: 'c1'",
        id="trailing",
    ),
]


@pytest.mark.parametrize(("history", "match"), INCONSISTENT)
async def test_inconsistent_history_raises_before_the_model_is_called(
    history: list[Any], match: str
) -> None:
    chat = ScriptedChat(["ok"])
    think = model_node("think", chat=chat)

    with pytest.raises(ChatHistoryError, match=match) as error:
        await _one_node_graph(think).ainvoke({"messages": history})

    assert "repair_history=True" in str(error.value)
    assert chat.requests == []


async def test_consistent_history_is_sent_as_is() -> None:
    chat = ScriptedChat(["ok"])
    think = model_node("think", chat=chat)
    history = [
        HumanMessage(content="pay"),
        _calls("c1", "c2"),
        ToolMessage(content="1", tool_call_id="c2"),
        ToolMessage(content="2", tool_call_id="c1"),
    ]

    await _one_node_graph(think).ainvoke({"messages": history})

    assert [
        (
            type(message).__name__,
            message.content,
            getattr(message, "tool_call_id", None),
        )
        for message in chat.requests[0].messages
    ] == [
        ("HumanMessage", "pay", None),
        ("AIMessage", None, None),
        ("ToolMessage", "1", "c2"),
        ("ToolMessage", "2", "c1"),
    ]


async def test_repair_history_answers_unanswered_calls_in_the_request_only() -> None:
    chat = ScriptedChat(["ok"])
    think = model_node("think", chat=chat, repair_history=True)
    history = [
        HumanMessage(content="deploy"),
        _calls("a", "b"),
        ToolMessage(content="1", tool_call_id="a"),
        HumanMessage(content="and now?"),
    ]

    result = await _one_node_graph(think).ainvoke({"messages": history})

    sent = chat.requests[0].messages
    assert [type(message).__name__ for message in sent] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "ToolMessage",
        "HumanMessage",
    ]
    assert isinstance(sent[3], ToolMessage)
    assert sent[3].tool_call_id == "b"
    assert sent[3].content == "Tool call was not executed."
    assert len(result.data["messages"]) == len(history) + 1


async def test_repair_history_drops_orphan_tool_messages() -> None:
    chat = ScriptedChat(["ok"])
    think = model_node("think", chat=chat, repair_history=True)

    await _one_node_graph(think).ainvoke(
        {
            "messages": [
                ToolMessage(content="stray", tool_call_id="zzz"),
                HumanMessage(content="hi"),
            ]
        }
    )

    assert [type(message).__name__ for message in chat.requests[0].messages] == [
        "HumanMessage"
    ]


async def test_the_node_decides_complete_even_when_tokens_are_asked_for() -> None:
    chat = FakeChat(ChatResponse(content="hello"))
    think = model_node("think", chat=chat)

    events = [
        event
        async for event in _one_node_graph(think).astream(
            _hi(), stream_mode=["tokens", "values"]
        )
    ]

    assert chat.calls == ["complete"]
    assert [event for event in events if event.mode == "tokens"] == []


async def test_stream_true_streams_under_ainvoke_without_token_events() -> None:
    chat = FakeChat(ChatResponse(content="hello"))
    think = model_node("think", chat=chat, stream=True)

    result = await _one_node_graph(think).ainvoke(_hi())

    assert chat.calls == ["stream"]
    assert result.data["final_text"] == "hello"


async def test_stream_true_emits_tokens_when_asked_for() -> None:
    chat = FakeChat(ChatResponse(content="hello"))
    think = model_node("think", chat=chat, stream=True)

    tokens = [
        event.data
        async for event in _one_node_graph(think).astream(_hi(), stream_mode="tokens")
        if event.mode == "tokens" and isinstance(event.data, ChatStreamChunk)
    ]

    assert [chunk.content_delta for chunk in tokens] == ["hello", None]


class Recorder(Middleware):
    def __init__(self, name: str, log: list[Any]) -> None:
        self.name = name
        self.log = log

    def before_model(self, ctx: ModelMiddlewareContext) -> None:
        self.log.append((self.name, "before", ctx.response))

    async def after_model(self, ctx: ModelMiddlewareContext) -> None:
        assert ctx.response is not None
        self.log.append((self.name, "after", ctx.response.content))


@pytest.mark.parametrize("stream", [False, True])
async def test_model_hooks_run_around_every_call_in_order(stream: bool) -> None:
    log: list[Any] = []
    chat = FakeChat(ChatResponse(content="hello"))
    think = model_node("think", chat=chat, stream=stream)
    graph = _one_node_graph(think, middleware=[Recorder("a", log), Recorder("b", log)])

    await graph.ainvoke(_hi())

    assert log == [
        ("a", "before", None),
        ("b", "before", None),
        ("b", "after", "hello"),
        ("a", "after", "hello"),
    ]


async def test_model_hook_context_describes_the_call() -> None:
    seen: list[ModelMiddlewareContext] = []

    class Capture(Middleware):
        def before_model(self, ctx: ModelMiddlewareContext) -> None:
            seen.append(ctx)

        def after_model(self, ctx: ModelMiddlewareContext) -> None:
            seen.append(ctx)

    think = model_node("think", chat=ScriptedChat(["hello"]), system_prompt="be nice")
    graph = Graph(State, name="g", middleware=[Capture()]).flow(
        START >> think, think >> END
    )

    result = await graph.ainvoke(_hi())

    before, after = seen
    assert (before.graph_name, before.node_name, before.step) == ("g", "think", 0)
    assert before.run_id
    assert before.root_run_id == before.run_id
    assert before.thread_id == result.thread_id
    assert before.task_id
    assert before.request.messages[0] == SystemMessage(content="be nice")
    assert before.response is None
    assert after.request == before.request
    assert after.response is not None
    assert after.response.content == "hello"
    with pytest.raises(AttributeError):
        before.response = after.response  # ty: ignore[invalid-assignment]


async def test_before_model_replaces_the_request() -> None:
    class Redact(Middleware):
        def before_model(self, ctx: ModelMiddlewareContext):
            return ctx.replace(
                ctx.request.model_copy(
                    update={"messages": [HumanMessage(content="[redacted]")]}
                )
            )

    chat = ScriptedChat(["ok"])
    think = model_node("think", chat=chat)

    result = await _one_node_graph(think, middleware=[Redact()]).ainvoke(_hi())

    assert chat.requests[0].messages == [HumanMessage(content="[redacted]")]
    assert result.data["messages"][0].content == "hi"


async def test_after_model_replaces_the_response() -> None:
    class Rewrite(Middleware):
        def after_model(self, ctx: ModelMiddlewareContext):
            assert ctx.response is not None
            return ctx.replace(ctx.response.model_copy(update={"content": "edited"}))

    think = model_node("think", chat=ScriptedChat(["raw"]))

    result = await _one_node_graph(think, middleware=[Rewrite()]).ainvoke(_hi())

    assert result.data["final_text"] == "edited"
    assert result.data["messages"][-1].content == "edited"


@pytest.mark.parametrize("hook", ["before_model", "after_model"])
async def test_a_model_hook_replacement_of_the_wrong_type_raises(hook: str) -> None:
    class Wrong(Middleware):
        pass

    setattr(Wrong, hook, lambda self, ctx: ctx.replace("text"))
    think = model_node("think", chat=ScriptedChat(["ok"]))

    with pytest.raises(TypeError, match=hook):
        await _one_node_graph(think, middleware=[Wrong()]).ainvoke(_hi())


async def test_react_agent_sends_each_tool_once() -> None:
    from nodestep import build_react_agent

    chat = ScriptedChat([ChatResponse(content="hi")])
    agent = build_react_agent(chat, tools=[example])

    await agent.ainvoke({"messages": [HumanMessage(content="go")]})

    assert [tool.name for tool in chat.requests[0].tools] == ["example"]


def test_the_chat_protocol_declares_the_model_name() -> None:
    from typing import get_type_hints

    from nodestep.chat.base import Chat

    assert get_type_hints(Chat)["model"] is str
    assert ScriptedChat().model == "scripted"


@pytest.mark.parametrize("stream", [False, True])
async def test_model_hooks_see_the_model_of_the_chat(stream: bool) -> None:
    seen: list[str] = []

    class Capture(Middleware):
        def before_model(self, ctx: ModelMiddlewareContext) -> None:
            seen.append(ctx.model)

        def after_model(self, ctx: ModelMiddlewareContext) -> None:
            seen.append(ctx.model)

    chat = FakeChat(ChatResponse(content="hello", model="served-model"))
    think = model_node("think", chat=chat, stream=stream)

    await _one_node_graph(think, middleware=[Capture()]).ainvoke(_hi())

    assert seen == ["fake-model", "fake-model"]


async def test_scripted_chat_reports_scripted_to_model_hooks() -> None:
    seen: list[str] = []

    class Capture(Middleware):
        def before_model(self, ctx: ModelMiddlewareContext) -> None:
            seen.append(ctx.model)

    think = model_node("think", chat=ScriptedChat(["hello"]))

    await _one_node_graph(think, middleware=[Capture()]).ainvoke(_hi())

    assert seen == ["scripted"]


def test_a_chat_without_a_model_name_is_refused() -> None:
    class Nameless:
        async def complete(self, request: ChatRequest) -> ChatResponse:
            return ChatResponse(content="hi")

        async def stream(self, request: ChatRequest):
            yield ChatStreamChunk(content_delta="hi")

    with pytest.raises(TypeError, match="Nameless has no model name"):
        model_node("think", chat=Nameless())  # ty: ignore[invalid-argument-type]
