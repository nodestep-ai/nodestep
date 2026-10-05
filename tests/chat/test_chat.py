import pytest
from pydantic import ValidationError

from nodestep.chat import (
    AIMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    HumanMessage,
    ToolCall,
    ToolDefinition,
    ToolMessage,
    assemble_stream,
)
from nodestep.exceptions import ModelProviderError


async def _chunks(items):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_assemble_stream_collects_content_and_tool_calls() -> None:
    chunks = [
        ChatStreamChunk(content_delta="he"),
        ChatStreamChunk(content_delta="llo"),
        ChatStreamChunk(
            tool_call_delta={
                "index": 0,
                "id": "tc-1",
                "function": {"name": "shell", "arguments": '{"cmd":'},
            }
        ),
        ChatStreamChunk(
            tool_call_delta={"index": 0, "function": {"arguments": '"ls"}'}}
        ),
    ]
    response = await assemble_stream(_chunks(chunks))
    assert response.content == "hello"
    assert response.tool_calls[0].name == "shell"
    assert response.tool_calls[0].arguments == {"cmd": "ls"}


@pytest.mark.asyncio
async def test_assemble_stream_keeps_malformed_arguments_as_an_error() -> None:
    chunks = [
        ChatStreamChunk(
            tool_call_delta={
                "index": 0,
                "id": "tc-1",
                "function": {"name": "shell", "arguments": '{"cmd": "l'},
            }
        ),
    ]

    response = await assemble_stream(_chunks(chunks))

    call = response.tool_calls[0]
    assert call.arguments == {}
    assert call.arguments_error is not None
    assert '{"cmd": "l' in call.arguments_error


@pytest.mark.asyncio
async def test_assemble_stream_raises_on_missing_tool_call_id() -> None:
    chunks = [
        ChatStreamChunk(
            tool_call_delta={
                "index": 0,
                "function": {"name": "shell", "arguments": "{}"},
            }
        ),
    ]
    with pytest.raises(ModelProviderError):
        await assemble_stream(_chunks(chunks))


@pytest.mark.asyncio
async def test_assemble_stream_raises_on_a_tool_call_without_a_name() -> None:
    chunks = [
        ChatStreamChunk(
            tool_call_delta={"index": 0, "id": "c1", "function": {"arguments": "{}"}}
        ),
    ]
    with pytest.raises(ModelProviderError, match=r"'c1'.*no name"):
        await assemble_stream(_chunks(chunks))


def test_tool_call_arguments_round_trip() -> None:
    call = ToolCall(id="1", name="shell", arguments={"command": "pwd"})
    assert call.arguments == {"command": "pwd"}


def test_tool_call_arguments_is_dict() -> None:
    tool_call = ToolCall(id="1", name="foo", arguments={"x": 1})
    assert tool_call.arguments == {"x": 1}
    assert tool_call.arguments_json == '{"x": 1}'


def test_tool_call_from_arguments_json() -> None:
    tool_call = ToolCall.from_arguments_json("1", "foo", '{"x": 1}')
    assert tool_call.arguments == {"x": 1}
    assert tool_call.arguments_error is None


def test_tool_call_from_invalid_arguments_json_records_the_error() -> None:
    tool_call = ToolCall.from_arguments_json("1", "foo", "{not json")
    assert tool_call.arguments == {}
    assert tool_call.arguments_error is not None
    assert "{not json" in tool_call.arguments_error


def test_tool_call_rejects_the_arguments_json_key() -> None:
    with pytest.raises(ValidationError, match="arguments_json"):
        ToolCall.model_validate(
            {"id": "1", "name": "foo", "arguments_json": '{"x": 1}'}
        )


def test_langchain_style_tool_call_args_raise() -> None:
    with pytest.raises(ValidationError, match="args"):
        ToolCall.model_validate({"id": "c1", "name": "search", "args": {"q": "x"}})
    with pytest.raises(ValidationError, match="args"):
        AIMessage.model_validate(
            {"tool_calls": [{"id": "c1", "name": "search", "args": {"q": "x"}}]}
        )


def test_messages_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="additional_kwargs"):
        HumanMessage.model_validate({"content": "hi", "additional_kwargs": {}})
    with pytest.raises(ValidationError, match="tool_call_ids"):
        ToolMessage.model_validate({"content": "r", "tool_call_ids": ["1"]})


def test_nodestep_model_forbids_extra_fields() -> None:
    from nodestep.models.base import NodestepModel

    class Record(NodestepModel):
        name: str

    with pytest.raises(ValidationError, match="other"):
        Record.model_validate({"name": "x", "other": 1})


def test_schema_and_usage_aliases_are_rejected() -> None:
    with pytest.raises(ValidationError, match="input_schema_json"):
        ToolDefinition.model_validate(
            {"name": "t", "description": "d", "input_schema_json": "{}"}
        )
    with pytest.raises(ValidationError, match="output_schema_json"):
        ChatRequest.model_validate({"messages": [], "output_schema_json": "{}"})
    with pytest.raises(ValidationError, match="usage_json"):
        ChatResponse.model_validate({"usage_json": "{}"})


def test_request_and_response_hold_plain_dicts() -> None:
    request = ChatRequest(messages=[], output_schema={"type": "object"})
    response = ChatResponse(usage={"total_tokens": 3})

    assert request.output_schema == {"type": "object"}
    assert request.tool_choice is None
    assert response.usage == {"total_tokens": 3}


def test_parse_tool_arguments_is_exported() -> None:
    from nodestep.chat import parse_tool_arguments

    assert parse_tool_arguments('{"x": 1}') == ({"x": 1}, None)


def test_tool_call_default_arguments() -> None:
    tool_call = ToolCall(id="1", name="foo")
    assert tool_call.arguments == {}
    assert tool_call.arguments_json == "{}"


def test_tool_definition_schema_round_trip() -> None:
    definition = ToolDefinition(
        name="shell", description="Run shell", input_schema={"type": "object"}
    )
    assert definition.input_schema == {"type": "object"}


def test_message_roles() -> None:
    assert HumanMessage(content="hi").role == "user"
    assert AIMessage(content="hello").role == "assistant"


def test_chat_request_round_trips_through_model_dump() -> None:
    from nodestep.chat import SystemMessage

    request = ChatRequest(
        messages=[
            SystemMessage(content="be nice"),
            HumanMessage(content="q"),
            AIMessage(content=None, tool_calls=[ToolCall(id="1", name="t")]),
            ToolMessage(content="r", name="t", tool_call_id="1"),
        ]
    )

    restored = ChatRequest.model_validate(request.model_dump())

    assert restored == request
