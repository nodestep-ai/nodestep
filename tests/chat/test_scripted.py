import subprocess
import sys
from collections.abc import AsyncIterator
from typing import Any

import pytest

import nodestep
from nodestep import ScriptedChat
from nodestep.chat import (
    AIMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    HumanMessage,
    ToolCall,
    assemble_stream,
)
from nodestep.exceptions import ModelProviderError


def _request(text: str = "hi") -> ChatRequest:
    return ChatRequest(messages=[HumanMessage(content=text)])


async def _replay(chunks: list[ChatStreamChunk]) -> AsyncIterator[ChatStreamChunk]:
    for chunk in chunks:
        yield chunk


def _full_response(model: str | None = None) -> ChatResponse:
    return ChatResponse(
        content="  Hello,\n  world!  ",
        tool_calls=[
            ToolCall(id="c1", name="search", arguments={"query": "nodestep"}),
            ToolCall(id="c2", name="fetch", arguments={"url": "https://x.dev", "n": 2}),
        ],
        finish_reason="tool_calls",
        usage={"completion_tokens": 5, "prompt_tokens": 12, "total_tokens": 17},
        model=model,
    )


def _kind(chunk: ChatStreamChunk) -> str:
    if chunk.content_delta is not None:
        return "content"
    if chunk.tool_call_delta is not None:
        return "tool_call"
    if chunk.usage is not None:
        return "usage"
    return "finish"


async def test_string_items_become_content_responses() -> None:
    chat = ScriptedChat(["hello", ChatResponse(content="world", finish_reason="stop")])

    first = await chat.complete(_request())
    second = await chat.complete(_request())

    assert first == ChatResponse(content="hello", model="scripted")
    assert second == ChatResponse(
        content="world", finish_reason="stop", model="scripted"
    )


async def test_responses_report_the_scripted_model_unless_they_name_one() -> None:
    chat = ScriptedChat(
        ["text", ChatResponse(content="object"), ChatResponse(content="x", model="m1")],
        default="fallback",
    )

    models = [(await chat.complete(_request())).model for _ in range(4)]

    assert models == ["scripted", "scripted", "m1", "scripted"]


async def test_requests_records_every_request() -> None:
    chat = ScriptedChat(["a", "b"])
    first, second = _request("one"), _request("two")

    await chat.complete(first)
    await assemble_stream(chat.stream(second))

    assert chat.requests == [first, second]


async def test_exhausted_script_raises() -> None:
    chat = ScriptedChat(["a", "b"])
    await chat.complete(_request())
    await chat.complete(_request())

    with pytest.raises(ModelProviderError, match="no response left for request 3"):
        await chat.complete(_request())
    assert len(chat.requests) == 3


async def test_default_is_returned_when_the_script_runs_out() -> None:
    chat = ScriptedChat(["first"], default=ChatResponse(content="again"))

    contents = [(await chat.complete(_request())).content for _ in range(3)]

    assert contents == ["first", "again", "again"]
    assert chat.responses == []


async def test_stream_yields_words_then_tool_calls_then_finish_then_usage() -> None:
    chat = ScriptedChat([_full_response()])

    chunks = [chunk async for chunk in chat.stream(_request())]

    assert [
        chunk.content_delta for chunk in chunks if chunk.content_delta is not None
    ] == [
        "  Hello,",
        "\n  world!",
        "  ",
    ]
    assert [_kind(chunk) for chunk in chunks] == [
        *["content"] * 3,
        *["tool_call"] * 2,
        "finish",
        "usage",
    ]


@pytest.mark.parametrize(
    "response",
    [
        _full_response(),
        _full_response(model="m1"),
        ChatResponse(refusal="I can't help with that.", finish_reason="stop"),
        ChatResponse(content=""),
        ChatResponse(),
    ],
    ids=["full", "named-model", "refusal", "empty-content", "bare"],
)
async def test_stream_reassembles_into_an_equal_response(
    response: ChatResponse,
) -> None:
    chat = ScriptedChat([response, response])
    expected = await chat.complete(_request())

    chunks = [chunk async for chunk in chat.stream(_request())]

    assert await assemble_stream(_replay(chunks)) == expected


@pytest.mark.parametrize(
    ("response", "field"),
    [
        (
            ChatResponse(
                tool_calls=[
                    ToolCall(id="bad", name="echo", arguments_error="not valid JSON")
                ]
            ),
            "arguments_error",
        ),
        (ChatResponse(content="hi", raw={"id": "cmpl-1"}), "raw"),
    ],
    ids=["arguments-error", "raw"],
)
async def test_stream_refuses_a_response_it_cannot_rebuild(
    response: ChatResponse, field: str
) -> None:
    chat = ScriptedChat([response])

    with pytest.raises(ValueError, match=field):
        await assemble_stream(chat.stream(_request()))


def test_items_must_be_responses_or_strings() -> None:
    items: list[Any] = [AIMessage(content="hi")]

    with pytest.raises(TypeError, match="AIMessage"):
        ScriptedChat(items)


def test_scripted_chat_is_exported_from_nodestep_and_integrations() -> None:
    import nodestep.chat.integrations as integrations

    assert nodestep.ScriptedChat is integrations.ScriptedChat
    assert "ScriptedChat" in nodestep.__all__
    assert "ScriptedChat" in integrations.__all__


def test_importing_nodestep_does_not_load_the_openai_integration() -> None:
    code = (
        "import sys; import nodestep; "
        "loaded = sorted(m for m in sys.modules "
        "if m.startswith('nodestep.chat.integrations.openai') "
        "or m.split('.')[0] in {'openai', 'pydantic_settings'}); "
        "assert not loaded, loaded"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8"
    )
    assert result.returncode == 0, result.stderr


def test_integrations_dir_lists_lazy_names_without_loading_the_integration() -> None:
    code = (
        "import sys; import nodestep.chat.integrations as integrations; "
        "names = dir(integrations); "
        "assert names == sorted([*integrations.__all__, 'OpenAIChat', 'OpenAISettings']), names; "
        "assert 'nodestep.chat.integrations.openai' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8"
    )
    assert result.returncode == 0, result.stderr
