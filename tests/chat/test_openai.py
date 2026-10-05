import sys
from typing import Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("openai")

from nodestep.chat import (
    AIMessage,
    ChatRequest,
    ChatStreamChunk,
    HumanMessage,
    ToolCall,
    ToolDefinition,
)
from nodestep.chat.integrations.openai import OpenAIChat, OpenAISettings
from nodestep.chat.integrations.openai.mapping import (
    message_to_payload,
    request_to_parameters,
    response_from_completion,
)
from nodestep.core.stream import FinalEventData


def test_integration_not_installed_error_message() -> None:
    from nodestep.exceptions import IntegrationNotInstalledError, NodestepError

    error = IntegrationNotInstalledError("openai", "openai")
    assert isinstance(error, NodestepError)
    assert isinstance(error, ImportError)
    assert error.integration == "openai"
    assert error.extra == "openai"
    assert str(error) == (
        "The openai chat integration needs the optional dependency. Install it with: "
        'uv add "nodestep[openai] @ git+https://github.com/nodestep-ai/nodestep" '
        '(or pip install "nodestep[openai] @ git+https://github.com/nodestep-ai/nodestep")'
    )


def test_openai_chat_without_the_sdk_raises_an_import_error(monkeypatch) -> None:
    import sys

    from nodestep.exceptions import IntegrationNotInstalledError

    monkeypatch.setitem(sys.modules, "openai", None)

    with pytest.raises(
        ImportError,
        match=r'uv add "nodestep\[openai\] @ git\+https://github\.com/nodestep-ai/nodestep" \(',
    ) as info:
        OpenAIChat(model="m", api_key="sk-test")
    assert isinstance(info.value, IntegrationNotInstalledError)


def test_integrations_package_exports_openai_names_lazily() -> None:
    import nodestep.chat.integrations as integrations
    from nodestep.chat.integrations import openai as openai_integration

    assert integrations.OpenAIChat is openai_integration.OpenAIChat
    assert integrations.OpenAISettings is openai_integration.OpenAISettings


def test_openai_settings_defaults(monkeypatch) -> None:
    for name in _OPENAI_ENV:
        monkeypatch.delenv(name, raising=False)

    settings = OpenAISettings()

    assert settings.api_key is None
    assert settings.base_url is None
    assert settings.organization is None
    assert settings.project is None
    assert settings.timeout == 60.0
    assert settings.max_retries == 3


def test_openai_message_mapping_serializes_tool_calls() -> None:
    message = AIMessage(
        content=None,
        tool_calls=[ToolCall(id="call-1", name="shell", arguments={"command": "pwd"})],
    )
    payload = message_to_payload(message)
    assert payload["role"] == "assistant"
    assert payload["tool_calls"][0]["function"]["name"] == "shell"
    assert payload["tool_calls"][0]["function"]["arguments"] == '{"command": "pwd"}'


def test_openai_message_mapping_sends_a_refusal_back() -> None:
    payload = message_to_payload(AIMessage(refusal="I can't help with that."))

    assert payload == {
        "role": "assistant",
        "content": "",
        "refusal": "I can't help with that.",
    }


def test_request_to_parameters_includes_tools_and_schema() -> None:
    request = ChatRequest(
        messages=[HumanMessage(content="hi")],
        tools=[
            ToolDefinition(
                name="shell", description="run", input_schema={"type": "object"}
            )
        ],
        output_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
    )
    parameters = request_to_parameters(request)
    assert parameters["messages"][0]["content"] == "hi"
    assert parameters["tools"][0]["function"]["name"] == "shell"
    assert "tool_choice" not in parameters
    assert parameters["response_format"]["type"] == "json_schema"
    assert parameters["response_format"]["json_schema"]["strict"] is True


def _real_completion(**kwargs: Any) -> Any:
    from openai.types.chat import ChatCompletion

    from openai_fakes import completion

    return ChatCompletion.model_validate(completion(**kwargs))


def test_response_from_completion_parses_content() -> None:
    response = response_from_completion(_real_completion(content="hello"))
    assert response.content == "hello"
    assert response.usage["prompt_tokens"] == 11
    assert response.model == "fake"


def test_response_from_completion_parses_tool_calls() -> None:
    response = response_from_completion(
        _real_completion(tool_calls=[("tc-1", "shell", {"command": "pwd"})])
    )
    assert response.tool_calls[0].name == "shell"
    assert response.tool_calls[0].arguments == {"command": "pwd"}


def test_response_from_completion_keeps_refusal_and_finish_reason() -> None:
    response = response_from_completion(
        _real_completion(refusal="no", finish_reason="length")
    )
    assert response.refusal == "no"
    assert response.finish_reason == "length"


@pytest.mark.asyncio
async def test_complete_with_fake_client() -> None:
    fake_completion = _real_completion(content="hi back")
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=fake_completion)

    chat = OpenAIChat(model="gpt-6-luna", client=mock_client)
    response = await chat.complete(ChatRequest(messages=[HumanMessage(content="hi")]))

    assert response.content == "hi back"
    mock_client.chat.completions.create.assert_awaited_once()
    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs["model"] == "gpt-6-luna"


@pytest.mark.asyncio
async def test_complete_wraps_openai_error() -> None:
    from openai import OpenAIError

    from nodestep.exceptions import ModelProviderError

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=OpenAIError("boom"))

    chat = OpenAIChat(model="gpt-6-luna", client=mock_client)
    with pytest.raises(ModelProviderError, match="boom"):
        await chat.complete(ChatRequest(messages=[HumanMessage(content="hi")]))


def _tool_request() -> ChatRequest:
    return ChatRequest(
        messages=[HumanMessage(content="hi")],
        tools=[
            ToolDefinition(
                name="shell",
                description="run",
                input_schema={
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                },
            )
        ],
    )


@pytest.mark.asyncio
async def test_stream_yields_content_deltas() -> None:
    from openai_fakes import FakeOpenAIServer, chunk

    server = FakeOpenAIServer([[chunk({"content": "hel"}), chunk({"content": "lo"})]])
    chat = OpenAIChat(model="m", client=server.client())

    chunks = [
        chunk
        async for chunk in chat.stream(
            ChatRequest(messages=[HumanMessage(content="hi")])
        )
    ]

    assert [chunk.content_delta for chunk in chunks if chunk.content_delta] == [
        "hel",
        "lo",
    ]


@pytest.mark.asyncio
async def test_stream_with_non_strict_tools_assembles_parallel_calls() -> None:
    from nodestep.chat import assemble_stream
    from openai_fakes import FakeOpenAIServer, chunk

    server = FakeOpenAIServer(
        [
            [
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "shell", "arguments": ""},
                            }
                        ]
                    }
                ),
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "call_b",
                                "type": "function",
                                "function": {"name": "shell", "arguments": ""},
                            }
                        ]
                    }
                ),
                chunk(
                    {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"cmd": "ls"}'}}
                        ]
                    }
                ),
                chunk(
                    {
                        "tool_calls": [
                            {"index": 1, "function": {"arguments": '{"cmd": "pwd"}'}}
                        ]
                    }
                ),
                chunk({}, finish_reason="tool_calls"),
                chunk(
                    None,
                    usage={
                        "prompt_tokens": 3,
                        "completion_tokens": 9,
                        "total_tokens": 12,
                    },
                ),
            ]
        ]
    )
    chat = OpenAIChat(model="m", client=server.client())

    response = await assemble_stream(chat.stream(_tool_request()))

    assert len(server.requests) == 1
    assert server.requests[0]["stream"] is True
    assert server.requests[0]["stream_options"] == {"include_usage": True}
    assert [
        (tool_call.id, tool_call.arguments) for tool_call in response.tool_calls
    ] == [
        ("call_a", {"cmd": "ls"}),
        ("call_b", {"cmd": "pwd"}),
    ]
    assert response.finish_reason == "tool_calls"
    assert response.usage["completion_tokens"] == 9
    assert response.model == "fake"


@pytest.mark.asyncio
async def test_stream_wraps_openai_error() -> None:
    from openai import OpenAIError

    from nodestep.exceptions import ModelProviderError

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=OpenAIError("stream boom")
    )

    chat = OpenAIChat(model="gpt-6-luna", client=mock_client)
    with pytest.raises(ModelProviderError, match="stream boom"):
        async for _ in chat.stream(ChatRequest(messages=[HumanMessage(content="hi")])):
            pass


@pytest.mark.asyncio
async def test_token_streaming_agent_with_tools_runs_end_to_end() -> None:
    from nodestep import build_react_agent, tool
    from openai_fakes import FakeOpenAIServer, chunk

    @tool
    def shell(cmd: str) -> str:
        """Run a shell command."""
        return f"ran {cmd}"

    server = FakeOpenAIServer(
        [
            [
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "type": "function",
                                "function": {
                                    "name": "shell",
                                    "arguments": '{"cmd": "ls"}',
                                },
                            }
                        ]
                    }
                ),
                chunk({}, finish_reason="tool_calls"),
            ],
            [
                chunk({"content": "Do"}),
                chunk({"content": "ne."}),
                chunk({}, finish_reason="stop"),
            ],
        ]
    )
    agent = build_react_agent(
        OpenAIChat(model="m", client=server.client()), tools=[shell], stream=True
    )

    tokens = []
    final = None
    async for event in agent.astream(
        {"messages": [HumanMessage(content="go")]}, stream_mode=["tokens"]
    ):
        if isinstance(event.data, ChatStreamChunk) and event.data.content_delta:
            tokens.append(event.data.content_delta)
        if isinstance(event.data, FinalEventData):
            final = event.data.state

    assert tokens == ["Do", "ne."]
    assert final is not None
    assert final["final_text"] == "Done."


def test_response_from_completion_empty_choices_raises() -> None:
    from openai.types.chat import ChatCompletion

    from nodestep.exceptions import ModelProviderError
    from openai_fakes import completion

    empty = ChatCompletion.model_validate({**completion(content="x"), "choices": []})
    with pytest.raises(ModelProviderError, match="no choices"):
        response_from_completion(empty)


def _completion_with_arguments(arguments: str) -> Any:
    return _real_completion(tool_calls=[("tc-1", "shell", arguments)])


def test_response_from_completion_keeps_malformed_arguments_as_an_error() -> None:
    response = response_from_completion(_completion_with_arguments("{not json"))

    call = response.tool_calls[0]
    assert call.arguments == {}
    assert call.arguments_error is not None
    assert "{not json" in call.arguments_error


def test_response_from_completion_rejects_null_arguments() -> None:
    response = response_from_completion(_completion_with_arguments("null"))

    assert response.tool_calls[0].arguments == {}
    assert response.tool_calls[0].arguments_error is not None
    assert "JSON object" in response.tool_calls[0].arguments_error


def test_response_from_completion_raises_for_a_custom_tool_call() -> None:
    from openai.types.chat import ChatCompletion

    from nodestep.exceptions import ModelProviderError
    from openai_fakes import completion

    body = completion(tool_calls=[("tc-1", "shell", {})])
    body["choices"][0]["message"]["tool_calls"] = [
        {"id": "tc-1", "type": "custom", "custom": {"name": "shell", "input": "ls"}}
    ]

    with pytest.raises(ModelProviderError, match="'custom' tool call 'tc-1'"):
        response_from_completion(ChatCompletion.model_validate(body))


def test_response_from_completion_raises_for_several_choices() -> None:
    from openai.types.chat import ChatCompletion

    from nodestep.exceptions import ModelProviderError
    from openai_fakes import completion

    body = completion(content="a")
    body["choices"] = [body["choices"][0], {**body["choices"][0], "index": 1}]

    with pytest.raises(ModelProviderError, match="2 choices"):
        response_from_completion(ChatCompletion.model_validate(body))


def test_stream_event_of_a_second_choice_raises() -> None:
    from openai.types.chat import ChatCompletionChunk

    from nodestep.chat.integrations.openai.mapping import chunks_from_stream_event
    from nodestep.exceptions import ModelProviderError
    from openai_fakes import chunk

    body = chunk({"content": "b"})
    body["choices"][0]["index"] = 1

    with pytest.raises(ModelProviderError, match="choice 1"):
        chunks_from_stream_event(ChatCompletionChunk.model_validate(body))


def test_response_from_completion_rejects_non_object_arguments() -> None:
    response = response_from_completion(_completion_with_arguments("[1, 2]"))

    assert response.tool_calls[0].arguments_error is not None


@pytest.mark.asyncio
async def test_openai_chat_context_manager() -> None:
    mock_client = AsyncMock()
    async with OpenAIChat(model="gpt-6-luna", client=mock_client) as chat:
        assert chat.model == "gpt-6-luna"
    mock_client.close.assert_awaited_once()


def test_constructor_settings_reach_the_client() -> None:
    chat = OpenAIChat(
        model="gpt-6-luna",
        api_key="sk-explicit",
        base_url="http://localhost:11434/v1",
        organization="org-x",
        project="proj-y",
        timeout=5,
        max_retries=0,
    )

    client = chat._client
    assert client.api_key == "sk-explicit"
    assert str(client.base_url).startswith("http://localhost:11434/v1")
    assert client.organization == "org-x"
    assert client.project == "proj-y"
    assert client.timeout == 5
    assert client.max_retries == 0


CASE_BLIND_ENVIRONMENT = pytest.mark.skipif(
    sys.platform == "win32", reason="Windows environment variable names ignore case"
)
_OPENAI_ENV = {
    "OPENAI_API_KEY": "sk-from-env",
    "OPENAI_BASE_URL": "http://evil.example/v1",
    "OPENAI_ORG_ID": "org-from-env",
    "OPENAI_PROJECT_ID": "proj-from-env",
    "OPENAI_ORGANIZATION": "org-legacy-env",
    "OPENAI_PROJECT": "proj-legacy-env",
    "OPENAI_TIMEOUT": "1.5",
    "OPENAI_MAX_RETRIES": "9",
}


@pytest.fixture
def openai_env(monkeypatch) -> dict[str, str]:
    for name, value in _OPENAI_ENV.items():
        monkeypatch.setenv(name, value)
    return _OPENAI_ENV


def test_constructor_reads_no_openai_settings_from_the_environment(openai_env) -> None:
    chat = OpenAIChat("gpt-6-luna", api_key="sk-mine")

    client = chat._client
    assert client.api_key == "sk-mine"
    assert str(client.base_url) == "https://api.openai.com/v1/"
    assert client.organization is None
    assert client.project is None
    assert client.timeout == 60.0
    assert client.max_retries == 3
    assert not isinstance(client.default_headers.get("OpenAI-Organization"), str)
    assert not isinstance(client.default_headers.get("OpenAI-Project"), str)


def test_constructor_without_api_key_raises_even_when_env_has_one(openai_env) -> None:
    from nodestep.exceptions import ModelProviderError

    with pytest.raises(ModelProviderError, match="from_env"):
        OpenAIChat("gpt-6-luna")


def test_model_is_required() -> None:
    arguments: dict[str, Any] = {"api_key": "sk-test"}
    with pytest.raises(TypeError):
        OpenAIChat(**arguments)
    with pytest.raises(TypeError):
        OpenAIChat.from_env(**arguments)


@pytest.mark.parametrize(
    "connection",
    [
        {"api_key": "sk-x"},
        {"base_url": "http://mine.invalid/v1"},
        {"organization": "org-x"},
        {"project": "proj-x"},
        {"timeout": 5.0},
        {"max_retries": 0},
        {"timeout": 60.0},
        {"max_retries": 3},
    ],
)
def test_client_combined_with_connection_arguments_raises(connection) -> None:
    from openai_fakes import FakeOpenAIServer

    with pytest.raises(TypeError, match="client="):
        OpenAIChat("gpt-6-luna", client=FakeOpenAIServer([]).client(), **connection)


def test_client_alone_reads_no_settings(monkeypatch) -> None:
    from openai_fakes import FakeOpenAIServer

    monkeypatch.setenv("OPENAI_TIMEOUT", "not-a-number")
    client = FakeOpenAIServer([]).client()

    chat = OpenAIChat("gpt-6-luna", client=client)

    assert chat._client is client


def test_from_env_reads_openai_variables(openai_env) -> None:
    chat = OpenAIChat.from_env(model="gpt-6-luna")

    client = chat._client
    assert chat.model == "gpt-6-luna"
    assert client.api_key == "sk-from-env"
    assert str(client.base_url) == "http://evil.example/v1/"
    assert client.organization == "org-from-env"
    assert client.project == "proj-from-env"
    assert client.timeout == 1.5
    assert client.max_retries == 9


def test_from_env_keyword_arguments_override_the_environment(openai_env) -> None:
    chat = OpenAIChat.from_env(
        "gpt-6-luna", base_url="http://mine.invalid/v1", temperature=0.1
    )

    assert str(chat._client.base_url) == "http://mine.invalid/v1/"
    assert chat._client.api_key == "sk-from-env"
    assert chat.request_parameters == {"temperature": 0.1}


def test_from_env_treats_empty_variables_as_unset(monkeypatch) -> None:
    for name in _OPENAI_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("OPENAI_ORG_ID", "")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "")

    chat = OpenAIChat.from_env(model="gpt-6-luna")

    assert str(chat._client.base_url) == "https://api.openai.com/v1/"
    assert chat._client.organization is None
    assert chat._client.project is None


def test_from_env_with_empty_api_key_raises(monkeypatch) -> None:
    from nodestep.exceptions import ModelProviderError

    monkeypatch.setenv("OPENAI_API_KEY", "")

    with pytest.raises(ModelProviderError, match="OPENAI_API_KEY is not set") as info:
        OpenAIChat.from_env("gpt-6-luna")
    assert "from_env(model=...)" not in str(info.value)


@pytest.mark.parametrize(
    "name",
    [
        "OPENAI_ORGANIZATION",
        "OPENAI_PROJECT",
        "OPENAI_organization",
        "OPENAI_project",
        pytest.param("OPENAI_api_key", marks=CASE_BLIND_ENVIRONMENT),
        pytest.param("OPENAI_base_url", marks=CASE_BLIND_ENVIRONMENT),
    ],
)
def test_openai_settings_read_only_the_documented_names(monkeypatch, name) -> None:
    for documented in _OPENAI_ENV:
        monkeypatch.delenv(documented, raising=False)
    monkeypatch.setenv(name, "from-undocumented-name")

    settings = OpenAISettings()

    assert settings.model_dump() == OpenAISettings.model_construct().model_dump()


@CASE_BLIND_ENVIRONMENT
def test_openai_settings_ignore_lowercase_names(monkeypatch) -> None:
    for name in _OPENAI_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("openai_timeout", "7")
    monkeypatch.setenv("openai_api_key", "sk-lowercase")

    settings = OpenAISettings()

    assert settings.timeout == 60.0
    assert settings.api_key is None


def test_from_env_without_api_key_raises(monkeypatch) -> None:
    from nodestep.exceptions import ModelProviderError

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ModelProviderError, match="OPENAI_API_KEY is not set") as info:
        OpenAIChat.from_env("gpt-6-luna")
    assert "from_env(model=...)" not in str(info.value)


@pytest.mark.parametrize(
    "key",
    [
        "messages",
        "tools",
        "tool_choice",
        "response_format",
        "stream",
        "stream_options",
        "model",
    ],
)
def test_request_parameters_owned_by_the_mapping_raise(key) -> None:
    from openai_fakes import FakeOpenAIServer

    parameters: dict[str, Any] = {key: "x"}
    with pytest.raises(TypeError, match=key):
        OpenAIChat("gpt-6-luna", client=FakeOpenAIServer([]).client(), **parameters)


@pytest.mark.parametrize(
    ("tool_choice", "wire"),
    [
        ("auto", "auto"),
        ("required", "required"),
        ("none", "none"),
        ("shell", {"type": "function", "function": {"name": "shell"}}),
    ],
)
def test_request_tool_choice_is_passed_through(tool_choice, wire) -> None:
    request = _tool_request().model_copy(update={"tool_choice": tool_choice})

    assert request_to_parameters(request)["tool_choice"] == wire


async def test_request_tool_choice_reaches_the_wire() -> None:
    from openai_fakes import FakeOpenAIServer, completion

    server = FakeOpenAIServer([completion(content="ok")])
    chat = OpenAIChat("gpt-6-luna", client=server.client())
    request = ChatRequest(
        messages=[HumanMessage(content="hi")],
        tools=_tool_request().tools,
        tool_choice="required",
    )

    await chat.complete(request)

    assert server.requests[0]["tool_choice"] == "required"


async def test_stream_usage_false_omits_stream_options() -> None:
    from openai_fakes import FakeOpenAIServer, chunk

    server = FakeOpenAIServer([[chunk({"content": "x"})]])
    chat = OpenAIChat("gpt-6-luna", client=server.client(), stream_usage=False)

    _ = [chunk async for chunk in chat.stream(_tool_request())]

    assert server.requests[0]["stream"] is True
    assert "stream_options" not in server.requests[0]


async def test_extra_keyword_arguments_are_sent_with_every_request() -> None:
    from openai_fakes import FakeOpenAIServer, completion

    server = FakeOpenAIServer([completion(content="ok"), completion(content="ok")])
    chat = OpenAIChat(
        model="m", client=server.client(), temperature=0.2, max_completion_tokens=50
    )

    await chat.complete(ChatRequest(messages=[HumanMessage(content="hi")]))
    await chat.complete(ChatRequest(messages=[HumanMessage(content="again")]))

    for sent in server.requests:
        assert sent["temperature"] == 0.2
        assert sent["max_completion_tokens"] == 50


def _object_nodes(schema: Any) -> list[dict]:
    found: list[dict] = []
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            found.append(schema)
        for value in schema.values():
            found.extend(_object_nodes(value))
    elif isinstance(schema, list):
        for item in schema:
            found.extend(_object_nodes(item))
    return found


def test_structured_output_schema_is_strict_compliant() -> None:
    from pydantic import BaseModel

    class Step(BaseModel):
        explanation: str
        output: str

    class Answer(BaseModel):
        steps: list[Step]
        final_answer: str
        confidence: float | None = None

    parameters = request_to_parameters(
        ChatRequest(
            messages=[HumanMessage(content="q")],
            output_schema=Answer.model_json_schema(),
        )
    )

    schema = parameters["response_format"]["json_schema"]["schema"]
    nodes = _object_nodes(schema)
    assert len(nodes) == 2
    for node in nodes:
        assert node["additionalProperties"] is False
        assert node["required"] == list(node["properties"])
    assert list(schema["properties"]) == ["steps", "final_answer", "confidence"]
    assert "default" not in schema["properties"]["confidence"]


def test_ref_with_sibling_keywords_is_inlined() -> None:
    from pydantic import BaseModel, Field

    from nodestep.chat.integrations.openai.mapping import to_strict_schema

    class Inner(BaseModel):
        x: int

    class Outer(BaseModel):
        inner: Inner = Field(description="the inner value")

    schema = to_strict_schema(Outer.model_json_schema())

    inner = schema["properties"]["inner"]
    assert "$ref" not in inner
    assert inner["description"] == "the inner value"
    assert inner["additionalProperties"] is False


def test_tool_parameter_order_is_preserved() -> None:
    from nodestep import tool

    @tool
    def book(city: str, nights: int, budget: float) -> str:
        """Book a hotel."""
        return city

    assert list(book.input_schema()["properties"]) == ["city", "nights", "budget"]
