from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

from pydantic import Field

from nodestep.exceptions import ModelProviderError
from nodestep.models.base import NodestepModel


def parse_tool_arguments(raw: str | None) -> tuple[dict[str, Any], str | None]:
    """Parse the JSON arguments of a tool call.

    Parameters
    ----------
    raw : str or None
        Argument text produced by the model.

    Returns
    -------
    tuple[dict, str or None]
        The parsed arguments and an error description when they are invalid.
        Empty text means no arguments; anything but a JSON object, ``null``
        included, is invalid.
    """
    if not raw:
        return {}, None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        return {}, f"Arguments are not valid JSON ({error}): {raw[:500]}"
    if not isinstance(value, dict):
        return {}, f"Arguments must be a JSON object, got: {raw[:500]}"
    return value, None


class ToolCall(NodestepModel):
    """A tool invocation requested by the model.

    Attributes
    ----------
    id : str
        Id the model gave the call; its ``ToolMessage`` answers with it.
    arguments : dict
        Parsed arguments; empty when ``arguments_error`` is set.
    arguments_error : str or None
        Why the argument text could not be parsed, or ``None``.
    """

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    arguments_error: str | None = None

    @classmethod
    def from_arguments_json(
        cls, id: str, name: str, arguments_json: str | None
    ) -> ToolCall:
        """Build a tool call from the JSON argument text a model produced.

        Parameters
        ----------
        id : str
            Tool call id.
        name : str
            Tool name.
        arguments_json : str or None
            Argument text, parsed with ``parse_tool_arguments``.

        Returns
        -------
        ToolCall
            The call, with ``arguments_error`` set when the text is invalid.
        """
        arguments, arguments_error = parse_tool_arguments(arguments_json)
        return cls(
            id=id, name=name, arguments=arguments, arguments_error=arguments_error
        )

    @property
    def arguments_json(self) -> str:
        """Arguments serialized as a JSON string."""
        return json.dumps(self.arguments, sort_keys=True)


class BaseMessage(NodestepModel, ABC):
    """Base class of all chat messages.

    Attributes
    ----------
    id : str or None
        Optional message id, for example to remove the message later.
    """

    id: str | None = None
    content: str | None = None

    @property
    @abstractmethod
    def role(self) -> str:
        """Role of the message as a chat integration sends it."""
        ...


class SystemMessage(BaseMessage):
    """Instruction message for the model."""

    type: Literal["system"] = "system"

    @property
    def role(self) -> str:
        """Always ``"system"``."""
        return "system"


class HumanMessage(BaseMessage):
    """Message written by the user."""

    type: Literal["human"] = "human"

    @property
    def role(self) -> str:
        """Always ``"user"``."""
        return "user"


class AIMessage(BaseMessage):
    """Reply produced by the model: text, a refusal, or tool calls.

    Attributes
    ----------
    refusal : str or None
        The model's refusal text, when it declined to answer.
    finish_reason : str or None
        Why the model stopped, as the chat integration reports it.
    usage : dict
        Token counts as the chat integration reports them.
    """

    type: Literal["ai"] = "ai"
    refusal: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)

    @property
    def role(self) -> str:
        """Always ``"assistant"``."""
        return "assistant"


class ToolMessage(BaseMessage):
    """Result of a tool call; ``arguments`` are the ones it ran with when they differ from the call.

    Attributes
    ----------
    name : str or None
        Name of the tool that ran.
    tool_call_id : str or None
        Id of the ``ToolCall`` this message answers.
    """

    type: Literal["tool"] = "tool"
    name: str | None = None
    tool_call_id: str | None = None
    arguments: dict[str, Any] | None = None

    @property
    def role(self) -> str:
        """Always ``"tool"``."""
        return "tool"


Message = Annotated[
    SystemMessage | HumanMessage | AIMessage | ToolMessage,
    Field(discriminator="type"),
]
"""Any chat message, told apart by its ``type`` field when validated."""


class ToolDefinition(NodestepModel):
    """Tool description sent to the model.

    Attributes
    ----------
    input_schema : dict
        JSON schema of the arguments.
    """

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(NodestepModel):
    """Request to a chat model, the same for every chat integration.

    Attributes
    ----------
    tool_choice : {"auto", "required", "none"} or str or None
        Whether the model may, must or must not call tools; another string
        names the tool to call, and ``None`` sends no choice.
    output_schema : dict or None
        JSON schema the answer must follow.
    """

    messages: list[Message]
    tools: list[ToolDefinition] = Field(default_factory=list)
    tool_choice: Literal["auto", "required", "none"] | str | None = None
    output_schema: dict[str, Any] | None = None


class ChatResponse(NodestepModel):
    """Reply from a chat model, the same for every chat integration.

    Attributes
    ----------
    content : str or None
        Answer text; ``None`` when the model only calls tools or refuses.
    refusal : str or None
        The model's refusal text, when it declined to answer.
    finish_reason : str or None
        Why the model stopped, as the chat integration reports it.
    usage : dict
        Token counts as the chat integration reports them.
    model : str or None
        Model that answered, as the chat integration names it.
    raw : Any
        The response object the chat integration received, for fields
        nodestep does not map.
    """

    content: str | None = None
    refusal: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None
    raw: Any = None


class ChatStreamChunk(NodestepModel):
    """One incremental piece of a streamed response.

    Attributes
    ----------
    content_delta : str or None
        Next part of the answer text.
    refusal_delta : str or None
        Next part of the refusal text.
    finish_reason : str or None
        Why the model stopped; set on a final chunk.
    usage : dict or None
        Token counts; set on the chunk that reports them.
    model : str or None
        Model that answered, as the chat integration names it.
    raw : Any
        The event object the chat integration received.
    """

    content_delta: str | None = None
    refusal_delta: str | None = None
    tool_call_delta: dict[str, Any] | None = None
    """Part of one tool call, shaped as in [Streamed tool calls](../../guides/chat-models.md#streamed-tool-calls)."""
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    model: str | None = None
    raw: Any = None


async def assemble_stream(chunks: AsyncIterator[ChatStreamChunk]) -> ChatResponse:
    """Collect streamed chunks into a single response.

    Parameters
    ----------
    chunks : AsyncIterator[ChatStreamChunk]
        Chunks produced by ``Chat.stream``.

    Returns
    -------
    ChatResponse
        Joined content, assembled tool calls, finish reason, usage and the
        model named by the chunks.

    Raises
    ------
    ModelProviderError
        If a streamed tool call has no id or no name.
    """
    content_parts: list[str] = []
    refusal_parts: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    model: str | None = None
    tool_calls_by_index: dict[int, dict[str, Any]] = {}

    async for chunk in chunks:
        if chunk.content_delta is not None:
            content_parts.append(chunk.content_delta)
        if chunk.refusal_delta is not None:
            refusal_parts.append(chunk.refusal_delta)
        if chunk.finish_reason is not None:
            finish_reason = chunk.finish_reason
        if chunk.usage is not None:
            usage = chunk.usage
        if chunk.model is not None:
            model = chunk.model
        if chunk.tool_call_delta is not None:
            delta = chunk.tool_call_delta
            index = delta.get("index", 0)
            if index not in tool_calls_by_index:
                tool_calls_by_index[index] = {
                    "id": delta.get("id"),
                    "name": "",
                    "arguments": "",
                }
            entry = tool_calls_by_index[index]
            function = delta.get("function", {})
            if function.get("name"):
                entry["name"] = function["name"]
            if delta.get("id"):
                entry["id"] = delta["id"]
            if function.get("arguments"):
                entry["arguments"] += function["arguments"]
    content = "".join(content_parts) if content_parts else None
    calls: list[ToolCall] = []
    for _, assembled in sorted(tool_calls_by_index.items()):
        if not assembled["name"]:
            raise ModelProviderError(
                f"Streamed tool call {assembled['id']!r} has no name"
            )
        if not assembled["id"]:
            raise ModelProviderError(
                f"Streamed tool call '{assembled['name']}' is missing an id"
            )
        calls.append(
            ToolCall.from_arguments_json(
                assembled["id"], assembled["name"], assembled["arguments"]
            )
        )
    return ChatResponse(
        content=content,
        refusal="".join(refusal_parts) if refusal_parts else None,
        tool_calls=calls,
        finish_reason=finish_reason,
        usage=usage,
        model=model,
    )


__all__ = [
    "AIMessage",
    "BaseMessage",
    "ChatRequest",
    "ChatResponse",
    "ChatStreamChunk",
    "HumanMessage",
    "Message",
    "SystemMessage",
    "ToolCall",
    "ToolDefinition",
    "ToolMessage",
    "assemble_stream",
    "parse_tool_arguments",
]
