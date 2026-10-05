from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

from nodestep.chat import (
    AIMessage,
    BaseMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ToolCall,
    ToolMessage,
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionChunk


def message_to_payload(message: BaseMessage) -> dict[str, Any]:
    """Convert a message to the OpenAI wire format.

    Parameters
    ----------
    message : BaseMessage

    Returns
    -------
    dict
    """
    payload: dict[str, Any] = {
        "role": message.role,
        "content": message.content or "",
    }
    if isinstance(message, ToolMessage):
        if message.name:
            payload["name"] = message.name
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
    if isinstance(message, AIMessage) and message.refusal is not None:
        payload["refusal"] = message.refusal
    if isinstance(message, AIMessage) and message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments),
                },
            }
            for tool_call in message.tool_calls
        ]
    return payload


def _resolve_ref(root: dict[str, Any], ref: str) -> dict[str, Any]:
    node: Any = root
    for part in ref.removeprefix("#/").split("/"):
        node = node[part]
    return node


def _ensure_strict(node: Any, root: dict[str, Any]) -> Any:
    if isinstance(node, list):
        return [_ensure_strict(item, root) for item in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node and len(node) > 1:
        resolved = copy.deepcopy(_resolve_ref(root, node["$ref"]))
        node = {
            **resolved,
            **{key: value for key, value in node.items() if key != "$ref"},
        }
    strict = dict(node)
    for key in ("$defs", "definitions"):
        if isinstance(strict.get(key), dict):
            strict[key] = {
                name: _ensure_strict(value, root) for name, value in strict[key].items()
            }
    if strict.get("type") == "object" and "additionalProperties" not in strict:
        strict["additionalProperties"] = False
    properties = strict.get("properties")
    if isinstance(properties, dict):
        strict["required"] = list(properties)
        strict["properties"] = {
            name: _ensure_strict(value, root) for name, value in properties.items()
        }
    for key in ("items", "anyOf", "oneOf", "allOf"):
        if key in strict:
            strict[key] = _ensure_strict(strict[key], root)
    if "default" in strict and strict["default"] is None:
        strict.pop("default")
    return strict


def to_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt a JSON schema to OpenAI strict structured-output rules.

    Every object forbids extra properties and lists all properties as required.

    Parameters
    ----------
    schema : dict
        JSON schema, for example from ``BaseModel.model_json_schema()``.

    Returns
    -------
    dict
    """
    return _ensure_strict(schema, schema)


def request_to_parameters(request: ChatRequest) -> dict[str, Any]:
    """Build Chat Completions arguments from a request.

    ``tool_choice`` is sent only when the request sets it; a value other than
    ``"auto"``, ``"required"`` or ``"none"`` names the tool to call.

    Parameters
    ----------
    request : ChatRequest

    Returns
    -------
    dict
    """
    parameters: dict[str, Any] = {
        "messages": [message_to_payload(message) for message in request.messages],
    }
    if request.tools:
        parameters["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in request.tools
        ]
    if request.tool_choice in ("auto", "required", "none"):
        parameters["tool_choice"] = request.tool_choice
    elif request.tool_choice is not None:
        parameters["tool_choice"] = {
            "type": "function",
            "function": {"name": request.tool_choice},
        }
    if request.output_schema is not None:
        parameters["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "final_output",
                "schema": to_strict_schema(request.output_schema),
                "strict": True,
            },
        }
    return parameters


def response_from_completion(completion: ChatCompletion) -> ChatResponse:
    """Convert an OpenAI completion into a ``ChatResponse``, keeping its model.

    Parameters
    ----------
    completion : ChatCompletion

    Returns
    -------
    ChatResponse

    Raises
    ------
    ModelProviderError
        If the completion has no choices or several (``n`` > 1), or a tool
        call that is not a function call.
    """
    from openai.types.chat import ChatCompletionMessageFunctionToolCall

    from nodestep.exceptions import ModelProviderError

    if not completion.choices:
        raise ModelProviderError("OpenAI response contains no choices")
    if len(completion.choices) > 1:
        raise ModelProviderError(
            f"OpenAI response contains {len(completion.choices)} choices; "
            "nodestep reads one, so do not pass n > 1"
        )
    choice = completion.choices[0]
    message = choice.message
    tool_calls: list[ToolCall] = []
    for item in message.tool_calls or []:
        if not isinstance(item, ChatCompletionMessageFunctionToolCall):
            raise ModelProviderError(
                f"OpenAI response contains a {item.type!r} tool call {item.id!r}; "
                "nodestep supports only function tool calls"
            )
        tool_calls.append(
            ToolCall.from_arguments_json(
                item.id, item.function.name, item.function.arguments
            )
        )
    usage = completion.usage.model_dump() if completion.usage is not None else {}
    return ChatResponse(
        content=message.content,
        refusal=message.refusal,
        tool_calls=tool_calls,
        finish_reason=choice.finish_reason,
        usage=usage,
        model=completion.model,
        raw=completion,
    )


def chunks_from_stream_event(event: ChatCompletionChunk) -> list[ChatStreamChunk]:
    """Convert one streamed OpenAI chunk into stream chunks that carry its model.

    Parameters
    ----------
    event : ChatCompletionChunk

    Returns
    -------
    list[ChatStreamChunk]

    Raises
    ------
    ModelProviderError
        If the chunk belongs to a choice other than the first (``n`` > 1).
    """
    from nodestep.exceptions import ModelProviderError

    chunks: list[ChatStreamChunk] = []
    model = event.model
    for choice in event.choices:
        if choice.index != 0:
            raise ModelProviderError(
                f"OpenAI stream contains choice {choice.index}; nodestep reads "
                "one, so do not pass n > 1"
            )
        delta = choice.delta
        if delta.content:
            chunks.append(
                ChatStreamChunk(content_delta=delta.content, model=model, raw=event)
            )
        if delta.refusal:
            chunks.append(
                ChatStreamChunk(refusal_delta=delta.refusal, model=model, raw=event)
            )
        for call in delta.tool_calls or []:
            chunks.append(
                ChatStreamChunk(
                    tool_call_delta={
                        "index": call.index,
                        "id": call.id,
                        "function": {
                            "name": call.function.name if call.function else None,
                            "arguments": call.function.arguments
                            if call.function
                            else None,
                        },
                    },
                    model=model,
                    raw=event,
                )
            )
        if choice.finish_reason:
            chunks.append(
                ChatStreamChunk(
                    finish_reason=choice.finish_reason, model=model, raw=event
                )
            )
    if event.usage is not None:
        chunks.append(
            ChatStreamChunk(usage=event.usage.model_dump(), model=model, raw=event)
        )
    return chunks


__all__ = [
    "chunks_from_stream_event",
    "message_to_payload",
    "request_to_parameters",
    "response_from_completion",
    "to_strict_schema",
]
