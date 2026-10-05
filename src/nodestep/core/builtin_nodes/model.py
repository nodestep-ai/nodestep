from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from nodestep.chat import (
    AIMessage,
    Chat,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    SystemMessage,
    ToolDefinition,
    ToolMessage,
    assemble_stream,
)
from nodestep.core.node import Node, node
from nodestep.core.stream import NodeContext
from nodestep.core.tool import Tool, collect_tools
from nodestep.exceptions import (
    ChatHistoryError,
    GraphConfigError,
    StructuredOutputError,
)
from nodestep.middleware.base import (
    ModelMiddlewareContext,
    run_after_hooks,
    run_before_hooks,
)
from nodestep.models.base import NodestepModel
from nodestep.utils.reducers import add_messages
from nodestep.utils.state import get_state_field as _get_field


class StructuredOutputEvent(NodestepModel):
    """Payload of the ``custom`` event sent by ``model_node(on_invalid="emit")``.

    Attributes
    ----------
    schema_name : str
        Name of the ``output_schema`` model.
    message : str
        The validation error.
    raw_content : str or None
        The answer text that did not validate.
    """

    kind: Literal["structured_output_error"] = "structured_output_error"
    schema_name: str
    message: str
    raw_content: str | None = None


def _check_fields(
    schema: Any, node_name: str, messages_field: str, written: Sequence[str]
) -> None:
    declared = {} if schema is None else schema.fields
    descriptor = declared.get(messages_field)
    if descriptor is None or descriptor.reducer is not add_messages:
        problem = (
            "is not declared" if descriptor is None else "does not use add_messages"
        )
        raise GraphConfigError(
            f"Node '{node_name}' appends to the state field '{messages_field}', "
            f"which {problem}; declare it as Annotated[list[Message], add_messages]"
        )
    for field_name in written:
        if field_name not in declared:
            raise GraphConfigError(
                f"Node '{node_name}' writes the state field '{field_name}', "
                "which the state schema does not declare"
            )


def _history_problems(messages: list[Any]) -> list[str]:
    problems: list[str] = []
    pending: list[str] | None = None
    owner = 0

    def close() -> None:
        if pending:
            unanswered = ", ".join(repr(call_id) for call_id in pending)
            problems.append(
                f"message {owner} (AIMessage) has tool calls without results: "
                f"{unanswered}"
            )

    for index, message in enumerate(messages):
        if isinstance(message, ToolMessage):
            if pending is not None and message.tool_call_id in pending:
                pending.remove(message.tool_call_id)
            else:
                problems.append(
                    f"message {index} (ToolMessage {message.tool_call_id!r}) "
                    "answers no tool call of the AIMessage before it"
                )
            continue
        close()
        pending = None
        if isinstance(message, AIMessage) and message.tool_calls:
            pending = [call.id for call in message.tool_calls]
            owner = index
    close()
    return problems


def _pair_tool_messages(messages: list[Any]) -> list[Any]:
    paired: list[Any] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        index += 1
        if isinstance(message, ToolMessage):
            continue
        paired.append(message)
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        expected = {call.id: call for call in message.tool_calls}
        while index < len(messages) and isinstance(messages[index], ToolMessage):
            reply = messages[index]
            index += 1
            if reply.tool_call_id in expected:
                expected.pop(reply.tool_call_id)
                paired.append(reply)
        paired.extend(
            ToolMessage(
                content="Tool call was not executed.",
                name=call.name,
                tool_call_id=call.id,
            )
            for call in expected.values()
        )
    return paired


def _require(value: Any, expected: type, hook_name: str) -> None:
    if not isinstance(value, expected):
        raise TypeError(
            f"{hook_name} must replace the value with a {expected.__name__}, "
            f"got {type(value).__name__}"
        )


async def _forward_tokens(
    chunks: AsyncIterator[ChatStreamChunk], ctx: NodeContext
) -> AsyncIterator[ChatStreamChunk]:
    async for chunk in chunks:
        ctx._emit_token(chunk)
        yield chunk


async def _call_model(
    chat: Chat, request: ChatRequest, ctx: NodeContext, stream: bool
) -> ChatResponse:
    def hook_context(
        current: ChatRequest, response: ChatResponse | None = None
    ) -> ModelMiddlewareContext:
        return ModelMiddlewareContext(
            graph_name=ctx.graph_name,
            node_name=ctx.node,
            run_id=ctx.run_id,
            thread_id=ctx.thread_id,
            step=ctx.step,
            task_id=ctx.task_id,
            root_run_id=ctx.root_run_id,
            model=chat.model,
            request=current,
            response=response,
        )

    def before_context(current: Any) -> ModelMiddlewareContext:
        _require(current, ChatRequest, "before_model")
        return hook_context(current)

    def after_context(current: Any) -> ModelMiddlewareContext:
        _require(current, ChatResponse, "after_model")
        return hook_context(request, current)

    request = await run_before_hooks(
        ctx.middleware, "before_model", request, before_context
    )
    _require(request, ChatRequest, "before_model")
    if stream:
        response = await assemble_stream(_forward_tokens(chat.stream(request), ctx))
    else:
        response = await chat.complete(request)
    response = await run_after_hooks(
        ctx.middleware, "after_model", response, after_context
    )
    _require(response, ChatResponse, "after_model")
    return response


def model_node(
    name: str,
    *,
    chat: Chat,
    system_prompt: str | None = None,
    tools: Sequence[Tool] = (),
    tool_choice: Literal["auto", "required", "none"] | str | None = None,
    stream: bool = False,
    repair_history: bool = False,
    output_schema: type[BaseModel] | None = None,
    on_invalid: Literal["raise", "emit"] = "raise",
    messages_field: str = "messages",
    tool_calls_field: str = "tool_calls",
    final_text_field: str = "final_text",
    final_output_field: str = "final_output",
) -> Node:
    """Create a node that calls a chat model.

    The node sends the messages of ``messages_field`` and appends the model's
    ``AIMessage`` to it. It writes the requested calls to ``tool_calls_field``
    and the text to ``final_text_field``, or ``None`` when the model calls
    tools or refuses. The ``before_model`` and ``after_model`` hooks run around
    the call.

    Parameters
    ----------
    name : str
        Node name.
    chat : Chat
        Chat model to call.
    system_prompt : str, optional
        System message prepended to the request.
    tools : Sequence[Tool], optional
        The node's complete tool list; middleware tools are not added.
    tool_choice : {"auto", "required", "none"} or str, optional
        The request's tool choice; another string names the tool to call.
        ``None`` sets none.
    stream : bool, optional
        Call ``chat.stream`` instead of ``chat.complete``; the chunks become
        ``"tokens"`` events.
    repair_history : bool, optional
        Send a repaired copy of an inconsistent history instead of raising.
        Tool results that answer no call of the ``AIMessage`` before them are
        dropped; calls without a result get ``"Tool call was not executed."``.
        The state is not changed.
    output_schema : type[BaseModel], optional
        Model the final answer is parsed into and stored in
        ``final_output_field``; ``None`` on turns that call tools or refuse.
    on_invalid : {"raise", "emit"}, optional
        For an answer that does not validate: raise ``StructuredOutputError``,
        or emit a ``"custom"`` event with a ``StructuredOutputEvent`` and store
        ``None``.
    messages_field : str, optional
        State field of the conversation; it must use ``add_messages``.
    tool_calls_field : str, optional
        State field the requested tool calls are written to.
    final_text_field : str, optional
        State field the answer text is written to.
    final_output_field : str, optional
        State field the parsed answer is written to; used only with
        ``output_schema``.

    Returns
    -------
    Node

    Raises
    ------
    GraphConfigError
        If two different tools share a name, or the same tool is listed twice.
    TypeError
        If ``chat`` has no ``model`` string.
    ValueError
        If ``on_invalid`` is neither ``"raise"`` nor ``"emit"``.
    GraphConfigError
        When the node runs, before the model is called: ``messages_field``
        does not use ``add_messages``, or a field it writes is not declared.
    ChatHistoryError
        When the node runs, before the model is called, with
        ``repair_history`` off: a tool call has no result right after it, or a
        tool result answers no call of the ``AIMessage`` before it.
    TypeError
        When the node runs: a model hook replaced the request or the response
        with another type.
    """
    if on_invalid not in ("raise", "emit"):
        raise ValueError(f"on_invalid must be 'raise' or 'emit', got {on_invalid!r}")
    if not isinstance(getattr(chat, "model", None), str):
        raise TypeError(
            f"{type(chat).__name__} has no model name; the Chat protocol declares "
            "model: str, e.g. a class attribute model = 'my-model'"
        )
    definitions = [
        ToolDefinition(
            name=item.name,
            description=item.description,
            input_schema=item.input_schema(),
        )
        for item in collect_tools(tools)
    ]
    written = [tool_calls_field, final_text_field]
    if output_schema is not None:
        written.append(final_output_field)

    def parse(
        schema: type[BaseModel], response: ChatResponse, ctx: NodeContext
    ) -> BaseModel | None:
        try:
            return schema.model_validate_json(response.content or "")
        except ValidationError as error:
            if on_invalid == "raise":
                raise StructuredOutputError(
                    schema_name=schema.__name__,
                    message=str(error),
                    raw_content=response.content,
                ) from error
            ctx.emit(
                StructuredOutputEvent(
                    schema_name=schema.__name__,
                    message=str(error),
                    raw_content=response.content,
                )
            )
            return None

    @node(name=name)
    async def run_model(state: Any, ctx: NodeContext) -> dict:
        _check_fields(ctx.state_schema, name, messages_field, written)
        messages = list(_get_field(state, messages_field))
        if repair_history:
            messages = _pair_tool_messages(messages)
        else:
            problems = _history_problems(messages)
            if problems:
                raise ChatHistoryError(
                    f"Node '{name}' cannot send the messages of '{messages_field}': "
                    + "; ".join(problems)
                    + "; fix the history, or pass repair_history=True to send a "
                    "repaired copy"
                )
        if system_prompt is not None:
            messages = [SystemMessage(content=system_prompt), *messages]
        request = ChatRequest(
            messages=messages,
            tools=definitions,
            tool_choice=tool_choice,
            output_schema=(
                None if output_schema is None else output_schema.model_json_schema()
            ),
        )
        response = await _call_model(chat, request, ctx, stream)
        answered = not response.tool_calls and response.refusal is None
        update: dict[str, Any] = {
            messages_field: [
                AIMessage(
                    content=response.content,
                    refusal=response.refusal,
                    tool_calls=list(response.tool_calls),
                    finish_reason=response.finish_reason,
                    usage=response.usage,
                )
            ],
            tool_calls_field: list(response.tool_calls),
            final_text_field: response.content if answered else None,
        }
        if output_schema is not None:
            update[final_output_field] = (
                parse(output_schema, response, ctx) if answered else None
            )
        return update

    return run_model


__all__ = ["StructuredOutputEvent", "model_node"]
