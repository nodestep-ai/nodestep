from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

from nodestep.core.builtin_nodes.model import _check_fields
from nodestep.core.stream import NodeContext
from nodestep.core.tool import Tool, ToolContext, call_tool, collect_tools
from nodestep.utils.state import get_state_field as _state_field

if TYPE_CHECKING:
    from nodestep.core.node import Node

ToolErrorPolicy = Literal["raise", "return"] | Callable[[Exception], str]
"""What a tool runner does when a tool call fails.

``"raise"`` raises the error, ``"return"`` sends its text to the model as the
tool result, and a callable turns the exception into that text.
"""


def _entry_id(entry: Any) -> str | None:
    value = entry.get("id") if isinstance(entry, dict) else getattr(entry, "id", None)
    return value if isinstance(value, str) else None


def tool_runner(
    name: str,
    *,
    tools: Sequence[Tool],
    tool_errors: ToolErrorPolicy = "raise",
    messages_field: str = "messages",
    tool_calls_field: str = "tool_calls",
) -> Node:
    """Create a node that runs the pending tool calls.

    The calls in ``tool_calls_field`` run concurrently. Their ``ToolMessage``
    results are appended to ``messages_field``, and ``tool_calls_field`` is
    cleared. When ``before_tool`` middleware changed the arguments, the
    ``ToolMessage`` records the ones the tool ran with in ``arguments``. A call
    denied with ``ToolDeniedError`` always becomes a ``ToolMessage`` with the
    reason. Calls that finished before a pause do not run again on resume.

    Parameters
    ----------
    name : str
        Node name.
    tools : Sequence[Tool]
        The node's complete tool list; middleware tools are not added.
    tool_errors : {"raise", "return"} or callable, optional
        What a failed call (an unknown tool, invalid arguments or an exception
        from the tool) does. ``"raise"`` raises it, several as
        ``ToolGroupExecutionError``. ``"return"`` sends
        ``"Tool error: <type>: <message>"`` to the model as the result. A
        callable turns the exception into that text. ``ContextNotProvidedError``
        and ``GraphConfigError`` from a tool, and an invalid call without a
        string ``id``, always raise.
    messages_field : str, optional
        State field the ``ToolMessage`` results are appended to; it must use
        ``add_messages``.
    tool_calls_field : str, optional
        State field the calls are read from; the node clears it.

    Returns
    -------
    Node

    Raises
    ------
    GraphConfigError
        If two different tools share a name, or the same tool is listed twice.
    ValueError
        If ``tool_errors`` is neither ``"raise"``, ``"return"`` nor a callable.
    GraphConfigError
        When the node runs, before any tool runs: ``messages_field`` does not
        use ``add_messages``, or ``tool_calls_field`` is not declared.
    """
    from nodestep.chat import ToolCall, ToolMessage
    from nodestep.core.command import GraphInterrupt, GraphInterruptGroup
    from nodestep.core.node import node
    from nodestep.exceptions import (
        ContextNotProvidedError,
        GraphConfigError,
        ToolDeniedError,
        ToolExecutionError,
        ToolGroupExecutionError,
    )

    if not callable(tool_errors) and tool_errors not in ("raise", "return"):
        raise ValueError(
            f"tool_errors must be 'raise', 'return' or a callable, got {tool_errors!r}"
        )
    tool_map = {item.name: item for item in collect_tools(tools)}

    def describe(error: Exception) -> str:
        if not callable(tool_errors):
            return f"Tool error: {type(error).__name__}: {error}"
        text = tool_errors(error)
        if not isinstance(text, str):
            raise TypeError(f"tool_errors must return a str, got {type(text).__name__}")
        return text

    @node(name=name)
    async def run_tools(state: Any, ctx: NodeContext) -> dict:
        _check_fields(ctx.state_schema, name, messages_field, [tool_calls_field])
        raw_calls = list(_state_field(state, tool_calls_field))
        finished: dict[str, Any] = ctx.cache.setdefault("tool_results", {})

        async def execute(call: ToolCall) -> ToolMessage:
            try:
                selected = tool_map.get(call.name)
                if selected is None:
                    raise ToolExecutionError(call.name, call.arguments, "Unknown tool")
                if call.arguments_error is not None:
                    raise ToolExecutionError(
                        call.name,
                        call.arguments,
                        f"Invalid tool call arguments: {call.arguments_error}",
                    )
                tool_ctx = ToolContext.from_node_context(
                    ctx, state, tool_call_id=call.id
                )
                result = await call_tool(selected, call.arguments, tool_ctx)
                content = result.value.model_dump_json()
                requested = selected.input_model.model_validate(call.arguments)
            except ToolDeniedError as error:
                return ToolMessage(
                    content=str(error), name=call.name, tool_call_id=call.id
                )
            except (ContextNotProvidedError, GraphConfigError):
                raise
            except Exception as error:
                if tool_errors == "raise":
                    raise
                return ToolMessage(
                    content=describe(error), name=call.name, tool_call_id=call.id
                )
            return ToolMessage(
                content=content,
                name=call.name,
                tool_call_id=call.id,
                arguments=(
                    None
                    if result.arguments == requested.model_dump(mode="json")
                    else result.arguments
                ),
            )

        async def run_one(entry: Any) -> ToolMessage:
            try:
                call = (
                    entry
                    if isinstance(entry, ToolCall)
                    else ToolCall.model_validate(entry)
                )
            except Exception as error:
                entry_id = _entry_id(entry)
                if tool_errors == "raise" or entry_id is None:
                    raise
                return ToolMessage(content=describe(error), tool_call_id=entry_id)
            done = finished.get(call.id)
            if done is not None:
                return ToolMessage.model_validate(done)
            message = await execute(call)
            finished[call.id] = message.model_dump(mode="json")
            return message

        outcomes = await asyncio.gather(
            *[run_one(entry) for entry in raw_calls], return_exceptions=True
        )
        messages: list[ToolMessage] = []
        collected_interrupts: list[GraphInterrupt] = []
        collected_errors: list[BaseException] = []
        for outcome in outcomes:
            if isinstance(outcome, GraphInterrupt):
                collected_interrupts.append(outcome)
            elif isinstance(outcome, GraphInterruptGroup):
                collected_interrupts.extend(outcome.interrupts)
            elif isinstance(outcome, BaseException):
                collected_errors.append(outcome)
            else:
                messages.append(outcome)
        if collected_interrupts:
            raise GraphInterruptGroup(collected_interrupts)
        if collected_errors:
            if len(collected_errors) == 1:
                raise collected_errors[0]
            raise ToolGroupExecutionError(collected_errors)
        return {messages_field: messages, tool_calls_field: []}

    return run_tools


__all__ = ["ToolErrorPolicy", "tool_runner"]
