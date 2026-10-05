"""`SummarizationMiddleware`, which replaces older messages with a summary.

See [Built-in middleware](../../concepts/middleware.md#built-in-middleware).
"""

from __future__ import annotations

import dataclasses
import math
import weakref
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field

from nodestep.chat import (
    AIMessage,
    BaseMessage,
    ChatRequest,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from nodestep.chat.base import Chat
from nodestep.core.command import Command
from nodestep.core.tool import Tool, ToolContext, tool
from nodestep.exceptions import GraphConfigError, ModelProviderError
from nodestep.middleware.base import (
    Middleware,
    ModelMiddlewareContext,
    NodeMiddlewareContext,
    Replacement,
)
from nodestep.utils.reducers import RemoveMessage

_PER_MESSAGE_OVERHEAD = 4


class TokenCounter(Protocol):
    """Protocol for counting message tokens."""

    def count_message(self, message: BaseMessage) -> int:
        """Return the token count of one message."""
        ...

    def count_messages(self, messages: list[BaseMessage]) -> int:
        """Return the token count of several messages."""
        ...


class ApproximateTokenCounter:
    """Estimate tokens as characters divided by four."""

    def count_message(self, message: BaseMessage) -> int:
        """Return the estimated token count of one message."""
        text_chars = 0
        content = getattr(message, "content", None) or ""
        text_chars += len(content)
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                text_chars += len(call.arguments_json or "")
                text_chars += len(call.name or "")
        if isinstance(message, ToolMessage):
            text_chars += len(message.name or "") + len(message.tool_call_id or "")
        approx = math.ceil(text_chars / 4) + _PER_MESSAGE_OVERHEAD
        return max(approx, 1)

    def count_messages(self, messages: list[BaseMessage]) -> int:
        """Return the estimated token count of several messages."""
        return sum(self.count_message(message) for message in messages)


class SummarizationDecision(BaseModel):
    """Whether to summarize, and why.

    Attributes
    ----------
    reason : str
        The check that was made, such as ``"token_limit:1200>960"``.
    token_count : int or None
        Tokens counted; ``None`` for ``MessageCountTrigger``.
    message_count : int
        Messages checked.
    """

    should_summarize: bool
    reason: str
    token_count: int | None = None
    message_count: int


class SummarizationTrigger(Protocol):
    """Protocol that decides when to summarize."""

    def should_summarize(
        self,
        messages: list[BaseMessage],
        token_counter: TokenCounter,
    ) -> SummarizationDecision:
        """Decide whether the messages should be summarized.

        Parameters
        ----------
        messages : list[BaseMessage]
        token_counter : TokenCounter

        Returns
        -------
        SummarizationDecision
        """
        ...


class TokenLimitTrigger(BaseModel):
    """Summarize above a share of the model's input limit.

    Attributes
    ----------
    max_input_tokens : int
        The model's input limit.
    threshold_ratio : float
        Share of the limit above which the trigger fires.
    """

    max_input_tokens: int = Field(gt=0)
    threshold_ratio: float = Field(default=0.8, gt=0, le=1)

    def should_summarize(
        self,
        messages: list[BaseMessage],
        token_counter: TokenCounter,
    ) -> SummarizationDecision:
        """Check the token count against the limit."""
        limit = math.floor(self.max_input_tokens * self.threshold_ratio)
        count = token_counter.count_messages(messages)
        return SummarizationDecision(
            should_summarize=count > limit,
            reason=(
                f"token_limit:{count}>{limit}"
                if count > limit
                else f"token_limit:{count}<={limit}"
            ),
            token_count=count,
            message_count=len(messages),
        )


class FixedTokenTrigger(BaseModel):
    """Summarize above a fixed number of tokens.

    Attributes
    ----------
    max_tokens : int
        Token count above which the trigger fires.
    """

    max_tokens: int = Field(gt=0)

    def should_summarize(
        self,
        messages: list[BaseMessage],
        token_counter: TokenCounter,
    ) -> SummarizationDecision:
        """Check the token count against the limit."""
        count = token_counter.count_messages(messages)
        return SummarizationDecision(
            should_summarize=count > self.max_tokens,
            reason=(
                f"fixed_token:{count}>{self.max_tokens}"
                if count > self.max_tokens
                else f"fixed_token:{count}<={self.max_tokens}"
            ),
            token_count=count,
            message_count=len(messages),
        )


class MessageCountTrigger(BaseModel):
    """Summarize above a number of messages.

    Attributes
    ----------
    max_messages : int
        Message count above which the trigger fires.
    """

    max_messages: int = Field(gt=0)

    def should_summarize(
        self,
        messages: list[BaseMessage],
        token_counter: TokenCounter,
    ) -> SummarizationDecision:
        """Check the message count against the limit."""
        count = len(messages)
        return SummarizationDecision(
            should_summarize=count > self.max_messages,
            reason=(
                f"message_count:{count}>{self.max_messages}"
                if count > self.max_messages
                else f"message_count:{count}<={self.max_messages}"
            ),
            message_count=count,
        )


class RecentMessagesPolicy(BaseModel):
    """Keep the last N messages out of the summary.

    Attributes
    ----------
    keep_recent : int
        Number of messages kept as they are.
    """

    keep_recent: int = Field(default=20, ge=0)

    def split(
        self,
        messages: list[BaseMessage],
        token_counter: TokenCounter,
    ) -> tuple[list[BaseMessage], list[BaseMessage]]:
        """Split messages into those to summarize and those to keep.

        Parameters
        ----------
        messages : list[BaseMessage]
        token_counter : TokenCounter

        Returns
        -------
        tuple[list, list]
            Older and recent messages.
        """
        if self.keep_recent == 0:
            return list(messages), []
        if len(messages) <= self.keep_recent:
            return [], list(messages)
        return list(messages[: -self.keep_recent]), list(messages[-self.keep_recent :])


class RecentTokensPolicy(BaseModel):
    """Keep the most recent messages within a token budget.

    Attributes
    ----------
    keep_recent_tokens : int
        Token budget of the messages kept as they are.
    """

    keep_recent_tokens: int = Field(gt=0)

    def split(
        self,
        messages: list[BaseMessage],
        token_counter: TokenCounter,
    ) -> tuple[list[BaseMessage], list[BaseMessage]]:
        """Split messages into those to summarize and those to keep.

        Parameters
        ----------
        messages : list[BaseMessage]
        token_counter : TokenCounter

        Returns
        -------
        tuple[list, list]
            Older and recent messages.
        """
        if not messages:
            return [], []
        recent: list[BaseMessage] = []
        running = 0
        for message in reversed(messages):
            cost = token_counter.count_message(message)
            if recent and running + cost > self.keep_recent_tokens:
                break
            recent.insert(0, message)
            running += cost
        kept = len(recent)
        older = list(messages[:-kept]) if kept > 0 else list(messages)
        return older, recent


class SummarizationResult(BaseModel):
    """Result of the ``summarize_context`` tool.

    Attributes
    ----------
    summary : str
        The summary; empty when no older message was left to summarize.
    original_message_count : int
        Messages before summarizing.
    summarized_message_count : int
        Older messages the summary replaces.
    kept_message_count : int
        Recent messages kept as they are.
    token_count : int or None
        Tokens counted by the trigger; ``None`` for ``MessageCountTrigger``.
    reason : str
        The trigger's ``reason``, or ``"no_older_messages"``.
    """

    summary: str
    original_message_count: int
    summarized_message_count: int
    kept_message_count: int
    token_count: int | None = None
    reason: str


def _clean_split(conversation: list[BaseMessage], split: int) -> int:
    while split > 0:
        opens_with_result = split < len(conversation) and isinstance(
            conversation[split], ToolMessage
        )
        previous = conversation[split - 1]
        closes_with_call = isinstance(previous, AIMessage) and bool(previous.tool_calls)
        if not (opens_with_result or closes_with_call):
            break
        split -= 1
    return split


def _transcript(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        content = message.content or ""
        if isinstance(message, AIMessage):
            calls = ", ".join(
                f"{call.name}({call.arguments_json})" for call in message.tool_calls
            )
            lines.append(
                f"assistant: {content}" + (f" [called {calls}]" if calls else "")
            )
        elif isinstance(message, ToolMessage):
            lines.append(f"tool {message.name or ''}: {content}")
        elif isinstance(message, SystemMessage):
            lines.append(f"system: {content}")
        else:
            lines.append(f"user: {content}")
    return "\n".join(lines)


class _Pending:
    def __init__(self, stored_ids: set[str]) -> None:
        self.stored_ids = stored_ids
        self.edits: list[Any] | None = None


def _with_summary(update: Any, field: str, edits: list[Any]) -> Any:
    if isinstance(update, Command):
        return dataclasses.replace(
            update, update=_with_summary(update.update, field, edits)
        )
    values = dict(update) if isinstance(update, dict) else {}
    values[field] = [*edits, *(values.get(field) or [])]
    return values


def _default_summary_message_factory(summary: str) -> BaseMessage:
    return SystemMessage(content=summary)


class SummarizationMiddleware(Middleware):
    """Replace older messages with a summary before a model call once a trigger fires.

    Only ``model_node`` calls are summarized, and not in a node that shares
    its superstep. The node's stored messages that ``recent`` does not keep go
    to ``chat``; one summary message replaces them in place, with any unstored
    messages between them, and other unstored messages, such as the system
    prompt, stay. The node's update stores the summary in place of the first
    replaced message and removes the others. See
    [Built-in middleware](../../concepts/middleware.md#built-in-middleware).

    Parameters
    ----------
    chat : Chat
        Model that writes the summary.
    trigger : SummarizationTrigger
        When to summarize, checked against the whole request, e.g.
        ``MessageCountTrigger(max_messages=40)``.
    recent : RecentMessagesPolicy or RecentTokensPolicy
        Which messages to keep verbatim; a tool call and its results are
        never split.
    messages_field : str, optional
        State field holding the messages.
    token_counter : TokenCounter, optional
        Counts tokens for the trigger and the recent policy; an
        ``ApproximateTokenCounter`` by default.
    summary_prompt : str, optional
        Instructions for the summarizer.
    summary_message_factory : callable, optional
        Builds the summary message; a ``SystemMessage`` by default.
    provide_tool : bool, optional
        Return a ``summarize_context`` tool from ``tools()``; list it in a
        node's ``tools=``.

    Raises
    ------
    ModelProviderError
        From ``before_model``, if the summarizer returns no text; the model is
        not called.
    GraphConfigError
        From ``before_node``, if the state schema does not declare
        ``messages_field``.
    """

    def __init__(
        self,
        chat: Chat,
        *,
        trigger: SummarizationTrigger,
        recent: RecentMessagesPolicy | RecentTokensPolicy,
        messages_field: str = "messages",
        token_counter: TokenCounter | None = None,
        summary_prompt: str = (
            "Summarize the prior conversation concisely. "
            "Preserve facts, decisions, and unresolved questions."
        ),
        summary_message_factory: Callable[[str], BaseMessage] | None = None,
        provide_tool: bool = False,
    ) -> None:
        self.chat = chat
        self.messages_field = messages_field
        self.trigger = trigger
        self.recent = recent
        self.token_counter = token_counter or ApproximateTokenCounter()
        self.summary_prompt = summary_prompt
        self.summary_message_factory = (
            summary_message_factory or _default_summary_message_factory
        )
        self.provide_tool = provide_tool
        self._scratch_key = f"summarization:{id(self)}"
        self._pending: weakref.WeakValueDictionary[tuple[str, int, str], _Pending] = (
            weakref.WeakValueDictionary()
        )
        self._summarize_context = self._build_summarize_context()

    def before_node(self, ctx: NodeMiddlewareContext) -> None:
        """Remember which messages the node's stored input holds.

        Returns
        -------
        None

        Raises
        ------
        GraphConfigError
            If the graph's state schema does not declare ``messages_field``.
        """
        if ctx.superstep_size > 1:
            return None
        schema = ctx.state_schema
        if schema is not None and self.messages_field not in schema.fields:
            raise GraphConfigError(
                f"SummarizationMiddleware reads the state field "
                f"'{self.messages_field}', which the state schema of graph "
                f"'{ctx.graph_name}' does not declare"
            )
        stored = ctx.stored_state
        unset = isinstance(stored, dict) and self.messages_field not in stored
        messages = [] if unset else self._read(stored)
        pending = _Pending(
            {message.id for message in messages if message.id is not None}
        )
        ctx.scratch[self._scratch_key] = pending
        self._pending[(ctx.run_id, ctx.step, ctx.task_id)] = pending
        return None

    async def before_model(self, ctx: ModelMiddlewareContext) -> Replacement | None:
        """Summarize the request's older stored messages when the trigger fires.

        Returns
        -------
        Replacement or None
            The request with the summary, or ``None`` to send it unchanged.

        Raises
        ------
        ModelProviderError
            If the summarizer returns no text.
        """
        pending = self._pending.get((ctx.run_id, ctx.step, ctx.task_id))
        if pending is None:
            return None
        stored_ids = pending.stored_ids
        messages: list[BaseMessage] = list(ctx.request.messages)
        if not self.trigger.should_summarize(
            messages, self.token_counter
        ).should_summarize:
            return None
        positions = [
            index
            for index, message in enumerate(messages)
            if message.id is not None and message.id in stored_ids
        ]
        stored = [messages[index] for index in positions]
        older, _ = self.recent.split(stored, self.token_counter)
        split = _clean_split(stored, len(older))
        if split == 0:
            return None
        start, end = positions[0], positions[split - 1] + 1
        summary = self.summary_message_factory(
            await self._summarize(messages[start:end])
        ).model_copy(update={"id": stored[0].id})
        pending.edits = [
            summary,
            *(RemoveMessage(str(message.id)) for message in stored[1:split]),
        ]
        return ctx.replace(
            ctx.request.model_copy(
                update={"messages": [*messages[:start], summary, *messages[end:]]}
            )
        )

    def after_node(self, ctx: NodeMiddlewareContext) -> Replacement | None:
        """Add the summary edits to the node's update.

        Returns
        -------
        Replacement or None
            The update with the edits, or ``None`` when nothing was summarized.
        """
        pending = ctx.scratch.pop(self._scratch_key, None)
        if pending is None or pending.edits is None:
            return None
        return ctx.replace(_with_summary(ctx.state, self.messages_field, pending.edits))

    def on_error(self, ctx: NodeMiddlewareContext, error: Exception) -> None:
        """Forget the node's pending summary.

        Returns
        -------
        None
        """
        ctx.scratch.pop(self._scratch_key, None)
        return None

    def tools(self) -> tuple[Tool, ...]:
        """Return this instance's ``summarize_context`` when ``provide_tool`` is set."""
        if self.provide_tool:
            return (self._summarize_context,)
        return ()

    def _build_summarize_context(self) -> Tool:
        middleware = self

        @tool
        async def summarize_context(ctx: ToolContext) -> SummarizationResult:
            """Summarize the older part of the conversation without changing it."""
            return await middleware.summarize_state(ctx.state)

        return summarize_context

    async def summarize_state(self, state: Any) -> SummarizationResult:
        """Summarize the messages of a state without changing it.

        Parameters
        ----------
        state : Any

        Returns
        -------
        SummarizationResult

        Raises
        ------
        GraphConfigError
            If ``state`` has no ``messages_field``.
        """
        messages = self._read(state)
        decision = self.trigger.should_summarize(messages, self.token_counter)
        older, _ = self.recent.split(messages, self.token_counter)
        split = _clean_split(messages, len(older))
        older, recent = messages[:split], messages[split:]
        if not older:
            return SummarizationResult(
                summary="",
                original_message_count=len(messages),
                summarized_message_count=0,
                kept_message_count=len(recent),
                token_count=decision.token_count,
                reason="no_older_messages",
            )
        summary_text = await self._summarize(older)
        return SummarizationResult(
            summary=summary_text,
            original_message_count=len(messages),
            summarized_message_count=len(older),
            kept_message_count=len(recent),
            token_count=decision.token_count,
            reason=decision.reason,
        )

    def _read(self, state: Any) -> list[BaseMessage]:
        if isinstance(state, dict):
            if self.messages_field in state:
                return list(state[self.messages_field])
        elif hasattr(state, self.messages_field):
            return list(getattr(state, self.messages_field))
        raise GraphConfigError(
            f"SummarizationMiddleware reads the state field "
            f"'{self.messages_field}', which a {type(state).__name__} state "
            "does not have"
        )

    async def _summarize(self, messages: list[BaseMessage]) -> str:
        request = ChatRequest(
            messages=[
                SystemMessage(content=self.summary_prompt),
                HumanMessage(content=_transcript(messages)),
            ],
        )
        response = await self.chat.complete(request)
        if not response.content:
            raise ModelProviderError(
                "The summarizer returned an empty summary; the older messages were kept"
            )
        return response.content


__all__ = [
    "ApproximateTokenCounter",
    "FixedTokenTrigger",
    "MessageCountTrigger",
    "RecentMessagesPolicy",
    "RecentTokensPolicy",
    "SummarizationDecision",
    "SummarizationMiddleware",
    "SummarizationResult",
    "SummarizationTrigger",
    "TokenCounter",
    "TokenLimitTrigger",
]
