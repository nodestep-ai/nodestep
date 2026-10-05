from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from nodestep.chat.messages import ChatRequest, ChatResponse, ChatStreamChunk


@runtime_checkable
class Chat(Protocol):
    """Protocol of a chat model: any object with ``model``, ``complete`` and ``stream``.

    Attributes
    ----------
    model : str
        Name of the model; middleware sees it as
        ``ModelMiddlewareContext.model``.
    """

    model: str

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Send a request and return the full response.

        Parameters
        ----------
        request : ChatRequest
            Messages, tool definitions and an optional output schema.

        Returns
        -------
        ChatResponse
            The model's reply.
        """
        ...

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Send a request and yield the response as it is generated.

        Parameters
        ----------
        request : ChatRequest
            Messages, tool definitions and an optional output schema.

        Returns
        -------
        AsyncIterator[ChatStreamChunk]
            Content, tool-call, finish and usage chunks in arrival order.
        """
        ...


__all__ = ["Chat"]
