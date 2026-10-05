from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterable

from nodestep.chat import ChatRequest, ChatResponse, ChatStreamChunk
from nodestep.exceptions import ModelProviderError

_WORDS = re.compile(r"\s*\S+|\s+")


def _as_response(item: ChatResponse | str) -> ChatResponse:
    if isinstance(item, str):
        return ChatResponse(content=item, model="scripted")
    if not isinstance(item, ChatResponse):
        raise TypeError(
            "ScriptedChat responses must be ChatResponse or str, "
            f"got {type(item).__name__}"
        )
    if item.model is None:
        return item.model_copy(update={"model": "scripted"})
    return item


def _check_streamable(response: ChatResponse) -> None:
    for call in response.tool_calls:
        if call.arguments_error is not None:
            raise ValueError(
                f"ScriptedChat cannot stream tool call '{call.id}' with "
                "arguments_error: assemble_stream parses streamed arguments again "
                "and would lose the error; script this response for complete() only"
            )
    if response.raw is not None:
        raise ValueError(
            "ScriptedChat cannot stream a response with raw set: assemble_stream "
            "does not rebuild raw; script this response for complete() only"
        )


class ScriptedChat:
    """Chat model that returns scripted responses, for tests and examples.

    Parameters
    ----------
    responses : Iterable[ChatResponse or str], optional
        Responses returned in order, one per request. A ``str`` means
        ``ChatResponse(content=...)``.
    default : ChatResponse or str, optional
        Response returned for every request after the script runs out.
        Without it, such a request raises ``ModelProviderError``.

    Attributes
    ----------
    model : str
        Always ``"scripted"``.
    responses : list[ChatResponse]
        Responses not returned yet, in order. A response without a ``model``
        reports ``"scripted"``.
    requests : list[ChatRequest]
        Every request received, in order.

    Raises
    ------
    TypeError
        If a response or ``default`` is neither a ``ChatResponse`` nor a ``str``.
    """

    model: str = "scripted"

    def __init__(
        self,
        responses: Iterable[ChatResponse | str] = (),
        *,
        default: ChatResponse | str | None = None,
    ) -> None:
        self.responses: list[ChatResponse] = [_as_response(item) for item in responses]
        self.requests: list[ChatRequest] = []
        self._default = None if default is None else _as_response(default)

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Record the request and return the next scripted response.

        Parameters
        ----------
        request : ChatRequest

        Returns
        -------
        ChatResponse

        Raises
        ------
        ModelProviderError
            If the script ran out and no ``default`` was given.
        """
        self.requests.append(request)
        if self.responses:
            return self.responses.pop(0)
        if self._default is not None:
            return self._default
        raise ModelProviderError(
            f"ScriptedChat has no response left for request {len(self.requests)}"
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Return the next scripted response as stream chunks.

        The chunks are the content word by word, whitespace kept, then the
        refusal, one ``tool_call_delta`` per tool call, the finish reason and
        the usage when set.

        Parameters
        ----------
        request : ChatRequest

        Returns
        -------
        AsyncIterator[ChatStreamChunk]

        Raises
        ------
        ModelProviderError
            If the script ran out and no ``default`` was given.
        ValueError
            If the response sets ``raw`` or has a tool call with
            ``arguments_error``; ``assemble_stream`` cannot rebuild those.
        """
        response = await self.complete(request)
        _check_streamable(response)
        model = response.model
        if response.content is not None:
            for word in _WORDS.findall(response.content) or [response.content]:
                yield ChatStreamChunk(content_delta=word, model=model)
        if response.refusal is not None:
            yield ChatStreamChunk(refusal_delta=response.refusal, model=model)
        for index, call in enumerate(response.tool_calls):
            yield ChatStreamChunk(
                tool_call_delta={
                    "index": index,
                    "id": call.id,
                    "function": {"name": call.name, "arguments": call.arguments_json},
                },
                model=model,
            )
        yield ChatStreamChunk(finish_reason=response.finish_reason, model=model)
        usage = response.usage
        if usage:
            yield ChatStreamChunk(usage=usage, model=model)


__all__ = ["ScriptedChat"]
