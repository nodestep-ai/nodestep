from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from nodestep.chat import ChatRequest, ChatResponse, ChatStreamChunk
from nodestep.chat.integrations.openai.mapping import (
    chunks_from_stream_event,
    request_to_parameters,
    response_from_completion,
)
from nodestep.chat.integrations.openai.settings import OpenAISettings
from nodestep.exceptions import IntegrationNotInstalledError, ModelProviderError

if TYPE_CHECKING:
    from openai import AsyncOpenAI


_MAPPING_PARAMETERS = frozenset(
    {
        "messages",
        "tools",
        "tool_choice",
        "response_format",
        "stream",
        "stream_options",
        "model",
    }
)
_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_TIMEOUT = 60.0
_DEFAULT_MAX_RETRIES = 3


class OpenAIChat:
    """Chat model for the OpenAI Chat Completions API, or a compatible one.

    The constructor reads no environment variable; ``from_env`` does. The
    OpenAI SDK reads some variables itself; the
    [module introduction][nodestep.chat.integrations.openai] lists them.
    ``model`` is always required:

    ```python
    from nodestep import OpenAIChat

    chat = OpenAIChat(
        model="gpt-6-luna",
        api_key="sk-your-key",
        timeout=30,
        max_retries=2,
        temperature=0.2,
    )
    print(chat.model, chat.request_parameters)
    ```

    ```text
    gpt-6-luna {'temperature': 0.2}
    ```

    - Extra keyword arguments, such as ``temperature``, are sent with every
      request.
    - Keys the request mapping sets itself (``messages``, ``tools``,
      ``tool_choice``, ``response_format``, ``stream``, ``stream_options``,
      ``model``) raise ``TypeError``; use ``model_node(tool_choice=...)`` and
      ``output_schema=``.
    - ``stream_usage=True`` (the default) asks for token usage at the end of
      streamed responses.

    Pass your own ``AsyncOpenAI`` client as ``client=``. The connection
    arguments (``api_key``, ``base_url``, ``organization``, ``project``,
    ``timeout``, ``max_retries``) then raise ``TypeError``, so set them on the
    client:

    ```python
    from openai import AsyncOpenAI

    from nodestep import OpenAIChat

    client = AsyncOpenAI(api_key="sk-your-key", base_url="http://localhost:8000/v1")
    chat = OpenAIChat(model="gpt-6-luna", client=client)
    print(chat.model)
    ```

    ```text
    gpt-6-luna
    ```

    ``await chat.aclose()``, or ``async with OpenAIChat(...) as chat:``, closes
    the HTTP client.

    Parameters
    ----------
    model : str
        Model name, e.g. ``"gpt-6-luna"``.
    api_key : str, optional
        API key; required unless ``client`` is given.
    base_url : str, optional
        Base URL of an OpenAI-compatible API; ``None`` means the OpenAI API.
    organization : str, optional
        OpenAI organization id, sent only when given.
    project : str, optional
        OpenAI project id, sent only when given.
    timeout : float, optional
        HTTP timeout in seconds; ``None`` means 60.
    max_retries : int, optional
        Retries of transient HTTP errors, done by the SDK; ``None`` means 3.
    client : AsyncOpenAI, optional
        Client to use instead of the connection arguments.
    stream_usage : bool
        Ask for token usage at the end of streamed responses.
    **request_parameters
        Sent with every request, such as ``temperature``.

    Raises
    ------
    IntegrationNotInstalledError
        If the ``openai`` extra is not installed.
    ModelProviderError
        If ``client`` is not given and ``api_key`` is missing or empty.
    TypeError
        If ``client`` is combined with connection arguments, or
        ``request_parameters`` sets a key the request mapping sets:
        ``messages``, ``tools``, ``tool_choice``, ``response_format``,
        ``stream``, ``stream_options`` or ``model``.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        client: AsyncOpenAI | None = None,
        stream_usage: bool = True,
        **request_parameters: Any,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ModuleNotFoundError as error:
            raise IntegrationNotInstalledError("openai", "openai") from error
        owned = sorted(_MAPPING_PARAMETERS & request_parameters.keys())
        if owned:
            raise TypeError(
                f"OpenAIChat does not accept {', '.join(owned)} as request "
                "parameters; the request mapping sets them (use "
                "ChatRequest.tool_choice and ChatRequest.output_schema)"
            )
        self.model = model
        self.stream_usage = stream_usage
        self.request_parameters = request_parameters
        if client is not None:
            connection = {
                "api_key": api_key is not None,
                "base_url": base_url is not None,
                "organization": organization is not None,
                "project": project is not None,
                "timeout": timeout is not None,
                "max_retries": max_retries is not None,
            }
            given = [name for name, is_given in connection.items() if is_given]
            if given:
                raise TypeError(
                    f"OpenAIChat got client= together with {', '.join(given)}; "
                    "configure the client instead"
                )
            self._client = client
            return
        if not api_key:
            raise ModelProviderError(
                "OpenAIChat needs api_key= or client=; use "
                "OpenAIChat.from_env(model=...) to read OPENAI_API_KEY"
            )
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=_DEFAULT_BASE_URL if base_url is None else base_url,
            organization=organization,
            project=project,
            timeout=_DEFAULT_TIMEOUT if timeout is None else timeout,
            max_retries=_DEFAULT_MAX_RETRIES if max_retries is None else max_retries,
        )
        self._client.organization = organization
        self._client.project = project

    @classmethod
    def from_env(cls, model: str, **kwargs: Any) -> OpenAIChat:
        """Create a chat model whose connection comes from ``OPENAI_*`` variables.

        Reads the variables in the table of the
        [module introduction][nodestep.chat.integrations.openai]; empty ones
        count as unset and keyword arguments win.

        Parameters
        ----------
        model : str
            Model name, e.g. ``"gpt-6-luna"``.
        **kwargs
            Passed to ``OpenAIChat``.

        Returns
        -------
        OpenAIChat

        Raises
        ------
        TypeError
            If ``client`` is passed; give it to the constructor instead.
        ModelProviderError
            If ``OPENAI_API_KEY`` is unset or empty and no ``api_key`` is passed.

        Examples
        --------
        >>> chat = OpenAIChat.from_env(model="gpt-6-luna")  # doctest: +SKIP
        """
        if "client" in kwargs:
            raise TypeError(
                "OpenAIChat.from_env() does not take client=; pass the client to "
                "OpenAIChat(model, client=...)"
            )
        arguments = {**OpenAISettings().model_dump(), **kwargs}
        if not arguments["api_key"]:
            raise ModelProviderError(
                "OpenAIChat.from_env found no API key: OPENAI_API_KEY is not set "
                "or is empty; set it, or pass api_key="
            )
        return cls(model, **arguments)

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    async def __aenter__(self) -> OpenAIChat:
        return self

    async def __aexit__(self, *error_info: object) -> None:
        await self.aclose()

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Send a request and wait for the full response.

        Parameters
        ----------
        request : ChatRequest
            Messages, tools and an optional output schema.

        Returns
        -------
        ChatResponse

        Raises
        ------
        ModelProviderError
            If the OpenAI SDK reports an error.
        """
        from openai import OpenAIError

        try:
            completion = await self._client.chat.completions.create(
                model=self.model,
                **{**self.request_parameters, **request_to_parameters(request)},
            )
        except OpenAIError as error:
            raise ModelProviderError(str(error)) from error
        return response_from_completion(completion)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Stream a response, with the token usage when ``stream_usage`` is set.

        Parameters
        ----------
        request : ChatRequest
            Messages, tools and an optional output schema.

        Returns
        -------
        AsyncIterator[ChatStreamChunk]

        Raises
        ------
        ModelProviderError
            If the OpenAI SDK reports an error.
        """
        from openai import OpenAIError

        parameters: dict[str, Any] = {
            **self.request_parameters,
            **request_to_parameters(request),
        }
        if self.stream_usage:
            parameters["stream_options"] = {"include_usage": True}
        try:
            events = await self._client.chat.completions.create(
                model=self.model, stream=True, **parameters
            )
            async for event in events:
                for chunk in chunks_from_stream_event(event):
                    yield chunk
        except OpenAIError as error:
            raise ModelProviderError(str(error)) from error


__all__ = ["OpenAIChat"]
