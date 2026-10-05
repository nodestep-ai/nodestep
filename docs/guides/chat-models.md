# Write a chat integration

A chat integration is a class with the `Chat` protocol; write one to use a model that has no built-in integration.

## When you need it

The built-in [chat integrations](../concepts/agents.md#chat-models) do not cover your model or its API. Any class with the three members of `Chat` works wherever a chat model does. Tests need no integration: `ScriptedChat` plays the model ([Test with ScriptedChat](testing.md)).

## Steps

1. Write a class with a `model` name, an `async def complete(request)` that returns a `ChatResponse`, and a `stream(request)` that yields `ChatStreamChunk`s.
2. Map the `ChatRequest` (`messages`, `tools`, `tool_choice`, `output_schema`) to your API.
3. Map the reply to a `ChatResponse` (`content`, `refusal`, `tool_calls`, `finish_reason`, `usage`, `model`).
4. In `stream()`, yield text as `content_delta` chunks and tool calls as `tool_call_delta` dicts.
5. Pass an instance to `build_react_agent` or `model_node`.

## Complete example

`EchoChat` answers with the last message it received:

```python
import asyncio
from collections.abc import AsyncIterator

from nodestep import build_react_agent, run_agent
from nodestep.chat import ChatRequest, ChatResponse, ChatStreamChunk


class EchoChat:
    model = "echo"

    async def complete(self, request: ChatRequest) -> ChatResponse:
        text = f"You said: {request.messages[-1].content}"
        return ChatResponse(content=text, finish_reason="stop", model=self.model)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        response = await self.complete(request)
        yield ChatStreamChunk(content_delta=response.content, model=self.model)
        yield ChatStreamChunk(finish_reason=response.finish_reason, model=self.model)


agent = build_react_agent(EchoChat(), tools=[])
print(asyncio.run(run_agent(agent, "hello")))
```

```text
You said: hello
```

The agent calls `complete()` once, gets an answer without tool calls, and returns its text. A node built with `stream=True` calls `stream()` instead.

## Streamed tool calls

A streamed tool call arrives in parts. Each part is a `tool_call_delta` dict shaped `{"index": 0, "id": ..., "function": {"name": ..., "arguments": ...}}`, and the argument text may be split across chunks. `assemble_stream` joins the chunks into one `ChatResponse`:

```python
import asyncio
from collections.abc import AsyncIterator

from nodestep.chat import ChatStreamChunk, assemble_stream


async def chunks() -> AsyncIterator[ChatStreamChunk]:
    yield ChatStreamChunk(
        tool_call_delta={
            "index": 0,
            "id": "call_1",
            "function": {"name": "get_weather", "arguments": '{"city": '},
        }
    )
    yield ChatStreamChunk(
        tool_call_delta={"index": 0, "function": {"arguments": '"Oslo"}'}}
    )
    yield ChatStreamChunk(finish_reason="tool_calls")


response = asyncio.run(assemble_stream(chunks()))
call = response.tool_calls[0]
print(call.id, call.name, call.arguments)
print(response.finish_reason)
```

```text
call_1 get_weather {'city': 'Oslo'}
tool_calls
```

## Good to know

| Rule or setting | What happens |
|---|---|
| `model: str` | The model name; middleware sees it as `ctx.model` |
| `async complete(request)` | Returns a `ChatResponse` |
| `stream(request)` | Yields `ChatStreamChunk`s; used by nodes built with `stream=True` |
| Parsing of streamed tool calls | Parsing is lenient in two places: a delta without `"index"` belongs to the first tool call, and empty argument text means no arguments |
| A streamed tool call without a name or id | `assemble_stream` raises `ModelProviderError` |
| Arguments that are not a JSON object | That tool call fails: its `arguments_error` is set, and the tool does not run |
