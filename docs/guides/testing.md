# Test with ScriptedChat

`ScriptedChat` plays the chat model in tests: it returns the responses you script, in order, and records every request.

## When you need it

Use it for unit tests of graphs and agents that call a model. The tests need no network and no API key.

## Steps

1. Build a `ScriptedChat` with the model's responses in order. Add `default=` to answer any request after them.
2. Build the agent or graph with it.
3. Run it.
4. Assert on the answer, on `chat.requests`, and on `chat.responses == []`.

## Complete example

Script the tool call and the answer, run the agent, then check the answer, the tool calls and the requests:

```python
import asyncio

from nodestep import ScriptedChat, build_react_agent, run_agent, tool
from nodestep.chat import ChatResponse, ToolCall, ToolMessage

looked_up: list[str] = []


@tool
def get_order_status(order_id: str) -> str:
    """Look up the shipping status of an order."""
    looked_up.append(order_id)
    return "shipped"


chat = ScriptedChat(
    [
        ChatResponse(
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="get_order_status",
                    arguments={"order_id": "A-1001"},
                )
            ]
        ),
        "Order A-1001 has shipped.",
    ]
)
agent = build_react_agent(
    chat,
    tools=[get_order_status],
    system_prompt="You answer questions about orders.",
)

answer = asyncio.run(run_agent(agent, "Where is order A-1001?"))

assert answer == "Order A-1001 has shipped."
assert looked_up == ["A-1001"]
assert chat.responses == []
first, second = chat.requests
assert first.messages[0].content == "You answer questions about orders."
assert [definition.name for definition in first.tools] == ["get_order_status"]
tool_result = second.messages[-1]
assert isinstance(tool_result, ToolMessage)
assert tool_result.content == '{"result":"shipped"}'
print("passed")
```

```text
passed
```

In pytest, the same lines go into a test function. With `pytest-asyncio`, write an `async def` test and `await run_agent(...)`.

## Test interrupts and approvals

Interrupts need a state store; in tests, `InMemoryStateStore` keeps the threads in memory. [Pause for approval](approval.md) shows the steps. This test denies one call and approves another:

```python
import asyncio

from nodestep import (
    Graph,
    InMemoryStateStore,
    Resume,
    ScriptedChat,
    build_react_agent,
    tool,
)
from nodestep.chat import ChatResponse, HumanMessage, ToolCall
from nodestep.middleware import InterruptRule, ToolInterrupt, ToolInterruptMiddleware

sent: list[str] = []


@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    sent.append(to)
    return "sent"


def make_agent() -> Graph:
    chat = ScriptedChat(
        [
            ChatResponse(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="send_email",
                        arguments={"to": "ada@example.com", "body": "Hi"},
                    )
                ]
            ),
            "Done.",
        ]
    )
    return build_react_agent(
        chat,
        tools=[send_email],
        middleware=[ToolInterruptMiddleware(rules=[InterruptRule(tool="send_email")])],
        state_store=InMemoryStateStore(),
    )


async def main() -> None:
    agent = make_agent()
    message = {"messages": [HumanMessage(content="Email Ada")]}

    paused = await agent.ainvoke(message, thread_id="t1")
    assert paused.status == "interrupted"
    (pending,) = paused.interrupts.values()
    assert isinstance(pending.payload, ToolInterrupt)
    assert pending.payload.arguments == {"to": "ada@example.com", "body": "Hi"}
    assert sent == []

    denied = await agent.ainvoke(None, thread_id="t1", resume=Resume(False))
    assert denied.status == "completed"
    assert sent == []
    assert "denied" in denied.state.messages[2].content

    agent = make_agent()
    await agent.ainvoke(message, thread_id="t2")
    await agent.ainvoke(None, thread_id="t2", resume=Resume(True))
    assert sent == ["ada@example.com"]
    print("passed")


asyncio.run(main())
```

```text
passed
```

## Good to know

- A `str` item means `ChatResponse(content=...)`; a `ChatResponse` can carry `tool_calls`, `refusal`, `finish_reason`, `usage` and `model`, as in `ChatResponse(content="ok", usage={"input_tokens": 10, "output_tokens": 2})`.
- Each request takes the next response, in order. When the script runs out, `default=` answers every further request.
- Without `default=`, the next request raises `ModelProviderError("ScriptedChat has no response left for request N")`, so an unexpected model call fails the test.
- `chat.requests` records every `ChatRequest`: the messages, the tools offered, `tool_choice` and the output schema. Assert on it to check prompts, offered tools and the history the model saw.
- `chat.responses` holds the responses not used yet; an empty list means the script was used up exactly.
- `stream()` yields the response in word chunks, then the tool calls, the finish reason and the usage, so a graph with `stream=True` is tested the same way.
- Build a new `ScriptedChat` and graph for every test, so scripts and stores do not leak between tests.
- Pass fakes through `context=` instead of patching: a fake database or client that nodes and tools read from `ctx.context`.
- A graph without a model needs no `ScriptedChat`. Run it through the graph, so the flow, the validation and the reducers are tested too. Assert on the `"updates"` events, which show what each node returned; see [Streaming](../concepts/streaming.md).
