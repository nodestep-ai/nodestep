# Agents and tools

An agent is a graph in which one node calls a chat model and another runs the tools the model asks for, until the model answers without a tool call.

```python
import asyncio

from nodestep import ScriptedChat, build_react_agent, run_agent, tool
from nodestep.chat import ChatResponse, ToolCall


@tool
def get_order_status(order_id: str) -> str:
    """Look up the shipping status of an order."""
    return "shipped" if order_id == "A-1001" else "unknown"


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
print(answer)
print(len(chat.requests), "model calls")
print(chat.requests[-1].messages[-1].content)
```

```text
Order A-1001 has shipped.
2 model calls
{"result":"shipped"}
```

`ScriptedChat` plays the model here. Its first response asks for the tool, and the agent runs it. The result goes back to the model as the last message of the second request, and the second response is the answer.

## Chat models

A chat model is any object with the `Chat` protocol: a `model` name, `complete(request)`, which returns a `ChatResponse`, and `stream(request)`, which yields `ChatStreamChunk`s. Middleware sees the model name as `ctx.model`.

nodestep has these chat integrations:

| Chat integration | Class | Status | Reference |
|---|---|---|---|
| OpenAI | `OpenAIChat` | Built in, in the `openai` extra | [nodestep.chat.integrations.openai](../reference/chat/integrations/openai.md) |
| Scripted model for tests | `ScriptedChat` | Built in | [nodestep.chat.integrations](../reference/chat/integrations/index.md) |
| Anthropic (Claude) | | Planned | |
| Ollama | | Planned | |

- `ScriptedChat` returns your responses in order and records every request, with no network; see [Test with ScriptedChat](../guides/testing.md).
- For any other model, write a small class; see [Write a chat integration](../guides/chat-models.md).

Messages are pydantic models from `nodestep.chat`: `SystemMessage`, `HumanMessage`, `AIMessage` and `ToolMessage`. An `AIMessage` also carries `tool_calls`, `refusal`, `finish_reason` and `usage`. A state field holds the messages as `Annotated[list[Message], add_messages]`.

## Tools

`@tool` turns a typed function into a tool. The model reads its name, its docstring and an argument schema built from the type hints:

```python
import asyncio
from dataclasses import dataclass

from nodestep import ScriptedChat, ToolContext, build_react_agent, run_agent, tool
from nodestep.chat import ChatResponse, ToolCall


@dataclass
class Inventory:
    stock: dict[str, int]


@tool
def check_stock(sku: str, ctx: ToolContext) -> int:
    """Return how many items of a product are in stock."""
    inventory: Inventory = ctx.context
    return inventory.stock.get(sku, 0)


chat = ScriptedChat(
    [
        ChatResponse(
            tool_calls=[
                ToolCall(id="call_1", name="check_stock", arguments={"sku": "MUG-1"})
            ]
        ),
        "There are 12 mugs in stock.",
    ]
)
agent = build_react_agent(chat, tools=[check_stock])

print(check_stock.input_schema())
answer = asyncio.run(
    run_agent(agent, "How many mugs are left?", context=Inventory(stock={"MUG-1": 12}))
)
print(answer)
print(chat.requests[1].messages[-1].content)
```

```text
{'additionalProperties': False, 'properties': {'sku': {'title': 'Sku', 'type': 'string'}}, 'required': ['sku'], 'title': 'CheckStockInput', 'type': 'object'}
There are 12 mugs in stock.
{"result":12}
```

The schema has only `sku`. A parameter annotated `ToolContext` is hidden from the model and gives the tool the run's `context=`, here the inventory.

To run a tool outside an agent, use `call_tool`.

## The ready-made agent

`build_react_agent(chat, tools=[...])` returns a graph with two nodes. `think` calls the model, and `act` runs the tools it asks for. They alternate until the model answers without tool calls.

The state is `AgentState`, with `messages`, `tool_calls`, `final_text` and `final_output`. `run_agent(graph, message)` sends one user message and returns the final text.

| Argument | Default | Meaning |
|---|---|---|
| `tools` | required | The tools the model may call; may be empty |
| `system_prompt` | `None` | System message sent before the history on every call |
| `middleware` | `()` | Extra middleware; their tools are added to the agent's tools |
| `tool_errors` | `"raise"` | What a failing tool does, see below |
| `stream` | `False` | Stream model calls as `"tokens"` events |
| `output_schema` | `None` | A pydantic model the final answer is parsed into |
| `max_steps` | `25` | Superstep limit per call |
| `max_tool_calls` | `50` | Tool-call budget for one run and its resumes, kept by a `ToolLimitMiddleware` placed before your middleware. Calls over it are denied and the model is told |
| `workspace`, `state_store` | `None` | As for `Graph` |

## Tool errors

A failing tool fails the run by default. `tool_errors` picks what happens instead:

- `"raise"`: the error propagates.
- `"return"`: the model gets the error as the tool result and can try again.
- a function `(error) -> str`: the model gets the text it returns.

## Structured output

With `output_schema=`, the final answer is parsed into that model and stored in `final_output`:

```python
import asyncio

from pydantic import BaseModel

from nodestep import ScriptedChat, build_react_agent
from nodestep.chat import HumanMessage


class Ticket(BaseModel):
    title: str
    priority: int


agent = build_react_agent(
    ScriptedChat(['{"title": "Checkout is down", "priority": 1}']),
    tools=[],
    output_schema=Ticket,
)

result = asyncio.run(
    agent.ainvoke({"messages": [HumanMessage(content="Checkout is down!")]})
)
print(result.state.final_output)
```

```text
title='Checkout is down' priority=1
```

## Your own agent graph

`model_node(name, chat=..., tools=[...])` and `tool_runner(name, tools=[...])` are the two nodes of an agent. Use them to add your own nodes, routes or state fields:

```python
import asyncio
from typing import Annotated

from pydantic import Field

from nodestep import (
    END,
    START,
    BaseState,
    Graph,
    ScriptedChat,
    add_messages,
    model_node,
    tool,
    tool_runner,
    when,
)
from nodestep.chat import ChatResponse, HumanMessage, Message, ToolCall


class Support(BaseState):
    messages: Annotated[list[Message], add_messages] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_text: str | None = None


@tool
def lookup_order(order_id: str) -> str:
    """Find an order by id."""
    raise LookupError(f"no order {order_id}")


chat = ScriptedChat(
    [
        ChatResponse(
            tool_calls=[
                ToolCall(
                    id="call_1", name="lookup_order", arguments={"order_id": "B-7"}
                )
            ]
        ),
        "I could not find order B-7. Please check the number.",
    ]
)
think = model_node("think", chat=chat, tools=[lookup_order])
act = tool_runner("act", tools=[lookup_order], tool_errors="return")


def has_tool_calls(state: Support) -> bool:
    return bool(state.tool_calls)


graph = Graph(Support).flow(
    START >> think,
    think >> when(has_tool_calls, act, otherwise=END),
    act >> think,
)

result = asyncio.run(
    graph.ainvoke({"messages": [HumanMessage(content="Where is B-7?")]})
)
print(result.state.messages[2].content)
print(result.state.final_text)
```

```text
Tool error: LookupError: no order B-7
I could not find order B-7. Please check the number.
```

- `model_node` sends the `messages` field to the model and appends its `AIMessage`. It writes the requested calls to `tool_calls` and the text to `final_text`.
- `tool_runner` runs the calls in `tool_calls`, appends their `ToolMessage` results and clears `tool_calls`.
- The route goes back to `think` until the model stops asking for tools.

## After a stopped tool turn

A run can stop between the model's tool request and the tool results: a tool failed, a limit or a timeout stopped the run, or the caller closed the stream. The thread then holds an `AIMessage` whose tool calls have no results. New input on that thread makes the next model call fail with `ChatHistoryError`.

Continue such a thread with `invoke(None, thread_id=...)`, which runs the unfinished tool step. Or `fork()` it at the event before the model's request, and send the new input on the branch; see [Time travel](persistence.md#time-travel).

## Rules and defaults

| Rule or setting | What happens |
|---|---|
| `Chat` protocol | A `model` name, `complete(request)` returning a `ChatResponse`, and `stream(request)` yielding `ChatStreamChunk`s. Middleware sees the name as `ctx.model` |
| Tool name | The function's `__name__`, or `name=`. It must match `^[a-zA-Z0-9_-]{1,64}$` |
| Tool description | The function's own docstring becomes the description, all of it, and is sent to the model. `description=` overrides it. A tool without a description is refused |
| Tool arguments | A pydantic model built from the type hints, which rejects unknown arguments. Every parameter needs an annotation. `input_model=` uses your own model instead |
| `ToolContext` parameter | Hidden from the model. `ctx.context` is the run's `context=`, `ctx.state` the state the node received, and `ctx.workspace` the graph's workspace |
| Tool result | Validated against the return annotation and sent to the model as JSON, wrapped as `{"result": ...}` |
| Plain `def` tools | Run in a worker thread |
| Tool calls of one model turn | They run concurrently |
| `call_tool(tool, arguments, ctx)` | Runs a tool outside an agent, with validation and the tool middleware, and returns the result and the arguments the tool ran with |
| `run_agent(graph, message, **kwargs)` | Sends one user message and returns the final text. The keyword arguments go to `ainvoke`. A run that pauses for an interrupt raises `RunInterruptedError` |
| `build_react_agent(middleware=...)` | Adds the tools of the middleware passed to it to the agent's tools |
| `tool_errors="raise"` (default) | The error propagates and fails the run; several failing calls raise `ToolGroupExecutionError` |
| `tool_errors="return"` | The model gets `"Tool error: <type>: <message>"` as the tool result and can try again |
| `tool_errors=` a function | The model gets the text that the function `(error) -> str` returns |
| Unknown tool or invalid arguments | A failure, raised as `ToolExecutionError`. Arguments that are not a JSON object count as invalid |
| Denied call | A call denied by an approval or a tool limit always reaches the model as a tool message |
| `ContextNotProvidedError`, `GraphConfigError` | Raised inside a tool, they always propagate, whatever `tool_errors` says |
| `output_schema=` | The final answer is parsed into the model and stored in `final_output` |
| Invalid structured answer | An answer that does not validate raises `StructuredOutputError`. With `model_node(..., on_invalid="emit")` the node emits a `"custom"` event with a `StructuredOutputEvent` and stores `None` instead |
| `model_node` writes | It sends the `messages` field and appends the model's `AIMessage`. It writes the requested calls to `tool_calls` and the text to `final_text`, which is `None` when the model calls tools or refuses |
| `model_node` field names | `messages_field=`, `tool_calls_field=` and the others rename the fields |
| `model_node` checks | Before the first model call, the node checks that the messages field uses `add_messages` and that every field it writes is declared. Otherwise it raises `GraphConfigError` |
| A node's `tools=` | It is the node's complete tool list. Middleware does not add tools to it, so list them yourself, as in `tools=[*todo.tools(), lookup_order]`. Only `build_react_agent` adds the tools of the middleware passed to it |
| `model_node(tool_choice=...)` | The request's tool choice: `"auto"`, `"required"`, `"none"` or a tool name |
| `model_node(stream=True)` | Streams the model call as `"tokens"` events |
| Consistent history | Every tool call must be answered by a tool message right after it. Otherwise the node raises `ChatHistoryError` without calling the model |
| `repair_history=True` | The node sends a repaired copy instead: results without a call are dropped, and calls without a result get `"Tool call was not executed."`. The state is not changed |
| New input after a stopped tool turn | It appends a user message after the tool calls that have no results, and the next model call raises `ChatHistoryError`. Continue with `invoke(None, thread_id=...)`, or fork the thread; `repair_history=True` is the opt-in alternative |
