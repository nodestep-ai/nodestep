# Pause for approval

`ToolInterruptMiddleware` stops an agent before a matching tool call runs, and the run continues when a person approves, edits or denies the call.

## When you need it

Use an approval when a tool has effects that a person must check first, such as refunds, emails or deletions.

## Steps

1. Give the agent a state store, such as `state_store=InMemoryStateStore()`, because a paused run is stored.
2. Add `ToolInterruptMiddleware(rules=[InterruptRule(tool="refund", match=gt("amount", 100))])` to its `middleware=`.
3. Run it with a `thread_id`. The result's status is `"interrupted"`.
4. Read the pending interrupt. Its payload is a `ToolInterrupt` with `tool_name` and `arguments`.
5. Resume on the same thread with `resume=Resume(answers={key: decision})`.

The decision is one of these:

- `True` or `ToolDecision(action="approve")` runs the call.
- `ToolDecision(action="edit", arguments={...})` runs it with changed arguments.
- `ToolDecision(action="deny", message=...)` refuses it, and the model gets the message.

## Complete example

The scripted model asks for a refund of 120.0. The rule pauses the call, and the person edits the amount to 100.0:

```python
from nodestep import InMemoryStateStore, Resume, ScriptedChat, build_react_agent, tool
from nodestep.chat import ChatResponse, HumanMessage, ToolCall
from nodestep.middleware import InterruptRule, ToolDecision, ToolInterruptMiddleware, gt


@tool
def refund(order_id: str, amount: float) -> str:
    """Refund an amount on an order."""
    return f"refunded {amount} on {order_id}"


chat = ScriptedChat(
    [
        ChatResponse(
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="refund",
                    arguments={"order_id": "A-1001", "amount": 120.0},
                )
            ]
        ),
        "The refund is done.",
    ]
)
approvals = ToolInterruptMiddleware(
    rules=[InterruptRule(tool="refund", match=gt("amount", 100))]
)
agent = build_react_agent(
    chat,
    tools=[refund],
    middleware=[approvals],
    state_store=InMemoryStateStore(),
)

paused = agent.invoke(
    {"messages": [HumanMessage(content="Refund order A-1001")]},
    thread_id="support-1",
)
print(paused.status)
key, pending = next(iter(paused.interrupts.items()))
print(key, pending.payload.tool_name, pending.payload.arguments)

decision = ToolDecision(action="edit", arguments={"amount": 100.0})
done = agent.invoke(None, thread_id="support-1", resume=Resume(answers={key: decision}))
print(done.status)
print(done.state.messages[2].content)
print(done.state.final_text)
```

```text
interrupted
act:approve_call_1 refund {'order_id': 'A-1001', 'amount': 120.0}
completed
{"result":"refunded 100.0 on A-1001"}
The refund is done.
```

The key is the `act` task and the interrupt id, `"approve_"` plus the tool call id. The third message is the tool result, so the refund ran with the edited amount.

## Good to know

- An edit is merged into the call's arguments, so it cannot remove an argument ([Tool approvals](../concepts/interrupts.md#tool-approvals)).
- A denied call is not run, and the model gets a tool message with the reason ([Tool approvals](../concepts/interrupts.md#tool-approvals)).
- A predicate that raises, for example on a missing argument, counts as a match, so the call is asked for ([Tool approvals](../concepts/interrupts.md#tool-approvals)).
- To pause inside your own node, call `interrupt()`; see [step 7 of the Quickstart](../quickstart.md#7-pause-for-a-person) and [Interrupts](../concepts/interrupts.md).
- In tests, use `InMemoryStateStore` and `ScriptedChat`; see [Test with ScriptedChat](testing.md#test-interrupts-and-approvals).
