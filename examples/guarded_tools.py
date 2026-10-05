from nodestep import (
    InMemoryStateStore,
    Resume,
    ScriptedChat,
    build_react_agent,
    tool,
)
from nodestep.chat import ChatResponse, HumanMessage, ToolCall, ToolMessage
from nodestep.middleware import (
    InterruptRule,
    ToolDecision,
    ToolInterruptMiddleware,
    ToolLimitMiddleware,
    gt,
)


@tool
def refund(order_id: str, amount: float) -> str:
    """Refund an amount on an order."""
    return f"refunded {amount} on {order_id}"


def refund_call(call_id: str, order_id: str, amount: float) -> ChatResponse:
    arguments = {"order_id": order_id, "amount": amount}
    return ChatResponse(
        tool_calls=[ToolCall(id=call_id, name="refund", arguments=arguments)]
    )


chat = ScriptedChat(
    [
        refund_call("call_1", "A-1001", 30.0),
        refund_call("call_2", "A-1002", 250.0),
        refund_call("call_3", "A-1003", 15.0),
    ],
    default="One refund went through; the others need a manager.",
)

graph = build_react_agent(
    chat,
    tools=[refund],
    middleware=[
        ToolLimitMiddleware(per_tool={"refund": 2}),
        ToolInterruptMiddleware(
            rules=[InterruptRule(tool="refund", match=gt("amount", 100))]
        ),
    ],
    state_store=InMemoryStateStore(),
)


def main() -> None:
    result = graph.invoke(
        {"messages": [HumanMessage(content="Refund orders A-1001 to A-1003")]},
        thread_id="refunds",
    )
    while result.status == "interrupted":
        answers = {}
        for key, pending in result.interrupts.items():
            print(
                "approval requested:",
                pending.payload.tool_name,
                pending.payload.arguments,
            )
            answers[key] = ToolDecision(
                action="deny", message="Refunds over 100 need a manager."
            )
        result = graph.invoke(None, thread_id="refunds", resume=Resume(answers=answers))

    for message in result.state.messages:
        if isinstance(message, ToolMessage):
            print(f"{message.tool_call_id}: {message.content}")
    print(result.state.final_text)


if __name__ == "__main__":
    main()
