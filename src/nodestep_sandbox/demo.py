"""A small support-desk agent for trying the sandbox without an API key.

Run it with ``nodestep sandbox nodestep_sandbox.demo:graph``. The graph is a
``build_react_agent`` agent whose model is ``SupportDeskChat``, a stand-in that
answers by fixed rules. Asked for a refund, it proposes an ``issue_refund``
tool call, and ``ToolInterruptMiddleware`` pauses the run until a person
approves, edits or denies the call. Edited arguments that do not fit
``issue_refund`` are refused by the tool's argument check.
"""

import re
from collections.abc import AsyncIterator

from pydantic import BaseModel, ValidationError

from nodestep import ScriptedChat, build_react_agent, tool
from nodestep.chat import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from nodestep.middleware import InterruptRule, ToolInterruptMiddleware

DEFAULT_ORDER_ID = "A-1001"
DEFAULT_AMOUNT = 25.0
_ORDER_ID = re.compile(r"\b[A-Z]-\d+\b")
_AMOUNT = re.compile(r"\d+(?:\.\d+)?")


class RefundReceipt(BaseModel):
    """What ``issue_refund`` returns."""

    order_id: str
    amount: float
    status: str


@tool
def issue_refund(order_id: str, amount: float) -> RefundReceipt:
    """Refund an order; the demo only pretends to."""
    return RefundReceipt(order_id=order_id, amount=round(amount, 2), status="refunded")


class SupportDeskChat:
    """A chat model that answers by fixed rules, so the demo needs no API key.

    A message that mentions a refund gets an ``issue_refund`` call for the order
    id (``X-1234``) and the first other number in it, or ``A-1001`` and 25.0; a
    tool result gets a reply about the refund; anything else gets a fixed
    answer.
    """

    model = "support-desk"

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Answer the last message of the request.

        Parameters
        ----------
        request : ChatRequest

        Returns
        -------
        ChatResponse
        """
        last = request.messages[-1] if request.messages else None
        if isinstance(last, ToolMessage):
            return ChatResponse(content=_outcome(last), model="support-desk")
        text = last.content if isinstance(last, HumanMessage) else None
        if text is None or "refund" not in text.lower():
            return ChatResponse(
                content=(
                    "Thanks for your message. This demo answers with a scripted "
                    "model; ask for a refund to see the approval step."
                ),
                model="support-desk",
            )
        order = _ORDER_ID.search(text)
        order_id = order.group() if order else DEFAULT_ORDER_ID
        amount = _AMOUNT.search(_ORDER_ID.sub("", text))
        return ChatResponse(
            content=f"I can refund order {order_id}.",
            tool_calls=[
                ToolCall(
                    id=f"refund_{order_id}",
                    name=issue_refund.name,
                    arguments={
                        "order_id": order_id,
                        "amount": float(amount.group()) if amount else DEFAULT_AMOUNT,
                    },
                )
            ],
            model="support-desk",
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Stream the answer of ``complete`` in chunks.

        Parameters
        ----------
        request : ChatRequest

        Returns
        -------
        AsyncIterator[ChatStreamChunk]
        """
        answer = ScriptedChat([await self.complete(request)])
        async for chunk in answer.stream(request):
            yield chunk


def _outcome(message: ToolMessage) -> str:
    try:
        receipt = RefundReceipt.model_validate_json(message.content or "")
    except ValidationError:
        return f"The refund was not issued: {message.content}"
    return f"Refunded {receipt.amount:.2f} for order {receipt.order_id}."


graph = build_react_agent(
    SupportDeskChat(),
    tools=[issue_refund],
    name="support_demo",
    system_prompt="You are the support desk of a small shop.",
    middleware=[
        ToolInterruptMiddleware(rules=[InterruptRule(tool=issue_refund.name)]),
    ],
    tool_errors="return",
)
