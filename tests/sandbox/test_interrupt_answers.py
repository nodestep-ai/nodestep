import json
from typing import Any

import pytest
from pydantic import BaseModel

from nodestep import Interrupt, ScriptedChat, build_react_agent, tool
from nodestep.chat import ChatResponse, HumanMessage, ToolCall
from nodestep.middleware import (
    InterruptRule,
    ToolDecision,
    ToolInterrupt,
    ToolInterruptMiddleware,
)
from nodestep_sandbox.errors import FormError
from nodestep_sandbox.interrupts import PendingInterrupt, ResumeForm
from nodestep_sandbox.runs import Run
from nodestep_sandbox.session import Sandbox


class ToolApproval(BaseModel):
    kind: str = "tool_interrupt"
    tool_name: str
    tool_call_id: str | None = None
    arguments: dict[str, object]


class Opaque:
    def __repr__(self) -> str:
        return "<opaque>"


def pending(
    payload: object, *, key: str = "node:ask", interrupt_id: str = "ask"
) -> PendingInterrupt:
    return PendingInterrupt.from_interrupt(
        Interrupt(
            key=key, id=interrupt_id, node="node", task_id="node", payload=payload
        )
    )


def test_a_tool_interrupt_payload_becomes_an_approval() -> None:
    item = pending(
        {
            "kind": "tool_interrupt",
            "tool_name": "refund",
            "tool_call_id": "c1",
            "arguments": {"amount": 5},
        }
    )
    assert item.approval is not None
    assert (item.approval.tool_name, item.approval.tool_call_id) == ("refund", "c1")
    assert json.loads(item.approval.arguments_json) == {"amount": 5}


def test_a_model_payload_of_the_same_shape_is_an_approval_too() -> None:
    item = pending(ToolApproval(tool_name="refund", arguments={"amount": 5}))
    assert item.approval is not None
    assert item.approval.tool_name == "refund"


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "tool_interrupt", "tool_name": 5, "arguments": {}},
        {"kind": "tool_interrupt", "tool_name": "refund", "arguments": [1]},
        {"question": "Why?"},
        "plain text",
    ],
)
def test_other_payloads_are_plain_interrupts(payload: object) -> None:
    assert pending(payload).approval is None


def test_a_payload_json_cannot_hold_is_shown_with_repr() -> None:
    assert json.loads(pending({"thing": Opaque()}).payload_json) == {
        "thing": "<opaque>"
    }


def test_answers_are_keyed_by_interrupt_key() -> None:
    items = [
        pending({"question": "Title?"}, key="a:title", interrupt_id="title"),
        pending(
            {
                "kind": "tool_interrupt",
                "tool_name": "publish",
                "arguments": {"draft": "x"},
            },
            key="b:approve",
            interrupt_id="approve",
        ),
    ]
    answers = ResumeForm(items).parse(
        {"answer.0": "Hello", "format.0": "text", "decision.1": "approve"}
    )
    assert answers == {"a:title": "Hello", "b:approve": {"action": "approve"}}


def test_a_json_answer_is_parsed() -> None:
    answers = ResumeForm([pending("?")]).parse(
        {"answer.0": '{"ok": true}', "format.0": "json"}
    )
    assert answers == {"node:ask": {"ok": True}}


def test_edit_and_deny_decisions() -> None:
    item = pending(
        {"kind": "tool_interrupt", "tool_name": "refund", "arguments": {"amount": 5}}
    )
    form = ResumeForm([item])
    assert form.parse({"decision.0": "edit", "arguments.0": '{"amount": 2}'}) == {
        "node:ask": {"action": "edit", "arguments": {"amount": 2}}
    }
    assert form.parse({"decision.0": "deny", "message.0": " too much "}) == {
        "node:ask": {"action": "deny", "message": "too much"}
    }
    assert form.parse({"decision.0": "deny", "message.0": ""}) == {
        "node:ask": {"action": "deny"}
    }


@pytest.mark.parametrize(
    ("values", "errors"),
    [
        ({"answer.0": "", "format.0": "text"}, {"answer.0": "Enter an answer"}),
        (
            {"answer.0": "{", "format.0": "json"},
            {
                "answer.0": "Not valid JSON: Expecting property name enclosed in double quotes (line 1, column 2)"
            },
        ),
        ({"answer.0": "x", "format.0": "yaml"}, {"format.0": "Choose text or JSON"}),
    ],
)
def test_bad_plain_answers_are_reported(
    values: dict[str, str], errors: dict[str, str]
) -> None:
    with pytest.raises(FormError) as error:
        ResumeForm([pending("?")]).parse(values)
    assert error.value.errors == errors


@pytest.mark.parametrize(
    ("values", "errors"),
    [
        ({}, {"decision.0": "Choose approve, edit or deny"}),
        (
            {"decision.0": "edit", "arguments.0": "[1]"},
            {"arguments.0": "Edited arguments must be a JSON object"},
        ),
    ],
)
def test_bad_approval_answers_are_reported(
    values: dict[str, str], errors: dict[str, str]
) -> None:
    item = pending({"kind": "tool_interrupt", "tool_name": "refund", "arguments": {}})
    with pytest.raises(FormError) as error:
        ResumeForm([item]).parse(values)
    assert error.value.errors == errors


def test_the_middleware_payload_becomes_an_approval() -> None:
    item = pending(
        ToolInterrupt(tool_name="refund", tool_call_id="c1", arguments={"amount": 5})
    )
    assert item.approval is not None
    assert (item.approval.tool_name, item.approval.tool_call_id) == ("refund", "c1")
    assert json.loads(item.approval.arguments_json) == {"amount": 5}


@pytest.mark.parametrize(
    ("values", "decision"),
    [
        ({"decision.0": "approve"}, ToolDecision(action="approve")),
        (
            {"decision.0": "edit", "arguments.0": '{"amount": 2}'},
            ToolDecision(action="edit", arguments={"amount": 2}),
        ),
        (
            {"decision.0": "deny", "message.0": "too much"},
            ToolDecision(action="deny", message="too much"),
        ),
        ({"decision.0": "deny"}, ToolDecision(action="deny")),
    ],
)
def test_every_approval_answer_is_a_tool_decision(
    values: dict[str, str], decision: ToolDecision
) -> None:
    item = pending(
        ToolInterrupt(tool_name="refund", tool_call_id="c1", arguments={"amount": 5})
    )
    [answer] = ResumeForm([item]).parse(values).values()
    assert ToolDecision.model_validate(answer) == decision


refunded: list[float] = []


@tool
def refund(amount: float) -> float:
    """Refund an amount."""
    refunded.append(amount)
    return amount


async def finish(run: Run) -> Run:
    async for _ in run.follow():
        pass
    return run


@pytest.mark.parametrize(
    ("values", "ran", "reply"),
    [
        ({"decision.0": "approve"}, [5.0], '{"result":5.0}'),
        (
            {"decision.0": "edit", "arguments.0": '{"amount": 2}'},
            [2.0],
            '{"result":2.0}',
        ),
        (
            {"decision.0": "deny", "message.0": "too much"},
            [],
            "Tool 'refund' denied: too much",
        ),
    ],
)
async def test_form_answers_drive_the_tool_interrupt_middleware(
    values: dict[str, str], ran: list[float], reply: str
) -> None:
    refunded.clear()
    chat = ScriptedChat(
        [
            ChatResponse(
                tool_calls=[ToolCall(id="c1", name="refund", arguments={"amount": 5})]
            ),
            "done",
        ]
    )
    graph = build_react_agent(
        chat,
        tools=[refund],
        middleware=[ToolInterruptMiddleware(rules=[InterruptRule(tool="refund")])],
    )
    sandbox = Sandbox(graph)
    run = await finish(sandbox.start({"messages": [HumanMessage(content="refund")]}))
    [item] = run.pending
    assert item.approval is not None
    assert item.approval.tool_name == "refund"

    resumed = await finish(sandbox.resume(run, ResumeForm(run.pending).parse(values)))

    assert resumed.status == "completed"
    assert refunded == ran
    messages: list[dict[str, Any]] = json.loads(resumed.final_state or "")["messages"]
    [tool_message] = [message for message in messages if message["type"] == "tool"]
    assert tool_message["content"] == reply


transfers: list[tuple[str, float]] = []


@tool
def transfer(account: str, amount: float) -> float:
    """Transfer an amount to an account."""
    transfers.append((account, amount))
    return amount


async def test_a_form_edit_keeps_the_arguments_it_leaves_out() -> None:
    transfers.clear()
    chat = ScriptedChat(
        [
            ChatResponse(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="transfer",
                        arguments={"account": "A-1", "amount": 5},
                    )
                ]
            ),
            "done",
        ]
    )
    graph = build_react_agent(
        chat,
        tools=[transfer],
        middleware=[ToolInterruptMiddleware(rules=[InterruptRule(tool="transfer")])],
    )
    sandbox = Sandbox(graph)
    run = await finish(sandbox.start({"messages": [HumanMessage(content="pay")]}))
    values = {"decision.0": "edit", "arguments.0": '{"amount": 2}'}

    resumed = await finish(sandbox.resume(run, ResumeForm(run.pending).parse(values)))

    assert resumed.status == "completed"
    assert transfers == [("A-1", 2.0)]
