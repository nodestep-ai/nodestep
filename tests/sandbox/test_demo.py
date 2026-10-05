import json
from pathlib import Path
from typing import Any

import pytest

from nodestep import AgentState
from nodestep.chat import HumanMessage
from nodestep.middleware import ToolInterruptMiddleware
from nodestep_sandbox import demo
from nodestep_sandbox.loader import TargetLoader
from nodestep_sandbox.runs import Run
from nodestep_sandbox.session import Sandbox


async def finish(run: Run) -> Run:
    async for _ in run.follow():
        pass
    return run


def state_of(run: Run) -> dict[str, Any]:
    return json.loads(run.final_state or "")


async def paused_refund(text: str) -> tuple[Sandbox, Run]:
    sandbox = Sandbox(demo.graph)
    run = await finish(sandbox.start({"messages": [HumanMessage(content=text)]}))
    assert run.status == "interrupted"
    return sandbox, run


async def test_the_demo_loads_from_its_documented_target() -> None:
    graph = await TargetLoader(Path.cwd()).graph("nodestep_sandbox.demo:graph")
    assert graph is demo.graph
    assert graph.name == "support_demo"
    assert graph.state_store is None


def test_the_demo_is_a_react_agent_with_a_real_approval_middleware() -> None:
    assert demo.graph.state_schema.schema_type is AgentState
    assert sorted(demo.graph.nodes) == ["act", "think"]
    assert any(
        isinstance(item, ToolInterruptMiddleware) for item in demo.graph.middleware
    )


async def test_a_question_gets_the_scripted_answer() -> None:
    sandbox = Sandbox(demo.graph)
    run = await finish(
        sandbox.start({"messages": [HumanMessage(content="Where is my parcel?")]})
    )
    assert run.status == "completed"
    assert "ask for a refund" in (state_of(run)["final_text"] or "")


async def test_a_refund_waits_for_the_middleware_approval() -> None:
    _, run = await paused_refund("Please refund 40 for order B-2002")
    [pending] = run.pending
    assert pending.id == "approve_refund_B-2002"
    assert pending.node == "act"
    assert pending.approval is not None
    assert pending.approval.tool_name == "issue_refund"
    assert pending.approval.tool_call_id == "refund_B-2002"
    assert json.loads(pending.approval.arguments_json) == {
        "order_id": "B-2002",
        "amount": 40.0,
    }


async def test_a_refund_without_details_uses_the_default_order() -> None:
    _, run = await paused_refund("I want a refund")
    [pending] = run.pending
    assert pending.approval is not None
    assert json.loads(pending.approval.arguments_json) == {
        "order_id": "A-1001",
        "amount": 25.0,
    }


async def test_an_approved_refund_is_issued() -> None:
    sandbox, run = await paused_refund("I want a refund")
    [pending] = run.pending
    resumed = await finish(sandbox.resume(run, {pending.key: {"action": "approve"}}))
    assert resumed.status == "completed"
    state = state_of(resumed)
    assert state["final_text"] == "Refunded 25.00 for order A-1001."
    tool_message = state["messages"][-2]
    assert tool_message["type"] == "tool"
    assert json.loads(tool_message["content"])["status"] == "refunded"


async def test_an_edit_changes_the_amount() -> None:
    sandbox, run = await paused_refund("I want a refund of 40")
    [pending] = run.pending
    resumed = await finish(
        sandbox.resume(
            run, {pending.key: {"action": "edit", "arguments": {"amount": 12.5}}}
        )
    )
    state = state_of(resumed)
    assert state["final_text"] == "Refunded 12.50 for order A-1001."
    assert state["messages"][-2]["arguments"] == {"order_id": "A-1001", "amount": 12.5}


async def test_a_denied_refund_is_not_issued() -> None:
    sandbox, run = await paused_refund("refund please")
    [pending] = run.pending
    resumed = await finish(
        sandbox.resume(
            run, {pending.key: {"action": "deny", "message": "not eligible"}}
        )
    )
    state = state_of(resumed)
    assert (
        state["messages"][-2]["content"] == "Tool 'issue_refund' denied: not eligible"
    )
    assert state["final_text"] == (
        "The refund was not issued: Tool 'issue_refund' denied: not eligible"
    )


@pytest.mark.parametrize(
    ("arguments", "problem"),
    [
        ({"amount": "lots"}, "amount\n  Input should be a valid number"),
        ({"amont": 5}, "amont\n  Extra inputs are not permitted"),
        ({"order_id": None}, "order_id\n  Input should be a valid string"),
    ],
)
async def test_an_edit_that_does_not_fit_the_tool_is_refused(
    arguments: dict[str, Any], problem: str
) -> None:
    sandbox, run = await paused_refund("refund please")
    [pending] = run.pending
    resumed = await finish(
        sandbox.resume(run, {pending.key: {"action": "edit", "arguments": arguments}})
    )
    assert resumed.status == "completed"
    state = state_of(resumed)
    content = state["messages"][-2]["content"]
    assert content.startswith("Tool error: ValidationError: ")
    assert problem in content
    assert state["final_text"].startswith("The refund was not issued: Tool error")
