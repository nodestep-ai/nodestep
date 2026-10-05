from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated
from uuid import UUID

import pytest
from pydantic import BaseModel, Field

from nodestep import (
    END,
    START,
    Graph,
    InMemoryStateStore,
    Resume,
    ScriptedChat,
    add,
    build_react_agent,
    interrupt,
    node,
    tool,
)
from nodestep.chat.messages import ChatResponse, HumanMessage, ToolCall, ToolMessage
from nodestep.exceptions import UnknownThreadError
from nodestep.middleware import InterruptRule, ToolInterruptMiddleware


class Request(BaseModel):
    query: str
    log: Annotated[list[str], add] = Field(default_factory=list)


@node
def echo(state: Request) -> dict:
    return {"log": [state.query]}


class Counter(BaseModel):
    count: int = 0
    note: str = ""
    log: Annotated[list[str], add] = Field(default_factory=list)


@node
def bump(state: Counter) -> dict:
    return {"count": state.count + 1}


@node
def set_note(state: Counter) -> dict:
    return {"note": "kept"}


@node
def ask(state: Counter) -> dict:
    answer = interrupt("ok?", id="ok")
    return {"log": [f"{answer}:{state.note}:{state.count}"]}


async def test_required_field_state_runs_with_a_store() -> None:
    graph = Graph(Request, state_store=InMemoryStateStore()).flow(
        START >> echo, echo >> END
    )

    result = await graph.ainvoke({"query": "hi"}, thread_id="t")

    assert result.data["log"] == ["hi"]


async def test_second_run_on_thread_keeps_replace_fields() -> None:
    graph = Graph(Counter, state_store=InMemoryStateStore()).flow(
        START >> bump, bump >> END
    )

    await graph.ainvoke({}, thread_id="t")
    result = await graph.ainvoke({}, thread_id="t")

    assert result.data["count"] == 2


async def test_explicit_input_still_overrides_on_existing_thread() -> None:
    graph = Graph(Counter, state_store=InMemoryStateStore()).flow(
        START >> bump, bump >> END
    )

    await graph.ainvoke({}, thread_id="t")
    result = await graph.ainvoke({"count": 10}, thread_id="t")

    assert result.data["count"] == 11


async def test_model_instance_input_writes_every_field() -> None:
    graph = Graph(Counter, state_store=InMemoryStateStore()).flow(
        START >> bump, bump >> END
    )

    await graph.ainvoke({"note": "first"}, thread_id="t")
    result = await graph.ainvoke(Counter(count=5), thread_id="t")

    assert result.data["count"] == 6
    assert result.data["note"] == ""


async def test_resume_keeps_fields_set_before_interrupt() -> None:
    graph = Graph(Counter, state_store=InMemoryStateStore()).flow(
        START >> set_note, set_note >> ask, ask >> END
    )

    await graph.ainvoke({"count": 5}, thread_id="h")
    result = await graph.ainvoke(resume=Resume("yes"), thread_id="h")

    assert result.data["log"] == ["yes:kept:5"]


async def test_load_of_unknown_thread_raises() -> None:
    graph = Graph(Request, state_store=InMemoryStateStore()).flow(
        START >> echo, echo >> END
    )

    with pytest.raises(UnknownThreadError):
        await graph.load("missing")


async def test_get_state_of_unknown_thread_raises() -> None:
    graph = Graph(Request, state_store=InMemoryStateStore()).flow(
        START >> echo, echo >> END
    )

    with pytest.raises(UnknownThreadError):
        await graph.get_state("missing")


executed: list[str] = []


@tool
def deploy(target: str) -> str:
    """Deploy."""
    executed.append(target)
    return f"deployed {target}"


async def test_react_agent_runs_approved_tool_after_resume() -> None:
    executed.clear()
    chat = ScriptedChat(
        [
            ChatResponse(
                tool_calls=[
                    ToolCall(id="c1", name="deploy", arguments={"target": "prod"})
                ]
            ),
            ChatResponse(content="Deployed."),
        ]
    )
    agent = build_react_agent(
        chat,
        tools=[deploy],
        middleware=[ToolInterruptMiddleware(rules=[InterruptRule(tool="deploy")])],
        state_store=InMemoryStateStore(),
    )

    paused = await agent.ainvoke(
        {"messages": [HumanMessage(content="ship it")]}, thread_id="a"
    )
    assert paused.status == "interrupted"
    result = await agent.ainvoke(resume=Resume(True), thread_id="a")

    assert executed == ["prod"]
    assert result.data["final_text"] == "Deployed."
    tool_messages = [
        message
        for message in result.data["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert [tool_message.tool_call_id for tool_message in tool_messages] == ["c1"]


class Typed(BaseModel):
    when: datetime
    ident: UUID
    amount: Decimal
    where: Path
    log: Annotated[list[str], add] = Field(default_factory=list)


@node
def touch(state: Typed) -> dict:
    return {"log": [f"seen {state.when.year}"]}


@node
def ask_with_date(state: Typed) -> dict:
    answer = interrupt({"deadline": state.when}, id="deadline")
    return {"log": [f"answer {answer}"]}


def _typed_values() -> dict:
    return {
        "when": datetime(2026, 1, 2, 3, 4, tzinfo=UTC),
        "ident": UUID(int=7),
        "amount": Decimal("1.50"),
        "where": Path("/tmp/report.txt"),
    }


async def test_state_with_non_json_native_values_persists() -> None:
    graph = Graph(Typed, state_store=InMemoryStateStore()).flow(
        START >> touch, touch >> END
    )

    await graph.ainvoke(_typed_values(), thread_id="t")
    await graph.update_state("t", {"when": datetime(2027, 5, 6, tzinfo=UTC)})
    loaded = await graph.load("t")

    assert loaded["ident"] == UUID(int=7)
    assert loaded["amount"] == Decimal("1.50")
    assert loaded["where"] == Path("/tmp/report.txt")
    assert loaded["when"] == datetime(2027, 5, 6, tzinfo=UTC)


async def test_interrupt_payloads_and_answers_may_hold_dates() -> None:
    graph = Graph(Typed, state_store=InMemoryStateStore()).flow(
        START >> ask_with_date, ask_with_date >> END
    )

    await graph.ainvoke(_typed_values(), thread_id="d")
    final = await graph.ainvoke(
        resume=Resume(datetime(2026, 2, 1, tzinfo=UTC)), thread_id="d"
    )

    assert final.status == "completed"
