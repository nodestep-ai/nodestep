from __future__ import annotations

import json

import pytest

from nodestep.chat.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from nodestep.core.builtin_nodes.agent import AgentState
from nodestep.core.command import END, START, Resume, interrupt
from nodestep.core.graph import Graph
from nodestep.core.node import node
from nodestep.state.integrations import InMemoryStateStore
from nodestep.state.schema import StateSchema


def test_message_type_discriminator_is_persisted() -> None:
    payload = StateSchema.from_type(AgentState).dump_state(
        {
            "messages": [
                SystemMessage(content="sys"),
                HumanMessage(content="hi"),
                AIMessage(
                    content=None,
                    tool_calls=[ToolCall(id="1", name="lookup", arguments={"q": "x"})],
                ),
                ToolMessage(content="ok", name="lookup", tool_call_id="1"),
            ]
        }
    )
    types = [message["type"] for message in payload["messages"]]
    assert types == ["system", "human", "ai", "tool"]


def test_agent_state_roundtrip_restores_concrete_message_types() -> None:
    schema = StateSchema.from_type(AgentState)
    state, _ = schema.merge_input(
        None,
        {
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(
                    content=None,
                    tool_calls=[ToolCall(id="1", name="lookup", arguments={"q": "x"})],
                ),
                ToolMessage(content="ok", name="lookup", tool_call_id="1"),
            ]
        },
    )

    reloaded = schema.restore_state(json.loads(json.dumps(schema.dump_state(state))))
    declared = schema.to_declared(reloaded)

    assert [type(message).__name__ for message in declared.messages] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
    ]
    assert [message.role for message in declared.messages] == [
        "user",
        "assistant",
        "tool",
    ]
    assert declared.messages[1].tool_calls[0].name == "lookup"
    assert declared.messages[2].tool_call_id == "1"


def test_restore_state_rebuilds_typed_messages_from_dicts() -> None:
    schema = StateSchema.from_type(AgentState)
    raw = {
        "messages": [
            {"type": "human", "content": "hi"},
            {
                "type": "ai",
                "content": None,
                "tool_calls": [{"id": "1", "name": "lookup", "arguments": {}}],
            },
            {"type": "tool", "content": "ok", "name": "lookup", "tool_call_id": "1"},
        ],
        "tool_calls": [],
        "final_text": None,
        "final_output": None,
    }

    coerced = schema.restore_state(raw)

    assert isinstance(coerced, dict)
    assert [type(message).__name__ for message in coerced["messages"]] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
    ]
    assert coerced["messages"][1].tool_calls[0].name == "lookup"


@node(name="ingest")
def ingest(state: AgentState) -> dict:
    return {"messages": [HumanMessage(id="u1", content="question")]}


@node(name="await_approval")
def await_approval(state: AgentState) -> dict:
    decision = interrupt({"prompt": "approve?"}, id="approve")
    return {
        "messages": [
            AIMessage(
                id="a1",
                content=None,
                tool_calls=[ToolCall(id="1", name="lookup", arguments={})],
            ),
            ToolMessage(
                id="t1", content=str(decision), name="lookup", tool_call_id="1"
            ),
        ]
    }


@pytest.mark.asyncio
async def test_message_bearing_thread_interrupts_and_resumes() -> None:
    store = InMemoryStateStore()
    graph = Graph(AgentState, state_store=store).flow(
        START >> ingest,
        ingest >> await_approval,
        await_approval >> END,
    )

    paused = await graph.ainvoke({}, thread_id="resume-thread")
    assert paused.status == "interrupted"

    loaded = await graph.load("resume-thread")
    assert [type(message).__name__ for message in loaded["messages"]] == [
        "HumanMessage"
    ]

    resumed = await graph.ainvoke(resume=Resume("yes"), thread_id="resume-thread")
    assert resumed.status == "completed"
    roles = [message.role for message in resumed.state.messages]
    assert roles == ["user", "assistant", "tool"]
    assert resumed.state.messages[2].content == "yes"


def test_add_messages_dedups_by_id_and_refuses_dicts() -> None:
    from nodestep.utils.reducers import add_messages

    existing = [HumanMessage(id="u1", content="hi")]

    by_object = add_messages(existing, [HumanMessage(id="u1", content="hi-edited")])
    assert len(by_object) == 1
    assert by_object[0].content == "hi-edited"

    with pytest.raises(TypeError, match="add_messages"):
        add_messages(existing, [{"type": "human", "id": "u1", "content": "hi-d"}])


@node(name="record")
def record(state: AgentState) -> dict:
    return {"messages": [AIMessage(content="answer")]}


@pytest.mark.asyncio
async def test_resume_completed_thread_appends_without_crash() -> None:
    store = InMemoryStateStore()
    graph = Graph(AgentState, state_store=store).flow(
        START >> ingest,
        ingest >> record,
        record >> END,
    )

    first = await graph.ainvoke({}, thread_id="append-thread")
    assert first.status == "completed"
    assert [message.role for message in first.state.messages] == ["user", "assistant"]

    second = await graph.ainvoke(
        {"messages": [HumanMessage(id="u2", content="follow up")]},
        thread_id="append-thread",
    )
    assert second.status == "completed"
    roles = [message.role for message in second.state.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert all(not isinstance(message, dict) for message in second.state.messages)
