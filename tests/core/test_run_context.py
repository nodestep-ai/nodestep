import asyncio
from dataclasses import dataclass
from typing import Annotated, TypedDict

import pytest
from pydantic import BaseModel, Field

import nodestep
from nodestep import (
    END,
    START,
    Graph,
    InMemoryStateStore,
    NodeContext,
    Resume,
    ScriptedChat,
    ToolContext,
    add,
    build_react_agent,
    call_tool,
    interrupt,
    node,
    subgraph,
    tool,
    tool_runner,
)
from nodestep.chat import ChatResponse, HumanMessage, Message, ToolCall, ToolMessage
from nodestep.exceptions import ContextNotProvidedError
from nodestep.utils.reducers import add_messages


@dataclass
class Database:
    dsn: str


class S(TypedDict, total=False):
    log: Annotated[list[str], add]


seen: list[object] = []


@node
def reads_context(state: S, ctx: NodeContext) -> dict:
    seen.append(ctx.context)
    return {"log": [ctx.context.dsn]}


def _reader() -> Graph:
    return Graph(S).flow(START >> reads_context, reads_context >> END)


async def test_context_reaches_the_node_as_the_same_object() -> None:
    seen.clear()
    database = Database("postgres://prod")

    result = await _reader().ainvoke({}, context=database)

    assert result.data["log"] == ["postgres://prod"]
    assert seen == [database]
    assert seen[0] is database


def test_context_reaches_the_node_through_invoke_and_stream() -> None:
    database = Database("sync")

    invoked = _reader().invoke({}, context=database)
    events = list(_reader().stream({}, stream_mode="values", context=database))

    assert invoked.data["log"] == ["sync"]
    assert events[-1].data.state["log"] == ["sync"]


async def test_context_reaches_the_node_through_astream() -> None:
    database = Database("async")

    events = [
        event async for event in _reader().astream({}, stream_mode=[], context=database)
    ]

    assert events[-1].data.state["log"] == ["async"]


async def test_reading_context_without_one_raises() -> None:
    with pytest.raises(ContextNotProvidedError, match="context="):
        await _reader().ainvoke({})


@node
def asks_then_reads(state: S, ctx: NodeContext) -> dict:
    answer = interrupt("go?", id="go")
    return {"log": [f"{answer}:{ctx.context.dsn}"]}


def _paused_reader(store: InMemoryStateStore) -> Graph:
    return Graph(S, state_store=store).flow(
        START >> asks_then_reads, asks_then_reads >> END
    )


async def test_resume_needs_the_context_again() -> None:
    graph = _paused_reader(InMemoryStateStore())
    await graph.ainvoke({}, thread_id="t", context=Database("first"))

    with pytest.raises(ContextNotProvidedError):
        await graph.ainvoke(None, thread_id="t", resume=Resume("yes"))


async def test_resume_uses_the_context_passed_with_it() -> None:
    store = InMemoryStateStore()
    graph = _paused_reader(store)
    await graph.ainvoke({}, thread_id="t", context=Database("first"))

    final = await graph.ainvoke(
        None, thread_id="t", resume=Resume("yes"), context=Database("second")
    )

    assert final.data["log"] == ["yes:second"]
    stored = " ".join(event.data_json or "" for event in store.events)
    assert "first" not in stored


async def test_subgraph_child_sees_the_parent_context() -> None:
    seen.clear()
    child = subgraph("child", _reader(), share=["log"], child_thread="fresh")
    parent = Graph(S).flow(START >> child, child >> END)
    database = Database("shared")

    result = await parent.ainvoke({}, context=database)

    assert result.data["log"] == ["shared"]
    assert seen[0] is database


@tool
def lookup(query: str, ctx: ToolContext) -> str:
    """Look up a value in the database."""
    return f"{ctx.context.dsn}?{query}"


class Chat(BaseModel):
    messages: Annotated[list[Message], add_messages] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)


async def test_context_reaches_tools() -> None:
    runner = tool_runner("act", tools=[lookup])
    graph = Graph(Chat).flow(START >> runner, runner >> END)
    call = ToolCall(id="c1", name="lookup", arguments={"query": "users"})

    result = await graph.ainvoke({"tool_calls": [call]}, context=Database("db"))

    message = result.state.messages[-1]
    assert isinstance(message, ToolMessage)
    assert "db?users" in (message.content or "")


async def test_context_reaches_tools_of_a_react_agent() -> None:
    chat = ScriptedChat(
        [
            ChatResponse(
                tool_calls=[ToolCall(id="c1", name="lookup", arguments={"query": "q"})]
            ),
            "done",
        ]
    )
    agent = build_react_agent(chat, tools=[lookup])

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]}, context=Database("agent-db")
    )

    replies = [
        message
        for message in result.data["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert "agent-db?q" in replies[0].content


async def test_a_tool_context_without_a_run_context_raises() -> None:
    with pytest.raises(ContextNotProvidedError):
        await call_tool(lookup, {"query": "q"}, ToolContext())


async def test_parallel_tasks_share_the_context_object() -> None:
    seen.clear()

    @node
    async def first(state: S, ctx: NodeContext) -> dict:
        seen.append(ctx.context)
        await asyncio.sleep(0)
        return {"log": ["first"]}

    @node
    async def second(state: S, ctx: NodeContext) -> dict:
        seen.append(ctx.context)
        return {"log": ["second"]}

    @node(goto=[first, second])
    def fan(state: S) -> nodestep.Command:
        return nodestep.Command(goto=[first, second])

    graph = Graph(S).flow(START >> fan, first >> END, second >> END)
    database = Database("x")

    await graph.ainvoke({}, context=database)

    assert len(seen) == 2
    assert all(item is database for item in seen)


def test_transient_state_markers_are_removed() -> None:
    import nodestep.state
    import nodestep.state.context

    for module in (nodestep, nodestep.state, nodestep.state.context):
        assert not hasattr(module, "TransientState")
        assert not hasattr(module, "SecretState")
