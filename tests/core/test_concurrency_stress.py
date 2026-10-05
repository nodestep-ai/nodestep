import asyncio
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from nodestep.core.agent import AgentRegistry, AgentTask, gather_agents, spawn_agents
from nodestep.core.command import END, START, Command, Send
from nodestep.core.graph import Graph
from nodestep.core.node import node
from nodestep.exceptions import AgentTimeoutError
from nodestep.state.integrations.inmemory import InMemoryStateStore
from nodestep.utils.reducers import add


async def test_registry_isolation_concurrent_graphs() -> None:
    @node
    async def increment(state):
        await asyncio.sleep(0.01)
        return {"count": state["count"] + 1}

    graph = Graph(dict).flow(START >> increment, increment >> END)

    results = await asyncio.gather(*[graph.ainvoke({"count": i}) for i in range(20)])
    for i, result in enumerate(results):
        assert result.data["count"] == i + 1


async def test_parallel_state_merge_10_tasks() -> None:
    class State(BaseModel):
        values: Annotated[list[str], add] = Field(default_factory=list)

    @node
    def append_node(state: State) -> dict:
        return {"values": [f"item-{state.values[-1] if state.values else 'none'}"]}

    @node(goto=[append_node])
    def fan_out(state: State) -> Command:
        return Command(
            goto=[Send(append_node, {"values": [f"task-{i}"]}) for i in range(10)]
        )

    graph = Graph(State).flow(
        START >> fan_out,
        append_node >> END,
    )
    result = await graph.ainvoke({})
    assert len(result.data["values"]) == 10


async def test_cancel_on_timeout_cleanup() -> None:
    @node
    async def very_slow(state):
        await asyncio.sleep(60)
        return {"v": 1}

    slow_graph = Graph(dict).flow(START >> very_slow, very_slow >> END)
    registry = AgentRegistry()
    handles = await spawn_agents(
        [AgentTask(graph=slow_graph, input={"v": 0}, name="slow")],
        registry=registry,
    )

    with pytest.raises(AgentTimeoutError):
        await gather_agents(handles, timeout=0.05, registry=registry)

    await asyncio.sleep(0.1)
    assert handles[0].task.cancelled()


async def test_concurrent_checkpoint_writes() -> None:
    store = InMemoryStateStore()

    @node
    async def increment(state):
        await asyncio.sleep(0.01)
        return {"count": state["count"] + 1}

    graph = Graph(dict, state_store=store).flow(START >> increment, increment >> END)

    await asyncio.gather(
        graph.ainvoke({"count": 0}, thread_id="thread-a"),
        graph.ainvoke({"count": 100}, thread_id="thread-b"),
    )

    history_a = await store.get_history("thread-a", "main")
    history_b = await store.get_history("thread-b", "main")
    assert len(history_a.events) > 0
    assert len(history_b.events) > 0

    cp_a = await store.latest_checkpoint("thread-a", "main")
    cp_b = await store.latest_checkpoint("thread-b", "main")
    assert cp_a is not None
    assert cp_b is not None
