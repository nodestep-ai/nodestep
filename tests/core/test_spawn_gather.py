import asyncio
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from nodestep.core.agent import (
    AgentHandle,
    AgentRegistry,
    AgentStatus,
    AgentTask,
    gather_agents,
    spawn_agents,
)
from nodestep.core.command import END, START, Resume, interrupt
from nodestep.core.graph import Graph
from nodestep.core.node import node
from nodestep.core.stream import NodeContext
from nodestep.exceptions import (
    AgentGroupError,
    AgentHandleLostError,
    AgentTimeoutError,
    GraphExecutionError,
)
from nodestep.state.integrations import InMemoryStateStore
from nodestep.utils.reducers import add


class InnerState(BaseModel):
    query: str = ""
    answer: str = ""


@node
def answer_query(state: InnerState) -> dict:
    return {"answer": f"result for {state.query}"}


def make_inner_graph() -> Graph:
    return Graph(InnerState, name="inner").flow(
        START >> answer_query, answer_query >> END
    )


class OuterState(BaseModel):
    queries: list[str] = Field(default_factory=list)
    findings: Annotated[list[str], add] = Field(default_factory=list)


async def test_spawn_and_gather_blocking() -> None:
    inner = make_inner_graph()

    @node
    async def research(state: OuterState, ctx: NodeContext) -> dict:
        handles = await ctx.spawn(
            [AgentTask(graph=inner, input={"query": query}) for query in state.queries]
        )
        results = await ctx.gather(handles)
        return {"findings": [result.data["answer"] for result in results]}

    graph = Graph(OuterState).flow(START >> research, research >> END)
    result = await graph.ainvoke({"queries": ["topic-a", "topic-b"]})

    assert "result for topic-a" in result.data["findings"]
    assert "result for topic-b" in result.data["findings"]


async def test_spawn_in_one_node_gather_in_another() -> None:
    inner = make_inner_graph()

    class ExtState(BaseModel):
        queries: list[str] = Field(default_factory=list)
        findings: Annotated[list[str], add] = Field(default_factory=list)

    @dataclass
    class Running:
        handles: list[AgentHandle] = field(default_factory=list)

    @node
    async def dispatch(state: ExtState, ctx: NodeContext) -> dict:
        ctx.context.handles = await ctx.spawn(
            [AgentTask(graph=inner, input={"query": "q1"})]
        )
        return {}

    @node
    async def collect(state: ExtState, ctx: NodeContext) -> dict:
        results = await ctx.gather(ctx.context.handles)
        return {"findings": [result.data["answer"] for result in results]}

    graph = Graph(ExtState).flow(START >> dispatch, dispatch >> collect, collect >> END)
    result = await graph.ainvoke({}, context=Running())
    assert "result for q1" in result.data["findings"]


async def test_gather_takes_handle_ids_kept_in_state() -> None:
    inner = make_inner_graph()

    class IdState(BaseModel):
        handle_ids: list[str] = Field(default_factory=list)
        findings: Annotated[list[str], add] = Field(default_factory=list)

    @node
    async def dispatch(state: IdState, ctx: NodeContext) -> dict:
        handles = await ctx.spawn(
            [AgentTask(graph=inner, input={"query": query}) for query in ("q1", "q2")]
        )
        return {"handle_ids": [handle.id for handle in handles]}

    @node
    async def collect(state: IdState, ctx: NodeContext) -> dict:
        results = await ctx.gather(state.handle_ids)
        return {"findings": [result.data["answer"] for result in results]}

    graph = Graph(IdState).flow(START >> dispatch, dispatch >> collect, collect >> END)
    result = await graph.ainvoke({})

    assert result.data["findings"] == ["result for q1", "result for q2"]


async def test_gather_refuses_an_id_this_run_did_not_spawn() -> None:
    @node
    async def collect(state: OuterState, ctx: NodeContext) -> dict:
        await ctx.gather(["not-spawned"])
        return {}

    graph = Graph(OuterState).flow(START >> collect, collect >> END)

    with pytest.raises(AgentHandleLostError, match="not-spawned"):
        await graph.ainvoke({})


async def test_gather_after_a_pause_says_the_ids_belong_to_the_earlier_run() -> None:
    inner = make_inner_graph()

    class PausedState(BaseModel):
        handle_ids: list[str] = Field(default_factory=list)

    @node
    async def dispatch(state: PausedState, ctx: NodeContext) -> dict:
        handles = await ctx.spawn([AgentTask(graph=inner, input={"query": "q1"})])
        return {"handle_ids": [handle.id for handle in handles]}

    @node
    def ask(state: PausedState) -> None:
        interrupt("go on?", id="go")

    @node
    async def collect(state: PausedState, ctx: NodeContext) -> None:
        await ctx.gather(state.handle_ids)

    graph = Graph(PausedState, state_store=InMemoryStateStore()).flow(
        START >> dispatch, dispatch >> ask, ask >> collect, collect >> END
    )
    with pytest.raises(GraphExecutionError, match="never gathered"):
        await graph.ainvoke({}, thread_id="t")

    with pytest.raises(AgentHandleLostError) as info:
        await graph.ainvoke(None, thread_id="t", resume=Resume(True))

    message = str(info.value)
    assert "was not spawned in this run or was already gathered" in message
    assert "a pause ends the run" in message
    assert "ctx.submit" in message
    assert "restarted" not in message


async def test_a_failed_subagent_raises_agent_group_error() -> None:
    @node
    def fail_node(state: InnerState) -> dict:
        raise ValueError("agent failed")

    error_graph = Graph(InnerState, name="errorer").flow(
        START >> fail_node, fail_node >> END
    )
    inner = make_inner_graph()

    registry = AgentRegistry()
    handles = await spawn_agents(
        [
            AgentTask(graph=inner, input={"query": "ok"}),
            AgentTask(graph=error_graph, input={}),
        ],
        registry=registry,
    )
    with pytest.raises(AgentGroupError, match="1 of 2 sub-agents failed") as info:
        await gather_agents(handles, registry=registry)

    assert [result.status for result in info.value.results] == ["completed", "error"]
    assert isinstance(info.value.results[1].error, ValueError)
    assert info.value.results[1].thread_id == handles[1].thread_id
    with pytest.raises(AgentHandleLostError):
        registry.resolve(handles[0].id)


async def test_return_exceptions_returns_the_failed_results() -> None:
    @node
    def fail_node(state: InnerState) -> dict:
        raise ValueError("agent failed")

    error_graph = Graph(InnerState, name="errorer").flow(
        START >> fail_node, fail_node >> END
    )

    @node
    async def run_it(state: OuterState, ctx: NodeContext) -> dict:
        handles = await ctx.spawn([AgentTask(graph=error_graph, input={})])
        results = await ctx.gather(handles, return_exceptions=True)
        return {"findings": [result.status for result in results]}

    graph = Graph(OuterState).flow(START >> run_it, run_it >> END)
    result = await graph.ainvoke({})
    assert result.data["findings"] == ["error"]


async def test_a_non_dict_state_out_fails_the_subagent() -> None:
    inner = make_inner_graph()
    handles = await spawn_agents(
        [
            AgentTask(
                graph=inner,
                input={"query": "x"},
                state_out=lambda data: data["answer"],
            )
        ]
    )

    results = await gather_agents(handles, return_exceptions=True)

    assert results[0].status == "error"
    assert isinstance(results[0].error, TypeError)
    assert "state_out" in str(results[0].error)


async def test_state_in_without_input_raises() -> None:
    inner = make_inner_graph()

    with pytest.raises(TypeError, match="state_in"):
        await spawn_agents([AgentTask(graph=inner, input=None, state_in=dict)])


async def test_spawn_passes_the_task_context() -> None:
    @dataclass
    class Database:
        dsn: str

    @node
    def read(state: InnerState, ctx: NodeContext) -> dict:
        return {"answer": ctx.context.dsn}

    reader = Graph(InnerState, name="reader").flow(START >> read, read >> END)
    handles = await spawn_agents(
        [AgentTask(graph=reader, input={}, context=Database("sqlite://x"))]
    )

    assert (await gather_agents(handles))[0].data["answer"] == "sqlite://x"


async def test_handle_status_follows_the_subagent() -> None:
    release = asyncio.Event()
    reached = asyncio.Event()

    @node
    def first(state: InnerState) -> dict:
        return {"query": "q"}

    @node
    async def wait(state: InnerState) -> dict:
        reached.set()
        await release.wait()
        return {"answer": "a"}

    graph = Graph(InnerState, name="waiting").flow(
        START >> first, first >> wait, wait >> END
    )
    handles = await spawn_agents(
        [AgentTask(graph=graph, input={}, name="w")], thread_prefix="p"
    )
    await reached.wait()
    running = await handles[0].status()
    release.set()
    await gather_agents(handles)
    finished = await handles[0].status()

    assert isinstance(running, AgentStatus)
    assert (running.status, running.step, running.node) == ("running", 1, "first")
    assert (finished.status, finished.step, finished.node) == (
        "completed",
        2,
        "wait",
    )
    assert finished.thread_id == handles[0].thread_id == f"p:sub:{handles[0].id}"
    assert finished.name == "w"


async def test_a_timed_out_subagent_reports_cancelled() -> None:
    @node
    async def slow(state: InnerState) -> dict:
        await asyncio.sleep(10)
        return {"answer": "done"}

    slow_graph = Graph(InnerState, name="slow").flow(START >> slow, slow >> END)
    seen: list[AgentStatus] = []
    handles = await spawn_agents(
        [AgentTask(graph=slow_graph, input={}, on_progress=seen.append)]
    )

    with pytest.raises(AgentTimeoutError):
        await gather_agents(handles, timeout=0.05)

    assert (await handles[0].status()).status == "cancelled"
    assert [status.status for status in seen] == ["cancelled"]


async def test_a_cancelled_subagent_is_a_failed_result() -> None:
    @node
    async def slow(state: InnerState) -> dict:
        await asyncio.sleep(10)
        return {"answer": "done"}

    slow_graph = Graph(InnerState, name="slow").flow(START >> slow, slow >> END)
    handles = await spawn_agents([AgentTask(graph=slow_graph, input={})])
    handles[0].task.cancel()

    with pytest.raises(AgentGroupError, match="CancelledError"):
        await gather_agents(handles)
    results = await gather_agents(handles, return_exceptions=True)

    assert [result.status for result in results] == ["cancelled"]


async def test_gather_timeout_raises() -> None:
    @node
    async def slow(state: InnerState) -> dict:
        await asyncio.sleep(10)
        return {"answer": "done"}

    slow_graph = Graph(InnerState, name="slow").flow(START >> slow, slow >> END)

    registry = AgentRegistry()
    handles = await spawn_agents(
        [AgentTask(graph=slow_graph, input={})], registry=registry
    )
    with pytest.raises(AgentTimeoutError) as exc_info:
        await gather_agents(handles, timeout=0.05, registry=registry)
    assert len(exc_info.value.pending) == 1


async def test_gather_timeout_cancels_pending_tasks() -> None:
    @node
    async def slow_node(state):
        await asyncio.sleep(10)
        return {"value": "done"}

    slow_graph = Graph(dict).flow(START >> slow_node, slow_node >> END)

    registry = AgentRegistry()
    handles = await spawn_agents(
        [
            AgentTask(graph=slow_graph, input={"value": "start"}, name="slow"),
        ],
        registry=registry,
    )
    with pytest.raises(AgentTimeoutError):
        await gather_agents(handles, timeout=0.05, registry=registry)

    await asyncio.sleep(0.1)
    assert handles[0].task.cancelled()


async def test_lost_handle_raises() -> None:
    inner = make_inner_graph()
    registry = AgentRegistry()
    handles = await spawn_agents(
        [AgentTask(graph=inner, input={"query": "x"})], registry=registry
    )
    await gather_agents(handles, registry=registry)

    with pytest.raises(AgentHandleLostError):
        await gather_agents(handles, registry=registry)


async def test_state_out_transforms_result() -> None:
    inner = make_inner_graph()
    registry = AgentRegistry()

    handles = await spawn_agents(
        [
            AgentTask(
                graph=inner,
                input={"query": "test"},
                state_out=lambda data: {"transformed": data.get("answer", "")},
            )
        ],
        registry=registry,
    )
    results = await gather_agents(handles, registry=registry)
    assert results[0].data == {"transformed": "result for test"}


async def test_on_progress_reports_each_superstep_and_the_end() -> None:
    inner = make_inner_graph()
    progress_events: list[AgentStatus] = []

    registry = AgentRegistry()
    handles = await spawn_agents(
        [
            AgentTask(
                graph=inner,
                input={"query": "test"},
                on_progress=progress_events.append,
            )
        ],
        registry=registry,
    )
    results = await gather_agents(handles, registry=registry)
    assert [
        (update.id, update.status, update.step, update.node)
        for update in progress_events
    ] == [
        (results[0].id, "running", 1, "answer_query"),
        (results[0].id, "completed", 1, "answer_query"),
    ]


async def test_on_progress_fires_on_error() -> None:
    @node
    def fail_node(state: InnerState) -> dict:
        raise ValueError("fail")

    error_graph = Graph(InnerState, name="err-progress").flow(
        START >> fail_node, fail_node >> END
    )
    progress_events: list[AgentStatus] = []

    registry = AgentRegistry()
    handles = await spawn_agents(
        [AgentTask(graph=error_graph, input={}, on_progress=progress_events.append)],
        registry=registry,
    )
    await gather_agents(handles, registry=registry, return_exceptions=True)
    assert [update.status for update in progress_events] == ["error"]


async def test_gather_timeout_removes_pending_handles_from_registry() -> None:
    @node
    async def slow(state: InnerState) -> dict:
        await asyncio.sleep(10)
        return {"answer": "done"}

    slow_graph = Graph(InnerState, name="slow-leak").flow(START >> slow, slow >> END)

    registry = AgentRegistry()
    handles = await spawn_agents(
        [AgentTask(graph=slow_graph, input={})], registry=registry
    )
    with pytest.raises(AgentTimeoutError):
        await gather_agents(handles, timeout=0.05, registry=registry)

    with pytest.raises(AgentHandleLostError):
        registry.resolve(handles[0].id)


async def test_spawn_state_in_failure_registers_nothing() -> None:
    inner = make_inner_graph()

    def boom(_state: object) -> dict:
        raise ValueError("state_in failed")

    registry = AgentRegistry()
    tasks = [
        AgentTask(graph=inner, input={"query": "ok"}),
        AgentTask(graph=inner, input={"query": "bad"}, state_in=boom),
    ]
    with pytest.raises(ValueError, match="state_in failed"):
        await spawn_agents(tasks, registry=registry)

    assert registry._handles == {}
    assert asyncio.all_tasks() == {asyncio.current_task()}


async def test_registry_isolation_between_runs() -> None:
    inner = make_inner_graph()

    @node
    async def research(state: OuterState, ctx: NodeContext) -> dict:
        handles = await ctx.spawn(
            [AgentTask(graph=inner, input={"query": query}) for query in state.queries]
        )
        results = await ctx.gather(handles)
        return {"findings": [result.data["answer"] for result in results]}

    graph = Graph(OuterState).flow(START >> research, research >> END)

    first_result, second_result = await asyncio.gather(
        graph.ainvoke({"queries": ["run1-a", "run1-b"]}),
        graph.ainvoke({"queries": ["run2-a", "run2-b"]}),
    )

    assert sorted(first_result.data["findings"]) == [
        "result for run1-a",
        "result for run1-b",
    ]
    assert sorted(second_result.data["findings"]) == [
        "result for run2-a",
        "result for run2-b",
    ]


async def test_gather_with_timeout_and_no_handles_returns_empty() -> None:
    from nodestep.core.agent import gather_agents as _gather

    assert await _gather([], timeout=1.0) == []


async def test_spawn_refuses_no_input_before_starting_any() -> None:
    inner = make_inner_graph()

    with pytest.raises(TypeError, match="no input"):
        await spawn_agents(
            [
                AgentTask(graph=inner, input={"query": "a"}),
                AgentTask(graph=inner, input=None),
            ]
        )
    assert [
        task for task in asyncio.all_tasks() if task.get_name().startswith("agent:")
    ] == []


async def test_a_failing_progress_callback_gives_an_error_result() -> None:
    calls: list[str] = []

    def broken(status: AgentStatus) -> None:
        calls.append(status.status)
        raise ValueError(f"callback failed at {status.status}")

    handles = await spawn_agents(
        [AgentTask(graph=make_inner_graph(), input={}, on_progress=broken)]
    )
    results = await gather_agents(handles, return_exceptions=True)

    assert results[0].status == "error"
    assert str(results[0].error) == "callback failed at running"
    assert calls == ["running"]
    assert (await handles[0].status()).status == "error"


async def test_spawn_with_an_empty_thread_prefix_raises() -> None:
    @node
    def answer(state: InnerState) -> dict:
        return {"answer": "a"}

    graph = Graph(InnerState).flow(START >> answer, answer >> END)

    with pytest.raises(ValueError, match="thread_prefix"):
        await spawn_agents(
            [AgentTask(graph=graph, input={})],
            parent_thread_id="parent",
            thread_prefix="",
        )


async def test_spawn_uses_the_parent_thread_without_a_prefix() -> None:
    @node
    def answer(state: InnerState) -> dict:
        return {"answer": "a"}

    graph = Graph(InnerState).flow(START >> answer, answer >> END)

    (handle,) = await spawn_agents(
        [AgentTask(graph=graph, input={})], parent_thread_id="parent"
    )
    result = (await gather_agents([handle]))[0]

    assert result.thread_id == f"parent:sub:{handle.id}"


async def test_gather_agents_resolves_handle_ids_through_the_registry() -> None:
    registry = AgentRegistry()
    handles = await spawn_agents(
        [AgentTask(graph=make_inner_graph(), input={"query": "q"})],
        registry=registry,
    )

    results = await gather_agents([handles[0].id], registry=registry)

    assert [result.data["answer"] for result in results] == ["result for q"]


async def test_gather_agents_needs_a_registry_for_handle_ids() -> None:
    with pytest.raises(TypeError, match="registry="):
        await gather_agents(["abc"])


def make_slow_graph() -> Graph:
    @node
    async def wait_long(state: InnerState) -> dict:
        await asyncio.sleep(10)
        return {"answer": "late"}

    return Graph(InnerState, name="slow").flow(START >> wait_long, wait_long >> END)


async def test_a_completed_run_cancels_and_names_sub_agents_it_never_gathered() -> None:
    slow = make_slow_graph()
    spawned: list[AgentHandle] = []

    @node
    async def dispatch(state: OuterState, ctx: NodeContext) -> dict:
        spawned.extend(
            await ctx.spawn([AgentTask(graph=slow, input={}, name="researcher")])
        )
        return {}

    graph = Graph(OuterState, name="outer").flow(START >> dispatch, dispatch >> END)

    with pytest.raises(GraphExecutionError) as info:
        await graph.ainvoke({})

    message = str(info.value)
    assert "Graph 'outer'" in message
    assert f"'researcher' ({spawned[0].id})" in message
    assert "never gathered" in message
    assert "ctx.gather" in message
    assert spawned[0].task.cancelled()


async def test_a_paused_run_cancels_sub_agents_it_never_gathered() -> None:
    slow = make_slow_graph()
    spawned: list[AgentHandle] = []

    @node
    async def dispatch(state: OuterState, ctx: NodeContext) -> dict:
        spawned.extend(await ctx.spawn([AgentTask(graph=slow, input={})]))
        return {}

    @node
    def ask(state: OuterState) -> None:
        interrupt("go on?", id="go")

    graph = Graph(OuterState, state_store=InMemoryStateStore()).flow(
        START >> dispatch, dispatch >> ask, ask >> END
    )

    with pytest.raises(GraphExecutionError, match=r"'slow' \("):
        await graph.ainvoke({}, thread_id="t")

    assert spawned[0].task.cancelled()


async def test_a_failed_run_cancels_its_sub_agents_and_raises_the_failure() -> None:
    slow = make_slow_graph()
    spawned: list[AgentHandle] = []

    @node
    async def dispatch(state: OuterState, ctx: NodeContext) -> dict:
        spawned.extend(await ctx.spawn([AgentTask(graph=slow, input={})]))
        raise ValueError("dispatch broke")

    graph = Graph(OuterState).flow(START >> dispatch, dispatch >> END)

    with pytest.raises(ValueError, match="dispatch broke"):
        await graph.ainvoke({})

    assert spawned[0].task.cancelled()


async def test_a_stream_closed_early_cancels_its_sub_agents() -> None:
    slow = make_slow_graph()
    spawned: list[AgentHandle] = []

    @node
    async def dispatch(state: OuterState, ctx: NodeContext) -> dict:
        spawned.extend(await ctx.spawn([AgentTask(graph=slow, input={})]))
        return {}

    @node
    async def collect(state: OuterState, ctx: NodeContext) -> dict:
        await ctx.gather(spawned)
        return {}

    graph = Graph(OuterState).flow(
        START >> dispatch, dispatch >> collect, collect >> END
    )

    async with aclosing(graph.astream({}, stream_mode="updates")) as events:
        async for _ in events:
            break

    assert spawned[0].task.cancelled()


async def test_a_run_that_gathers_every_sub_agent_completes() -> None:
    inner = make_inner_graph()

    @node
    async def research(state: OuterState, ctx: NodeContext) -> dict:
        handles = await ctx.spawn([AgentTask(graph=inner, input={"query": "q"})])
        await ctx.gather([handles[0].id])
        return {}

    graph = Graph(OuterState).flow(START >> research, research >> END)

    result = await graph.ainvoke({})

    assert result.data["findings"] == []
