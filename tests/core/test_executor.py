import asyncio
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from nodestep import InMemoryStateStore
from nodestep.core.agent import (
    AgentResult,
    AgentStatus,
    AgentTask,
    DurableAgentHandle,
)
from nodestep.core.command import END, START
from nodestep.core.executor import AsyncioExecutor
from nodestep.core.graph import Graph
from nodestep.core.node import node
from nodestep.core.stream import NodeContext
from nodestep.exceptions import AgentHandleLostError


class State(BaseModel):
    value: str = ""


@node
def set_value(state: State) -> dict:
    return {"value": "done"}


def make_graph() -> Graph:
    return Graph(State, name="test-bg").flow(START >> set_value, set_value >> END)


async def _finished(executor: AsyncioExecutor, handle_id: str) -> AgentResult:
    for _ in range(500):
        if (result := await executor.poll(handle_id)) is not None:
            return result
        await asyncio.sleep(0.01)
    raise AssertionError(f"agent {handle_id} did not finish")


async def test_submit_returns_durable_handle() -> None:
    executor = AsyncioExecutor()
    handle = await executor.submit(AgentTask(graph=make_graph(), input={}))
    assert isinstance(handle, DurableAgentHandle)
    assert handle.executor_id == "asyncio"
    assert handle.graph_name == "test-bg"
    assert handle.thread_id == f"bg:{handle.id}"


async def test_poll_returns_none_then_result() -> None:
    release = asyncio.Event()

    @node
    async def slow_set(state: State) -> dict:
        await release.wait()
        return {"value": "slow-done"}

    slow_graph = Graph(State, name="slow-bg").flow(START >> slow_set, slow_set >> END)
    executor = AsyncioExecutor()
    handle = await executor.submit(AgentTask(graph=slow_graph, input={}))

    assert await executor.poll(handle.id) is None
    release.set()
    result = await _finished(executor, handle.id)
    assert result.data["value"] == "slow-done"
    assert result.status == "completed"


async def test_cancel_stops_agent_and_reports_cancelled() -> None:
    @node
    async def forever(state: State) -> dict:
        await asyncio.sleep(100)
        return {"value": "never"}

    stuck_graph = Graph(State, name="stuck").flow(START >> forever, forever >> END)
    executor = AsyncioExecutor()
    handle = await executor.submit(AgentTask(graph=stuck_graph, input={}))
    await asyncio.sleep(0)

    assert await executor.cancel(handle.id)
    result = await executor.poll(handle.id)
    status = await executor.status(handle.id)

    assert result is not None
    assert result.status == "cancelled"
    assert result.thread_id == handle.thread_id
    assert status.status == "cancelled"


async def test_cancel_before_the_agent_started_reports_cancelled() -> None:
    executor = AsyncioExecutor()
    handle = await executor.submit(AgentTask(graph=make_graph(), input={}))

    assert await executor.cancel(handle.id)
    assert (await executor.status(handle.id)).status == "cancelled"


async def test_list_running_shows_active() -> None:
    @node
    async def slow(state: State) -> dict:
        await asyncio.sleep(10)
        return {"value": "x"}

    slow_graph = Graph(State, name="list-test").flow(START >> slow, slow >> END)
    executor = AsyncioExecutor()
    handle = await executor.submit(AgentTask(graph=slow_graph, input={}))
    running = await executor.list_running()
    assert len(running) == 1
    assert running[0].id == handle.id

    await executor.cancel(handle.id)
    running_after = await executor.list_running()
    assert len(running_after) == 0


async def test_submit_with_custom_executor_id() -> None:
    executor = AsyncioExecutor(executor_id="my-cluster")
    handle = await executor.submit(AgentTask(graph=make_graph(), input={}))
    assert handle.executor_id == "my-cluster"
    assert (await _finished(executor, handle.id)).status == "completed"


async def test_error_in_background_agent() -> None:
    @node
    def fail(state: State) -> dict:
        raise RuntimeError("bg fail")

    failing_graph = Graph(State, name="err-bg").flow(START >> fail, fail >> END)
    executor = AsyncioExecutor()
    handle = await executor.submit(AgentTask(graph=failing_graph, input={}))
    result = await _finished(executor, handle.id)
    assert result.status == "error"
    assert isinstance(result.error, RuntimeError)
    assert result.thread_id == handle.thread_id
    assert (await executor.status(handle.id)).status == "error"


async def test_unknown_handle_ids_raise() -> None:
    executor = AsyncioExecutor()

    with pytest.raises(AgentHandleLostError, match="this executor does not know it"):
        await executor.poll("typo-id")
    with pytest.raises(AgentHandleLostError):
        await executor.status("typo-id")


async def test_forget_releases_a_finished_run() -> None:
    executor = AsyncioExecutor()
    handle = await executor.submit(AgentTask(graph=make_graph(), input={}))
    await _finished(executor, handle.id)

    await executor.forget(handle.id)

    with pytest.raises(AgentHandleLostError):
        await executor.poll(handle.id)
    with pytest.raises(AgentHandleLostError):
        await executor.status(handle.id)
    assert executor._runs == {}
    assert executor._handles == {}


async def test_forget_releases_a_cancelled_run() -> None:
    executor = AsyncioExecutor()
    handle = await executor.submit(AgentTask(graph=make_graph(), input={}))
    await executor.cancel(handle.id)

    await executor.forget(handle.id)

    with pytest.raises(AgentHandleLostError):
        await executor.poll(handle.id)


async def test_forget_refuses_a_running_agent() -> None:
    release = asyncio.Event()

    @node
    async def wait_for_release(state: State) -> dict:
        await release.wait()
        return {"value": "released"}

    slow_graph = Graph(State, name="slow-forget").flow(
        START >> wait_for_release, wait_for_release >> END
    )
    executor = AsyncioExecutor()
    handle = await executor.submit(AgentTask(graph=slow_graph, input={}))

    with pytest.raises(ValueError, match="still running"):
        await executor.forget(handle.id)

    release.set()
    assert (await _finished(executor, handle.id)).data["value"] == "released"


async def test_forget_raises_for_an_unknown_id() -> None:
    executor = AsyncioExecutor()

    with pytest.raises(AgentHandleLostError):
        await executor.forget("typo-id")


async def test_cancel_raises_for_an_unknown_id() -> None:
    executor = AsyncioExecutor()

    with pytest.raises(AgentHandleLostError, match="this executor does not know it"):
        await executor.cancel("typo-id")


async def test_an_empty_thread_prefix_raises() -> None:
    executor = AsyncioExecutor()

    with pytest.raises(ValueError, match="thread_prefix"):
        await executor.submit(AgentTask(graph=make_graph(), input={}), thread_prefix="")


async def test_thread_ids_follow_the_explicit_id_then_the_prefix() -> None:
    executor = AsyncioExecutor()

    explicit = await executor.submit(
        AgentTask(graph=make_graph(), input={}), thread_id="mine"
    )
    prefixed = await executor.submit(
        AgentTask(graph=make_graph(), input={}), thread_prefix="job"
    )

    assert explicit.thread_id == "mine"
    assert prefixed.thread_id == f"job:bg:{prefixed.id}"
    assert (await _finished(executor, explicit.id)).thread_id == "mine"


async def test_repeated_ctx_submits_get_their_own_threads() -> None:
    class Log(BaseModel):
        log: list[str] = []

    @node
    def child(state: Log) -> dict:
        return {"log": [*state.log, "child ran"]}

    store = InMemoryStateStore()
    worker = Graph(Log, name="worker", state_store=store).flow(
        START >> child, child >> END
    )
    executor = AsyncioExecutor()
    submitted: list[DurableAgentHandle] = []

    @node
    async def launch(state: State, ctx: NodeContext) -> dict:
        submitted.extend(
            await ctx.submit(
                [AgentTask(graph=worker, input={"log": ["task input"]})],
                executor=executor,
            )
        )
        return {}

    parent = Graph(State, name="parent").flow(START >> launch, launch >> END)
    await parent.ainvoke({}, thread_id="job-1")
    await parent.ainvoke({}, thread_id="job-1")
    results = [await _finished(executor, handle.id) for handle in submitted]

    assert [handle.thread_id for handle in submitted] == [
        f"job-1:bg:{handle.id}" for handle in submitted
    ]
    assert len({handle.thread_id for handle in submitted}) == 2
    assert [result.data["log"] for result in results] == [
        ["task input", "child ran"],
        ["task input", "child ran"],
    ]


async def test_status_reports_every_superstep_and_the_end() -> None:
    @node
    def first(state: State) -> dict:
        return {"value": "a"}

    @node
    def second(state: State) -> dict:
        return {"value": "b"}

    graph = Graph(State, name="two-steps").flow(
        START >> first, first >> second, second >> END
    )
    seen: list[AgentStatus] = []
    executor = AsyncioExecutor()
    handle = await executor.submit(
        AgentTask(graph=graph, input={}, name="stepper", on_progress=seen.append)
    )
    await _finished(executor, handle.id)

    assert [(status.status, status.step, status.node) for status in seen] == [
        ("running", 1, "first"),
        ("running", 2, "second"),
        ("completed", 2, "second"),
    ]
    assert {(status.id, status.name, status.thread_id) for status in seen} == {
        (handle.id, "stepper", handle.thread_id)
    }
    assert await executor.status(handle.id) == seen[-1]


async def test_status_is_running_while_the_agent_works() -> None:
    release = asyncio.Event()
    reached = asyncio.Event()

    @node
    def first(state: State) -> dict:
        return {"value": "a"}

    @node
    async def wait(state: State) -> dict:
        reached.set()
        await release.wait()
        return {"value": "b"}

    graph = Graph(State, name="waiting").flow(
        START >> first, first >> wait, wait >> END
    )
    executor = AsyncioExecutor()
    handle = await executor.submit(AgentTask(graph=graph, input={}))
    await reached.wait()

    status = await executor.status(handle.id)
    release.set()
    await _finished(executor, handle.id)

    assert (status.status, status.step, status.node) == ("running", 1, "first")


async def test_an_interrupted_agent_reports_interrupted() -> None:
    from nodestep import interrupt

    @node
    def ask(state: State) -> dict:
        return {"value": interrupt("ok?", id="ok")}

    graph = Graph(State, name="asker", state_store=InMemoryStateStore()).flow(
        START >> ask, ask >> END
    )
    executor = AsyncioExecutor()
    handle = await executor.submit(AgentTask(graph=graph, input={}))
    result = await _finished(executor, handle.id)

    assert result.status == "interrupted"
    assert (await executor.status(handle.id)).status == "interrupted"


async def test_submit_passes_the_task_context() -> None:
    @dataclass
    class Database:
        dsn: str

    @node
    def read(state: State, ctx: NodeContext) -> dict:
        return {"value": ctx.context.dsn}

    graph = Graph(State, name="reader").flow(START >> read, read >> END)
    executor = AsyncioExecutor()
    handle = await executor.submit(
        AgentTask(graph=graph, input={}, context=Database("sqlite://x"))
    )

    assert (await _finished(executor, handle.id)).data["value"] == "sqlite://x"


async def test_submit_refuses_state_in_without_input() -> None:
    executor = AsyncioExecutor()

    with pytest.raises(TypeError, match="state_in"):
        await executor.submit(
            AgentTask(graph=make_graph(), input=None, state_in=lambda value: value)
        )
    assert await executor.list_running() == []


async def test_a_new_agent_has_finished_no_superstep() -> None:
    release = asyncio.Event()

    @node
    async def wait(state: State) -> dict:
        await release.wait()
        return {}

    graph = Graph(State, name="waiting").flow(START >> wait, wait >> END)
    executor = AsyncioExecutor()
    handle = await executor.submit(AgentTask(graph=graph, input={}))

    status = await executor.status(handle.id)
    release.set()
    await _finished(executor, handle.id)

    assert (status.status, status.step, status.node) == ("running", 0, None)
    assert (await executor.status(handle.id)).step == 1


async def test_a_failing_progress_callback_fails_the_agent_once() -> None:
    calls: list[str] = []

    def broken(status: AgentStatus) -> None:
        calls.append(status.status)
        raise ValueError(f"callback failed at {status.status}")

    executor = AsyncioExecutor()
    handle = await executor.submit(
        AgentTask(graph=make_graph(), input={}, on_progress=broken)
    )
    result = await _finished(executor, handle.id)

    assert result.status == "error"
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "callback failed at running"
    assert calls == ["running"]
    assert (await executor.status(handle.id)).status == "error"


async def test_submit_refuses_no_input_on_a_new_thread() -> None:
    executor = AsyncioExecutor()

    with pytest.raises(TypeError, match="thread_id="):
        await executor.submit(AgentTask(graph=make_graph(), input=None))
    assert await executor.list_running() == []


async def test_submit_without_input_runs_on_the_given_thread() -> None:
    from nodestep import interrupt
    from nodestep.exceptions import ResumeError

    @node
    def ask(state: State) -> dict:
        interrupt("ok?", id="q")
        return {"value": "answered"}

    store = InMemoryStateStore()
    graph = Graph(State, name="asker", state_store=store).flow(START >> ask, ask >> END)
    await graph.ainvoke({}, thread_id="t")
    executor = AsyncioExecutor()

    handle = await executor.submit(AgentTask(graph=graph, input=None), thread_id="t")

    result = await _finished(executor, handle.id)
    assert isinstance(result.error, ResumeError)
    assert "thread 't' is paused" in str(result.error)


async def test_ctx_submit_checks_every_task_before_starting_any() -> None:
    class Recording(AsyncioExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.submitted: list[AgentTask] = []

        async def submit(
            self,
            task: AgentTask,
            *,
            thread_id: str | None = None,
            thread_prefix: str | None = None,
        ) -> DurableAgentHandle:
            self.submitted.append(task)
            return DurableAgentHandle(
                id="x", name="x", executor_id="recording", graph_name="x"
            )

    executor = Recording()

    @node
    async def launch(state: State, ctx: NodeContext) -> dict:
        await ctx.submit(
            [
                AgentTask(graph=make_graph(), input={}),
                AgentTask(graph=make_graph(), input=None, state_in=dict),
            ],
            executor=executor,
        )
        return {}

    parent = Graph(State, name="parent").flow(START >> launch, launch >> END)

    with pytest.raises(TypeError, match="state_in"):
        await parent.ainvoke({})
    assert executor.submitted == []
