import asyncio
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from nodestep.core.command import END, START, Command, Resume, Send, interrupt
from nodestep.core.flow import branch
from nodestep.core.graph import Graph
from nodestep.core.node import node
from nodestep.middleware.base import Middleware, NodeMiddlewareContext, Replacement
from nodestep.state.integrations.inmemory import InMemoryStateStore
from nodestep.utils.reducers import add


class State(BaseModel):
    values: Annotated[list[str], add] = Field(default_factory=list)
    counter: int = 0


execution_order: list[str] = []


@node
async def slow_a(state: State) -> dict:
    execution_order.append("a_start")
    await asyncio.sleep(0.05)
    execution_order.append("a_end")
    return {"values": ["a"]}


@node
async def slow_b(state: State) -> dict:
    execution_order.append("b_start")
    await asyncio.sleep(0.05)
    execution_order.append("b_end")
    return {"values": ["b"]}


@node(goto=[slow_a, slow_b])
def dispatch(state: State) -> Command:
    return Command(goto=[Send(slow_a), Send(slow_b)])


async def test_send_runs_tasks_concurrently() -> None:
    execution_order.clear()
    graph = Graph(State).flow(
        START >> dispatch,
        slow_a >> END,
        slow_b >> END,
    )
    result = await graph.ainvoke({})
    assert "a" in result.data["values"]
    assert "b" in result.data["values"]
    assert execution_order[0] in ("a_start", "b_start")
    assert execution_order[1] in ("a_start", "b_start")


@node
def error_node(state: State) -> dict:
    raise ValueError("boom")


@node
def ok_node(state: State) -> dict:
    return {"values": ["ok"]}


async def test_sibling_failure_fails_the_run() -> None:
    graph = Graph(State).flow(
        START >> dispatch_mixed,
        ok_node >> END,
        error_node >> END,
    )

    with pytest.raises(ValueError, match="boom"):
        await graph.ainvoke({})


@node
def asking_node(state: State) -> dict:
    return {"values": [interrupt("continue?", id="continue")]}


@node(goto=[error_node, asking_node])
def dispatch_error_and_interrupt(state: State) -> Command:
    return Command(goto=[Send(error_node), Send(asking_node)])


async def test_sibling_failure_wins_over_a_sibling_interrupt() -> None:
    graph = Graph(State, state_store=InMemoryStateStore()).flow(
        START >> dispatch_error_and_interrupt,
        error_node >> END,
        asking_node >> END,
    )

    with pytest.raises(ValueError, match="boom"):
        await graph.ainvoke({}, thread_id="mixed")


async def test_on_error_middleware_can_still_recover_a_sibling() -> None:
    class Recover(Middleware):
        def on_error(self, ctx: NodeMiddlewareContext, error: Exception) -> dict:
            return {"values": ["recovered"]}

    graph = Graph(State, middleware=[Recover()]).flow(
        START >> dispatch_mixed,
        ok_node >> END,
        error_node >> END,
    )

    result = await graph.ainvoke({})

    assert sorted(result.data["values"]) == ["ok", "recovered"]


@node(goto=[ok_node, error_node])
def dispatch_mixed(state: State) -> Command:
    return Command(goto=[Send(ok_node), Send(error_node)])


@node(goto=[error_node])
def dispatch_all_fail(state: State) -> Command:
    return Command(goto=[Send(error_node)])


async def test_all_tasks_fail_raises() -> None:
    graph = Graph(State).flow(
        START >> dispatch_all_fail,
        error_node >> END,
    )
    import pytest

    with pytest.raises(ValueError, match="boom"):
        await graph.ainvoke({})


async def test_single_task_unchanged() -> None:
    @node
    def single(state: State) -> dict:
        return {"values": ["single"], "counter": 1}

    graph = Graph(State).flow(START >> single, single >> END)
    result = await graph.ainvoke({})
    assert result.data["values"] == ["single"]
    assert result.data["counter"] == 1


async def test_parallel_tasks_share_same_step_index() -> None:
    store = InMemoryStateStore()
    graph = Graph(State, state_store=store).flow(
        START >> dispatch,
        slow_a >> END,
        slow_b >> END,
    )
    execution_order.clear()
    await graph.ainvoke({}, thread_id="step-test")

    import json

    node_started_events = [
        event
        for event in store.events
        if event.type == "node_started" and event.node in ("slow_a", "slow_b")
    ]
    assert len(node_started_events) == 2
    steps = [
        json.loads(event.data_json)["step"]
        for event in node_started_events
        if event.data_json
    ]
    assert steps[0] == steps[1], f"Parallel tasks should share step index, got {steps}"


async def test_middleware_hooks_fire_for_each_parallel_task() -> None:
    hook_nodes: list[str] = []

    class TrackingMiddleware(Middleware):
        def before_node(self, ctx: NodeMiddlewareContext) -> Replacement | None:
            hook_nodes.append(ctx.node_name)
            return None

    graph = Graph(State, middleware=[TrackingMiddleware()]).flow(
        START >> dispatch,
        slow_a >> END,
        slow_b >> END,
    )
    execution_order.clear()
    await graph.ainvoke({})

    assert "slow_a" in hook_nodes
    assert "slow_b" in hook_nodes


class RouteState(BaseModel):
    seen: Annotated[list[str], add] = Field(default_factory=list)
    picked: str = ""


@node
async def emit_x(state: RouteState) -> dict:
    return {"seen": ["x"]}


@node
async def emit_y(state: RouteState) -> dict:
    return {"seen": ["y"]}


@node(goto=[emit_x, emit_y])
def route_dispatch(state: RouteState) -> Command:
    return Command(goto=[Send(emit_x), Send(emit_y)])


def _pick(state: RouteState) -> str:
    return "both" if len(state.seen) >= 2 else "one"


@node
def both_node(state: RouteState) -> dict:
    return {"picked": "both"}


@node
def one_node(state: RouteState) -> dict:
    return {"picked": "one"}


async def test_route_selection_sees_merged_superstep_state() -> None:
    graph = Graph(RouteState).flow(
        START >> route_dispatch,
        emit_x >> branch(_pick, {"both": both_node, "one": one_node}),
        emit_y >> branch(_pick, {"both": both_node, "one": one_node}),
        both_node >> END,
        one_node >> END,
    )
    result = await graph.ainvoke({})
    assert result.data["picked"] == "both"


class InterruptFanState(BaseModel):
    values: Annotated[list[str], add] = Field(default_factory=list)


@node
def fan_sibling(state: InterruptFanState) -> dict:
    return {"values": ["sib"]}


@node
def fan_asker(state: InterruptFanState) -> dict:
    answer = interrupt({"q": "?"}, id="q")
    return {"values": [f"ask:{answer}"]}


@node(goto=[fan_sibling, fan_asker])
def fan_dispatch(state: InterruptFanState) -> Command:
    return Command(goto=[Send(fan_sibling), Send(fan_asker)])


async def test_interrupt_merges_successful_siblings() -> None:
    store = InMemoryStateStore()
    graph = Graph(InterruptFanState, state_store=store).flow(
        START >> fan_dispatch,
        fan_sibling >> END,
        fan_asker >> END,
    )
    first = await graph.ainvoke({}, thread_id="int-merge")
    assert first.status == "interrupted"

    final = await graph.ainvoke(
        None,
        thread_id="int-merge",
        resume=Resume("yes"),
    )
    assert "sib" in final.data["values"]
    assert "ask:yes" in final.data["values"]
    assert sorted(final.data["values"]) == ["ask:yes", "sib"]


class ConflictState(BaseModel):
    values: Annotated[list[str], add] = Field(default_factory=list)
    winner: str = ""


@node
def write_x(state: ConflictState) -> dict:
    return {"winner": "x"}


@node
def write_y(state: ConflictState) -> dict:
    return {"winner": "y"}


@node(goto=[write_x, write_y])
def dispatch_conflict(state: ConflictState) -> Command:
    return Command(goto=[Send(write_x), Send(write_y)])


async def test_parallel_replace_conflict_raises_through_graph() -> None:
    import pytest

    from nodestep.exceptions import InvalidUpdateError

    graph = Graph(ConflictState).flow(
        START >> dispatch_conflict,
        write_x >> END,
        write_y >> END,
    )
    with pytest.raises(InvalidUpdateError) as info:
        await graph.ainvoke({})

    assert info.value.field == "winner"
    assert sorted(info.value.nodes) == ["write_x", "write_y"]


async def test_parallel_reducer_field_combines_through_graph() -> None:
    execution_order.clear()
    graph = Graph(State).flow(
        START >> dispatch,
        slow_a >> END,
        slow_b >> END,
    )
    result = await graph.ainvoke({})
    assert sorted(result.data["values"]) == ["a", "b"]


@node
def write_winner_only(state: ConflictState) -> dict:
    return {"winner": "solo"}


@node
def write_values_only(state: ConflictState) -> dict:
    return {"values": ["v"]}


@node(goto=[write_winner_only, write_values_only])
def dispatch_distinct(state: ConflictState) -> Command:
    return Command(goto=[Send(write_winner_only), Send(write_values_only)])


async def test_parallel_distinct_replace_field_single_writer_ok() -> None:
    graph = Graph(ConflictState).flow(
        START >> dispatch_distinct,
        write_winner_only >> END,
        write_values_only >> END,
    )
    result = await graph.ainvoke({})
    assert result.data["winner"] == "solo"
    assert result.data["values"] == ["v"]


class SendInterruptState(BaseModel):
    results: Annotated[list[str], add] = Field(default_factory=list)
    item: str = ""


@node
def send_worker(state: SendInterruptState) -> dict:
    answer = interrupt({"item": state.item}, id="item")
    return {"results": [f"{state.item}:{answer}"]}


@node(goto=[send_worker])
def send_fan(state: SendInterruptState) -> Command:
    return Command(
        goto=[
            Send(send_worker, {"item": "a"}),
            Send(send_worker, {"item": "b"}),
        ]
    )


async def test_parallel_sends_to_same_node_resume_with_payloads() -> None:
    store = InMemoryStateStore()
    graph = Graph(SendInterruptState, state_store=store).flow(
        START >> send_fan,
        send_worker >> END,
    )
    first = await graph.ainvoke({}, thread_id="send-int")
    assert first.status == "interrupted"
    assert len(first.interrupts) == 2

    answers = {
        key: f"ok-{pending.payload['item']}"
        for key, pending in first.interrupts.items()
    }
    final = await graph.ainvoke(
        None, thread_id="send-int", resume=Resume(answers=answers)
    )
    assert final.status == "completed"
    assert sorted(final.data["results"]) == ["a:ok-a", "b:ok-b"]


@node
def maybe_ask(state: SendInterruptState) -> dict:
    if state.item == "ask":
        answer = interrupt({"item": state.item}, id="item")
        return {"results": [f"ask:{answer}"]}
    return {"results": [state.item]}


@node(goto=[maybe_ask])
def mixed_fan(state: SendInterruptState) -> Command:
    return Command(
        goto=[
            Send(maybe_ask, {"item": "plain"}),
            Send(maybe_ask, {"item": "ask"}),
        ]
    )


async def test_completed_sibling_does_not_clear_pending_interrupt() -> None:
    store = InMemoryStateStore()
    graph = Graph(SendInterruptState, state_store=store).flow(
        START >> mixed_fan,
        maybe_ask >> END,
    )
    first = await graph.ainvoke({}, thread_id="mixed-int")
    assert first.status == "interrupted"
    assert len(first.interrupts) == 1

    final = await graph.ainvoke(None, thread_id="mixed-int", resume=Resume("yes"))
    assert final.status == "completed"
    assert sorted(final.data["results"]) == ["ask:yes", "plain"]


class JoinState(BaseModel):
    log: Annotated[list[str], add] = Field(default_factory=list)


@node
def join_worker_a(state: JoinState) -> dict:
    return {"log": ["a"]}


@node
def join_worker_b(state: JoinState) -> dict:
    return {"log": ["b"]}


@node(goto=[join_worker_a, join_worker_b])
def fan_to_workers(state: JoinState) -> Command:
    return Command(goto=[Send(join_worker_a), Send(join_worker_b)])


@node
def join_node(state: JoinState) -> dict:
    return {"log": ["JOIN"]}


async def test_fan_in_runs_join_once() -> None:
    graph = Graph(JoinState).flow(
        START >> fan_to_workers,
        join_worker_a >> join_node,
        join_worker_b >> join_node,
        join_node >> END,
    )

    result = await graph.ainvoke({})

    assert result.data["log"] == ["a", "b", "JOIN"]


@node
def echo_payload(state: JoinState) -> dict:
    return {"log": ["echo"]}


@node(goto=[echo_payload])
def send_twice(state: JoinState) -> Command:
    return Command(
        goto=[Send(echo_payload, {"log": []}), Send(echo_payload, {"log": []})]
    )


async def test_send_fan_out_to_one_node_still_runs_per_send() -> None:
    graph = Graph(JoinState).flow(START >> send_twice, echo_payload >> END)

    result = await graph.ainvoke({})

    assert result.data["log"] == ["echo", "echo"]


@node
def blocking_a(state: JoinState) -> dict:
    import time

    time.sleep(0.3)
    return {"log": ["a"]}


@node
def blocking_b(state: JoinState) -> dict:
    import time

    time.sleep(0.3)
    return {"log": ["b"]}


@node(goto=[blocking_a, blocking_b])
def fan_blocking(state: JoinState) -> Command:
    return Command(goto=[Send(blocking_a), Send(blocking_b)])


async def test_sync_nodes_run_in_parallel_off_the_loop() -> None:
    import time

    graph = Graph(JoinState).flow(
        START >> fan_blocking,
        blocking_a >> END,
        blocking_b >> END,
    )
    started = time.monotonic()

    await graph.ainvoke({})

    assert time.monotonic() - started < 0.5


@node(timeout=0.1)
def blocking_with_timeout(state: JoinState) -> dict:
    import time

    time.sleep(0.4)
    return {"log": ["late"]}


async def test_sync_node_timeout_is_enforced() -> None:
    from nodestep.exceptions import NodeTimeoutError

    graph = Graph(JoinState).flow(
        START >> blocking_with_timeout, blocking_with_timeout >> END
    )

    with pytest.raises(NodeTimeoutError):
        await graph.ainvoke({})


sibling_finished: list[str] = []


@node
async def failing_fast(state: JoinState) -> dict:
    await asyncio.sleep(0.01)
    raise RuntimeError("boom early")


@node
async def slow_sibling(state: JoinState) -> dict:
    await asyncio.sleep(1)
    sibling_finished.append("slow")
    return {"log": ["slow"]}


@node
async def bad_return(state: JoinState) -> int:
    await asyncio.sleep(0.01)
    return 42


@node(goto=[failing_fast, slow_sibling])
def fan_fail(state: JoinState) -> Command:
    return Command(goto=[Send(failing_fast), Send(slow_sibling)])


@node(goto=[bad_return, slow_sibling])
def fan_bad_return(state: JoinState) -> Command:
    return Command(goto=[Send(bad_return), Send(slow_sibling)])


async def test_node_failure_cancels_running_siblings() -> None:
    import time

    sibling_finished.clear()
    graph = Graph(JoinState).flow(
        START >> fan_fail, failing_fast >> END, slow_sibling >> END
    )
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="boom early"):
        await graph.ainvoke({})

    assert time.monotonic() - started < 0.5
    await asyncio.sleep(1.1)
    assert sibling_finished == []


async def test_errors_outside_a_node_body_also_cancel_siblings() -> None:
    sibling_finished.clear()
    graph = Graph(JoinState).flow(
        START >> fan_bad_return,
        bad_return >> END,
        slow_sibling >> END,
    )

    with pytest.raises(TypeError, match="Node 'bad_return' returned int;"):
        await graph.ainvoke({})

    await asyncio.sleep(1.1)
    assert sibling_finished == []


async def test_external_cancellation_leaves_no_tasks_behind() -> None:
    graph = Graph(JoinState).flow(START >> slow_sibling, slow_sibling >> END)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(graph.ainvoke({}), timeout=0.05)
    await asyncio.sleep(0.05)

    leftovers = [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and not task.done()
    ]
    assert leftovers == []
