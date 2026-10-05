import asyncio
import json
from collections import Counter
from typing import Annotated, TypedDict

import pytest

from nodestep import (
    END,
    START,
    Command,
    Graph,
    InMemoryStateStore,
    NodeContext,
    Resume,
    Send,
    add,
    interrupt,
    node,
)
from nodestep.exceptions import (
    GraphConfigError,
    GraphTimeoutError,
    InvalidUpdateError,
    ResumeError,
    RunLimitExceededError,
    StateStoreError,
    UnknownThreadError,
)
from nodestep.state import HistoryEvent

calls: Counter[str] = Counter()


class Log(TypedDict, total=False):
    log: Annotated[list[str], add]


def _step(name: str, delay: float = 0.0):
    async def run(state: Log) -> dict:
        calls[name] += 1
        await asyncio.sleep(delay)
        return {"log": [name]}

    return node(run, name=name)


step_a, step_b, step_c, step_d = _step("a"), _step("b"), _step("c"), _step("d")
slow_b = _step("b", delay=0.3)


@node
def pre(state: Log) -> dict:
    calls["pre"] += 1
    return {"log": ["pre"]}


@node
def ask(state: Log) -> dict:
    calls["ask"] += 1
    return {"log": [f"answer={interrupt('ok?', id='ok')}"]}


def _paused_graph() -> Graph:
    return Graph(Log, state_store=InMemoryStateStore()).flow(
        START >> pre, pre >> ask, ask >> END
    )


def _linear_graph(**options) -> Graph:
    return Graph(Log, state_store=InMemoryStateStore(), **options).flow(
        START >> step_a,
        step_a >> step_b,
        step_b >> step_c,
        step_c >> step_d,
        step_d >> END,
    )


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    calls.clear()


async def test_a_store_requires_a_thread_id() -> None:
    graph = _paused_graph()

    with pytest.raises(
        GraphConfigError, match=r"a state store is configured; pass thread_id="
    ):
        await graph.ainvoke({})
    with pytest.raises(GraphConfigError, match=r"pass thread_id="):
        [event async for event in graph.astream({}, stream_mode=[])]

    assert calls == {}


async def test_a_blank_thread_id_does_not_become_a_random_thread() -> None:
    graph = _paused_graph()

    with pytest.raises(GraphConfigError, match=r"pass thread_id="):
        await graph.ainvoke({}, thread_id="")

    assert calls == {}


async def test_a_per_call_store_requires_a_thread_id() -> None:
    graph = Graph(Log).flow(START >> pre, pre >> END)

    with pytest.raises(GraphConfigError, match=r"pass thread_id="):
        await graph.ainvoke({}, state_store=InMemoryStateStore())


def test_sync_wrappers_require_a_thread_id_with_a_store() -> None:
    graph = _paused_graph()

    with pytest.raises(GraphConfigError, match=r"pass thread_id="):
        graph.invoke({})
    with pytest.raises(GraphConfigError, match=r"pass thread_id="):
        list(graph.stream({}, stream_mode=[]))


async def test_without_a_store_a_thread_id_is_generated() -> None:
    graph = Graph(Log).flow(START >> pre, pre >> END)

    first = await graph.ainvoke({})
    second = await graph.ainvoke({})

    assert first.thread_id
    assert second.thread_id
    assert first.thread_id != second.thread_id


async def test_none_input_on_a_paused_thread_raises_and_runs_nothing() -> None:
    graph = _paused_graph()
    await graph.ainvoke({"log": ["turn1"]}, thread_id="i")
    events_before = len((await graph.history("i")).events)

    with pytest.raises(
        ResumeError,
        match=r"thread 'i' is paused on \['ask:ok'\]; pass resume=Resume\(\.\.\.\)",
    ):
        await graph.ainvoke(None, thread_id="i")

    assert calls == {"pre": 1, "ask": 1}
    assert len((await graph.history("i")).events) == events_before
    final = await graph.ainvoke(None, thread_id="i", resume=Resume("yes"))
    assert final.data["log"] == ["turn1", "pre", "answer=yes"]


async def test_new_input_on_a_paused_thread_raises_and_keeps_the_pause() -> None:
    graph = _paused_graph()
    await graph.ainvoke({"log": ["turn1"]}, thread_id="p")
    events_before = len((await graph.history("p")).events)

    with pytest.raises(
        ResumeError,
        match=(
            r"^thread 'p' is paused on \['ask:ok'\]; pass resume=Resume\(\.\.\.\), "
            r"or fork to start over$"
        ),
    ):
        await graph.ainvoke({"log": ["turn2"]}, thread_id="p")

    assert calls == {"pre": 1, "ask": 1}
    assert len((await graph.history("p")).events) == events_before
    final = await graph.ainvoke(None, thread_id="p", resume=Resume("yes"))
    assert final.data["log"] == ["turn1", "pre", "answer=yes"]


async def test_none_input_on_a_completed_thread_raises_and_runs_nothing() -> None:
    graph = _paused_graph()
    await graph.ainvoke({"log": ["t"]}, thread_id="c")
    await graph.ainvoke(None, thread_id="c", resume=Resume("yes"))
    calls.clear()

    with pytest.raises(
        ResumeError,
        match=r"thread 'c' has no unfinished run; pass input to start a new turn",
    ):
        await graph.ainvoke(None, thread_id="c")

    assert calls == {}
    assert (await graph.load("c"))["log"] == ["t", "pre", "answer=yes"]


async def test_none_input_on_an_unknown_thread_raises() -> None:
    store = InMemoryStateStore()
    graph = Graph(Log, state_store=store).flow(START >> pre, pre >> END)

    with pytest.raises(UnknownThreadError, match="Thread 'k' does not exist") as info:
        await graph.ainvoke(None, thread_id="k")

    assert info.value.thread_id == "k"
    assert calls == {}
    assert store.events == []


async def test_none_input_on_an_unknown_branch_raises() -> None:
    graph = Graph(Log, state_store=InMemoryStateStore()).flow(START >> pre, pre >> END)
    await graph.ainvoke({}, thread_id="t")

    with pytest.raises(UnknownThreadError, match="Branch 'typo' of thread 't'"):
        await graph.ainvoke(None, thread_id="t", branch_id="typo")


async def test_resume_on_an_unknown_thread_raises() -> None:
    graph = _paused_graph()

    with pytest.raises(UnknownThreadError):
        await graph.ainvoke(None, thread_id="k", resume=Resume("yes"))


async def test_a_branch_without_a_store_raises() -> None:
    graph = Graph(Log).flow(START >> pre, pre >> END)

    with pytest.raises(GraphConfigError, match="branch_id='typo' needs a state store"):
        await graph.ainvoke({"log": []}, branch_id="typo")

    assert calls == {}


async def test_none_input_without_a_store_raises() -> None:
    graph = Graph(Log).flow(START >> pre, pre >> END)

    with pytest.raises(GraphConfigError, match="without a state_store; pass input"):
        await graph.ainvoke(None)

    assert calls == {}


async def test_continuation_after_max_steps_does_not_rerun_finished_steps() -> None:
    graph = _linear_graph(max_steps=2)

    with pytest.raises(RunLimitExceededError):
        await graph.ainvoke({"log": []}, thread_id="lin")
    assert (await graph.load("lin"))["log"] == ["a", "b"]

    graph.max_steps = 25
    final = await graph.ainvoke(None, thread_id="lin")

    assert final.data["log"] == ["a", "b", "c", "d"]
    assert calls == {"a": 1, "b": 1, "c": 1, "d": 1}


async def test_continuation_after_a_break_between_steps_runs_only_what_is_left() -> (
    None
):
    graph = _linear_graph()

    async for event in graph.astream(
        {"log": []}, thread_id="brk", stream_mode="updates"
    ):
        if event.node == "b":
            break
    assert (await graph.load("brk"))["log"] == ["a", "b"]

    final = await graph.ainvoke(None, thread_id="brk")

    assert final.data["log"] == ["a", "b", "c", "d"]
    assert calls == {"a": 1, "b": 1, "c": 1, "d": 1}


async def test_break_after_the_last_step_leaves_nothing_to_continue() -> None:
    graph = Graph(Log, state_store=InMemoryStateStore()).flow(START >> pre, pre >> END)

    async for event in graph.astream(
        {"log": []}, thread_id="brk", stream_mode="updates"
    ):
        if event.mode == "updates":
            break

    with pytest.raises(ResumeError, match="no unfinished run"):
        await graph.ainvoke(None, thread_id="brk")
    assert calls == {"pre": 1}
    assert (await graph.load("brk"))["log"] == ["pre"]


@node
async def waits_the_first_time(state: Log, ctx: NodeContext) -> dict:
    calls["waits"] += 1
    ctx.emit("working")
    if calls["waits"] == 1:
        await asyncio.sleep(5)
    return {"log": ["waited"]}


async def test_break_inside_a_step_continues_that_step() -> None:
    graph = Graph(Log, state_store=InMemoryStateStore()).flow(
        START >> step_a, step_a >> waits_the_first_time, waits_the_first_time >> END
    )

    async for event in graph.astream(
        {"log": []}, thread_id="mid", stream_mode="custom"
    ):
        if event.mode == "custom":
            break
    final = await graph.ainvoke(None, thread_id="mid")

    assert final.data["log"] == ["a", "waited"]
    assert calls == {"a": 1, "waits": 2}


async def test_continuation_after_a_graph_timeout_inside_a_step() -> None:
    graph = Graph(Log, state_store=InMemoryStateStore(), timeout=0.2).flow(
        START >> step_a, step_a >> slow_b, slow_b >> step_c, step_c >> END
    )

    with pytest.raises(GraphTimeoutError):
        await graph.ainvoke({"log": []}, thread_id="tmo")
    graph.timeout = None
    final = await graph.ainvoke(None, thread_id="tmo")

    assert final.data["log"] == ["a", "b", "c"]
    assert calls == {"a": 1, "b": 2, "c": 1}


async def test_every_committed_step_is_marked_completed() -> None:
    graph = _linear_graph()

    await graph.ainvoke({"log": []}, thread_id="m")

    markers = [
        (event.type, json.loads(event.data_json or "{}").get("step"))
        for event in (await graph.history("m")).events
        if event.type in ("superstep_started", "superstep_completed")
    ]
    assert markers == [
        ("superstep_started", 0),
        ("superstep_completed", 0),
        ("superstep_started", 1),
        ("superstep_completed", 1),
        ("superstep_started", 2),
        ("superstep_completed", 2),
        ("superstep_started", 3),
        ("superstep_completed", 3),
    ]


async def test_new_input_after_max_steps_starts_a_new_turn() -> None:
    graph = _linear_graph(max_steps=1)
    with pytest.raises(RunLimitExceededError):
        await graph.ainvoke({"log": []}, thread_id="n")
    graph.max_steps = 25

    final = await graph.ainvoke({"log": ["again"]}, thread_id="n")

    assert final.data["log"] == ["a", "again", "a", "b", "c", "d"]


async def test_command_routes_are_continued_after_max_steps() -> None:
    @node(goto=[step_c])
    def jumps(state: Log) -> Command:
        calls["jumps"] += 1
        return Command(goto=step_c)

    graph = Graph(Log, state_store=InMemoryStateStore(), max_steps=1).flow(
        START >> jumps, step_c >> END
    )
    with pytest.raises(RunLimitExceededError):
        await graph.ainvoke({"log": []}, thread_id="j")
    graph.max_steps = 25

    final = await graph.ainvoke(None, thread_id="j")

    assert final.data["log"] == ["c"]
    assert calls == {"jumps": 1, "c": 1}


@node
async def fast(state: Log) -> dict:
    calls["fast"] += 1
    return {"log": ["fast"]}


@node
async def slow_first_time(state: Log) -> dict:
    calls["slow"] += 1
    if calls["slow"] == 1:
        await asyncio.sleep(5)
    return {"log": ["slow"]}


@node
async def flaky(state: Log) -> dict:
    calls["flaky"] += 1
    await asyncio.sleep(0.05)
    if calls["flaky"] == 1:
        raise RuntimeError("flaky")
    return {"log": ["flaky"]}


@node
async def emits_then_waits(state: Log, ctx: NodeContext) -> dict:
    calls["emits"] += 1
    await asyncio.sleep(0.05)
    ctx.emit("working")
    if calls["emits"] == 1:
        await asyncio.sleep(5)
    return {"log": ["emits"]}


@node(goto=[step_c])
async def fast_jump(state: Log) -> Command:
    calls["fast_jump"] += 1
    return Command(update={"log": ["jump"]}, goto=step_c)


@node(goto=[fast, slow_first_time])
def fan_slow(state: Log) -> Command:
    return Command(goto=[fast, slow_first_time])


@node(goto=[fast, flaky])
def fan_flaky(state: Log) -> Command:
    return Command(goto=[fast, flaky])


@node(goto=[fast, emits_then_waits])
def fan_emits(state: Log) -> Command:
    return Command(goto=[fast, emits_then_waits])


@node(goto=[fast_jump, slow_first_time])
def fan_jump(state: Log) -> Command:
    return Command(goto=[fast_jump, slow_first_time])


async def test_continuation_after_a_timeout_does_not_rerun_finished_siblings() -> None:
    graph = Graph(Log, state_store=InMemoryStateStore(), timeout=0.2).flow(
        START >> fan_slow, fast >> END, slow_first_time >> END
    )
    with pytest.raises(GraphTimeoutError):
        await graph.ainvoke({"log": []}, thread_id="par")
    graph.timeout = None

    final = await graph.ainvoke(None, thread_id="par")

    assert final.data["log"] == ["fast", "slow"]
    assert calls == {"fast": 1, "slow": 2}


async def test_continuation_keeps_the_command_routes_of_finished_siblings() -> None:
    graph = Graph(Log, state_store=InMemoryStateStore(), timeout=0.2).flow(
        START >> fan_jump, step_c >> END, slow_first_time >> END
    )
    with pytest.raises(GraphTimeoutError):
        await graph.ainvoke({"log": []}, thread_id="jmp")
    graph.timeout = None

    final = await graph.ainvoke(None, thread_id="jmp")

    assert final.data["log"] == ["jump", "slow", "c"]
    assert calls == {"fast_jump": 1, "slow": 2, "c": 1}


async def test_continuation_after_a_failure_does_not_rerun_finished_siblings() -> None:
    graph = Graph(Log, state_store=InMemoryStateStore()).flow(
        START >> fan_flaky, fast >> END, flaky >> END
    )
    with pytest.raises(RuntimeError, match="flaky"):
        await graph.ainvoke({"log": []}, thread_id="fl")

    final = await graph.ainvoke(None, thread_id="fl")

    assert final.data["log"] == ["fast", "flaky"]
    assert calls == {"fast": 1, "flaky": 2}


async def test_continuation_after_a_break_does_not_rerun_finished_siblings() -> None:
    graph = Graph(Log, state_store=InMemoryStateStore()).flow(
        START >> fan_emits, fast >> END, emits_then_waits >> END
    )
    async for event in graph.astream(
        {"log": []}, thread_id="pbrk", stream_mode="custom"
    ):
        if event.mode == "custom":
            break

    final = await graph.ainvoke(None, thread_id="pbrk")

    assert final.data["log"] == ["emits", "fast"]
    assert calls == {"fast": 1, "emits": 2}


class _FailsOnce(InMemoryStateStore):
    def __init__(
        self, event_type: str, node_name: str | None = None, *, skip: int = 0
    ) -> None:
        super().__init__()
        self.armed = True
        self.event_type = event_type
        self.node_name = node_name
        self.skip = skip

    async def append_next_event(self, event: HistoryEvent) -> HistoryEvent:
        if (
            self.armed
            and event.type == self.event_type
            and self.node_name in (None, event.node)
        ):
            if self.skip:
                self.skip -= 1
            else:
                self.armed = False
                raise StateStoreError("disk full")
        return await super().append_next_event(event)


async def test_committed_writes_without_a_step_marker_are_not_applied_twice() -> None:
    store = _FailsOnce("superstep_completed")
    graph = Graph(Log, state_store=store).flow(
        START >> step_a, step_a >> step_b, step_b >> END
    )
    with pytest.raises(StateStoreError, match="disk full"):
        await graph.ainvoke({"log": []}, thread_id="atom")
    assert (await graph.load("atom"))["log"] == ["a"]

    final = await graph.ainvoke(None, thread_id="atom")

    assert final.data["log"] == ["a", "b"]
    assert calls == {"a": 1, "b": 1}


@node(goto=[step_c, step_d])
def fan_c_d(state: Log) -> Command:
    return Command(goto=[step_c, step_d])


async def test_a_partly_committed_parallel_step_commits_only_the_rest() -> None:
    store = _FailsOnce("state_delta", node_name="d")
    graph = Graph(Log, state_store=store).flow(
        START >> fan_c_d, step_c >> END, step_d >> END
    )
    with pytest.raises(StateStoreError, match="disk full"):
        await graph.ainvoke({"log": []}, thread_id="part")
    assert (await graph.load("part"))["log"] == ["c"]

    final = await graph.ainvoke(None, thread_id="part")

    assert final.data["log"] == ["c", "d"]
    assert (await graph.load("part"))["log"] == ["c", "d"]
    assert calls == {"c": 1, "d": 1}


async def test_resume_on_an_unfinished_thread_says_how_to_continue() -> None:
    graph = _linear_graph(max_steps=1)
    with pytest.raises(RunLimitExceededError):
        await graph.ainvoke({"log": []}, thread_id="u")

    with pytest.raises(
        ResumeError,
        match=(
            r"^thread 'u' has no pending interrupts; continue its unfinished "
            r"superstep with ainvoke\(None, thread_id='u'\) without resume=$"
        ),
    ):
        await graph.ainvoke(None, thread_id="u", resume=Resume("x"))


async def test_resume_on_a_completed_thread_says_to_pass_input() -> None:
    graph = Graph(Log, state_store=InMemoryStateStore()).flow(START >> pre, pre >> END)
    await graph.ainvoke({"log": []}, thread_id="done")

    with pytest.raises(
        ResumeError,
        match=(
            r"^thread 'done' has no pending interrupts; pass input to start a new "
            r"turn$"
        ),
    ):
        await graph.ainvoke(None, thread_id="done", resume=Resume("x"))


@node
async def approves_slowly(state: Log, ctx: NodeContext) -> dict:
    calls["approves"] += 1
    if "quote" not in ctx.cache:
        calls["quote"] += 1
        ctx.cache["quote"] = 42
    answer = interrupt("go?", id="go")
    ctx.emit("working")
    if calls["approves"] == 2:
        await asyncio.sleep(5)
    return {"log": [f"{answer}@{ctx.cache['quote']}"]}


async def test_continuing_a_cut_off_resumed_step_keeps_its_answers() -> None:
    graph = Graph(Log, state_store=InMemoryStateStore()).flow(
        START >> approves_slowly, approves_slowly >> END
    )
    await graph.ainvoke({"log": []}, thread_id="rs")
    async for event in graph.astream(
        None, thread_id="rs", resume=Resume("approved"), stream_mode="custom"
    ):
        if event.mode == "custom":
            break

    final = await graph.ainvoke(None, thread_id="rs")

    assert final.status == "completed"
    assert final.data["log"] == ["approved@42"]
    assert calls == {"approves": 3, "quote": 1}


async def test_a_lost_step_start_leaves_a_continuable_thread() -> None:
    store = _FailsOnce("superstep_started", skip=1)
    graph = Graph(Log, state_store=store).flow(
        START >> step_a, step_a >> step_b, step_b >> step_c, step_c >> END
    )
    with pytest.raises(StateStoreError, match="disk full"):
        await graph.ainvoke({"log": []}, thread_id="gap")
    assert (await graph.load("gap"))["log"] == ["a"]

    final = await graph.ainvoke(None, thread_id="gap")

    assert final.data["log"] == ["a", "b", "c"]
    assert calls == {"a": 1, "b": 1, "c": 1}


class Item(Log, total=False):
    item: str


@node
def collects(state: Item) -> dict:
    return {"log": [f"got {state['item']}"]}


@node(goto=[collects])
def sends_items(state: Item) -> Command:
    return Command(goto=[Send(collects, {"item": "x"}), Send(collects, {"item": "y"})])


async def test_a_lost_step_start_keeps_the_send_payloads() -> None:
    store = _FailsOnce("superstep_started", skip=1)
    graph = Graph(Item, state_store=store).flow(START >> sends_items, collects >> END)
    with pytest.raises(StateStoreError, match="disk full"):
        await graph.ainvoke({"log": []}, thread_id="gap")

    final = await graph.ainvoke(None, thread_id="gap")

    assert final.data["log"] == ["got x", "got y"]


class Claim(TypedDict, total=False):
    owner: str


@node
def claims_left(state: Claim) -> dict:
    calls["left"] += 1
    return {"owner": "left"}


@node
def claims_right(state: Claim) -> dict:
    calls["right"] += 1
    return {"owner": "right"}


@node(goto=[claims_left, claims_right])
def splits(state: Claim) -> Command:
    return Command(goto=[claims_left, claims_right])


async def test_a_step_whose_writes_cannot_be_merged_is_not_continued() -> None:
    graph = Graph(Claim, state_store=InMemoryStateStore()).flow(
        START >> splits, claims_left >> END, claims_right >> END
    )
    with pytest.raises(InvalidUpdateError, match="'owner'"):
        await graph.ainvoke({}, thread_id="m")
    calls.clear()

    with pytest.raises(
        ResumeError,
        match=(
            r"^thread 'm' cannot continue: the writes of its last superstep cannot "
            r"be merged \(InvalidUpdateError: Conflicting .*\); pass input to start "
            r"a new turn, or fork the thread at an earlier event$"
        ),
    ):
        await graph.ainvoke(None, thread_id="m")
    with pytest.raises(ResumeError, match="pass input to start a new turn"):
        await graph.ainvoke(None, thread_id="m", resume=Resume("x"))

    assert calls == {}


async def test_a_fork_of_a_step_whose_writes_cannot_be_merged_keeps_the_reason() -> (
    None
):
    graph = Graph(Claim, state_store=InMemoryStateStore()).flow(
        START >> splits, claims_left >> END, claims_right >> END
    )
    with pytest.raises(InvalidUpdateError):
        await graph.ainvoke({}, thread_id="m")
    failed = next(
        event
        for event in (await graph.history("m")).events
        if event.type == "superstep_failed"
    )
    fork = await graph.fork("m", from_=failed.id)

    with pytest.raises(
        ResumeError, match=r"cannot be merged \(InvalidUpdateError: Conflicting "
    ):
        await graph.ainvoke(None, thread_id="m", branch_id=fork.id)


class Owned(TypedDict, total=False):
    owner: str
    log: Annotated[list[str], add]


@node
def owns(state: Owned) -> dict:
    return {"owner": "node"}


@node
def reads_owner(state: Owned) -> dict:
    return {"log": [f"owner={state['owner']}"]}


async def test_a_manual_update_after_a_committed_write_of_its_field_is_kept() -> None:
    store = _FailsOnce("superstep_completed")
    graph = Graph(Owned, state_store=store).flow(
        START >> owns, owns >> reads_owner, reads_owner >> END
    )
    with pytest.raises(StateStoreError, match="disk full"):
        await graph.ainvoke({}, thread_id="gap")
    await graph.update_state("gap", {"owner": "human"})

    final = await graph.ainvoke(None, thread_id="gap")

    assert final.data["owner"] == "human"
    assert final.data["log"] == ["owner=human"]


@node
def asks_owner(state: Owned) -> dict:
    return {"log": [f"asked {interrupt('?', id='q')}"]}


async def test_a_manual_update_while_paused_ends_with_its_step_after_a_lost_start() -> (
    None
):
    store = _FailsOnce("superstep_started")
    store.armed = False
    graph = Graph(Owned, state_store=store).flow(
        START >> asks_owner, asks_owner >> owns, owns >> END
    )
    await graph.ainvoke({}, thread_id="gap")
    await graph.update_state("gap", {"owner": "human"})
    store.armed, store.skip = True, 1
    with pytest.raises(StateStoreError, match="disk full"):
        await graph.ainvoke(None, thread_id="gap", resume=Resume("ok"))

    final = await graph.ainvoke(None, thread_id="gap")

    assert final.data["owner"] == "node"
    assert final.data["log"] == ["asked ok"]
