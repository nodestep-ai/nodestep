import asyncio
import dataclasses
import inspect
import re
from typing import Annotated, Any, TypedDict, cast

import pytest

from nodestep import (
    END,
    START,
    Command,
    Graph,
    InMemoryStateStore,
    Interrupt,
    Middleware,
    Resume,
    Send,
    add,
    interrupt,
    node,
)
from nodestep.core.command import GraphInterrupt, GraphInterruptGroup
from nodestep.core.stream import InterruptEventData
from nodestep.exceptions import GraphExecutionError, ResumeError


class S(TypedDict, total=False):
    log: Annotated[list[str], add]
    mode: str


side_effects: list[str] = []


@node
def asks_approval(state: S) -> dict:
    return {"log": [f"answer={interrupt('approve?', id='approve')}"]}


def _single(store: InMemoryStateStore | None = None) -> Graph:
    return Graph(S, state_store=store or InMemoryStateStore()).flow(
        START >> asks_approval, asks_approval >> END
    )


def test_interrupt_takes_a_required_keyword_id() -> None:
    parameter = inspect.signature(interrupt).parameters["id"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


async def test_interrupt_key_is_the_task_id_and_the_id() -> None:
    result = await _single().ainvoke({}, thread_id="t")

    assert result.interrupts == {
        "asks_approval:approve": Interrupt(
            key="asks_approval:approve",
            id="approve",
            node="asks_approval",
            task_id="asks_approval",
            payload="approve?",
        )
    }


async def test_interrupt_event_carries_the_interrupt_records() -> None:
    events = [
        event async for event in _single().astream({}, thread_id="t", stream_mode=[])
    ]

    data = events[-1].data
    assert isinstance(data, InterruptEventData)
    assert list(data.interrupts) == ["asks_approval:approve"]
    assert data.interrupts["asks_approval:approve"].payload == "approve?"


@pytest.mark.parametrize("bad", ["", "has space", "a:b", "a/b", "ümlaut"])
async def test_interrupt_id_must_match_the_allowed_characters(bad: str) -> None:
    @node
    def asks(state: S) -> dict:
        interrupt("x", id=bad)
        return {}

    graph = Graph(S, state_store=InMemoryStateStore()).flow(START >> asks, asks >> END)

    with pytest.raises(ValueError, match=re.escape("[A-Za-z0-9_.-]+")):
        await graph.ainvoke({}, thread_id="t")


@node
def asks_twice_with_one_id(state: S) -> dict:
    first = interrupt("first?", id="same")
    second = interrupt("second?", id="same")
    return {"log": [f"{first}{second}"]}


async def test_an_id_repeated_within_one_node_run_raises() -> None:
    graph = Graph(S, state_store=InMemoryStateStore()).flow(
        START >> asks_twice_with_one_id, asks_twice_with_one_id >> END
    )
    await graph.ainvoke({}, thread_id="t")

    with pytest.raises(GraphExecutionError, match="'same'"):
        await graph.ainvoke(None, thread_id="t", resume=Resume("yes"))


@node
def refund(state: S) -> dict:
    return {"log": [f"refund={interrupt('approve $10 refund?', id='approve')}"]}


@node
def delete(state: S) -> dict:
    return {"log": [f"delete={interrupt('DELETE the account?', id='approve')}"]}


@node(goto=[refund, delete])
def fan(state: S) -> Command:
    return Command(goto=[refund, delete])


def _two_pending() -> Graph:
    return Graph(S, state_store=InMemoryStateStore()).flow(
        START >> fan, refund >> END, delete >> END
    )


async def _event_count(graph: Graph, thread_id: str) -> int:
    return len((await graph.history(thread_id)).events)


async def test_one_value_cannot_answer_two_pending_interrupts() -> None:
    graph = _two_pending()
    paused = await graph.ainvoke({}, thread_id="t")
    before = await _event_count(graph, "t")

    with pytest.raises(ResumeError, match=r"delete:approve.*refund:approve"):
        await graph.ainvoke(None, thread_id="t", resume=Resume(True))

    assert sorted(paused.interrupts) == ["delete:approve", "refund:approve"]
    assert await _event_count(graph, "t") == before


async def test_answers_by_key_resume_every_pending_interrupt() -> None:
    graph = _two_pending()
    await graph.ainvoke({}, thread_id="t")

    final = await graph.ainvoke(
        None,
        thread_id="t",
        resume=Resume(answers={"refund:approve": True, "delete:approve": False}),
    )

    assert final.status == "completed"
    assert sorted(final.data["log"]) == ["delete=False", "refund=True"]


async def test_every_pending_interrupt_must_be_answered() -> None:
    graph = _two_pending()
    await graph.ainvoke({}, thread_id="t")

    with pytest.raises(ResumeError, match="delete:approve"):
        await graph.ainvoke(
            None, thread_id="t", resume=Resume(answers={"refund:approve": True})
        )


async def test_an_answer_for_an_unknown_key_raises() -> None:
    graph = _two_pending()
    await graph.ainvoke({}, thread_id="t")

    with pytest.raises(ResumeError, match="nope:x"):
        await graph.ainvoke(
            None,
            thread_id="t",
            resume=Resume(
                answers={"refund:approve": 1, "delete:approve": 2, "nope:x": 3}
            ),
        )


async def test_a_single_pending_interrupt_accepts_a_value_or_its_key() -> None:
    by_value = _single()
    by_key = _single()
    await by_value.ainvoke({}, thread_id="t")
    await by_key.ainvoke({}, thread_id="t")

    first = await by_value.ainvoke(None, thread_id="t", resume=Resume("yes"))
    second = await by_key.ainvoke(
        None, thread_id="t", resume=Resume(answers={"asks_approval:approve": "yes"})
    )

    assert first.data["log"] == ["answer=yes"]
    assert second.data["log"] == ["answer=yes"]


def test_resume_needs_exactly_one_of_value_and_answers() -> None:
    with pytest.raises(ResumeError):
        Resume()  # ty: ignore[no-matching-overload]
    with pytest.raises(ResumeError):
        Resume(True, answers={"a:b": True})  # ty: ignore[no-matching-overload]


def test_resume_with_empty_answers_raises() -> None:
    with pytest.raises(ResumeError, match="answers"):
        Resume(answers={})


def test_resume_none_is_a_value() -> None:
    assert Resume(None).value is None
    assert Resume(None).answers is None


async def test_a_command_is_not_a_resume() -> None:
    graph = _single()
    await graph.ainvoke({}, thread_id="t")
    command: Any = Command(update={"mode": "x"})

    with pytest.raises(TypeError, match="Resume"):
        await graph.ainvoke(None, thread_id="t", resume=command)


def test_command_has_only_update_and_goto() -> None:
    assert [item.name for item in dataclasses.fields(Command)] == ["update", "goto"]


@node
def conditional_asks(state: S) -> dict:
    first = None
    if state.get("mode") == "strict":
        first = interrupt("A: approve refund?", id="refund")
    second = interrupt("B: delete account?", id="delete")
    return {"log": [f"A={first}", f"B={second}"]}


async def test_answers_follow_their_id_when_control_flow_changes() -> None:
    graph = Graph(S, state_store=InMemoryStateStore()).flow(
        START >> conditional_asks, conditional_asks >> END
    )
    paused = await graph.ainvoke({"mode": "strict"}, thread_id="t")
    await graph.update_state("t", {"mode": "lenient"})

    again = await graph.ainvoke(None, thread_id="t", resume=Resume("yes-to-A"))
    final = await graph.ainvoke(None, thread_id="t", resume=Resume("yes-to-B"))

    assert list(paused.interrupts) == ["conditional_asks:refund"]
    assert again.status == "interrupted"
    assert list(again.interrupts) == ["conditional_asks:delete"]
    assert final.data["log"] == ["A=None", "B=yes-to-B"]


@node
def charge_then_ask(state: S) -> dict:
    side_effects.append("charged card")
    return {"log": [f"answer={interrupt('confirm shipping?', id='ship')}"]}


@node
async def charge_then_ask_async(state: S) -> dict:
    side_effects.append("charged card")
    return {"log": [f"answer={interrupt('confirm shipping?', id='ship')}"]}


class AutoApprove(Middleware):
    def before_node(self, ctx):
        side_effects.append(f"before_node({ctx.node_name})")

    def on_interrupt(self, ctx, payload):
        side_effects.append(f"on_interrupt({payload})")
        return "auto-yes"


@pytest.mark.parametrize("asker", [charge_then_ask, charge_then_ask_async])
async def test_middleware_answer_lets_the_node_continue(asker) -> None:
    side_effects.clear()
    graph = Graph(S, middleware=[AutoApprove()]).flow(START >> asker, asker >> END)

    result = await graph.ainvoke({})

    assert result.data["log"] == ["answer=auto-yes"]
    assert side_effects == [
        f"before_node({asker.name})",
        "charged card",
        "on_interrupt(confirm shipping?)",
    ]


class AsyncAnswer(Middleware):
    async def on_interrupt(self, ctx, payload):
        return "late"


async def test_an_async_on_interrupt_hook_raises() -> None:
    graph = Graph(S, middleware=[AsyncAnswer()]).flow(
        START >> asks_approval, asks_approval >> END
    )

    with pytest.raises(TypeError, match="on_interrupt"):
        await graph.ainvoke({})


@node
def swallow(state: S) -> dict:
    try:
        answer = interrupt("need approval", id="gate")
    except Exception:
        answer = "fallback"
    return {"log": [f"answer={answer}"]}


async def test_except_exception_does_not_swallow_the_pause() -> None:
    graph = Graph(S, state_store=InMemoryStateStore()).flow(
        START >> swallow, swallow >> END
    )

    paused = await graph.ainvoke({}, thread_id="t")
    final = await graph.ainvoke(None, thread_id="t", resume=Resume("ok"))

    assert paused.status == "interrupted"
    assert list(paused.interrupts) == ["swallow:gate"]
    assert final.data["log"] == ["answer=ok"]


def test_interrupt_signals_are_not_exceptions() -> None:
    for signal in (GraphInterrupt, GraphInterruptGroup):
        assert issubclass(signal, BaseException)
        assert not issubclass(signal, Exception)


@node
async def cancels_itself(state: S) -> dict:
    raise asyncio.CancelledError


async def test_a_cancelled_node_is_not_recorded_as_an_interrupt() -> None:
    graph = Graph(S, state_store=InMemoryStateStore()).flow(
        START >> cancels_itself, cancels_itself >> END
    )

    with pytest.raises(asyncio.CancelledError):
        await graph.ainvoke({}, thread_id="t")

    history = await graph.history("t")
    assert [event for event in history.events if event.type == "interrupted"] == []


@node
async def slow_asker(state: S) -> dict:
    await asyncio.sleep(5)
    return {"log": [interrupt("never", id="never")]}


@node
def fails(state: S) -> dict:
    raise RuntimeError("boom")


@node(goto=[slow_asker, fails])
def fan_slow_and_failing(state: S) -> Command:
    return Command(goto=[Send(slow_asker, {}), Send(fails, {})])


async def test_a_sibling_cancelled_by_a_failure_is_not_an_interrupt() -> None:
    graph = Graph(S, state_store=InMemoryStateStore()).flow(
        START >> fan_slow_and_failing, slow_asker >> END, fails >> END
    )

    with pytest.raises(RuntimeError, match="boom"):
        await graph.ainvoke({}, thread_id="t")

    history = await graph.history("t")
    assert [event for event in history.events if event.type == "interrupted"] == []


class AsksOnError(Middleware):
    def on_error(self, ctx, error):
        answer = interrupt(f"retry after {error}?", id="retry")
        return {"log": [f"recovered={answer}"]}


async def test_an_interrupt_in_on_error_pauses_the_run() -> None:
    graph = Graph(S, state_store=InMemoryStateStore(), middleware=[AsksOnError()]).flow(
        START >> fails, fails >> END
    )

    paused = await graph.ainvoke({}, thread_id="e")
    final = await graph.ainvoke(None, thread_id="e", resume=Resume("yes"))

    assert paused.status == "interrupted"
    assert list(paused.interrupts) == ["fails:retry"]
    assert paused.interrupts["fails:retry"].payload == "retry after boom?"
    assert final.data["log"] == ["recovered=yes"]


class RecoversThenAsks(Middleware):
    def on_error(self, ctx, error):
        return {"log": ["recovered"]}

    def after_node(self, ctx):
        side_effects.append(f"after_node={interrupt('keep it?', id='keep')}")


async def test_an_interrupt_in_after_node_during_recovery_pauses_the_run() -> None:
    side_effects.clear()
    graph = Graph(
        S, state_store=InMemoryStateStore(), middleware=[RecoversThenAsks()]
    ).flow(START >> fails, fails >> END)

    paused = await graph.ainvoke({}, thread_id="a")
    final = await graph.ainvoke(None, thread_id="a", resume=Resume("yes"))

    assert list(paused.interrupts) == ["fails:keep"]
    assert final.data["log"] == ["recovered"]
    assert side_effects == ["after_node=yes"]


def test_resume_answers_must_be_a_mapping() -> None:
    with pytest.raises(TypeError, match="answers= takes a mapping"):
        Resume(answers=cast(Any, [("a:b", 1)]))


class ReplacesInputThenAnswers(Middleware):
    def __init__(self) -> None:
        self.seen: list[Any] = []

    def before_node(self, ctx):
        return ctx.replace({**ctx.state, "mode": "replaced"})

    def on_interrupt(self, ctx, payload):
        self.seen.append(ctx.state.get("mode"))
        return "auto"


async def test_on_interrupt_sees_the_input_the_node_received() -> None:
    middleware = ReplacesInputThenAnswers()
    graph = Graph(S, middleware=[middleware]).flow(
        START >> asks_approval, asks_approval >> END
    )

    await graph.ainvoke({"mode": "original"})

    assert middleware.seen == ["replaced"]


@node
def picks(state: S) -> dict:
    if state["mode"] == "x":
        interrupt("pick x?", id="x")
    return {"mode": f"picked-{state['mode']}"}


@node(goto=[picks])
def sends_two_picks(state: S) -> Command:
    return Command(goto=[Send(picks, {"mode": "x"}), Send(picks, {"mode": "y"})])


async def test_a_resumed_step_keeps_last_wins_in_scheduling_order() -> None:
    graph = Graph(S, state_store=InMemoryStateStore()).flow(
        START >> sends_two_picks, picks >> END
    )

    await graph.ainvoke({}, thread_id="lw")
    final = await graph.ainvoke(None, thread_id="lw", resume=Resume("ok"))

    assert final.data["mode"] == "picked-y"


_INPUT_WITH_RESUME = r"^resume= answers pending interrupts; pass initial_state=None$"


async def test_resume_together_with_input_raises() -> None:
    graph = _single()
    await graph.ainvoke({}, thread_id="t")
    before = await _event_count(graph, "t")

    with pytest.raises(TypeError, match=_INPUT_WITH_RESUME):
        await graph.ainvoke({"mode": "x"}, thread_id="t", resume=Resume("yes"))
    with pytest.raises(TypeError, match=_INPUT_WITH_RESUME):
        [
            event
            async for event in graph.astream(
                {}, stream_mode=[], thread_id="t", resume=Resume("yes")
            )
        ]

    assert await _event_count(graph, "t") == before
    final = await graph.ainvoke(None, thread_id="t", resume=Resume("yes"))
    assert final.data == {"log": ["answer=yes"]}


def test_sync_resume_together_with_input_raises() -> None:
    graph = _single()
    graph.invoke({}, thread_id="t")

    with pytest.raises(TypeError, match=_INPUT_WITH_RESUME):
        graph.invoke({"mode": "x"}, thread_id="t", resume=Resume("yes"))
    with pytest.raises(TypeError, match=_INPUT_WITH_RESUME):
        list(graph.stream({}, stream_mode=[], thread_id="t", resume=Resume("yes")))
