import asyncio

import pytest
from pydantic import ValidationError

from nodestep import END, START, BaseState, Graph, InMemoryStateStore, node
from nodestep.exceptions import StateUpdateError
from nodestep.state.schema import StateSnapshot


class CounterState(BaseState):
    count: int = 0
    note: str = ""


@node
def increment(state: CounterState) -> dict:
    return {"count": state.count + 1}


@pytest.mark.asyncio
async def test_get_state_returns_snapshot() -> None:
    store = InMemoryStateStore()
    graph = Graph(CounterState, state_store=store).flow(
        START >> increment, increment >> END
    )
    await graph.ainvoke({"count": 0}, thread_id="t1")

    snap = await graph.get_state("t1")
    assert isinstance(snap, StateSnapshot)
    assert snap.value.count == 1


@pytest.mark.asyncio
async def test_update_state_appends_delta_and_is_visible() -> None:
    store = InMemoryStateStore()
    graph = Graph(CounterState, state_store=store).flow(
        START >> increment, increment >> END
    )
    await graph.ainvoke({"count": 0}, thread_id="t2")

    event_id = await graph.update_state("t2", {"note": "manual"})
    assert isinstance(event_id, str)
    assert event_id

    snap = await graph.get_state("t2")
    assert snap.value.note == "manual"
    assert snap.value.count == 1


@pytest.mark.asyncio
async def test_update_state_rejects_unknown_field() -> None:
    store = InMemoryStateStore()
    graph = Graph(CounterState, state_store=store).flow(
        START >> increment, increment >> END
    )
    await graph.ainvoke({"count": 0}, thread_id="t-unknown")

    with pytest.raises(StateUpdateError, match="Unknown field"):
        await graph.update_state("t-unknown", {"bogus": 1})

    snap = await graph.get_state("t-unknown")
    assert snap.value.count == 1


@pytest.mark.asyncio
async def test_update_state_rejects_invalid_value() -> None:
    store = InMemoryStateStore()
    graph = Graph(CounterState, state_store=store).flow(
        START >> increment, increment >> END
    )
    await graph.ainvoke({"count": 0}, thread_id="t-invalid")

    with pytest.raises(StateUpdateError, match="'count'") as info:
        await graph.update_state("t-invalid", {"count": "not-a-number"})

    assert isinstance(info.value.__cause__, ValidationError)

    snap = await graph.get_state("t-invalid")
    assert snap.value.count == 1


@node
async def slow_inc(state: CounterState) -> dict:
    await asyncio.sleep(0.05)
    return {"count": state.count + 1}


@pytest.mark.asyncio
async def test_update_state_during_run_keeps_sequences_unique() -> None:
    store = InMemoryStateStore()
    graph = Graph(CounterState, state_store=store).flow(
        START >> slow_inc, slow_inc >> END
    )

    run = asyncio.ensure_future(graph.ainvoke({"count": 0}, thread_id="t-conc"))
    await asyncio.sleep(0.01)
    await graph.update_state("t-conc", {"note": "manual"})
    await run

    history = await graph.history("t-conc")
    sequences = [event.sequence for event in history.events]
    assert len(sequences) == len(set(sequences))

    snap = await graph.get_state("t-conc")
    assert snap.value.count == 1


@pytest.mark.asyncio
async def test_get_state_requires_store() -> None:
    from nodestep.exceptions import GraphConfigError

    graph = Graph(CounterState).flow(START >> increment, increment >> END)
    with pytest.raises(GraphConfigError, match="no state_store"):
        await graph.get_state("missing")
