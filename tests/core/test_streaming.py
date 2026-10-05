import asyncio
import time
from typing import Annotated, TypedDict

from nodestep import END, START, Graph, NodeContext, add, node


class S(TypedDict, total=False):
    log: Annotated[list[str], add]


@node
async def ticker(state: S, ctx: NodeContext) -> dict:
    for index in range(3):
        ctx.emit(f"chunk-{index}")
        await asyncio.sleep(0.1)
    return {}


async def test_emitted_events_arrive_while_the_node_runs() -> None:
    graph = Graph(S).flow(START >> ticker, ticker >> END)
    arrivals: list[float] = []
    started = time.monotonic()

    async for event in graph.astream({}, stream_mode="custom"):
        if event.mode == "custom":
            arrivals.append(time.monotonic() - started)

    assert len(arrivals) == 3
    assert arrivals[1] - arrivals[0] >= 0.08
    assert arrivals[2] - arrivals[1] >= 0.08


@node
def sync_ticker(state: S, ctx: NodeContext) -> dict:
    for index in range(2):
        ctx.emit(f"sync-{index}")
        time.sleep(0.1)
    return {}


async def test_sync_node_emits_live_too() -> None:
    graph = Graph(S).flow(START >> sync_ticker, sync_ticker >> END)
    arrivals: list[float] = []
    started = time.monotonic()

    async for event in graph.astream({}, stream_mode="custom"):
        if event.mode == "custom":
            arrivals.append(time.monotonic() - started)

    assert len(arrivals) == 2
    assert arrivals[1] - arrivals[0] >= 0.08


@node
def fast(state: S) -> dict:
    return {"log": ["fast"]}


@node
async def slow(state: S) -> dict:
    await asyncio.sleep(0.3)
    return {"log": ["slow"]}


def test_sync_stream_yields_before_the_graph_finishes() -> None:
    graph = Graph(S).flow(START >> fast, fast >> slow, slow >> END)
    started = time.monotonic()

    stream = graph.stream({}, stream_mode="updates")
    first = next(stream)
    first_at = time.monotonic() - started
    rest = list(stream)

    assert first.mode == "updates"
    assert first_at < 0.2
    assert rest[-1].mode == "final"


cancelled: list[bool] = []


@node
async def long_running(state: S, ctx: NodeContext) -> dict:
    ctx.emit("started")
    try:
        await asyncio.sleep(5)
    except asyncio.CancelledError:
        cancelled.append(True)
        raise
    return {}


async def test_closing_a_stream_cancels_the_running_step() -> None:
    cancelled.clear()
    graph = Graph(S).flow(START >> long_running, long_running >> END)

    events = graph.astream({}, stream_mode="custom")
    async for _ in events:
        break
    await events.aclose()

    assert cancelled == [True]
