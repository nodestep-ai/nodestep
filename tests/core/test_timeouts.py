import asyncio
import time
from typing import Annotated, TypedDict

import pytest

from nodestep import END, START, Graph, Middleware, add, node
from nodestep.exceptions import GraphTimeoutError, NodeTimeoutError


class Log(TypedDict, total=False):
    log: Annotated[list[str], add]


received: list[BaseException] = []


class Captures(Middleware):
    def on_error(self, ctx, error):
        received.append(error)


@pytest.fixture(autouse=True)
def _reset() -> None:
    received.clear()


@node
def db_call(state: Log) -> dict:
    raise TimeoutError("database did not answer within 30s")


@node(timeout=5)
async def db_call_with_budget(state: Log) -> dict:
    raise TimeoutError("database did not answer within 30s")


@pytest.mark.parametrize("handler", [db_call, db_call_with_budget])
async def test_a_foreign_timeout_error_propagates_unchanged(handler) -> None:
    graph = Graph(Log, middleware=[Captures()]).flow(START >> handler, handler >> END)

    with pytest.raises(TimeoutError, match="database did not answer") as info:
        await graph.ainvoke({})

    assert type(info.value) is TimeoutError
    assert [type(error) for error in received] == [TimeoutError]


@node(timeout=0.05)
async def too_slow(state: Log) -> dict:
    await asyncio.sleep(5)
    return {"log": ["never"]}


async def test_the_node_timeout_becomes_node_timeout_error_with_its_cause() -> None:
    graph = Graph(Log, middleware=[Captures()]).flow(START >> too_slow, too_slow >> END)

    with pytest.raises(
        NodeTimeoutError, match=r"'too_slow' timed out after 0\.05s"
    ) as info:
        await graph.ainvoke({})

    assert isinstance(info.value.__cause__, TimeoutError)
    assert received == [info.value]


@node
def instant_a(state: Log) -> dict:
    return {"log": ["a"]}


@node
def instant_b(state: Log) -> dict:
    return {"log": ["b"]}


def _instant_graph() -> Graph:
    return Graph(Log, timeout=0.1).flow(
        START >> instant_a, instant_a >> instant_b, instant_b >> END
    )


async def test_graph_timeout_ignores_time_the_async_consumer_spends() -> None:
    modes = []
    async for event in _instant_graph().astream({}, stream_mode="updates"):
        modes.append(event.mode)
        await asyncio.sleep(0.15)

    assert modes == ["updates", "updates", "final"]


def test_graph_timeout_ignores_time_the_sync_consumer_spends() -> None:
    modes = []
    for event in _instant_graph().stream({}, stream_mode="updates"):
        modes.append(event.mode)
        time.sleep(0.15)

    assert modes == ["updates", "updates", "final"]


def _sleeper(name: str):
    async def run(state: Log) -> dict:
        await asyncio.sleep(0.1)
        return {"log": [name]}

    return node(run, name=name)


async def test_graph_timeout_adds_up_the_time_of_every_step() -> None:
    first, second, third = _sleeper("first"), _sleeper("second"), _sleeper("third")
    graph = Graph(Log, timeout=0.25).flow(
        START >> first, first >> second, second >> third, third >> END
    )

    with pytest.raises(GraphTimeoutError, match=r"timed out after 0\.25s"):
        await graph.ainvoke({})
