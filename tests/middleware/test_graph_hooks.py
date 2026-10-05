from typing import Annotated, Any, TypedDict

import pytest

from nodestep import (
    END,
    START,
    Command,
    Graph,
    InMemoryStateStore,
    Middleware,
    Resume,
    add,
    interrupt,
    node,
)
from nodestep.exceptions import GraphExecutionError
from nodestep.middleware import GraphMiddlewareContext, Replacement

seen: list[Any] = []


class Tagged(TypedDict, total=False):
    log: Annotated[list[str], add]
    tag: str
    seen: list[str]


@node
def reads_tag(state: Tagged) -> dict:
    seen.append(("node", state.get("tag")))
    return {"log": [f"tag={state.get('tag')}"]}


@pytest.fixture(autouse=True)
def _reset_seen() -> None:
    seen.clear()


class ReplacesBefore(Middleware):
    def before_graph(self, ctx):
        return Replacement({**ctx.state, "tag": "hook"})


class ReplacesAfter(Middleware):
    def after_graph(self, ctx):
        return Replacement({**ctx.state, "tag": "hook"})


class ReplacesBeforeAsync(Middleware):
    async def before_graph(self, ctx):
        return Replacement({**ctx.state, "tag": "hook"})


@pytest.mark.parametrize(
    ("middleware", "hook"),
    [
        (ReplacesBefore(), "before_graph"),
        (ReplacesAfter(), "after_graph"),
        (ReplacesBeforeAsync(), "before_graph"),
    ],
)
async def test_graph_hook_replacements_raise(middleware: Middleware, hook: str) -> None:
    graph = Graph(Tagged, middleware=[middleware]).flow(
        START >> reads_tag, reads_tag >> END
    )

    with pytest.raises(GraphExecutionError, match=f"{hook}.*graph hooks are read-only"):
        await graph.ainvoke({"tag": "input"})


async def test_a_before_graph_replacement_runs_no_node() -> None:
    store = InMemoryStateStore()
    graph = Graph(Tagged, state_store=store, middleware=[ReplacesBefore()]).flow(
        START >> reads_tag, reads_tag >> END
    )

    with pytest.raises(GraphExecutionError):
        await graph.ainvoke({"tag": "input"}, thread_id="t")

    assert seen == []
    assert store.events == []


def test_graph_context_has_no_replace() -> None:
    assert not hasattr(GraphMiddlewareContext, "replace")


class MutatesInPlace(Middleware):
    def before_graph(self, ctx):
        ctx.state["log"].append("before_graph")
        ctx.state["tag"] = "before_graph"

    def after_graph(self, ctx):
        ctx.state["log"].append("after_graph")


async def test_graph_hooks_get_a_copy_of_the_state() -> None:
    graph = Graph(
        Tagged, state_store=InMemoryStateStore(), middleware=[MutatesInPlace()]
    ).flow(START >> reads_tag, reads_tag >> END)

    result = await graph.ainvoke({"log": [], "tag": "input"}, thread_id="t")

    assert seen == [("node", "input")]
    assert result.data == {"log": ["tag=input"], "tag": "input"}
    assert await graph.load("t") == result.data


class Records(Middleware):
    def before_graph(self, ctx):
        seen.append(("before_graph", ctx.resuming, ctx.state.get("log")))

    def after_graph(self, ctx):
        seen.append(("after_graph", ctx.resuming, ctx.state.get("log")))


@node
def asks(state: Tagged) -> dict:
    seen.append(("asks", state.get("log")))
    return {"log": [f"answer={interrupt('go?', id='go')}"]}


async def test_before_graph_runs_before_resumed_tasks() -> None:
    graph = Graph(
        Tagged, state_store=InMemoryStateStore(), middleware=[Records()]
    ).flow(START >> asks, asks >> END)
    await graph.ainvoke({"log": ["first"]}, thread_id="r")
    seen.clear()

    await graph.ainvoke(None, thread_id="r", resume=Resume("yes"))

    assert seen == [
        ("before_graph", True, ["first"]),
        ("asks", ["first"]),
        ("after_graph", True, ["first", "answer=yes"]),
    ]


class Peek(Middleware):
    def before_node(self, ctx):
        ctx.state["seen"].append(f"peeked by {ctx.node_name}")


@node
def left(state: Tagged) -> dict:
    return {"log": ["left"]}


@node
def right(state: Tagged) -> dict:
    return {"log": [f"right saw {state['seen']}"]}


@node(goto=[left, right])
def fan(state: Tagged) -> Command:
    return Command(goto=[left, right])


async def test_before_node_changes_in_place_do_not_leak() -> None:
    graph = Graph(Tagged, middleware=[Peek()]).flow(
        START >> fan, left >> END, right >> END
    )

    result = await graph.ainvoke({"log": [], "seen": []})

    assert result.data == {"log": ["left", "right saw []"], "seen": []}


class ReplacesInput(Middleware):
    def before_node(self, ctx):
        return ctx.replace({**ctx.state, "tag": f"set for {ctx.node_name}"})


async def test_before_node_replacement_still_reaches_the_node() -> None:
    graph = Graph(Tagged, middleware=[ReplacesInput()]).flow(
        START >> reads_tag, reads_tag >> END
    )

    result = await graph.ainvoke({"tag": "input"})

    assert result.data["log"] == ["tag=set for reads_tag"]
    assert result.data["tag"] == "input"


class AnswersAndMutates(Middleware):
    def on_interrupt(self, ctx, payload):
        ctx.state["seen"].append("on_interrupt")
        return "auto"


async def test_on_interrupt_changes_in_place_do_not_leak() -> None:
    graph = Graph(Tagged, middleware=[AnswersAndMutates()]).flow(
        START >> asks, asks >> END
    )

    result = await graph.ainvoke({"log": [], "seen": []})

    assert result.data == {"log": ["answer=auto"], "seen": []}
