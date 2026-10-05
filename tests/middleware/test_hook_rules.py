from typing import Any

import pytest

from nodestep import END, START, Graph, ScriptedChat, node
from nodestep.chat import HumanMessage
from nodestep.core import model_node
from nodestep.core.builtin_nodes.agent import AgentState
from nodestep.exceptions import GraphExecutionError
from nodestep.middleware import (
    Middleware,
    Replacement,
    run_after_hooks,
    run_before_hooks,
)


@pytest.mark.parametrize(
    "name", ["before_modle", "after_tools", "on_interupt", "before_step", "on_start"]
)
def test_unknown_hook_names_raise_when_the_class_is_defined(name: str) -> None:
    with pytest.raises(TypeError, match=name):
        type("Misspelled", (Middleware,), {name: lambda self, ctx: None})


def test_unknown_hook_names_raise_for_static_and_class_methods() -> None:
    with pytest.raises(TypeError, match="after_tools"):
        type("Static", (Middleware,), {"after_tools": staticmethod(lambda ctx: None)})
    with pytest.raises(TypeError, match="on_start"):
        type("Class", (Middleware,), {"on_start": classmethod(lambda cls: None)})


def test_known_hooks_and_other_names_are_accepted() -> None:
    class Fine(Middleware):
        on_label = "not a method"

        def before_model(self, ctx):
            return None

        def on_error(self, ctx, error):
            return None

        def _before_helper(self) -> None:
            return None

        def summarize(self) -> None:
            return None

    assert Fine().on_label == "not a method"


@node
def passthrough(state: dict) -> dict:
    return {}


@pytest.mark.parametrize("hook", ["before_node", "after_node"])
def test_node_hook_returning_a_plain_value_raises(hook: str) -> None:
    def returns_dict(self, ctx):
        return {"count": 1}

    plain = type("Plain", (Middleware,), {hook: returns_dict})
    graph = Graph(dict, middleware=[plain()]).flow(
        START >> passthrough, passthrough >> END
    )

    with pytest.raises(TypeError, match=rf"Plain\.{hook} returned dict"):
        graph.invoke({})


def test_async_hook_returning_a_plain_value_raises() -> None:
    class AsyncPlain(Middleware):
        async def before_node(self, ctx):
            return {"count": 1}

    graph = Graph(dict, middleware=[AsyncPlain()]).flow(
        START >> passthrough, passthrough >> END
    )

    with pytest.raises(TypeError, match=r"AsyncPlain\.before_node returned dict"):
        graph.invoke({})


@pytest.mark.parametrize("hook", ["before_graph", "after_graph"])
def test_graph_hook_returning_a_plain_value_raises(hook: str) -> None:
    def returns_dict(self, ctx):
        return {"count": 1}

    plain = type("Plain", (Middleware,), {hook: returns_dict})
    graph = Graph(dict, middleware=[plain()]).flow(
        START >> passthrough, passthrough >> END
    )

    with pytest.raises(TypeError, match=rf"Plain\.{hook} returned dict"):
        graph.invoke({})


def test_graph_hook_returning_a_replacement_still_raises_graph_execution_error() -> (
    None
):
    class Replacing(Middleware):
        def before_graph(self, ctx):
            return Replacement({})

    graph = Graph(dict, middleware=[Replacing()]).flow(
        START >> passthrough, passthrough >> END
    )

    with pytest.raises(GraphExecutionError, match="read-only"):
        graph.invoke({})


async def test_model_hook_returning_a_plain_request_raises() -> None:
    class Plain(Middleware):
        def before_model(self, ctx):
            return ctx.request

    think = model_node("think", chat=ScriptedChat(["ok"]))
    graph = Graph(AgentState, middleware=[Plain()]).flow(START >> think, think >> END)

    with pytest.raises(TypeError, match=r"Plain\.before_model returned ChatRequest"):
        await graph.ainvoke({"messages": [HumanMessage(content="hi")]})


@pytest.mark.parametrize("hook", ["before_graph", "after_graph"])
async def test_value_hook_runners_refuse_graph_hooks(hook: Any) -> None:
    for runner in (run_before_hooks, run_after_hooks):
        with pytest.raises(ValueError, match=hook):
            await runner([Middleware()], hook, {}, lambda value: value)


class _PlainHooks:
    def before_nodes(self, ctx: Any) -> None:
        return None


def test_graph_middleware_must_subclass_middleware() -> None:
    with pytest.raises(TypeError, match=r"_PlainHooks.*subclass nodestep\.Middleware"):
        Graph(AgentState, middleware=[_PlainHooks()])  # ty: ignore[invalid-argument-type]


def test_react_agent_middleware_must_subclass_middleware() -> None:
    from nodestep import build_react_agent

    with pytest.raises(TypeError, match=r"_PlainHooks.*subclass nodestep\.Middleware"):
        build_react_agent(
            ScriptedChat(default="done"),
            tools=[],
            middleware=[_PlainHooks()],  # ty: ignore[invalid-argument-type]
        )
