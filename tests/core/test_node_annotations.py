from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import pytest

from nodestep import END, START, Graph, NodeContext, node


async def test_string_annotation_resolves_to_node_context() -> None:
    @node
    def reader(state: dict, ctx: NodeContext) -> dict:
        return {"node": ctx.node}

    result = await Graph(dict).flow(START >> reader, reader >> END).ainvoke({})

    assert reader.accepts_context is True
    assert result.data == {"node": "reader"}


def test_unresolvable_annotation_raises_naming_the_parameter() -> None:
    from nodestep import NodeContext as LocalContext

    with pytest.raises(
        TypeError, match=r"annotation 'LocalContext' of parameter 'ctx'"
    ):

        @node
        def reader(state: dict, ctx: LocalContext) -> dict:
            return {}


class Reader:
    def __call__(self, state: dict, ctx: NodeContext) -> dict:
        return {"node": ctx.node}


def read_node(state: dict, ctx: NodeContext) -> dict:
    return {"node": ctx.node}


def test_annotation_that_fails_to_evaluate_raises_naming_the_parameter() -> None:
    def reader(state: dict, ctx: NodeContext) -> dict:
        return {}

    reader.__annotations__["ctx"] = "list["

    with pytest.raises(
        TypeError, match=r"Cannot resolve the annotation 'list\[' of parameter 'ctx'"
    ):
        node(reader)


@pytest.mark.parametrize(
    "handler",
    [Reader(), functools.partial(read_node), Reader],
    ids=["instance", "partial", "class"],
)
@pytest.mark.parametrize("name", [None, "reader"])
def test_a_non_function_callable_is_rejected(
    handler: Callable[..., Any], name: str | None
) -> None:
    with pytest.raises(TypeError, match="pass a function or bound method"):
        node(handler, name=name)


def test_a_bound_method_is_named_after_the_method() -> None:
    class Holder:
        def step(self, state: dict) -> dict:
            return {}

    assert node(Holder().step).name == "step"
