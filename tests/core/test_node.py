import asyncio
from typing import Any, cast

import pytest

from nodestep import END, START, Graph, NodeContext
from nodestep.core.node import get_node_metadata, get_node_name, node


@node
def classify(state: dict) -> dict:
    return {"intent": "support"}


@node(name="custom")
async def named(state: dict) -> dict:
    return {"ok": True}


@node
def with_context(state: dict, ctx: NodeContext) -> dict:
    return {"ctx": ctx is not None}


def test_node_rejects_ambiguous_second_argument() -> None:
    with pytest.raises(TypeError, match=r"annotated with nodestep\.NodeContext"):

        @node
        def invalid_context(state: dict, extra: object) -> dict:
            return {"extra": extra is not None}


def test_node_decorator_attaches_metadata() -> None:
    assert get_node_name(classify) == "classify"
    assert get_node_metadata(classify).name == "classify"
    assert get_node_metadata(classify).accepts_context is False


def test_node_decorator_custom_name() -> None:
    assert get_node_name(named) == "custom"


def test_node_accepts_context_when_two_positional_args() -> None:
    assert get_node_metadata(with_context).accepts_context is True


def test_node_rejects_varargs() -> None:
    with pytest.raises(TypeError, match="Graph nodes cannot use"):

        @node
        def invalid(*values: str) -> dict:
            return {"values": values}


def test_node_rejects_too_many_positional_args() -> None:
    with pytest.raises(TypeError, match="one state parameter"):

        @node
        def invalid_too_many(state: dict, ctx: NodeContext, extra: object) -> dict:
            return {"extra": extra}


def test_node_accepts_keyword_only_context() -> None:
    @node
    def keyword_ctx(state: dict, *, ctx: NodeContext) -> dict:
        return {"ok": ctx is not None}

    assert get_node_metadata(keyword_ctx).accepts_context is True


def test_node_rejects_misnamed_keyword_only_parameter() -> None:
    with pytest.raises(TypeError, match=r"annotated with nodestep\.NodeContext"):

        @node
        def misnamed_keyword(state: dict, *, extra: object) -> dict:
            return {"extra": extra}


@pytest.mark.asyncio
async def test_keyword_only_context_is_injected_through_graph() -> None:
    @node
    def reads_ctx(state: dict, *, ctx: NodeContext) -> dict:
        return {"node_name": ctx.node}

    graph = Graph(dict).flow(START >> reads_ctx, reads_ctx >> END)
    result = await graph.ainvoke({})
    assert result.data["node_name"] == "reads_ctx"


@pytest.mark.asyncio
async def test_node_timeout():
    from nodestep.exceptions import NodeTimeoutError

    @node(timeout=0.05)
    async def slow(state):
        await asyncio.sleep(10)
        return {"v": 1}

    graph = Graph(dict).flow(START >> slow, slow >> END)
    with pytest.raises(NodeTimeoutError):
        await graph.ainvoke({"v": 0})


def test_context_name_without_annotation_is_an_ordinary_parameter() -> None:
    def positional(state, ctx):
        return {}

    def keyword(state, *, context):
        return {}

    for handler in (positional, keyword):
        with pytest.raises(TypeError, match=r"annotated with nodestep\.NodeContext"):
            node(handler)


def test_context_name_with_another_annotation_is_rejected() -> None:
    with pytest.raises(TypeError, match=r"annotated with nodestep\.NodeContext"):

        @node
        def by_name(state: dict, context: dict) -> dict:
            return {}

    with pytest.raises(TypeError, match=r"annotated with nodestep\.NodeContext"):

        @node
        def with_default(state: dict, ctx: int = 5) -> dict:
            return {}


def test_a_different_class_named_node_context_is_rejected() -> None:
    class NodeContext:
        pass

    with pytest.raises(TypeError, match=r"annotated with nodestep\.NodeContext"):

        @node
        def own_class(state: dict, settings: NodeContext) -> dict:
            return {}


async def test_context_is_injected_by_annotation_under_any_name() -> None:
    @node
    def sync_reader(state: dict, run_info: NodeContext) -> dict:
        return {"sync": run_info.node}

    @node
    async def async_reader(state: dict, *, info: NodeContext) -> dict:
        return {"async": info.node}

    graph = Graph(dict).flow(
        START >> sync_reader, sync_reader >> async_reader, async_reader >> END
    )

    result = await graph.ainvoke({})

    assert result.data == {"sync": "sync_reader", "async": "async_reader"}


def test_lambda_node_needs_a_name() -> None:
    with pytest.raises(TypeError, match="name="):
        node(lambda state: {})

    assert node(lambda state: {}, name="named").name == "named"


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_node_name_is_rejected(blank: str) -> None:
    def step(state: dict) -> dict:
        return {}

    with pytest.raises(TypeError, match="must not be empty"):
        node(step, name=blank)


def test_node_name_is_kept_verbatim() -> None:
    def step(state: dict) -> dict:
        return {}

    assert node(step, name="  spaced  ").name == "  spaced  "


def test_goto_declares_dynamic_targets() -> None:
    @node(goto=[classify, END])
    def router(state: dict) -> dict:
        return {}

    assert router.goto == (classify, END)
    assert get_node_metadata(router).goto == ("classify", "END")
    assert classify.goto is None


def test_goto_accepts_node_names() -> None:
    @node(goto=["classify", END])
    def router(state: dict) -> dict:
        return {}

    assert router.goto == ("classify", END)
    assert get_node_metadata(router).goto == ("classify", "END")


def test_goto_rejects_non_node_targets() -> None:
    with pytest.raises(
        TypeError, match="goto targets must be nodes, node names or END"
    ):

        @node(goto=cast(Any, [42]))
        def router(state: dict) -> dict:
            return {}


def test_goto_rejects_a_single_name() -> None:
    with pytest.raises(TypeError, match="must be a list"):

        @node(goto=cast(Any, "classify"))
        def router(state: dict) -> dict:
            return {}


def test_goto_rejects_an_empty_declaration() -> None:
    with pytest.raises(TypeError, match="at least one target"):

        @node(goto=[])
        def router(state: dict) -> dict:
            return {}


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_keyboard_interrupt_and_system_exit_propagate_from_a_node(
    error: type[BaseException], asynchronous: bool
) -> None:
    if asynchronous:

        @node
        async def stop(state: dict) -> dict:
            raise error

    else:

        @node
        def stop(state: dict) -> dict:
            raise error

    graph = Graph(dict).flow(START >> stop, stop >> END)

    with pytest.raises(error):
        graph.invoke({})
