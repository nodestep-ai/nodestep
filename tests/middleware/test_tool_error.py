import asyncio
from typing import Annotated, Any, TypedDict

import pytest

from nodestep import END, START, Graph, Middleware, node, tool
from nodestep.chat import Message, ToolCall
from nodestep.core import ToolContext, call_tool, tool_runner
from nodestep.exceptions import ToolDeniedError
from nodestep.middleware import ToolMiddlewareContext
from nodestep.utils.reducers import add_messages


class ToolState(TypedDict, total=False):
    messages: Annotated[list[Message], add_messages]
    tool_calls: list[ToolCall]


@tool
def explode(value: int) -> int:
    """Fail on purpose."""
    raise ValueError(f"exploded on {value}")


@tool
def wrong_result(value: int) -> int:
    """Return a value the output model rejects."""
    return "not a number"  # ty: ignore[invalid-return-type]


@tool
def fine(value: int) -> int:
    """Return the value."""
    return value


def calls(name: str) -> Any:
    @node(name=f"request_{name}")
    def request(_: dict) -> dict:
        return {
            "tool_calls": [ToolCall(id="call_1", name=name, arguments={"value": 3})]
        }

    return request


def tool_graph(name: str, *middleware: Middleware, tool_errors: Any = "raise") -> Graph:
    request = calls(name)
    runner = tool_runner(
        "tools", tools=[explode, wrong_result, fine], tool_errors=tool_errors
    )
    return Graph(ToolState, middleware=list(middleware)).flow(
        START >> request, request >> runner, runner >> END
    )


class Observer(Middleware):
    def __init__(self, label: str = "observer", log: list[Any] | None = None) -> None:
        self.label = label
        self.log: list[Any] = [] if log is None else log
        self.errors: list[tuple[ToolMiddlewareContext, Exception]] = []

    def before_node(self, ctx):
        if ctx.node_name == "tools":
            self.log.append((self.label, "before_node", ctx.task_id))

    def on_tool_error(self, ctx, error):
        self.log.append((self.label, "on_tool_error"))
        self.errors.append((ctx, error))


async def test_the_hook_sees_the_tool_error_and_the_error_still_raises() -> None:
    observer = Observer()

    with pytest.raises(ValueError, match="exploded on 3") as info:
        await tool_graph("explode", observer).ainvoke({"messages": []})

    [(ctx, error)] = observer.errors
    assert error is info.value
    assert ctx.tool_name == "explode"
    assert ctx.node_name == "tools"
    assert ctx.tool_call_id == "call_1"
    assert ctx.value.value == 3


async def test_the_hook_runs_before_the_tool_errors_policy() -> None:
    observer = Observer()

    result = await tool_graph("explode", observer, tool_errors="return").ainvoke(
        {"messages": []}
    )

    [(_, error)] = observer.errors
    assert isinstance(error, ValueError)
    assert "exploded on 3" in result.data["messages"][0].content


async def test_the_context_carries_the_task_id_of_the_node() -> None:
    observer = Observer()

    await tool_graph("explode", observer, tool_errors="return").ainvoke(
        {"messages": []}
    )

    [(ctx, _)] = observer.errors
    [(_, _, task_id)] = [entry for entry in observer.log if len(entry) == 3]
    assert ctx.task_id == task_id
    assert ctx.task_id


async def test_a_result_the_output_model_rejects_counts_as_a_tool_error() -> None:
    observer = Observer()

    await tool_graph("wrong_result", observer, tool_errors="return").ainvoke(
        {"messages": []}
    )

    [(ctx, _)] = observer.errors
    assert ctx.tool_name == "wrong_result"


async def test_a_successful_tool_calls_no_hook() -> None:
    observer = Observer()

    await tool_graph("fine", observer).ainvoke({"messages": []})

    assert observer.errors == []


async def test_a_denied_tool_call_is_not_a_tool_error() -> None:
    class Denies(Middleware):
        def before_tool(self, ctx):
            raise ToolDeniedError(ctx.tool_name, "not allowed")

    observer = Observer()

    await tool_graph("explode", observer, Denies()).ainvoke({"messages": []})

    assert observer.errors == []


async def test_hooks_run_in_reverse_middleware_order() -> None:
    log: list[Any] = []

    await tool_graph(
        "explode", Observer("first", log), Observer("second", log), tool_errors="return"
    ).ainvoke({"messages": []})

    assert [entry for entry in log if entry[1] == "on_tool_error"] == [
        ("second", "on_tool_error"),
        ("first", "on_tool_error"),
    ]


async def test_async_hooks_are_awaited() -> None:
    seen: list[str] = []

    class Async(Middleware):
        async def on_tool_error(self, ctx, error):
            await asyncio.sleep(0)
            seen.append(str(error))

    await tool_graph("explode", Async(), tool_errors="return").ainvoke({"messages": []})

    assert seen == ["exploded on 3"]


async def test_a_hook_error_is_a_note_on_the_tool_error() -> None:
    class Raises(Middleware):
        def on_tool_error(self, ctx, error):
            raise RuntimeError("observer broke")

    observer = Observer()

    with pytest.raises(ValueError, match="exploded on 3") as info:
        await tool_graph("explode", observer, Raises()).ainvoke({"messages": []})

    assert info.value.__notes__ == [
        "Raises.on_tool_error raised RuntimeError: observer broke"
    ]
    assert len(observer.errors) == 1


async def test_a_return_value_is_a_note_on_the_tool_error() -> None:
    class Recovers(Middleware):
        def on_tool_error(self, ctx, error):
            return 0

    with pytest.raises(ValueError, match="exploded on 3") as info:
        await tool_graph("explode", Recovers()).ainvoke({"messages": []})

    [note] = info.value.__notes__
    assert note.startswith("Recovers.on_tool_error raised TypeError:")
    assert "read-only and returns None" in note


async def test_a_replacement_is_a_graph_execution_error_note() -> None:
    class Replaces(Middleware):
        def on_tool_error(self, ctx, error):
            return ctx.replace(0)

    with pytest.raises(ValueError, match="exploded on 3") as info:
        await tool_graph("explode", Replaces()).ainvoke({"messages": []})

    [note] = info.value.__notes__
    assert note.startswith("Replaces.on_tool_error raised GraphExecutionError:")


async def test_a_hook_cancellation_propagates_after_every_hook_ran() -> None:
    class Cancels(Middleware):
        def on_tool_error(self, ctx, error):
            raise asyncio.CancelledError("observer cancelled")

    observer = Observer()
    ctx = ToolContext(middleware=(observer, Cancels()))

    with pytest.raises(asyncio.CancelledError, match="observer cancelled") as info:
        await call_tool(explode, {"value": 1}, ctx)

    assert len(observer.errors) == 1
    assert isinstance(info.value.__context__, ValueError)


async def test_call_tool_runs_the_hook_of_the_context_middleware() -> None:
    observer = Observer()
    ctx = ToolContext(middleware=(observer,), task_id="task-1")

    with pytest.raises(ValueError, match="exploded on 1"):
        await call_tool(explode, {"value": 1}, ctx)

    [(hook_ctx, _)] = observer.errors
    assert hook_ctx.task_id == "task-1"


async def test_tool_hooks_see_the_task_id() -> None:
    seen: list[str] = []

    class Captures(Middleware):
        def before_tool(self, ctx):
            seen.append(ctx.task_id)

        def after_tool(self, ctx):
            seen.append(ctx.task_id)

    await tool_graph("fine", Captures()).ainvoke({"messages": []})

    first, second = seen
    assert first == second
    assert first
