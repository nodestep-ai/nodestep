import functools
from typing import Annotated, Any, TypedDict

import pytest
from pydantic import BaseModel, ValidationError

from nodestep.core.tool import Tool, ToolCallResult, ToolContext, call_tool, tool
from nodestep.middleware.base import Middleware, Replacement, ToolMiddlewareContext


class Greeting(BaseModel):
    """A greeting."""

    message: str


@tool
async def greet(name: str) -> Greeting:
    """Greet someone."""
    return Greeting(message=f"Hello {name}")


@tool(name="add_numbers")
def add(left: int, right: int) -> int:
    """Add two numbers."""
    return left + right


@tool
async def read_state(ctx: ToolContext, key: str) -> str:
    """Read a state key."""
    return str(ctx.state[key])


def test_tool_metadata() -> None:
    assert greet.name == "greet"
    assert add.name == "add_numbers"


def test_tool_is_instance() -> None:
    assert isinstance(greet, Tool)
    assert isinstance(add, Tool)


def test_tool_repr() -> None:
    assert repr(greet) == "Tool('greet')"


async def test_call_tool_model_output() -> None:
    result = await call_tool(greet, {"name": "Ada"})
    assert result == ToolCallResult(
        value=Greeting(message="Hello Ada"), arguments={"name": "Ada"}
    )


async def test_call_tool_scalar_output() -> None:
    result = await call_tool(add, {"left": 2, "right": 3})
    assert result.value.model_dump() == {"result": 5}
    assert result.arguments == {"left": 2, "right": 3}


async def test_call_tool_injects_context() -> None:
    ctx = ToolContext(graph_name="graph", node="act", state={"answer": 42})
    result = await call_tool(read_state, {"key": "answer"}, ctx)
    assert result.value.model_dump() == {"result": "42"}
    assert result.arguments == {"key": "answer"}


def test_tool_name_is_function_name_verbatim() -> None:
    def get_weather(city: str) -> str:
        """Return the weather."""
        return city

    def http_get(url: str) -> str:
        """Fetch a URL."""
        return url

    get_weather.__name__ = "getWeather"
    http_get.__name__ = "HTTPGet"

    assert tool(get_weather).name == "getWeather"
    assert tool(http_get).name == "HTTPGet"
    assert tool(http_get).input_model.__name__ == "HTTPGetInput"


def test_lambda_requires_name() -> None:
    with pytest.raises(TypeError, match="name="):
        tool(description="Answer.")(lambda: 42)
    assert tool(name="answer", description="Answer.")(lambda: 42).name == "answer"


class Adder:
    """Add one."""

    def __call__(self, value: int) -> int:
        """Add one to a value."""
        return value + 1

    def add_two(self, value: int) -> int:
        """Add two to a value."""
        return value + 2


def test_callable_instances_are_rejected() -> None:
    for decorate in (
        tool,
        tool(name="adder"),
        tool(name="adder", description="Add one."),
    ):
        with pytest.raises(TypeError, match="pass a function or bound method"):
            decorate(Adder())


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (Adder(), r"tool\(\) got Adder <"),
        (Adder, r"tool\(\) got type <class "),
        (functools.partial(len), r"tool\(\) got partial functools\.partial\("),
    ],
    ids=["instance", "class", "partial"],
)
def test_the_rejection_reads_like_the_node_one(value: Any, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        tool(value)


def test_a_bound_method_is_a_tool() -> None:
    made = tool(Adder().add_two)

    assert made.name == "add_two"
    assert made.input_schema()["required"] == ["value"]


def test_empty_name_raises() -> None:
    def lookup(key: str) -> str:
        """Look up a key."""
        return key

    with pytest.raises(TypeError, match="name"):
        tool(name="")(lookup)


def test_missing_description_raises() -> None:
    def undocumented(key: str) -> str:
        return key

    with pytest.raises(TypeError, match="description"):
        tool(undocumented)


def test_empty_description_raises() -> None:
    def blank(key: str) -> str:
        """ """
        return key

    with pytest.raises(TypeError, match="description"):
        tool(description="")(blank)
    with pytest.raises(TypeError, match="description"):
        tool(blank)


def test_description_argument_wins_over_docstring() -> None:
    def documented(key: str) -> str:
        """Docstring text."""
        return key

    assert tool(description="Given text.")(documented).description == "Given text."


def test_full_docstring_is_description() -> None:
    def documented(city: str) -> str:
        """Return the weather.

        Parameters
        ----------
        city : str
            City name.
        """
        return city

    assert tool(documented).description == (
        "Return the weather.\n\nParameters\n----------\ncity : str\n    City name."
    )


def test_inherited_docstring_does_not_count() -> None:
    class Base:
        def run(self, query: str) -> str:
            """Base docstring: deletes production data."""
            return query

    class Child(Base):
        def run(self, query: str) -> str:
            return query.upper()

    with pytest.raises(TypeError, match="description"):
        tool(Child().run)
    assert tool(Base().run).description == "Base docstring: deletes production data."


async def test_generated_input_model_forbids_unknown_arguments() -> None:
    @tool
    def bump(a: int, flag: bool = False) -> int:
        """Add one."""
        return a + 1

    with pytest.raises(ValidationError, match="hallucinated"):
        await call_tool(bump, {"a": 1, "hallucinated": "rm -rf /"})
    assert bump.input_schema()["additionalProperties"] is False


def test_unannotated_parameter_raises() -> None:
    def untyped(query: str, limit) -> str:
        """Untyped limit."""
        return query

    with pytest.raises(TypeError, match="limit"):
        tool(untyped)


class Query(BaseModel):
    """Search arguments."""

    query: str
    limit: int = 5


async def test_input_model_is_used_as_argument_model() -> None:
    def search(query, limit):
        """Search."""
        return f"{query}:{limit}"

    searcher = tool(input_model=Query)(search)

    assert searcher.input_model is Query
    result = await call_tool(searcher, {"query": "x"})
    assert result.value.model_dump() == {"result": "x:5"}
    assert result.arguments == {"query": "x", "limit": 5}


async def test_input_model_with_lambda_and_context() -> None:
    searcher = tool(name="search", description="Search.", input_model=Query)(
        lambda query, limit: f"{query}:{limit}"
    )
    assert (await call_tool(searcher, {"query": "y"})).value.model_dump() == {
        "result": "y:5"
    }
    with pytest.raises(TypeError, match="ctx"):
        tool(name="search", description="Search.", input_model=Query)(
            lambda query, limit, ctx: query
        )

    typed = tool(input_model=Query)(_search_with_context)
    result = await call_tool(typed, {"query": "x"}, ToolContext(thread_id="t-1"))
    assert result.value.model_dump() == {"result": "x:5:t-1"}


def _search_with_context(query: str, limit: int, context: ToolContext) -> str:
    """Search with context."""
    return f"{query}:{limit}:{context.thread_id}"


def test_input_model_must_match_parameters() -> None:
    def search(query: str) -> str:
        """Search."""
        return query

    with pytest.raises(TypeError, match="limit"):
        tool(input_model=Query)(search)


def test_optional_tool_context_raises() -> None:
    def optional_context(query: str, ctx: ToolContext | None = None) -> str:
        """Opt ctx."""
        return query

    with pytest.raises(TypeError, match="ctx"):
        tool(optional_context)


def test_annotated_tool_context_raises() -> None:
    def annotated_ctx(query: str, ctx: Annotated[ToolContext, "injected"]) -> str:
        """Annotated ctx."""
        return query

    with pytest.raises(TypeError, match="ctx"):
        tool(annotated_ctx)


def test_untyped_context_parameter_raises() -> None:
    def untyped_ctx(query: str, ctx) -> str:
        """Untyped ctx."""
        return query

    def untyped_context(query: str, context) -> str:
        """Untyped context."""
        return query

    for function, parameter in ((untyped_ctx, "ctx"), (untyped_context, "context")):
        with pytest.raises(TypeError, match=parameter):
            tool(function)


def test_two_tool_context_parameters_raise() -> None:
    def twice(first: ToolContext, second: ToolContext) -> str:
        """Two contexts."""
        return "x"

    with pytest.raises(TypeError, match="second"):
        tool(twice)


def test_typed_tool_context_under_any_name_is_injected() -> None:
    def typed(query: str, run: ToolContext) -> str:
        """Typed ctx."""
        return query

    made = tool(typed)
    assert made.metadata.context_parameter == "run"
    assert list(made.input_model.model_fields) == ["query"]


async def test_call_tool_accepts_only_tool() -> None:
    def plain(value: int) -> int:
        """Negate."""
        return -value

    vars(plain)["__nodestep_tool__"] = add.metadata
    candidate: Any = plain

    with pytest.raises(TypeError, match="Tool"):
        await call_tool(candidate, {"value": 3})


class _DoubleLeft(Middleware):
    def before_tool(self, ctx: ToolMiddlewareContext) -> Replacement | None:
        return Replacement(ctx.value.model_copy(update={"left": ctx.value.left * 2}))


async def test_call_tool_returns_effective_arguments_after_middleware() -> None:
    ctx = ToolContext(middleware=(_DoubleLeft(),))

    result = await call_tool(add, {"left": 2, "right": 3}, ctx)

    assert result.arguments == {"left": 4, "right": 3}
    assert result.value.model_dump() == {"result": 7}
    assert not hasattr(ctx, "edited_arguments")


def test_tool_call_result_serializes_value() -> None:
    result = ToolCallResult(value=Greeting(message="hi"), arguments={"name": "x"})
    dumped: dict[str, Any] = result.model_dump()
    assert dumped == {"value": {"message": "hi"}, "arguments": {"name": "x"}}


async def test_sync_tools_run_concurrently_in_the_tool_runner() -> None:
    import time

    from nodestep import END, START, Graph, add_messages, node, tool, tool_runner
    from nodestep.chat import Message, ToolCall

    class ToolState(TypedDict, total=False):
        messages: Annotated[list[Message], add_messages]
        tool_calls: list[ToolCall]

    @tool
    def slow_lookup(key: str) -> str:
        """Look up slowly."""
        time.sleep(0.2)
        return key

    @node
    def request(state: dict) -> dict:
        return {
            "tool_calls": [
                ToolCall(id=f"c{i}", name="slow_lookup", arguments={"key": str(i)})
                for i in range(4)
            ]
        }

    runner = tool_runner("tools", tools=[slow_lookup])
    graph = Graph(ToolState).flow(START >> request, request >> runner, runner >> END)
    started = time.monotonic()

    result = await graph.ainvoke({"messages": []})

    assert len(result.data["messages"]) == 4
    assert time.monotonic() - started < 0.5


def test_tool_context_subclass_raises() -> None:
    class RunContext(ToolContext):
        pass

    def sub_ctx(query: str, ctx: RunContext) -> str:
        """Subclass ctx."""
        return query

    with pytest.raises(TypeError, match="exactly as ToolContext"):
        tool(sub_ctx)


@pytest.mark.parametrize("bad", ["  bad name ", "has.dot", "x" * 65, "naïve"])
def test_tool_name_outside_the_provider_pattern_raises(bad: str) -> None:
    def lookup(key: str) -> str:
        """Look up a key."""
        return key

    with pytest.raises(TypeError, match="a-zA-Z0-9_-"):
        tool(name=bad)(lookup)


def test_tool_name_of_64_allowed_characters_is_accepted() -> None:
    name = "Tool_name-1" + "x" * 53
    made = tool(name=name, description="d")(lambda: 1)
    assert made.name == name


class _BreakLeft(Middleware):
    def before_tool(self, ctx: ToolMiddlewareContext) -> Replacement | None:
        return Replacement(ctx.value.model_copy(update={"left": "notint"}))


async def test_before_tool_model_copy_is_revalidated() -> None:
    with pytest.raises(ValidationError, match="left"):
        await call_tool(
            add, {"left": 2, "right": 3}, ToolContext(middleware=(_BreakLeft(),))
        )


def test_same_tool_listed_twice_raises() -> None:
    from nodestep import ScriptedChat, build_react_agent, tool_runner
    from nodestep.exceptions import GraphConfigError

    class WithAdd(Middleware):
        def tools(self) -> list[Tool]:
            return [add]

    with pytest.raises(GraphConfigError, match="'add_numbers' is listed twice"):
        tool_runner("act", tools=[add, add])
    with pytest.raises(GraphConfigError, match="'add_numbers' is listed twice"):
        build_react_agent(ScriptedChat(["ok"]), tools=[add], middleware=[WithAdd()])


class _Point(BaseModel):
    x: int


@tool
def returns_point_as_any() -> Any:
    """Return a point from a tool annotated with Any."""
    return _Point(x=1)


@tool
def returns_point_as_dict() -> dict:
    """Return a point from a tool annotated with dict."""
    return _Point(x=1)  # ty: ignore[invalid-return-type]


@tool
def returns_point_without_annotation():
    """Return a point from a tool without a return annotation."""
    return _Point(x=1)


@pytest.mark.parametrize(
    "point_tool", [returns_point_as_any, returns_point_without_annotation]
)
async def test_a_model_result_is_wrapped_unless_the_annotation_is_a_model(
    point_tool: Tool,
) -> None:
    result = await call_tool(point_tool, {})

    assert result.value.model_dump() == {"result": {"x": 1}}


async def test_a_model_result_is_checked_against_a_dict_annotation() -> None:
    with pytest.raises(
        ValidationError, match=r"result\n\s+Input should be a valid dict"
    ):
        await call_tool(returns_point_as_dict, {})


@tool
def returns_point() -> _Point:
    """Return a point."""
    return _Point(x=1)


async def test_a_model_annotation_uses_the_model_as_output() -> None:
    result = await call_tool(returns_point, {})

    assert result.value == _Point(x=1)
