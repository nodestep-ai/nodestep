from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
    get_args,
    get_type_hints,
    overload,
)

from pydantic import BaseModel, ConfigDict, SerializeAsAny, create_model

from nodestep.exceptions import ContextNotProvidedError, GraphConfigError
from nodestep.utils.json import dump_json_object, load_json_object

if TYPE_CHECKING:
    from nodestep.core.stream import NodeContext


class ToolMetadata(BaseModel):
    """Name, description and schemas of a tool.

    Attributes
    ----------
    input_model : type[BaseModel]
        Model the arguments are validated against.
    output_model : type[BaseModel]
        Model the result is validated against.
    context_parameter : str or None
        Name of the parameter that receives the ``ToolContext``.
    returns_model : bool
        The function's return type is the output model itself; otherwise the
        value is wrapped in the model's ``result`` field.
    """

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    context_parameter: str | None = None
    returns_model: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ToolCallResult(BaseModel):
    """Validated result of a tool call and the arguments it ran with.

    Attributes
    ----------
    value : BaseModel
        The result as an instance of the tool's ``output_model``.
    arguments : dict
        The arguments after ``before_tool`` middleware.
    """

    value: SerializeAsAny[BaseModel]
    arguments: dict[str, Any]


@dataclass
class ToolContext:
    """Run information passed to tools that declare a ``ToolContext``.

    Attributes
    ----------
    graph_name : str or None
        Name of the graph.
    node : str or None
        Name of the node that calls the tool.
    state : Any
        State that node received.
    run_id : str or None
        Id of the run.
    emit : callable or None
        Sends a ``"custom"`` stream event, like ``NodeContext.emit``.
    middleware : tuple
        Middleware of the run, in order.
    workspace : Workspace or None
        The graph's workspace.
    tool_call_id : str or None
        Id of the tool call being executed.
    root_run_id : str or None
        Id of the outermost run when the graph runs inside another one.
    task_id : str
        Task of the node that calls the tool.
    """

    graph_name: str | None = None
    node: str | None = None
    state: Any = None
    thread_id: str | None = None
    run_id: str | None = None
    emit: Callable[[Any], None] | None = None
    middleware: tuple[Any, ...] = ()
    workspace: Any = None
    tool_call_id: str | None = None
    root_run_id: str | None = None
    task_id: str = ""
    _context: Any = None

    @property
    def context(self) -> Any:
        """The ``context=`` object of the run the tool is called in.

        Returns
        -------
        Any

        Raises
        ------
        ContextNotProvidedError
            If the run was started without ``context=``.
        """
        if self._context is None:
            raise ContextNotProvidedError
        return self._context

    @classmethod
    def from_node_context(
        cls, ctx: NodeContext, state: Any, *, tool_call_id: str | None = None
    ) -> ToolContext:
        """Build a tool context from the running node.

        Parameters
        ----------
        ctx : NodeContext
            Context of the node that calls the tool.
        state : Any
            State the node received.
        tool_call_id : str, optional
            Id of the tool call being executed.

        Returns
        -------
        ToolContext
        """
        return cls(
            graph_name=ctx.graph_name,
            node=ctx.node,
            state=state,
            thread_id=ctx.thread_id,
            run_id=ctx.run_id,
            emit=ctx.emit,
            middleware=ctx.middleware,
            workspace=ctx.workspace,
            tool_call_id=tool_call_id,
            root_run_id=ctx.root_run_id,
            task_id=ctx.task_id,
            _context=ctx._context,
        )


class Tool:
    """A callable exposed to models, with validated input and output.

    ``tool()`` builds one from a function and its type hints.

    Parameters
    ----------
    function : Callable
        The function the tool calls.
    metadata : ToolMetadata
        Name, description and schemas of the tool, kept as ``metadata``.
    """

    __slots__ = ("_function", "metadata")

    def __init__(self, function: Callable[..., Any], metadata: ToolMetadata) -> None:
        self._function = function
        self.metadata = metadata

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function directly.

        No validation, middleware or ``ToolContext`` injection; ``call_tool``
        does those.

        Parameters
        ----------
        *args, **kwargs
            Passed to the function as they are.

        Returns
        -------
        Any
            What the function returns.
        """
        return self._function(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Tool({self.name!r})"

    @property
    def name(self) -> str:
        """Tool name shown to the model."""
        return self.metadata.name

    @property
    def description(self) -> str:
        """Tool description shown to the model."""
        return self.metadata.description

    @property
    def input_model(self) -> type[BaseModel]:
        """Pydantic model of the tool arguments."""
        return self.metadata.input_model

    @property
    def output_model(self) -> type[BaseModel]:
        """Pydantic model of the tool result."""
        return self.metadata.output_model

    def input_schema_json(self) -> str:
        """JSON schema of the arguments, as a string."""
        return (
            dump_json_object(self.input_model.model_json_schema(), sort_keys=False)
            or "{}"
        )

    def input_schema(self) -> dict[str, Any]:
        """JSON schema of the arguments."""
        return load_json_object(self.input_schema_json()) or {}

    def output_schema_json(self) -> str:
        """JSON schema of the result, as a string."""
        return (
            dump_json_object(self.output_model.model_json_schema(), sort_keys=False)
            or "{}"
        )

    def output_schema(self) -> dict[str, Any]:
        """JSON schema of the result."""
        return load_json_object(self.output_schema_json()) or {}

    def supports_context(self) -> bool:
        """Whether the tool takes a ``ToolContext`` argument."""
        return self.metadata.context_parameter is not None


def collect_tools(*groups: Iterable[Tool]) -> list[Tool]:
    """Merge tool lists into one list with unique names.

    Parameters
    ----------
    *groups : Iterable[Tool]

    Returns
    -------
    list[Tool]

    Raises
    ------
    GraphConfigError
        If two different tools share a name, or the same tool is listed twice.
    """
    collected: dict[str, Tool] = {}
    for group in groups:
        for item in group:
            existing = collected.get(item.name)
            if existing is item:
                raise GraphConfigError(f"Tool '{item.name}' is listed twice")
            if existing is not None:
                raise GraphConfigError(f"Two different tools are named '{item.name}'")
            collected[item.name] = item
    return list(collected.values())


_CONTEXT_NAMES = frozenset({"ctx", "context"})


def _mentions_tool_context(annotation: Any) -> bool:
    if isinstance(annotation, type) and issubclass(annotation, ToolContext):
        return True
    return any(_mentions_tool_context(argument) for argument in get_args(annotation))


def _find_context_parameter(
    function: Callable[..., Any], hints: dict[str, Any]
) -> str | None:
    found: str | None = None
    for name, parameter in inspect.signature(function).parameters.items():
        annotation = hints.get(name, parameter.annotation)
        if annotation is ToolContext:
            if found is not None:
                raise TypeError(
                    f"Tool parameter '{name}' is a second ToolContext; "
                    f"'{found}' already receives it"
                )
            found = name
        elif _mentions_tool_context(annotation):
            raise TypeError(
                f"Tool parameter '{name}' must be annotated exactly as ToolContext "
                f"to be injected, not {annotation!r}"
            )
        elif annotation is inspect.Parameter.empty and name in _CONTEXT_NAMES:
            raise TypeError(
                f"Tool parameter '{name}' has no annotation; annotate it as "
                "ToolContext to receive the tool context"
            )
    return found


def _argument_parameters(
    function: Callable[..., Any], context_parameter: str | None
) -> list[inspect.Parameter]:
    parameters = []
    for name, parameter in inspect.signature(function).parameters.items():
        if name == context_parameter:
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise TypeError("Tool functions cannot use *args or **kwargs")
        parameters.append(parameter)
    return parameters


def _model_prefix(name: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in name.split("_"))


def _build_input_model(
    parameters: list[inspect.Parameter], hints: dict[str, Any], tool_name: str
) -> type[BaseModel]:
    fields: dict[str, tuple[Any, Any]] = {}
    for parameter in parameters:
        annotation = hints.get(parameter.name, parameter.annotation)
        if annotation is inspect.Parameter.empty:
            raise TypeError(
                f"Tool '{tool_name}' parameter '{parameter.name}' has no type "
                "annotation; annotate it or pass input_model="
            )
        default = (
            ... if parameter.default is inspect.Parameter.empty else parameter.default
        )
        fields[parameter.name] = (annotation, default)
    return cast(Any, create_model)(
        f"{_model_prefix(tool_name)}Input",
        __config__=ConfigDict(arbitrary_types_allowed=True, extra="forbid"),
        **fields,
    )


def _check_input_model(
    parameters: list[inspect.Parameter],
    input_model: type[BaseModel],
    tool_name: str,
) -> None:
    expected = {parameter.name for parameter in parameters}
    declared = set(input_model.model_fields)
    if expected != declared:
        raise TypeError(
            f"input_model {input_model.__name__} of tool '{tool_name}' must declare "
            f"exactly the parameters {sorted(expected)}; "
            f"missing {sorted(expected - declared)}, extra {sorted(declared - expected)}"
        )


def _returns_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _build_output_model(annotation: Any, tool_name: str) -> type[BaseModel]:
    if _returns_model(annotation):
        return annotation
    if annotation is inspect.Signature.empty:
        annotation = Any
    return cast(Any, create_model)(
        f"{_model_prefix(tool_name)}Output",
        __config__=ConfigDict(arbitrary_types_allowed=True),
        result=(annotation, ...),
    )


_TOOL_NAME = re.compile(r"[a-zA-Z0-9_-]{1,64}")


def _tool_name(function: Callable[..., Any], name: str | None) -> str:
    if name is None:
        name = getattr(function, "__name__", None)
        if name is None or name == "<lambda>":
            raise TypeError(
                f"{function!r} has no usable __name__; pass name= to tool()"
            )
    if not _TOOL_NAME.fullmatch(name):
        raise TypeError(
            f"Tool name {name!r} must match ^[a-zA-Z0-9_-]{{1,64}}$, the names "
            "chat providers accept"
        )
    return name


def _tool_description(
    function: Callable[..., Any], description: str | None, tool_name: str
) -> str:
    text = (
        description
        if description is not None
        else inspect.cleandoc(function.__doc__ or "")
    )
    if not text.strip():
        raise TypeError(
            f"Tool '{tool_name}' has no description; give the function its own "
            "docstring or pass description="
        )
    return text


def _make_tool_context(
    metadata: ToolMetadata, tool_ctx: ToolContext
) -> Callable[[Any], Any]:
    from nodestep.middleware.base import ToolMiddlewareContext

    async def factory(value: Any) -> ToolMiddlewareContext:
        return ToolMiddlewareContext(
            graph_name=tool_ctx.graph_name,
            node_name=tool_ctx.node,
            tool_name=metadata.name,
            value=value,
            state=tool_ctx.state,
            run_id=tool_ctx.run_id,
            thread_id=tool_ctx.thread_id,
            tool_call_id=tool_ctx.tool_call_id,
            root_run_id=tool_ctx.root_run_id,
            task_id=tool_ctx.task_id,
        )

    return factory


async def call_tool(
    tool: Tool, arguments: BaseModel | dict[str, Any], ctx: ToolContext | None = None
) -> ToolCallResult:
    """Validate arguments, run tool middleware and call a tool.

    Parameters
    ----------
    tool : Tool
        A tool made with ``tool()``.
    arguments : BaseModel or dict
        Tool arguments.
    ctx : ToolContext, optional
        Run information and middleware; an empty context is used when omitted.

    Returns
    -------
    ToolCallResult
        The validated tool result and the arguments the tool ran with, after
        ``before_tool`` middleware.

    Raises
    ------
    TypeError
        If ``tool`` is not a ``Tool``.
    pydantic.ValidationError
        If the arguments or the result do not match the tool's models; the
        ``on_tool_error`` hooks see a result error first.
    Exception
        What the tool raised, after the ``on_tool_error`` hooks saw it.
    """
    from nodestep.middleware.base import (
        run_after_hooks,
        run_before_hooks,
        run_tool_error_hooks,
    )

    if not isinstance(tool, Tool):
        raise TypeError(
            f"call_tool expects a Tool, got {type(tool).__name__}; "
            "wrap the function with tool()"
        )
    metadata = tool.metadata
    input_model = metadata.input_model
    output_model = metadata.output_model
    tool_ctx = ctx or ToolContext()
    validated_arguments = (
        arguments
        if isinstance(arguments, BaseModel)
        else input_model.model_validate(arguments)
    )

    context_factory = _make_tool_context(metadata, tool_ctx)

    validated_arguments = await run_before_hooks(
        tool_ctx.middleware,
        "before_tool",
        validated_arguments,
        context_factory,
    )
    validated_arguments = input_model.model_validate(
        validated_arguments.model_dump(warnings=False)
        if isinstance(validated_arguments, BaseModel)
        else validated_arguments
    )
    kwargs = {
        name: getattr(validated_arguments, name)
        for name in type(validated_arguments).model_fields
    }
    if metadata.context_parameter is not None:
        kwargs[metadata.context_parameter] = tool_ctx
    try:
        if inspect.iscoroutinefunction(tool._function):
            result = tool(**kwargs)
        else:
            result = await asyncio.to_thread(tool, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        if metadata.returns_model:
            output = output_model.model_validate(result)
        else:
            output = output_model.model_validate({"result": result})
    except Exception as error:
        await run_tool_error_hooks(
            tool_ctx.middleware, await context_factory(validated_arguments), error
        )
        raise

    output = await run_after_hooks(
        tool_ctx.middleware,
        "after_tool",
        output,
        context_factory,
    )
    return ToolCallResult(
        value=output_model.model_validate(output),
        arguments=validated_arguments.model_dump(mode="json"),
    )


@overload
def tool(
    function: Callable[..., Any],
    /,
) -> Tool: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    input_model: type[BaseModel] | None = None,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    function: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    input_model: type[BaseModel] | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Turn a function into a tool.

    The arguments are validated against a model built from the type hints,
    which rejects unknown arguments. A parameter annotated exactly as
    ``ToolContext`` receives the tool context and is hidden from the model.
    Sync functions run in a worker thread.

    Parameters
    ----------
    function : callable, optional
        Function to wrap; omit it to use the decorator with options.
    name : str, optional
        Tool name, matching ``^[a-zA-Z0-9_-]{1,64}$``; defaults to
        ``function.__name__``. Lambdas need one.
    description : str, optional
        Description sent to the model; defaults to the function's own
        docstring, all of it.
    input_model : type[BaseModel], optional
        Argument model to use instead of the generated one; its fields must be
        exactly the function's parameters, without the ``ToolContext`` one.

    Returns
    -------
    Tool or callable

    Raises
    ------
    TypeError
        If ``function`` is not a function or bound method, the name is missing
        or invalid, the description is missing or blank, the function takes
        ``*args`` or ``**kwargs``, a parameter has no annotation and no
        ``input_model`` is given, ``input_model`` does not match the
        parameters, or ``ToolContext`` appears other than as one parameter
        annotated exactly ``ToolContext`` (an unannotated ``ctx`` or
        ``context`` counts).
    """

    def decorate(inner: Callable[..., Any]) -> Tool:
        if not (inspect.isfunction(inner) or inspect.ismethod(inner)):
            raise TypeError(
                f"tool() got {type(inner).__name__} {inner!r}; pass a function or "
                "bound method"
            )
        tool_name = _tool_name(inner, name)
        tool_description = _tool_description(inner, description, tool_name)
        hints = get_type_hints(inner, include_extras=True)
        context_parameter = _find_context_parameter(inner, hints)
        parameters = _argument_parameters(inner, context_parameter)
        if input_model is None:
            arguments_model = _build_input_model(parameters, hints, tool_name)
        else:
            _check_input_model(parameters, input_model, tool_name)
            arguments_model = input_model
        return_annotation = hints.get(
            "return", inspect.signature(inner).return_annotation
        )
        metadata = ToolMetadata(
            name=tool_name,
            description=tool_description,
            input_model=arguments_model,
            output_model=_build_output_model(return_annotation, tool_name),
            context_parameter=context_parameter,
            returns_model=_returns_model(return_annotation),
        )
        return Tool(inner, metadata)

    if function is not None:
        return decorate(function)

    return decorate


__all__ = [
    "Tool",
    "ToolCallResult",
    "ToolContext",
    "ToolMetadata",
    "call_tool",
    "collect_tools",
    "tool",
]
