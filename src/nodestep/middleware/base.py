from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, TypeVar, get_args

from nodestep.exceptions import GraphExecutionError
from nodestep.state.context import isolate

if TYPE_CHECKING:
    from nodestep.chat.messages import ChatRequest, ChatResponse
    from nodestep.core.command import Interrupt
    from nodestep.core.tool import Tool
    from nodestep.state.schema import StateSchema

S = TypeVar("S")
"""The state type of the graph a middleware context belongs to."""

HookName = Literal[
    "before_graph",
    "after_graph",
    "before_node",
    "after_node",
    "before_tool",
    "after_tool",
    "before_model",
    "after_model",
    "on_interrupt",
    "on_error",
    "on_tool_error",
    "on_run_end",
]
"""Names of the hooks a ``Middleware`` subclass may define."""

_HOOK_NAMES = frozenset(get_args(HookName))
_HOOK_PREFIXES = ("before_", "after_", "on_")


@dataclass(frozen=True, slots=True)
class Replacement:
    """Value returned by a hook to replace the state, update or tool value.

    Attributes
    ----------
    value : Any
        What the hook puts in place of the value it received.
    """

    value: Any


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """How a run ended, passed to ``Middleware.on_run_end``.

    Attributes
    ----------
    status : {"completed", "paused", "failed", "cancelled"}
        ``"cancelled"`` also covers a stream closed before its last event.
    interrupts : Mapping[str, Interrupt]
        Pending interrupts by key, read-only; empty unless the run paused.
    error : BaseException or None
        What the run raised; set only when ``status`` is ``"failed"``.
    """

    status: Literal["completed", "paused", "failed", "cancelled"]
    interrupts: Mapping[str, Interrupt]
    error: BaseException | None


@dataclass(frozen=True)
class GraphMiddlewareContext(Generic[S]):
    """Context of ``before_graph``, ``after_graph`` and ``on_run_end``.

    Attributes
    ----------
    state : S
        A copy of the run's state.
    root_run_id : str
        Id of the outermost run when the graph runs inside another one.
    resuming : bool
        Whether the run was started with ``resume=``.
    """

    graph_name: str
    state: S
    run_id: str
    thread_id: str
    branch_id: str
    root_run_id: str = ""
    resuming: bool = False


@dataclass(frozen=True)
class NodeMiddlewareContext(Generic[S]):
    """Context of the node hooks.

    Attributes
    ----------
    state : S
        The node input in ``before_node``, after earlier replacements; the
        node's update in ``after_node``.
    stored_state : S
        A copy of the input as stored, before any ``before_node`` replacement.
    step : int
        Superstep, counted from 0.
    task_id : str
        Id of the node task.
    root_run_id : str
        Id of the outermost run when the graph runs inside another one.
    superstep_size : int
        Number of node tasks in this superstep.
    scratch : dict[str, Any]
        A dict shared by the hooks of one node task.
    state_schema : StateSchema or None
        The graph's state schema.
    """

    graph_name: str
    node_name: str
    state: S
    stored_state: S
    run_id: str
    thread_id: str
    step: int
    task_id: str = ""
    root_run_id: str = ""
    superstep_size: int = 1
    scratch: dict[str, Any] = field(default_factory=dict)
    state_schema: StateSchema | None = None

    def replace(self, value: S) -> Replacement:
        """Wrap a new node input or update as the hook's return value."""
        return Replacement(value)


@dataclass(frozen=True)
class ToolMiddlewareContext(Generic[S]):
    """Context of ``before_tool``, ``after_tool`` and ``on_tool_error``.

    Attributes
    ----------
    value : Any
        The arguments, or the result in ``after_tool``.
    state : S or None
        State the calling node received; ``None`` outside a graph run.
    root_run_id : str or None
        Id of the outermost run when the graph runs inside another one.
    task_id : str
        The calling node's task, ``""`` outside a graph run.
    """

    graph_name: str | None
    node_name: str | None
    tool_name: str
    value: Any
    state: S | None = None
    run_id: str | None = None
    thread_id: str | None = None
    tool_call_id: str | None = None
    root_run_id: str | None = None
    task_id: str = ""

    def replace(self, value: Any) -> Replacement:
        """Wrap new tool arguments or a new result as the hook's return value."""
        return Replacement(value)


@dataclass(frozen=True, slots=True)
class ModelMiddlewareContext:
    """Context of ``before_model`` and ``after_model``.

    Attributes
    ----------
    step : int
        Superstep, counted from 0.
    task_id : str
        Id of the node task that calls the model.
    root_run_id : str
        Id of the outermost run when the graph runs inside another one.
    model : str
        The chat model's ``model`` name.
    request : ChatRequest
        The request; in ``after_model``, what was sent.
    response : ChatResponse or None
        The response, ``None`` in ``before_model``.
    """

    graph_name: str
    node_name: str
    run_id: str
    thread_id: str | None
    step: int
    task_id: str
    root_run_id: str
    model: str
    request: ChatRequest
    response: ChatResponse | None = None

    def replace(self, value: ChatRequest | ChatResponse) -> Replacement:
        """Wrap a new request or response as the hook's return value."""
        return Replacement(value)


class ToolProvider(Protocol):
    """Protocol for objects that contribute tools."""

    def tools(self) -> Iterable[Tool]:
        """Return the tools this object provides."""
        ...


class Middleware:
    """Base class for hooks around graphs, nodes, model calls and tools.

    Override the hooks you need. The order they run in and what each may
    return: [Middleware](../../concepts/middleware.md).

    Raises
    ------
    TypeError
        If a subclass defines a ``before_*``, ``after_*`` or ``on_*`` method
        that is not a hook, such as ``after_tools``.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name, value in vars(cls).items():
            is_method = inspect.isfunction(value) or isinstance(
                value, (staticmethod, classmethod)
            )
            if (
                is_method
                and name.startswith(_HOOK_PREFIXES)
                and name not in _HOOK_NAMES
            ):
                known = ", ".join(sorted(_HOOK_NAMES))
                raise TypeError(
                    f"{cls.__name__}.{name} is not a middleware hook; the hooks are "
                    f"{known}"
                )

    def before_graph(self, ctx: GraphMiddlewareContext) -> Awaitable[None] | None:
        """Observe the state before the first superstep, also on a resume.

        Returns
        -------
        None
            Returning a ``Replacement`` raises ``GraphExecutionError``; any
            other value raises ``TypeError``.
        """
        return None

    def after_graph(self, ctx: GraphMiddlewareContext) -> Awaitable[None] | None:
        """Observe the final state; called only when the run completes.

        Returns
        -------
        None
            Returning a ``Replacement`` raises ``GraphExecutionError``; any
            other value raises ``TypeError``.
        """
        return None

    def on_run_end(
        self, ctx: GraphMiddlewareContext, outcome: RunOutcome
    ) -> Awaitable[None] | None:
        """Observe the end of a run, however it ended.

        Called once per run for each middleware whose ``before_graph`` was
        called; every hook runs, even when one raises. See
        [on_run_end](../../concepts/middleware.md#on_run_end).

        Parameters
        ----------
        ctx : GraphMiddlewareContext
        outcome : RunOutcome

        Returns
        -------
        None
            Returning a ``Replacement`` raises ``GraphExecutionError``; any
            other value raises ``TypeError``.
        """
        return None

    def before_node(
        self, ctx: NodeMiddlewareContext
    ) -> Replacement | Awaitable[Replacement | None] | None:
        """Called with a copy of the node's input before the node runs.

        Return ``ctx.replace(value)`` to give the node a different input; it is
        not stored.

        Returns
        -------
        Replacement or None
            A replacement value, or ``None`` to keep the current one.
        """
        return None

    def after_node(
        self, ctx: NodeMiddlewareContext
    ) -> Replacement | Awaitable[Replacement | None] | None:
        """Called with the node's update before it is merged.

        Returns
        -------
        Replacement or None
            A replacement value, or ``None`` to keep the current one.
        """
        return None

    def before_tool(
        self, ctx: ToolMiddlewareContext
    ) -> Replacement | Awaitable[Replacement | None] | None:
        """Called with validated tool arguments; may raise ``ToolDeniedError``.

        Returns
        -------
        Replacement or None
            A replacement value, or ``None`` to keep the current one.
        """
        return None

    def after_tool(
        self, ctx: ToolMiddlewareContext
    ) -> Replacement | Awaitable[Replacement | None] | None:
        """Called with the validated tool result.

        Returns
        -------
        Replacement or None
            A replacement value, or ``None`` to keep the current one.
        """
        return None

    def before_model(
        self, ctx: ModelMiddlewareContext
    ) -> Replacement | Awaitable[Replacement | None] | None:
        """Called by ``model_node`` with the request before every model call.

        Returns
        -------
        Replacement or None
            ``ctx.replace(request)`` with a ``ChatRequest`` to send instead, or
            ``None`` to keep the current one.
        """
        return None

    def after_model(
        self, ctx: ModelMiddlewareContext
    ) -> Replacement | Awaitable[Replacement | None] | None:
        """Called by ``model_node`` with the response after every model call.

        With ``model_node(stream=True)`` it runs after the stream ended; the
        ``"tokens"`` events carried the model's own chunks.

        Returns
        -------
        Replacement or None
            ``ctx.replace(response)`` with a ``ChatResponse`` the node uses
            instead, or ``None`` to keep the current one.
        """
        return None

    def on_interrupt(self, ctx: NodeMiddlewareContext, payload: Any) -> Any | None:
        """Answer an interrupt without pausing.

        Must be sync: returning an awaitable raises ``TypeError``. A
        ``subgraph`` child's unanswered interrupts come here with the subgraph
        node's context. See
        [Answering in code](../../concepts/interrupts.md#answering-in-code).

        Parameters
        ----------
        ctx : NodeMiddlewareContext
            ``state`` is a copy of the node's input, after ``before_node``
            replacements.
        payload : Any
            Payload passed to ``interrupt``.

        Returns
        -------
        Any or None
            The answer, or ``None`` to pass to the next middleware and finally
            pause the run.
        """
        return None

    def on_error(
        self, ctx: NodeMiddlewareContext, error: Exception
    ) -> Any | Awaitable[Any | None] | None:
        """Recover from a node error.

        Parameters
        ----------
        ctx : NodeMiddlewareContext
            Its ``state`` is a private copy of the node's input.
        error : Exception
            What the node raised.

        Returns
        -------
        Any or None
            A result to use instead, in any form a node may return (``ctx.state``
            counts as the received state), or ``None`` to fail the run. Changing
            ``ctx.state`` and returning anything else raises
            ``GraphExecutionError``.
        """
        return None

    def on_tool_error(
        self, ctx: ToolMiddlewareContext, error: Exception
    ) -> Awaitable[None] | None:
        """Observe a tool that raised, or returned a value its output model rejects.

        It cannot recover: the tool's error is raised after every hook ran,
        with each hook's own error added as a note. See
        [on_tool_error](../../concepts/middleware.md#on_tool_error).

        Parameters
        ----------
        ctx : ToolMiddlewareContext
            ``value`` holds the arguments the tool ran with.
        error : Exception
            What the tool raised, or the validation error of its result.

        Returns
        -------
        None
            A ``Replacement`` becomes a ``GraphExecutionError`` note on the
            error, any other value a ``TypeError`` note.
        """
        return None

    def tools(self) -> Iterable[Tool]:
        """Return the tools this middleware provides.

        No node gets them implicitly; see
        [Middleware tools](../../concepts/middleware.md#middleware-tools).
        """
        return ()


async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _invalid_return(item: Middleware, hook_name: str, result: Any) -> TypeError:
    return TypeError(
        f"{type(item).__name__}.{hook_name} returned {type(result).__name__}; "
        "a hook returns None or ctx.replace(value)"
    )


async def _run_hooks(
    middleware: Iterable[Middleware],
    hook_name: HookName,
    value: Any,
    make_context: Callable[[Any], Any | Awaitable[Any]],
) -> Any:
    if hook_name in ("before_graph", "after_graph"):
        raise ValueError(
            f"{hook_name} is read-only and cannot replace a value; use run_graph_hooks"
        )
    current = value
    for item in middleware:
        ctx = await _resolve(make_context(current))
        hook = getattr(item, hook_name)
        result = await _resolve(hook(ctx))
        if isinstance(result, Replacement):
            current = result.value
        elif result is not None:
            raise _invalid_return(item, hook_name, result)
    return current


async def run_graph_hooks(
    middleware: Iterable[Middleware],
    hook_name: Literal["before_graph", "after_graph"],
    state: Any,
    make_context: Callable[[Any], Any | Awaitable[Any]],
) -> None:
    """Run the read-only ``before_graph`` or ``after_graph`` hook of each middleware.

    ``before_graph`` runs in list order and ``after_graph`` in reverse order.
    Each hook gets its own copy of ``state``.

    Parameters
    ----------
    middleware : Iterable[Middleware]
    hook_name : {"before_graph", "after_graph"}
    state : Any
        The run's state; it is never changed.
    make_context : callable
        Builds the hook context for a copy of the state.

    Raises
    ------
    GraphExecutionError
        If a hook returns a ``Replacement``.
    TypeError
        If a hook returns anything else but ``None``.
    """
    ordered = tuple(middleware)
    if hook_name == "after_graph":
        ordered = ordered[::-1]
    for item in ordered:
        ctx = await _resolve(make_context(isolate(state)))
        result = await _resolve(getattr(item, hook_name)(ctx))
        if isinstance(result, Replacement):
            raise GraphExecutionError(
                f"{type(item).__name__}.{hook_name} returned a Replacement; "
                "graph hooks are read-only"
            )
        if result is not None:
            raise TypeError(
                f"{type(item).__name__}.{hook_name} returned "
                f"{type(result).__name__}; graph hooks return None"
            )


def _check_observer_return(item: Middleware, hook_name: str, result: Any) -> None:
    if isinstance(result, Replacement):
        raise GraphExecutionError(
            f"{type(item).__name__}.{hook_name} returned a Replacement; "
            f"{hook_name} is read-only"
        )
    if result is not None:
        raise TypeError(
            f"{type(item).__name__}.{hook_name} returned {type(result).__name__}; "
            f"{hook_name} is read-only and returns None"
        )


@dataclass(slots=True)
class _Observed:
    failures: list[tuple[Middleware, Exception]] = field(default_factory=list)
    stop: BaseException | None = None


async def _observe(
    middleware: Iterable[Middleware],
    hook_name: Literal["on_run_end", "on_tool_error"],
    call: Callable[[Middleware], Any],
) -> _Observed:
    observed = _Observed()
    for item in reversed(tuple(middleware)):
        try:
            result = await _resolve(call(item))
            _check_observer_return(item, hook_name, result)
        except Exception as error:
            observed.failures.append((item, error))
        except BaseException as stop:
            if observed.stop is None:
                observed.stop = stop
    return observed


def _note(
    error: BaseException,
    hook_name: str,
    failures: Iterable[tuple[Middleware, Exception]],
) -> None:
    for item, failure in failures:
        error.add_note(
            f"{type(item).__name__}.{hook_name} raised "
            f"{type(failure).__name__}: {failure}"
        )


async def run_end_hooks(
    middleware: Iterable[Middleware],
    state: Any,
    make_context: Callable[[Any], Any | Awaitable[Any]],
    outcome: RunOutcome,
) -> None:
    """Run the read-only ``on_run_end`` hook of each middleware in reverse order.

    Every hook runs, and each gets its own copy of ``state``.

    Parameters
    ----------
    middleware : Iterable[Middleware]
    state : Any
        The run's last merged state; it is never changed.
    make_context : callable
        Builds the hook context for a copy of the state.
    outcome : RunOutcome

    Raises
    ------
    BaseException
        The first exception a hook raised that does not subclass
        ``Exception``, such as ``asyncio.CancelledError``.
    Exception
        The first error a hook raised, with the later ones as notes, unless
        ``outcome.error`` is set: then the hook errors become notes on it and
        nothing is raised here, so the caller raises the run's error.
    """

    async def call(item: Middleware) -> Any:
        ctx = await _resolve(make_context(isolate(state)))
        return await _resolve(item.on_run_end(ctx, outcome))

    observed = await _observe(middleware, "on_run_end", call)
    failures = observed.failures
    if outcome.error is not None:
        _note(outcome.error, "on_run_end", failures)
    if observed.stop is not None:
        if outcome.error is None:
            _note(observed.stop, "on_run_end", failures)
        raise observed.stop
    if not failures or outcome.error is not None:
        return
    (_, first), *later = failures
    _note(first, "on_run_end", later)
    raise first


async def run_tool_error_hooks(
    middleware: Iterable[Middleware],
    ctx: ToolMiddlewareContext,
    error: Exception,
) -> None:
    """Run the observe-only ``on_tool_error`` hook of each middleware in reverse order.

    Every hook runs; an error a hook raises, or a value it returns, is added
    to ``error`` as a note.

    Parameters
    ----------
    middleware : Iterable[Middleware]
    ctx : ToolMiddlewareContext
    error : Exception
        The tool's error; the caller raises it.

    Raises
    ------
    BaseException
        The first exception a hook raised that does not subclass
        ``Exception``, such as ``asyncio.CancelledError``.
    """
    observed = await _observe(
        middleware,
        "on_tool_error",
        lambda item: item.on_tool_error(ctx, error),
    )
    _note(error, "on_tool_error", observed.failures)
    if observed.stop is not None:
        raise observed.stop


async def run_before_hooks(
    middleware: Iterable[Middleware],
    hook_name: HookName,
    value: Any,
    make_context: Callable[[Any], Any | Awaitable[Any]],
) -> Any:
    """Run a ``before_*`` hook of each middleware in order.

    Parameters
    ----------
    middleware : Iterable[Middleware]
    hook_name : str
        The ``before_*`` hook to run, such as ``"before_node"``.
    value : Any
        Current value, replaced by each hook's ``Replacement``.
    make_context : callable
        Builds the hook context for the current value.

    Returns
    -------
    Any

    Raises
    ------
    ValueError
        If ``hook_name`` is ``"before_graph"`` or ``"after_graph"``.
    TypeError
        If a hook returns anything but ``None`` or a ``Replacement``.
    """
    return await _run_hooks(middleware, hook_name, value, make_context)


async def run_after_hooks(
    middleware: Iterable[Middleware],
    hook_name: HookName,
    value: Any,
    make_context: Callable[[Any], Any | Awaitable[Any]],
) -> Any:
    """Run an ``after_*`` hook of each middleware in reverse order.

    Parameters
    ----------
    middleware : Iterable[Middleware]
    hook_name : str
        The ``after_*`` hook to run, such as ``"after_node"``.
    value : Any
        Current value, replaced by each hook's ``Replacement``.
    make_context : callable
        Builds the hook context for the current value.

    Returns
    -------
    Any

    Raises
    ------
    ValueError
        If ``hook_name`` is ``"before_graph"`` or ``"after_graph"``.
    TypeError
        If a hook returns anything but ``None`` or a ``Replacement``.
    """
    return await _run_hooks(reversed(tuple(middleware)), hook_name, value, make_context)


def _checked_middleware(middleware: Iterable[Any]) -> tuple[Middleware, ...]:
    items = tuple(middleware)
    for item in items:
        if not isinstance(item, Middleware):
            raise TypeError(
                f"{type(item).__name__} is not middleware; middleware must "
                "subclass nodestep.Middleware"
            )
    return items


def collect_middleware_tools(middleware: Iterable[Middleware]) -> list[Any]:
    """Return the tools of a middleware list, in list order.

    Parameters
    ----------
    middleware : Iterable[Middleware]

    Returns
    -------
    list[Tool]

    Raises
    ------
    TypeError
        If an item does not subclass ``Middleware``.
    """
    tools: list[Any] = []
    for item in _checked_middleware(middleware):
        tools.extend(item.tools())
    return tools
