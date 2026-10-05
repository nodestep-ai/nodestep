from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nodestep.exceptions import ContextNotProvidedError

if TYPE_CHECKING:
    from nodestep.core.agent import (
        AgentHandle,
        AgentResult,
        AgentTask,
        DurableAgentHandle,
    )
    from nodestep.core.command import Interrupt
    from nodestep.core.executor import AgentExecutor

StreamMode = str


@dataclass(frozen=True, slots=True)
class InterruptEventData:
    """Data of an ``interrupt`` event: state, thread and pending interrupts.

    Attributes
    ----------
    state : dict
        State of the paused run.
    thread_id : str
        Thread of the run.
    interrupts : dict[str, Interrupt]
        Pending interrupts keyed by ``Interrupt.key``.
    """

    state: dict[str, Any]
    thread_id: str
    interrupts: dict[str, Interrupt]


@dataclass(frozen=True, slots=True)
class FinalEventData:
    """Data of the ``final`` event: final state and thread."""

    state: dict[str, Any]
    thread_id: str


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One event produced while a graph runs.

    Attributes
    ----------
    mode : str
        ``"updates"``, ``"values"``, ``"custom"``, ``"tokens"``, ``"debug"``, or
        the terminal ``"interrupt"`` and ``"final"``.
    data : Any
        A task's update dict or ``None`` for ``"updates"``, the merged state
        for ``"values"``, the emitted value for ``"custom"`` and ``"tokens"``,
        ``InterruptEventData`` or ``FinalEventData`` for the terminal events.
        Events carry copies: changing them does not change the run.
    node : str, optional
        Node of the event; ``None`` for ``"values"``, the ``"state_merge"``
        debug event and the terminal events.
    run_id : str, optional
        Run that produced the event.
    checkpoint_id : str, optional
        History event to load the event's state at: ``superstep_completed``
        for ``"updates"`` and ``"values"``, the run's last event for the
        terminal events; ``None`` without a store.
    step : int, optional
        Superstep of the run, counted from 0.
    task_id : str, optional
        Task the event belongs to; ``None`` whenever ``node`` is ``None``.
    """

    mode: str
    data: InterruptEventData | FinalEventData | Any = None
    node: str | None = None
    run_id: str | None = None
    checkpoint_id: str | None = None
    step: int | None = None
    task_id: str | None = None


EmitFunction = Callable[[StreamEvent], None]


@dataclass(frozen=True, slots=True)
class NodeContext:
    """Run information and helpers for a running node.

    Attributes
    ----------
    graph_name : str
        Name of the graph.
    node : str
        Name of the node.
    run_id : str
        Id of this run.
    thread_id : str or None
        Thread of the run.
    branch_id : str
        History branch of the run.
    step : int
        Superstep, counted from 0.
    root_run_id : str
        Id of the outermost run when the graph runs inside another one.
    task_id : str
        Id of this task within the superstep.
    middleware : tuple
        Middleware of the run, in order.
    workspace : Workspace or None
        The graph's workspace.
    state_store : StateStore or None
        Store of the run's history.
    state_schema : StateSchema or None
        Schema of the graph state.
    stream_modes : tuple[str, ...]
        Stream modes the caller asked for.
    cache : dict
        Saved with the task when it pauses and given back on resume.
    resume_values : Mapping
        Answers to this task's interrupts by interrupt key, read-only.
    """

    graph_name: str
    run_id: str
    thread_id: str | None
    step: int
    node: str
    branch_id: str = "main"
    root_run_id: str = ""
    task_id: str = ""
    middleware: tuple[Any, ...] = ()
    workspace: Any = None
    state_store: Any = None
    state_schema: Any = None
    stream_modes: tuple[str, ...] = ()
    cache: dict[str, Any] = field(default_factory=dict)
    resume_values: Mapping[str, Any] = field(default_factory=dict)
    _agent_registry: Any = None
    _emit: Callable[[StreamEvent], None] | None = None
    _context: Any = None

    @property
    def context(self) -> Any:
        """The ``context=`` object of the run; never stored, so pass it again on resume.

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

    def emit(self, data: Any) -> None:
        """Emit a ``"custom"`` stream event with this node, step and task.

        Parameters
        ----------
        data : Any
            Event payload.
        """
        self._send("custom", data)

    def _emit_token(self, chunk: Any) -> None:
        self._send("tokens", chunk)

    def _send(self, mode: StreamMode, data: Any) -> None:
        if self._emit is None:
            return
        self._emit(
            StreamEvent(
                mode=mode,
                data=data,
                node=self.node,
                run_id=self.run_id,
                step=self.step,
                task_id=self.task_id,
            )
        )

    async def spawn(
        self,
        tasks: list[AgentTask],
        *,
        thread_prefix: str | None = None,
    ) -> list[AgentHandle]:
        """Start sub-agents concurrently; they belong to this run.

        A run that finishes or pauses with sub-agents never gathered cancels
        them and raises ``GraphExecutionError``; a run that fails or is closed
        early cancels them too. To wait in a later run, use ``submit``.

        Parameters
        ----------
        tasks : list[AgentTask]
            One task per sub-agent: the graph, its input and options.
        thread_prefix : str, optional
            Prefix of the threads ``"<prefix>:sub:<handle id>"``; defaults to
            this run's thread id.

        Returns
        -------
        list[AgentHandle]

        Raises
        ------
        TypeError
            If a task has no ``input``.
        ValueError
            If ``thread_prefix`` is empty.
        """
        from nodestep.core.agent import spawn_agents

        return await spawn_agents(
            tasks,
            parent_thread_id=self.thread_id,
            thread_prefix=thread_prefix,
            registry=self._agent_registry,
        )

    async def gather(
        self,
        handles: Sequence[AgentHandle | str],
        *,
        timeout: float | None = None,
        return_exceptions: bool = False,
    ) -> list[AgentResult]:
        """Wait for sub-agents started with ``spawn``.

        Parameters
        ----------
        handles : Sequence[AgentHandle | str]
            Handles, or ids of sub-agents spawned in this run, by any node.
            Ids do not resolve after a pause.
        timeout : float, optional
            Seconds to wait before cancelling the unfinished sub-agents.
        return_exceptions : bool, optional
            Return failed and cancelled results instead of raising.

        Returns
        -------
        list[AgentResult]

        Raises
        ------
        AgentGroupError
            If a sub-agent failed or was cancelled and ``return_exceptions``
            is False.
        AgentTimeoutError
            If the timeout expires first.
        AgentHandleLostError
            If a handle was not spawned in this run or was already gathered.
        """
        from nodestep.core.agent import gather_agents

        return await gather_agents(
            handles,
            timeout=timeout,
            return_exceptions=return_exceptions,
            registry=self._agent_registry,
        )

    async def submit(
        self,
        tasks: list[AgentTask],
        *,
        executor: AgentExecutor,
        thread_prefix: str | None = None,
    ) -> list[DurableAgentHandle]:
        """Start sub-agents on a background executor.

        Parameters
        ----------
        tasks : list[AgentTask]
            One task per sub-agent: the graph, its input and options.
        executor : AgentExecutor
            Executor that runs the sub-agents, such as ``AsyncioExecutor``.
        thread_prefix : str, optional
            Prefix of the threads ``"<prefix>:bg:<handle id>"``; defaults to
            this run's thread id.

        Returns
        -------
        list[DurableAgentHandle]

        Raises
        ------
        TypeError
            If a task has no ``input``; checked before any task starts.
        """
        from nodestep.core.agent import _check_task

        for task in tasks:
            _check_task(task, new_thread=True)
        prefix = thread_prefix if thread_prefix is not None else self.thread_id
        return [await executor.submit(task, thread_prefix=prefix) for task in tasks]

    async def poll_agents(
        self,
        handles: list[DurableAgentHandle],
        *,
        executor: AgentExecutor,
    ) -> list[AgentResult | None]:
        """Check background sub-agents without waiting.

        Parameters
        ----------
        handles : list[DurableAgentHandle]
            Handles returned by ``submit``.
        executor : AgentExecutor
            The executor the sub-agents were submitted to.

        Returns
        -------
        list[AgentResult or None]
            ``None`` for sub-agents that are still running.

        Raises
        ------
        AgentHandleLostError
            If the executor does not know a handle.
        """
        return [await executor.poll(handle.id) for handle in handles]


VALID_MODES = frozenset({"updates", "values", "debug", "custom", "tokens"})
TERMINAL_MODES = frozenset({"interrupt", "final"})


def normalize_modes(stream_mode: str | Sequence[str]) -> list[StreamMode]:
    """Validate and normalize a ``stream_mode`` argument.

    Parameters
    ----------
    stream_mode : str or Sequence[str]
        One mode or several; ``[]`` asks for the terminal events only.

    Returns
    -------
    list[str]

    Raises
    ------
    TypeError
        If ``stream_mode`` is neither a mode name nor a sequence of them.
    ValueError
        If a mode is unknown, or is a terminal mode, which is always emitted.
    """
    if isinstance(stream_mode, str):
        modes = [stream_mode]
    elif isinstance(stream_mode, Sequence):
        modes = list(stream_mode)
    else:
        raise TypeError(
            "stream_mode must be a mode name or a list of mode names, got "
            f"{type(stream_mode).__name__}; pass [] for the terminal events only"
        )
    for mode in modes:
        if mode in TERMINAL_MODES:
            raise ValueError(
                f"stream_mode '{mode}' cannot be requested; 'interrupt' and "
                "'final' events are always emitted"
            )
        if mode not in VALID_MODES:
            raise ValueError(
                f"Unknown stream_mode '{mode}'. Valid: {sorted(VALID_MODES)}"
            )
    return modes


__all__ = [
    "EmitFunction",
    "NodeContext",
    "StreamEvent",
    "StreamMode",
    "normalize_modes",
]
