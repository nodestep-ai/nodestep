from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from nodestep.core.stream import FinalEventData, InterruptEventData, StreamEvent
from nodestep.exceptions import (
    AgentGroupError,
    AgentHandleLostError,
    AgentTimeoutError,
    GraphExecutionError,
)

if TYPE_CHECKING:
    from nodestep.core.graph import Graph


@dataclass(frozen=True, slots=True)
class AgentStatus:
    """Progress of a sub-agent, derived from its own stream events.

    Attributes
    ----------
    id : str
        Handle id of the sub-agent.
    name : str
        Display name of the sub-agent.
    status : {"running", "completed", "interrupted", "error", "cancelled"}
        Where the sub-agent is.
    step : int
        Number of supersteps the sub-agent's thread has finished; 0 before
        the first one finishes.
    node : str, optional
        Last node that finished; ``None`` before the first superstep.
    thread_id : str, optional
        Thread the sub-agent runs on.
    updated_at : float
        ``time.time()`` of the last change.
    """

    id: str
    name: str
    status: Literal["running", "completed", "interrupted", "error", "cancelled"]
    step: int
    node: str | None
    thread_id: str | None
    updated_at: float


@dataclass(frozen=True, slots=True)
class AgentTask:
    """A graph to run as a sub-agent.

    Attributes
    ----------
    graph : Graph
        Graph to run.
    input : dict or BaseModel or None
        Input of the sub-agent's run. ``None`` continues an existing thread and
        is accepted only by ``AgentExecutor.submit`` with ``thread_id=``.
    name : str, optional
        Display name; defaults to the graph name.
    state_in : callable, optional
        Maps ``input`` before the run; needs an ``input``.
    state_out : callable, optional
        Maps the final state dict to the result's ``data``; must return a dict.
    on_progress : callable, optional
        Called with the ``AgentStatus`` after every superstep and once at the
        end. If it raises, the sub-agent stops with an ``"error"`` result
        holding that exception and the callback is not called again; an
        exception raised for ``"cancelled"`` is dropped.
    context : object, optional
        ``context=`` of the sub-agent's run; the parent's context is not passed.
    """

    graph: Graph
    input: Any
    name: str | None = None
    state_in: Callable[[Any], Any] | None = None
    state_out: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    on_progress: Callable[[AgentStatus], None] | None = None
    context: object | None = None


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Outcome of a sub-agent run.

    Attributes
    ----------
    id : str
        Id of the handle the sub-agent was started with.
    state : Any
        Final state as the graph's state type; ``None`` when the run failed.
    data : dict
        Final state as a dict, or what ``state_out`` returned; empty when the
        run failed.
    status : str
        ``"completed"``, ``"interrupted"``, ``"error"`` or ``"cancelled"``.
    error : BaseException or None
        What stopped a failed or cancelled run.
    thread_id : str or None
        Thread the sub-agent ran on.
    checkpoint_id : str or None
        History event to load the final or interrupted state at; ``None``
        without a state store.
    """

    id: str
    name: str
    state: Any
    data: dict[str, Any]
    status: str
    error: BaseException | None = None
    thread_id: str | None = None
    checkpoint_id: str | None = None

    def __repr__(self) -> str:
        return (
            f"AgentResult(id={self.id!r}, name={self.name!r}, status={self.status!r})"
        )


def _check_task(task: AgentTask, *, new_thread: bool) -> None:
    """Refuse a task that cannot start.

    Parameters
    ----------
    task : AgentTask
    new_thread : bool
        Whether the sub-agent runs on a newly generated thread.

    Raises
    ------
    TypeError
        If the task has a ``state_in`` but no ``input``, or no ``input`` for a
        new thread.
    """
    if task.input is not None:
        return
    if task.state_in is not None:
        raise TypeError(
            f"AgentTask for graph '{task.graph.name}' has state_in but no "
            "input; pass input= to map, or drop state_in"
        )
    if new_thread:
        raise TypeError(
            f"AgentTask for graph '{task.graph.name}' has no input, but it would "
            "run on a new thread; input=None continues an existing thread, "
            "which AgentExecutor.submit takes as thread_id="
        )


class _AgentRun:
    def __init__(
        self, handle_id: str, task: AgentTask, thread_id: str, *, new_thread: bool
    ) -> None:
        _check_task(task, new_thread=new_thread)
        self.id = handle_id
        self.task = task
        self.name = task.name or task.graph.name
        self.thread_id = thread_id
        self.input = task.state_in(task.input) if task.state_in else task.input
        self._callback_failed = False
        self._status = AgentStatus(
            id=handle_id,
            name=self.name,
            status="running",
            step=0,
            node=None,
            thread_id=thread_id,
            updated_at=time.time(),
        )

    def status(self, task: asyncio.Task[AgentResult]) -> AgentStatus:
        """Return the current status of the run.

        Parameters
        ----------
        task : asyncio.Task
            The asyncio task running ``run()``.

        Returns
        -------
        AgentStatus
        """
        if task.cancelled() and self._status.status != "cancelled":
            return replace(self._status, status="cancelled")
        return self._status

    def outcome(self, task: asyncio.Task[AgentResult]) -> AgentResult:
        """Return the result of the finished asyncio task running ``run()``.

        Parameters
        ----------
        task : asyncio.Task
            A finished task.

        Returns
        -------
        AgentResult
            A ``"cancelled"`` result, holding a ``CancelledError``, for a
            cancelled task.
        """
        if task.cancelled():
            return self._result(None, {}, "cancelled", error=asyncio.CancelledError())
        return task.result()

    async def run(self) -> AgentResult:
        """Run the sub-agent and report its progress.

        Returns
        -------
        AgentResult
            Also for a sub-agent that failed; the exception is in ``error``.

        Raises
        ------
        asyncio.CancelledError
            When the run is cancelled, after reporting ``"cancelled"``.
        """
        node: str | None = None
        terminal: StreamEvent | None = None
        try:
            async for event in self.task.graph.astream(
                self.input,
                stream_mode=["updates", "values"],
                thread_id=self.thread_id,
                context=self.task.context,
            ):
                if event.mode == "updates":
                    node = event.node
                elif event.mode == "values":
                    step = None if event.step is None else event.step + 1
                    self._report("running", step, node)
                elif event.mode in ("final", "interrupt"):
                    terminal = event
            if terminal is None:
                raise GraphExecutionError(
                    f"Graph '{self.task.graph.name}' produced no final event"
                )
            data = terminal.data
            assert isinstance(data, FinalEventData | InterruptEventData)
            output = (
                self.task.state_out(data.state) if self.task.state_out else data.state
            )
            if not isinstance(output, dict):
                raise TypeError(
                    f"state_out of sub-agent '{self.name}' returned "
                    f"{type(output).__name__}; it must return a dict"
                )
            status = "interrupted" if terminal.mode == "interrupt" else "completed"
            result = self._result(
                self.task.graph.state_schema.to_declared(data.state),
                output,
                status,
                checkpoint_id=terminal.checkpoint_id,
            )
            self._report(status, None, node)
            return result
        except asyncio.CancelledError:
            with suppress(Exception):
                self._report("cancelled", None, node)
            raise
        except Exception as error:
            try:
                self._report("error", None, node)
            except Exception as callback_error:
                return self._result(None, {}, "error", error=callback_error)
            return self._result(None, {}, "error", error=error)

    def _report(
        self,
        status: Literal["running", "completed", "interrupted", "error", "cancelled"],
        step: int | None,
        node: str | None,
    ) -> None:
        self._status = replace(
            self._status,
            status=status,
            step=self._status.step if step is None else step,
            node=node,
            updated_at=time.time(),
        )
        if self.task.on_progress is None or self._callback_failed:
            return
        try:
            self.task.on_progress(self._status)
        except Exception:
            self._callback_failed = True
            raise

    def _result(
        self,
        state: Any,
        data: dict[str, Any],
        status: str,
        *,
        error: BaseException | None = None,
        checkpoint_id: str | None = None,
    ) -> AgentResult:
        return AgentResult(
            id=self.id,
            name=self.name,
            state=state,
            data=data,
            status=status,
            error=error,
            thread_id=self.thread_id,
            checkpoint_id=checkpoint_id,
        )


@dataclass
class AgentHandle:
    """Handle to a sub-agent running in the current process.

    Attributes
    ----------
    task : asyncio.Task[AgentResult]
        The asyncio task running the sub-agent.
    thread_id : str
        Thread the sub-agent runs on.
    """

    id: str
    name: str
    task: asyncio.Task[AgentResult]
    graph_name: str
    thread_id: str
    _run: _AgentRun = field(init=False, repr=False)

    def done(self) -> bool:
        """Whether the sub-agent has finished."""
        return self.task.done()

    async def status(self) -> AgentStatus:
        """Return the current status of the sub-agent.

        Returns
        -------
        AgentStatus
        """
        return self._run.status(self.task)

    def __deepcopy__(self, memo: dict[int, Any]) -> AgentHandle:
        """Return the handle itself, e.g. when an error holding it is copied."""
        return self

    def __repr__(self) -> str:
        return f"AgentHandle(id={self.id!r}, name={self.name!r}, done={self.done()})"


@dataclass(frozen=True, slots=True)
class DurableAgentHandle:
    """Handle to a sub-agent submitted to an ``AgentExecutor``.

    Attributes
    ----------
    executor_id : str
        ``executor_id`` of the executor that runs the sub-agent.
    thread_id : str or None
        Thread the sub-agent runs on.
    """

    id: str
    name: str
    executor_id: str
    graph_name: str
    thread_id: str | None = None


class AgentRegistry:
    """Tracks the sub-agents spawned during one graph run."""

    def __init__(self) -> None:
        self._handles: dict[str, AgentHandle] = {}

    def register(self, handle: AgentHandle) -> None:
        """Remember a spawned sub-agent.

        Parameters
        ----------
        handle : AgentHandle
        """
        self._handles[handle.id] = handle

    def resolve(self, handle_id: str) -> AgentHandle:
        """Look up a spawned sub-agent.

        Parameters
        ----------
        handle_id : str

        Returns
        -------
        AgentHandle

        Raises
        ------
        AgentHandleLostError
            If the handle is not registered.
        """
        if handle_id not in self._handles:
            raise AgentHandleLostError(
                handle_id,
                "it was not spawned in this run or was already gathered; "
                "sub-agents started with ctx.spawn belong to the run that "
                "started them, and a pause ends the run, so use ctx.submit "
                "with an AgentExecutor to wait for them across runs",
            )
        return self._handles[handle_id]

    def remove(self, handle_id: str) -> None:
        """Forget a sub-agent.

        Parameters
        ----------
        handle_id : str
        """
        self._handles.pop(handle_id, None)

    async def cancel_all(self) -> list[AgentHandle]:
        """Cancel and forget every sub-agent that was not gathered.

        Returns
        -------
        list[AgentHandle]
            The sub-agents that were still registered, in spawn order,
            including finished ones whose results nobody collected.
        """
        handles = list(self._handles.values())
        self._handles.clear()
        for handle in handles:
            handle.task.cancel()
        await asyncio.gather(
            *(handle.task for handle in handles), return_exceptions=True
        )
        return handles


async def spawn_agents(
    tasks: list[AgentTask],
    *,
    parent_thread_id: str | None = None,
    thread_prefix: str | None = None,
    registry: AgentRegistry | None = None,
) -> list[AgentHandle]:
    """Start sub-agents concurrently on their own threads.

    Each sub-agent runs on the thread ``"<prefix>:sub:<handle id>"``, or
    ``"sub:<handle id>"`` without a prefix. Every ``state_in`` runs before any
    sub-agent starts.

    Parameters
    ----------
    tasks : list[AgentTask]
        Sub-agents to start.
    parent_thread_id : str, optional
        Prefix for the generated thread ids.
    thread_prefix : str, optional
        Thread prefix; wins over ``parent_thread_id``.
    registry : AgentRegistry, optional
        Registry that tracks the handles.

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
    if thread_prefix == "":
        raise ValueError("thread_prefix must not be empty; pass None for no prefix")
    prefix = thread_prefix if thread_prefix is not None else parent_thread_id
    runs: list[_AgentRun] = []
    for agent_task in tasks:
        handle_id = uuid4().hex[:12]
        thread_id = f"{prefix}:sub:{handle_id}" if prefix else f"sub:{handle_id}"
        runs.append(_AgentRun(handle_id, agent_task, thread_id, new_thread=True))

    handles: list[AgentHandle] = []
    for run in runs:
        handle = AgentHandle(
            id=run.id,
            name=run.name,
            task=asyncio.create_task(run.run(), name=f"agent:{run.name}:{run.id}"),
            graph_name=run.task.graph.name,
            thread_id=run.thread_id,
        )
        handle._run = run
        if registry is not None:
            registry.register(handle)
        handles.append(handle)
    return handles


async def gather_agents(
    handles: Sequence[AgentHandle | str],
    *,
    timeout: float | None = None,
    return_exceptions: bool = False,
    registry: AgentRegistry | None = None,
) -> list[AgentResult]:
    """Wait for sub-agents and collect their results.

    Parameters
    ----------
    handles : Sequence[AgentHandle | str]
        Handles returned by ``spawn_agents``, or their ids, which need
        ``registry``.
    timeout : float, optional
        Seconds to wait before cancelling the unfinished sub-agents.
    return_exceptions : bool, optional
        Return failed and cancelled results, with the exception in
        ``AgentResult.error``, instead of raising.
    registry : AgentRegistry, optional
        Registry the handles were registered with.

    Returns
    -------
    list[AgentResult]
        Results in the order of ``handles``.

    Raises
    ------
    AgentTimeoutError
        If the timeout expires first.
    AgentGroupError
        If a sub-agent failed or was cancelled and ``return_exceptions`` is
        False; it holds every result.
    AgentHandleLostError
        If a handle is not in ``registry``.
    TypeError
        If a handle id is given without ``registry``.
    """
    if registry is not None:
        resolved = [
            registry.resolve(handle if isinstance(handle, str) else handle.id)
            for handle in handles
        ]
    else:
        ids = [handle for handle in handles if isinstance(handle, str)]
        if ids:
            raise TypeError(
                f"gather_agents got the handle ids {ids} without registry=; "
                "pass the AgentRegistry they were spawned with, or the handles"
            )
        resolved = [handle for handle in handles if isinstance(handle, AgentHandle)]

    tasks = [handle.task for handle in resolved]
    if not tasks:
        return []
    if timeout is not None:
        done, pending_tasks = await asyncio.wait(tasks, timeout=timeout)
        completed: list[AgentResult] = []
        pending_handles: list[AgentHandle] = []
        for handle in resolved:
            if handle.task in done:
                completed.append(handle._run.outcome(handle.task))
                if registry is not None:
                    registry.remove(handle.id)
            else:
                pending_handles.append(handle)
        if pending_tasks:
            for task in pending_tasks:
                task.cancel()
            await asyncio.gather(*pending_tasks, return_exceptions=True)
            if registry is not None:
                for handle in pending_handles:
                    registry.remove(handle.id)
            raise AgentTimeoutError(completed, pending_handles)
        results = completed
    else:
        await asyncio.gather(*tasks, return_exceptions=True)
        results = [handle._run.outcome(handle.task) for handle in resolved]
        if registry is not None:
            for handle in resolved:
                registry.remove(handle.id)
    if not return_exceptions and any(result.error is not None for result in results):
        raise AgentGroupError(results)
    return results


__all__ = [
    "AgentHandle",
    "AgentRegistry",
    "AgentResult",
    "AgentStatus",
    "AgentTask",
    "DurableAgentHandle",
    "gather_agents",
    "spawn_agents",
]
