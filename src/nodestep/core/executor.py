from __future__ import annotations

import asyncio
from typing import Protocol
from uuid import uuid4

from nodestep.core.agent import (
    AgentResult,
    AgentStatus,
    AgentTask,
    DurableAgentHandle,
    _AgentRun,
)
from nodestep.exceptions import AgentHandleLostError


class AgentExecutor(Protocol):
    """Protocol for running sub-agents in the background."""

    async def submit(
        self,
        task: AgentTask,
        *,
        thread_id: str | None = None,
        thread_prefix: str | None = None,
    ) -> DurableAgentHandle:
        """Start a sub-agent.

        Parameters
        ----------
        task : AgentTask
        thread_id : str, optional
            Thread to run on.
        thread_prefix : str, optional
            Without ``thread_id``, the thread is ``"<prefix>:bg:<handle id>"``;
            without either, ``"bg:<handle id>"``.

        Returns
        -------
        DurableAgentHandle

        Raises
        ------
        TypeError
            If the task has a ``state_in`` but no ``input``, or no ``input``
            and no ``thread_id``.
        """
        ...

    async def poll(self, handle_id: str) -> AgentResult | None:
        """Return the result if the sub-agent finished, else ``None``.

        Parameters
        ----------
        handle_id : str

        Returns
        -------
        AgentResult or None
            A ``"cancelled"`` result for a cancelled sub-agent.

        Raises
        ------
        AgentHandleLostError
            If the executor does not know ``handle_id``.
        """
        ...

    async def status(self, handle_id: str) -> AgentStatus:
        """Return the current status of a sub-agent.

        Parameters
        ----------
        handle_id : str

        Returns
        -------
        AgentStatus

        Raises
        ------
        AgentHandleLostError
            If the executor does not know ``handle_id``.
        """
        ...

    async def cancel(self, handle_id: str) -> bool:
        """Cancel a running sub-agent.

        Parameters
        ----------
        handle_id : str

        Returns
        -------
        bool
            Whether a sub-agent was cancelled.

        Raises
        ------
        AgentHandleLostError
            If the executor does not know ``handle_id``.
        """
        ...

    async def list_running(self) -> list[DurableAgentHandle]:
        """List the sub-agents that are still running."""
        ...


class AsyncioExecutor:
    """Background executor that runs sub-agents as asyncio tasks in this process.

    Keeps every sub-agent it started, finished or not, until ``forget``.

    Parameters
    ----------
    executor_id : str, optional
        Id stored on the handles it creates; ``"asyncio"`` by default.
    """

    def __init__(
        self,
        *,
        executor_id: str = "asyncio",
    ) -> None:
        self.executor_id = executor_id
        self._runs: dict[str, tuple[_AgentRun, asyncio.Task[AgentResult]]] = {}
        self._handles: dict[str, DurableAgentHandle] = {}

    async def submit(
        self,
        task: AgentTask,
        *,
        thread_id: str | None = None,
        thread_prefix: str | None = None,
    ) -> DurableAgentHandle:
        """Start a sub-agent as an asyncio task.

        Parameters
        ----------
        task : AgentTask
        thread_id : str, optional
            Thread to run on.
        thread_prefix : str, optional
            Without ``thread_id``, the thread is ``"<prefix>:bg:<handle id>"``;
            without either, ``"bg:<handle id>"``.

        Returns
        -------
        DurableAgentHandle

        Raises
        ------
        TypeError
            If the task has a ``state_in`` but no ``input``, or no ``input``
            and no ``thread_id``.
        ValueError
            If ``thread_prefix`` is empty.
        """
        if thread_prefix == "":
            raise ValueError("thread_prefix must not be empty; pass None for no prefix")
        handle_id = uuid4().hex[:12]
        new_thread = thread_id is None
        if thread_id is None:
            thread_id = (
                f"{thread_prefix}:bg:{handle_id}"
                if thread_prefix is not None
                else f"bg:{handle_id}"
            )
        run = _AgentRun(handle_id, task, thread_id, new_thread=new_thread)
        asyncio_task = asyncio.create_task(
            run.run(), name=f"bg-agent:{run.name}:{handle_id}"
        )
        handle = DurableAgentHandle(
            id=handle_id,
            name=run.name,
            executor_id=self.executor_id,
            graph_name=task.graph.name,
            thread_id=thread_id,
        )
        self._runs[handle_id] = (run, asyncio_task)
        self._handles[handle_id] = handle
        return handle

    def _run(self, handle_id: str) -> tuple[_AgentRun, asyncio.Task[AgentResult]]:
        if handle_id not in self._runs:
            raise AgentHandleLostError(
                handle_id,
                "this executor does not know it; it was forgotten, or was "
                "submitted to another executor or before a restart",
            )
        return self._runs[handle_id]

    async def poll(self, handle_id: str) -> AgentResult | None:
        """Return the result if the sub-agent finished, else ``None``.

        Parameters
        ----------
        handle_id : str

        Returns
        -------
        AgentResult or None
            A ``"cancelled"`` result for a cancelled sub-agent.

        Raises
        ------
        AgentHandleLostError
            If the executor does not know ``handle_id``.
        """
        run, task = self._run(handle_id)
        if not task.done():
            return None
        return run.outcome(task)

    async def status(self, handle_id: str) -> AgentStatus:
        """Return the current status of a sub-agent.

        Parameters
        ----------
        handle_id : str

        Returns
        -------
        AgentStatus

        Raises
        ------
        AgentHandleLostError
            If the executor does not know ``handle_id``.
        """
        run, task = self._run(handle_id)
        return run.status(task)

    async def cancel(self, handle_id: str) -> bool:
        """Cancel a running sub-agent and wait until it stopped.

        Parameters
        ----------
        handle_id : str

        Returns
        -------
        bool
            Whether a sub-agent was cancelled; False for sub-agents that had
            already finished.

        Raises
        ------
        AgentHandleLostError
            If the executor does not know ``handle_id``.
        """
        _, task = self._run(handle_id)
        if task.done():
            return False
        task.cancel()
        await asyncio.wait([task])
        return task.cancelled()

    async def forget(self, handle_id: str) -> None:
        """Release a finished or cancelled sub-agent; its id is unknown afterwards.

        Parameters
        ----------
        handle_id : str

        Raises
        ------
        AgentHandleLostError
            If the executor does not know ``handle_id``.
        ValueError
            If the sub-agent is still running.
        """
        _, task = self._run(handle_id)
        if not task.done():
            raise ValueError(
                f"Sub-agent {handle_id!r} is still running; cancel it or wait "
                "for its result before forget()"
            )
        del self._runs[handle_id]
        del self._handles[handle_id]

    async def list_running(self) -> list[DurableAgentHandle]:
        """List the sub-agents that are still running."""
        return [
            self._handles[handle_id]
            for handle_id, (_, task) in self._runs.items()
            if not task.done()
        ]


__all__ = ["AgentExecutor", "AsyncioExecutor"]
