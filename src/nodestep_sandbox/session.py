import asyncio
import copy
from collections.abc import AsyncIterator, Mapping
from typing import Any
from uuid import uuid4

from nodestep import Graph, InMemoryStateStore, Resume, StreamEvent
from nodestep.state import History, StateStore
from nodestep_sandbox.errors import RunStateError
from nodestep_sandbox.records import ThreadSummary, json_text
from nodestep_sandbox.runs import Run, RunKind

STREAM_MODES = ("updates", "custom", "debug")
STOP_TIMEOUT = 2.0


class Sandbox:
    """Runs a copy of one graph and keeps the runs of this session.

    The sandbox runs a shallow copy of the graph. When the graph has no state
    store, the copy gets a new ``InMemoryStateStore``; the graph passed in is
    never changed. Every run starts a new thread, streams
    ``["updates", "custom", "debug"]`` and gets ``context``.

    Parameters
    ----------
    graph : Graph
    context : object, optional
        Passed as ``context=`` to every run and resume.

    Attributes
    ----------
    source : Graph
        The graph that was loaded.
    graph : Graph
        The shallow copy the sandbox runs.
    store : StateStore
        The copy's state store: the graph's own, or the new in-memory one.
    in_memory_store : bool
        Whether the sandbox created ``store``.
    runs : dict[str, Run]
        Runs of this session by id.
    """

    def __init__(self, graph: Graph[Any], *, context: object | None = None) -> None:
        self.source = graph
        self.in_memory_store = graph.state_store is None
        self.store: StateStore = (
            InMemoryStateStore() if graph.state_store is None else graph.state_store
        )
        self.graph: Graph[Any] = copy.copy(graph)
        self.graph.state_store = self.store
        self.context = context
        self.runs: dict[str, Run] = {}
        self._threads: dict[str, list[Run]] = {}
        self._tasks: dict[asyncio.Task[None], Run] = {}

    def start(self, value: Mapping[str, Any]) -> Run:
        """Start a run on a new thread with ``value`` as input.

        Parameters
        ----------
        value : Mapping[str, Any]
            Run input, passed to ``Graph.astream``.

        Returns
        -------
        Run
            The run, recording in the background.
        """
        thread_id = f"sandbox-{uuid4().hex[:12]}"
        run = self._new_run(thread_id, kind="start", input_json=json_text(value))
        self._launch(
            run,
            self.graph.astream(
                dict(value),
                stream_mode=list(STREAM_MODES),
                thread_id=thread_id,
                state_store=self.store,
                context=self.context,
            ),
        )
        return run

    def resume(self, run: Run, answers: Mapping[str, Any]) -> Run:
        """Resume the thread of a paused run with ``Resume(answers=answers)``.

        Parameters
        ----------
        run : Run
            A paused run that nothing has answered yet.
        answers : Mapping[str, Any]
            One answer per pending interrupt key.

        Returns
        -------
        Run
            The new run on the same thread.

        Raises
        ------
        RunStateError
            If the run was answered already or is not paused.
        """
        self.check_answerable(run)
        resumed = self._new_run(
            run.thread_id,
            kind="resume",
            input_json=json_text(answers),
            previous=run.id,
        )
        self._launch(
            resumed,
            self.graph.astream(
                None,
                stream_mode=list(STREAM_MODES),
                thread_id=run.thread_id,
                resume=Resume(answers=dict(answers)),
                state_store=self.store,
                context=self.context,
            ),
        )
        return resumed

    def check_answerable(self, run: Run) -> None:
        """Check that ``run`` is paused and that nothing has answered it yet.

        Parameters
        ----------
        run : Run

        Raises
        ------
        RunStateError
            If the run was answered already or is not paused.
        """
        follow_up = self.next_run(run)
        if follow_up is not None:
            raise RunStateError(
                f"Run {run.id} was already answered; see run {follow_up.id}"
            )
        if run.status != "interrupted":
            raise RunStateError(
                f"Run {run.id} is {run.status}, so there is nothing to answer"
            )

    def has_thread(self, thread_id: str) -> bool:
        """Whether this session started ``thread_id``."""
        return thread_id in self._threads

    def thread_runs(self, thread_id: str) -> list[Run]:
        """Return the runs of a thread, oldest first."""
        return list(self._threads[thread_id])

    def latest_run(self, thread_id: str) -> Run:
        """Return the newest run of a thread."""
        return self._threads[thread_id][-1]

    def next_run(self, run: Run) -> Run | None:
        """Return the run that resumed ``run``, if there is one."""
        runs = self._threads[run.thread_id]
        position = runs.index(run)
        return runs[position + 1] if position + 1 < len(runs) else None

    def thread_summaries(self) -> list[ThreadSummary]:
        """Describe the threads of this session, newest first.

        Returns
        -------
        list[ThreadSummary]
        """
        return [
            ThreadSummary(
                id=thread_id,
                status=runs[-1].status,
                runs=len(runs),
                started_at=runs[0].started_at,
                latest_run=runs[-1].id,
            )
            for thread_id, runs in reversed(self._threads.items())
        ]

    async def history(self, thread_id: str) -> History:
        """Return the stored history of a thread.

        It comes from ``graph.history(thread_id, state_store=store)``.

        Parameters
        ----------
        thread_id : str

        Returns
        -------
        History
        """
        return await self.graph.history(thread_id, state_store=self.store)

    def _new_run(
        self,
        thread_id: str,
        *,
        kind: RunKind,
        input_json: str,
        previous: str | None = None,
    ) -> Run:
        run = Run(
            run_id=uuid4().hex[:12],
            thread_id=thread_id,
            kind=kind,
            input_json=input_json,
            previous=previous,
        )
        self.runs[run.id] = run
        self._threads.setdefault(thread_id, []).append(run)
        return run

    async def stop(self) -> list[Run]:
        """Cancel the runs that are still going and wait for them to end.

        Each stopped run fails with "The sandbox stopped before the run
        finished". The wait is capped at ``STOP_TIMEOUT`` seconds. A sync node
        keeps running in its worker thread after its run is cancelled, because
        a thread cannot be stopped from outside.

        Returns
        -------
        list[Run]
            The runs that were still going.
        """
        tasks = {task: run for task, run in self._tasks.items() if not run.finished}
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=STOP_TIMEOUT)
        return list(tasks.values())

    def _launch(self, run: Run, events: AsyncIterator[StreamEvent]) -> None:
        task = asyncio.create_task(run.consume(events))
        self._tasks[task] = run
        task.add_done_callback(self._forget)

    def _forget(self, task: asyncio.Task[None]) -> None:
        self._tasks.pop(task, None)
