import asyncio
import time
import traceback
from collections.abc import AsyncIterator
from functools import partial
from typing import Any, Literal

from nodestep import StreamEvent
from nodestep.core.stream import FinalEventData, InterruptEventData
from nodestep_sandbox.interrupts import PendingInterrupt
from nodestep_sandbox.records import Row, TaskRecord, json_text

RunKind = Literal["start", "resume"]
RunStatus = Literal["running", "completed", "interrupted", "error"]


def _route_label(target: Any) -> str:
    if not isinstance(target, dict):
        return str(target)
    if target.get("kind") == "end":
        return "END"
    if target.get("kind") == "send":
        return f"Send to {target.get('node')}"
    return str(target.get("node"))


class Run:
    """One ``astream`` call of the sandbox, recorded for the run pages.

    Parameters
    ----------
    run_id : str
    thread_id : str
    kind : {"start", "resume"}
    input_json : str
        The run input, or the resume answers, as JSON.
    previous : str, optional
        Id of the run this one resumes.

    Attributes
    ----------
    rows : list[Row]
        One row per stream event, in the order they arrived.
    tasks : list[TaskRecord]
        One record per task that started or reported an update.
    status : {"running", "completed", "interrupted", "error"}
    pending : list[PendingInterrupt]
        The interrupts the run paused on.
    final_state : str, optional
        The state at the end or at the pause, as JSON.
    error, traceback : str, optional
        Set when the run raised.
    started_at : float
    finished_at : float, optional
    """

    def __init__(
        self,
        *,
        run_id: str,
        thread_id: str,
        kind: RunKind,
        input_json: str,
        previous: str | None = None,
    ) -> None:
        self.id = run_id
        self.thread_id = thread_id
        self.kind: RunKind = kind
        self.input_json = input_json
        self.previous = previous
        self.rows: list[Row] = []
        self.tasks: list[TaskRecord] = []
        self.status: RunStatus = "running"
        self.pending: list[PendingInterrupt] = []
        self.final_state: str | None = None
        self.error: str | None = None
        self.traceback: str | None = None
        self.started_at = time.time()
        self.finished_at: float | None = None
        self._merged: dict[int | None, str] = {}
        self._open: dict[tuple[int | None, str | None], int] = {}
        self._changed = asyncio.Condition()

    @property
    def finished(self) -> bool:
        """Whether the run has ended: completed, paused or failed."""
        return self.status != "running"

    async def consume(self, events: AsyncIterator[StreamEvent]) -> None:
        """Record ``events`` until they end.

        An exception from the stream marks the run failed; a stream that stops
        without a terminal event, or a cancelled ``consume``, does too.

        Parameters
        ----------
        events : AsyncIterator[StreamEvent]
        """
        stopped = False
        try:
            async for event in events:
                async with self._changed:
                    self._record(event)
                    self._changed.notify_all()
        except asyncio.CancelledError:
            stopped = True
            raise
        except Exception as error:
            async with self._changed:
                self._fail(error)
                self._changed.notify_all()
        finally:
            async with self._changed:
                if self.status == "running":
                    self.status = "error"
                    self.error = (
                        "The sandbox stopped before the run finished"
                        if stopped
                        else "The run stopped before it finished"
                    )
                self.finished_at = time.time()
                self._changed.notify_all()

    async def follow(self) -> AsyncIterator[Row]:
        """Yield every row, waiting for new ones until the run has finished.

        Returns
        -------
        AsyncIterator[Row]
        """
        index = 0
        while True:
            async with self._changed:
                await self._changed.wait_for(partial(self._has_news, index))
                batch = self.rows[index:]
                done = self.finished
            for row in batch:
                yield row
            index += len(batch)
            if done:
                return

    def _has_news(self, index: int) -> bool:
        return index < len(self.rows) or self.finished

    def _record(self, event: StreamEvent) -> None:
        now = time.time()
        if event.mode == "debug":
            self._record_debug(event, now)
        elif event.mode == "updates":
            self._record_update(event, now)
        elif event.mode == "custom":
            self.rows.append(
                Row(
                    kind="custom",
                    title="custom",
                    node=event.node,
                    step=event.step,
                    data=json_text(event.data),
                )
            )
        elif isinstance(event.data, InterruptEventData):
            self.pending = [
                PendingInterrupt.from_interrupt(item)
                for item in event.data.interrupts.values()
            ]
            self.final_state = json_text(event.data.state)
            self.status = "interrupted"
        elif isinstance(event.data, FinalEventData):
            self.final_state = json_text(event.data.state)
            self.status = "completed"

    def _record_debug(self, event: StreamEvent, now: float) -> None:
        data: dict[str, Any] = (
            event.data if isinstance(event.data, dict) else {"data": event.data}
        )
        kind = str(data.get("type", "debug"))
        if kind == "node_input":
            state = json_text(data.get("state"))
            self._open[(event.step, event.task_id)] = len(self.tasks)
            self.tasks.append(
                TaskRecord(
                    node=event.node or "",
                    step=event.step,
                    task_id=event.task_id,
                    state_before=state,
                    started_at=now,
                )
            )
            summary, shown = "task started with this state", state
        elif kind == "node_output":
            summary, shown = (
                "the node returned this update",
                json_text(data.get("updates")),
            )
        elif kind == "routes":
            targets = data.get("targets")
            if not isinstance(targets, list):
                targets = []
            labels = ", ".join(_route_label(target) for target in targets)
            summary, shown = f"next: {labels or 'nothing'}", json_text(targets)
        elif kind == "state_merge":
            shown = json_text(data.get("state"))
            self._merged[event.step] = shown
            summary = "state after the superstep"
        else:
            summary, shown = kind, json_text(data)
        self.rows.append(
            Row(
                kind="debug",
                title=kind,
                summary=summary,
                node=event.node,
                step=event.step,
                data=shown,
            )
        )

    def _record_update(self, event: StreamEvent, now: float) -> None:
        index = self._open.pop((event.step, event.task_id), None)
        if index is None:
            index = len(self.tasks)
            self.tasks.append(
                TaskRecord(
                    node=event.node or "", step=event.step, task_id=event.task_id
                )
            )
        task = self.tasks[index]
        task.update = json_text(event.data)
        task.state_after = self._merged.get(event.step)
        task.finished_at = now
        self.rows.append(
            Row(
                kind="update",
                title="update",
                summary="no update" if event.data is None else "",
                node=event.node,
                step=event.step,
                data=task.update,
                state_after=task.state_after,
                task=index,
            )
        )

    def _fail(self, error: BaseException) -> None:
        self.status = "error"
        self.error = f"{type(error).__name__}: {error}"
        self.traceback = "".join(traceback.format_exception(error))
