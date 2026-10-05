import json
from typing import Any, Literal, Self

from pydantic import BaseModel

from nodestep.state import HistoryEvent, to_json_value


def json_text(value: Any) -> str:
    """Return ``value`` as indented JSON.

    Values JSON cannot hold are shown with ``repr``.

    Parameters
    ----------
    value : Any

    Returns
    -------
    str
    """
    return json.dumps(to_json_value(value, fallback=repr), indent=2, ensure_ascii=False)


def parse_json(text: str) -> Any:
    """Parse JSON text.

    Parameters
    ----------
    text : str

    Returns
    -------
    Any

    Raises
    ------
    ValueError
        With a short message naming the line and column, if the text is not JSON.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Not valid JSON: {error.msg} (line {error.lineno}, column {error.colno})"
        ) from None


class Row(BaseModel):
    """One stream event as the run page shows it.

    Attributes
    ----------
    kind : {"update", "custom", "debug"}
    title : str
        ``update``, ``custom`` or the type of a debug event.
    summary : str
        One line about the event.
    node : str, optional
    step : int, optional
    data : str
        The event's data as JSON.
    state_after : str, optional
        For an update, the state after its superstep as JSON.
    task : int, optional
        For an update, the position of its task in ``Run.tasks``.
    """

    kind: Literal["update", "custom", "debug"]
    title: str
    summary: str = ""
    node: str | None = None
    step: int | None = None
    data: str = ""
    state_after: str | None = None
    task: int | None = None


class TaskRecord(BaseModel):
    """What one task of a run received and returned, with its timing.

    Attributes
    ----------
    node : str
    step : int, optional
    task_id : str, optional
    state_before : str, optional
        The task's input state as JSON; unknown for a task that finished before
        the pause this run resumes.
    update : str, optional
        The task's update as JSON.
    state_after : str, optional
        The state after the task's superstep as JSON.
    started_at, finished_at : float, optional
        Wall-clock times of the task's start and of its update event.
    """

    node: str
    step: int | None = None
    task_id: str | None = None
    state_before: str | None = None
    update: str | None = None
    state_after: str | None = None
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def elapsed(self) -> float | None:
        """Seconds from the task's start to the end of its superstep, when known."""
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at


class ThreadSummary(BaseModel):
    """A thread started in this sandbox session."""

    id: str
    status: str
    runs: int
    started_at: float
    latest_run: str


class HistoryRow(BaseModel):
    """One stored history event of a thread, ready to show."""

    sequence: int
    type: str
    node: str | None = None
    created_at: float
    data: str = ""
    error: str | None = None

    @classmethod
    def from_event(cls, event: HistoryEvent) -> Self:
        """Build a row from a stored ``HistoryEvent``.

        Parameters
        ----------
        event : HistoryEvent

        Returns
        -------
        HistoryRow
        """
        data = (
            ""
            if event.data_json is None
            else json.dumps(json.loads(event.data_json), indent=2, ensure_ascii=False)
        )
        error = ": ".join(part for part in (event.error_type, event.message) if part)
        return cls(
            sequence=event.sequence,
            type=event.type,
            node=event.node,
            created_at=event.created_at,
            data=data,
            error=error or None,
        )
