from __future__ import annotations

import time
from typing import Protocol
from uuid import uuid4

from pydantic import Field

from nodestep.models.base import NodestepModel


class HistoryEvent(NodestepModel):
    """One recorded event of a thread.

    Attributes
    ----------
    sequence : int
        Position on its branch, counted from 1; ``append_next_event`` sets it.
    type : str
        What happened, such as ``"run_started"``, ``"superstep_completed"``,
        ``"interrupted"`` or ``"error"``.
    node : str or None
        Node the event belongs to.
    parent_id : str or None
        Not set by nodestep.
    data_json : str or None
        Details of the event as JSON, such as the step and the updates.
    error_type : str or None
        Exception class name, on ``"error"`` and ``"superstep_failed"`` events.
    message : str or None
        The exception message, with ``error_type``.
    created_at : float
        Unix time in seconds.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    thread_id: str
    branch_id: str = "main"
    sequence: int
    type: str
    node: str | None = None
    parent_id: str | None = None
    data_json: str | None = None
    error_type: str | None = None
    message: str | None = None
    created_at: float = Field(default_factory=time.time)


class CheckpointRecord(NodestepModel):
    """Full state snapshot at a point in a thread's history.

    Attributes
    ----------
    event_id : str
        Id of the event the snapshot was taken at.
    sequence : int
        ``sequence`` of that event.
    state_json : str
        The state as JSON, written by ``StateSchema.dump_state``.
    active_json : str
        JSON list of the tasks still to run at that point.
    status : str
        ``"running"``, ``"completed"``, or ``"forked"`` for the first
        snapshot of a branch.
    created_at : float
        Unix time in seconds.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    thread_id: str
    branch_id: str = "main"
    event_id: str
    sequence: int
    state_json: str
    active_json: str = "[]"
    status: str = "running"
    created_at: float = Field(default_factory=time.time)


class BranchRecord(NodestepModel):
    """A history branch forked from an event.

    Attributes
    ----------
    id : str
        Id of the branch, the ``name`` given to ``fork`` or a random id.
    parent_branch_id : str
        Branch it was forked from.
    from_event_id : str
        Event of that branch it starts at.
    created_at : float
        Unix time in seconds.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    thread_id: str
    parent_branch_id: str = "main"
    from_event_id: str
    created_at: float = Field(default_factory=time.time)


class History(NodestepModel):
    """Events and checkpoints of one thread branch."""

    thread_id: str
    branch_id: str
    events: list[HistoryEvent] = Field(default_factory=list)
    checkpoints: list[CheckpointRecord] = Field(default_factory=list)


class StateStore(Protocol):
    """Protocol for storing thread history."""

    async def append_event(self, event: HistoryEvent) -> HistoryEvent:
        """Store an event as given.

        Parameters
        ----------
        event : HistoryEvent

        Returns
        -------
        HistoryEvent
        """
        ...

    async def append_next_event(self, event: HistoryEvent) -> HistoryEvent:
        """Store an event with the next sequence number of its thread.

        Parameters
        ----------
        event : HistoryEvent

        Returns
        -------
        HistoryEvent
            The stored event with its sequence number.
        """
        ...

    async def save_checkpoint(self, checkpoint: CheckpointRecord) -> CheckpointRecord:
        """Store a state snapshot.

        Parameters
        ----------
        checkpoint : CheckpointRecord

        Returns
        -------
        CheckpointRecord
        """
        ...

    async def get_history(self, thread_id: str, branch_id: str = "main") -> History:
        """Return the events and checkpoints of a thread branch.

        Parameters
        ----------
        thread_id : str
        branch_id : str, optional

        Returns
        -------
        History
        """
        ...

    async def latest_checkpoint(
        self,
        thread_id: str,
        branch_id: str = "main",
        at: str | None = None,
    ) -> CheckpointRecord | None:
        """Return the latest checkpoint, optionally at or before an event.

        Parameters
        ----------
        thread_id : str
        branch_id : str, optional
        at : str, optional
            Event id.

        Returns
        -------
        CheckpointRecord or None
        """
        ...

    async def fork(
        self,
        thread_id: str,
        *,
        from_: str,
        branch_id: str = "main",
        name: str | None = None,
    ) -> BranchRecord:
        """Record a new branch starting at an event.

        Parameters
        ----------
        thread_id : str
        from_ : str
            Event id to branch from.
        branch_id : str, optional
            Branch to fork.
        name : str, optional
            Id of the new branch; a random id when omitted.

        Returns
        -------
        BranchRecord

        Raises
        ------
        StateStoreError
            If the thread already has a branch with that id, including
            ``"main"``.
        """
        ...

    async def list_branches(self, thread_id: str) -> list[BranchRecord]:
        """Return the branches forked from a thread, oldest first.

        Parameters
        ----------
        thread_id : str

        Returns
        -------
        list[BranchRecord]
            Every recorded fork of the thread; ``"main"`` is not listed.
        """
        ...


__all__ = [
    "BranchRecord",
    "CheckpointRecord",
    "History",
    "HistoryEvent",
    "StateStore",
]
