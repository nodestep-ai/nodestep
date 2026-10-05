"""`InMemoryStateStore`, which keeps each thread's history in memory.

See [State stores](../../../concepts/persistence.md#state-stores).
"""

from __future__ import annotations

import threading

from nodestep.exceptions import StateStoreError
from nodestep.state.history import (
    BranchRecord,
    CheckpointRecord,
    History,
    HistoryEvent,
)


class InMemoryStateStore:
    """Thread-safe state store that keeps history in memory."""

    def __init__(self) -> None:
        self._events: dict[tuple[str, str], list[HistoryEvent]] = {}
        self._checkpoints: dict[tuple[str, str], list[CheckpointRecord]] = {}
        self._branches: list[BranchRecord] = []
        self._last_sequence: dict[tuple[str, str], int] = {}
        self._write_lock = threading.RLock()

    def _event_list(self, thread_id: str, branch_id: str) -> list[HistoryEvent]:
        key = (thread_id, branch_id)
        if key not in self._events:
            self._events[key] = []
        return self._events[key]

    def _checkpoint_list(
        self, thread_id: str, branch_id: str
    ) -> list[CheckpointRecord]:
        key = (thread_id, branch_id)
        if key not in self._checkpoints:
            self._checkpoints[key] = []
        return self._checkpoints[key]

    @property
    def events(self) -> list[HistoryEvent]:
        """All stored events."""
        return [event for bucket in self._events.values() for event in bucket]

    @property
    def checkpoints(self) -> list[CheckpointRecord]:
        """All stored checkpoints."""
        return [
            checkpoint for bucket in self._checkpoints.values() for checkpoint in bucket
        ]

    def _record_event(self, event: HistoryEvent) -> None:
        key = (event.thread_id, event.branch_id)
        self._event_list(*key).append(event)
        self._last_sequence[key] = max(self._last_sequence.get(key, 0), event.sequence)

    def _with_next_sequence(self, event: HistoryEvent) -> HistoryEvent:
        key = (event.thread_id, event.branch_id)
        return event.model_copy(
            update={"sequence": self._last_sequence.get(key, 0) + 1}
        )

    async def append_event(self, event: HistoryEvent) -> HistoryEvent:
        """Store an event as given.

        Parameters
        ----------
        event : HistoryEvent

        Returns
        -------
        HistoryEvent
            The same event.
        """
        with self._write_lock:
            self._record_event(event)
        return event

    async def append_next_event(self, event: HistoryEvent) -> HistoryEvent:
        """Store an event with the next sequence number of its thread.

        Parameters
        ----------
        event : HistoryEvent
            Event whose ``sequence`` is replaced.

        Returns
        -------
        HistoryEvent
            The stored event with its sequence number.
        """
        with self._write_lock:
            allocated = self._with_next_sequence(event)
            self._record_event(allocated)
        return allocated

    async def save_checkpoint(self, checkpoint: CheckpointRecord) -> CheckpointRecord:
        """Store a state snapshot.

        Parameters
        ----------
        checkpoint : CheckpointRecord

        Returns
        -------
        CheckpointRecord
            The same checkpoint.
        """
        with self._write_lock:
            self._checkpoint_list(checkpoint.thread_id, checkpoint.branch_id).append(
                checkpoint
            )
        return checkpoint

    async def get_history(self, thread_id: str, branch_id: str = "main") -> History:
        """Return the events and checkpoints of a thread branch.

        Parameters
        ----------
        thread_id : str
        branch_id : str, optional
            Branch to read; ``"main"`` by default.

        Returns
        -------
        History
            Empty lists for a thread or branch the store does not know.
        """
        with self._write_lock:
            return History(
                thread_id=thread_id,
                branch_id=branch_id,
                events=list(self._event_list(thread_id, branch_id)),
                checkpoints=list(self._checkpoint_list(thread_id, branch_id)),
            )

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
            Branch to read; ``"main"`` by default.
        at : str, optional
            Event id; only checkpoints up to that event count.

        Returns
        -------
        CheckpointRecord or None
            ``None`` when the branch has no such checkpoint.

        Raises
        ------
        StateUpdateError
            If ``at`` is not an event of the branch.
        """
        from nodestep.exceptions import StateUpdateError

        with self._write_lock:
            candidates = list(self._checkpoint_list(thread_id, branch_id))
            events = list(self._event_list(thread_id, branch_id))
        if at is not None:
            event_by_id = {event.id: event for event in events}
            if at not in event_by_id:
                raise StateUpdateError(
                    f"Event id '{at}' not found in thread '{thread_id}'"
                )
            at_sequence = event_by_id[at].sequence
            candidates = [
                checkpoint
                for checkpoint in candidates
                if checkpoint.sequence <= at_sequence
            ]
        if not candidates:
            return None
        return max(candidates, key=lambda checkpoint: checkpoint.sequence)

    def _new_branch(
        self, thread_id: str, from_: str, branch_id: str, name: str | None
    ) -> BranchRecord:
        branch = BranchRecord(
            thread_id=thread_id, parent_branch_id=branch_id, from_event_id=from_
        )
        if name is not None:
            branch = branch.model_copy(update={"id": name})
        key = (thread_id, branch.id)
        if (
            branch.id == "main"
            or self._events.get(key)
            or self._checkpoints.get(key)
            or any(
                other.thread_id == thread_id and other.id == branch.id
                for other in self._branches
            )
        ):
            raise StateStoreError(
                f"Branch '{branch.id}' of thread '{thread_id}' already exists"
            )
        return branch

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
            Branch to fork; ``"main"`` by default.
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
        with self._write_lock:
            branch = self._new_branch(thread_id, from_, branch_id, name)
            self._branches.append(branch)
        return branch

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
        with self._write_lock:
            return [
                branch for branch in self._branches if branch.thread_id == thread_id
            ]


__all__ = ["InMemoryStateStore"]
