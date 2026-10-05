"""`FilesystemStateStore`, which keeps each thread's history in a JSON Lines file.

See [State stores](../../../concepts/persistence.md#state-stores).
"""

from __future__ import annotations

import errno
import json
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any

from nodestep.exceptions import StateStoreError
from nodestep.models.base import NodestepModel
from nodestep.state.history import (
    BranchRecord,
    CheckpointRecord,
    History,
    HistoryEvent,
)
from nodestep.state.integrations.inmemory import InMemoryStateStore

_HEADER = {"kind": "nodestep-state-store", "version": 1}
_HEADER_LINE = json.dumps(_HEADER, separators=(",", ":")).encode("utf-8") + b"\n"

_file_lock: Callable[[Path], AbstractContextManager[None]] | None = None

if sys.platform == "win32":
    try:
        import msvcrt
    except ImportError:
        pass
    else:

        @contextmanager
        def _msvcrt_lock(path: Path) -> Iterator[None]:
            descriptor = os.open(f"{path}.lock", os.O_RDWR | os.O_CREAT)
            try:
                while True:
                    try:
                        msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
                        break
                    except OSError as error:
                        if error.errno != errno.EDEADLOCK:
                            raise
                try:
                    yield
                finally:
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            finally:
                os.close(descriptor)

        _file_lock = _msvcrt_lock
else:
    try:
        import fcntl
    except ImportError:
        pass
    else:

        @contextmanager
        def _fcntl_lock(path: Path) -> Iterator[None]:
            with path.open("ab") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        _file_lock = _fcntl_lock


class FilesystemStateStore(InMemoryStateStore):
    """State store that appends history to a JSON Lines file.

    Processes can share the file. See
    [State stores](../../../concepts/persistence.md#state-stores) for its
    header record, its lock and the repair of an interrupted write.

    Parameters
    ----------
    path : str or Path
        Store file; missing parent directories are created.

    Raises
    ------
    StateStoreError
        If the runtime has no file lock, ``path`` is a directory, the file is
        not a state store of this version, or a complete record is corrupt.
    """

    def __init__(self, path: str | Path) -> None:
        if _file_lock is None:
            raise StateStoreError(
                "FilesystemStateStore needs a file lock (fcntl or msvcrt), "
                "which this runtime does not provide"
            )
        self._lock = _file_lock
        self.path = Path(path)
        if self.path.is_dir():
            raise StateStoreError(f"'{self.path}' is not a file (got a directory)")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__()
        self._offset = 0
        self._has_header = False
        self._repaired_bytes = 0
        with self._exclusive():
            pass

    @property
    def repaired_bytes(self) -> int:
        """Bytes truncated by the last repair, or ``0`` if nothing was repaired.

        Returns
        -------
        int
        """
        return self._repaired_bytes

    def _corrupt(self, detail: object) -> StateStoreError:
        return StateStoreError(f"Corrupt state-store file '{self.path.name}': {detail}")

    def _foreign(self) -> StateStoreError:
        return StateStoreError(
            f"'{self.path}' is not a nodestep state store: it does not start with "
            f"the header record {json.dumps(_HEADER)}"
        )

    def _check_header(self, raw: bytes) -> None:
        try:
            header = json.loads(raw)
        except json.JSONDecodeError as error:
            raise self._foreign() from error
        if not isinstance(header, dict) or header.get("kind") != _HEADER["kind"]:
            raise self._foreign()
        if header != _HEADER:
            raise StateStoreError(
                f"'{self.path}' is a nodestep state store with version "
                f"{header.get('version')!r}; this nodestep reads version "
                f"{_HEADER['version']}"
            )

    def _apply(self, record: Any) -> None:
        if not isinstance(record, dict):
            raise self._corrupt("record is not an object")
        kind = record.get("kind")
        data = record.get("data")
        if kind == "event":
            self._record_event(HistoryEvent.model_validate(data))
        elif kind == "checkpoint":
            checkpoint = CheckpointRecord.model_validate(data)
            self._checkpoint_list(checkpoint.thread_id, checkpoint.branch_id).append(
                checkpoint
            )
        elif kind == "branch":
            self._branches.append(BranchRecord.model_validate(data))
        else:
            raise self._corrupt(f"unknown record kind {kind!r}")

    def _consume(self, raw: bytes) -> None:
        if not self._has_header:
            self._check_header(raw)
            self._has_header = True
            return
        if not raw.strip():
            return
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as error:
            raise self._corrupt(error) from error
        self._apply(record)

    def _append(self, data: bytes) -> None:
        with self.path.open("ab") as handle:
            handle.write(data)

    def _repair(self, fragment: bytes) -> None:
        try:
            json.loads(fragment)
        except json.JSONDecodeError:
            if not self._has_header and not _HEADER_LINE.startswith(fragment):
                raise self._foreign() from None
            os.truncate(self.path, self._offset)
            self._repaired_bytes = len(fragment)
            return
        self._consume(fragment)
        self._append(b"\n")
        self._offset += len(fragment) + 1

    def _refresh(self, *, repair: bool = False) -> None:
        chunk = b""
        if self.path.exists():
            with self.path.open("rb") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
        complete, newline, fragment = chunk.rpartition(b"\n")
        if newline:
            for raw in complete.split(b"\n"):
                self._consume(raw)
                self._offset += len(raw) + 1
        if not repair:
            return
        if fragment:
            self._repair(fragment)
        if not self._has_header:
            self._append(_HEADER_LINE)
            self._offset += len(_HEADER_LINE)
            self._has_header = True

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        with self._lock(self.path):
            self._refresh(repair=True)
            yield

    def _write(self, kind: str, model: NodestepModel) -> None:
        line = json.dumps(
            {"kind": kind, "data": model.model_dump(mode="json")},
            separators=(",", ":"),
        )
        encoded = (line + "\n").encode("utf-8")
        self._append(encoded)
        self._offset += len(encoded)

    async def append_event(self, event: HistoryEvent) -> HistoryEvent:
        """Store an event as given.

        Parameters
        ----------
        event : HistoryEvent

        Returns
        -------
        HistoryEvent
            The same event.

        Raises
        ------
        StateStoreError
            If a record another writer appended to the file is corrupt.
        """
        with self._write_lock, self._exclusive():
            self._write("event", event)
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

        Raises
        ------
        StateStoreError
            If a record another writer appended to the file is corrupt.
        """
        with self._write_lock, self._exclusive():
            allocated = self._with_next_sequence(event)
            self._write("event", allocated)
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

        Raises
        ------
        StateStoreError
            If a record another writer appended to the file is corrupt.
        """
        with self._write_lock, self._exclusive():
            self._write("checkpoint", checkpoint)
            self._checkpoint_list(checkpoint.thread_id, checkpoint.branch_id).append(
                checkpoint
            )
        return checkpoint

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
            ``"main"``, or a record another writer appended is corrupt.
        """
        with self._write_lock, self._exclusive():
            branch = self._new_branch(thread_id, from_, branch_id, name)
            self._write("branch", branch)
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

        Raises
        ------
        StateStoreError
            If a record another writer appended to the file is corrupt.
        """
        with self._write_lock:
            self._refresh()
        return await super().list_branches(thread_id)

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

        Raises
        ------
        StateStoreError
            If a record another writer appended to the file is corrupt.
        """
        with self._write_lock:
            self._refresh()
        return await super().get_history(thread_id, branch_id)

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
        StateStoreError
            If a record another writer appended to the file is corrupt.
        """
        with self._write_lock:
            self._refresh()
        return await super().latest_checkpoint(thread_id, branch_id, at)


__all__ = ["FilesystemStateStore"]
