import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nodestep.exceptions import StateStoreError
from nodestep.state import HistoryEvent
from nodestep.state.integrations import FilesystemStateStore

HEADER = b'{"kind":"nodestep-state-store","version":1}\n'


def _event(thread_id: str = "t", sequence: int = 1) -> HistoryEvent:
    return HistoryEvent(thread_id=thread_id, sequence=sequence, type="state_delta")


def _event_line(thread_id: str = "t", sequence: int = 1) -> bytes:
    data = _event(thread_id, sequence).model_dump(mode="json")
    return json.dumps({"kind": "event", "data": data}, separators=(",", ":")).encode()


def test_a_new_store_file_starts_with_the_header(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "store.jsonl"

    FilesystemStateStore(path)

    assert path.read_bytes() == HEADER


def test_an_empty_file_becomes_a_store(tmp_path: Path) -> None:
    path = tmp_path / "store.jsonl"
    path.touch()

    FilesystemStateStore(path)

    assert path.read_bytes() == HEADER


async def test_records_follow_the_header(tmp_path: Path) -> None:
    path = tmp_path / "store.jsonl"
    store = FilesystemStateStore(path)

    await store.append_event(_event())

    lines = path.read_bytes().splitlines(keepends=True)
    assert lines[0] == HEADER
    assert json.loads(lines[1])["kind"] == "event"
    assert len((await FilesystemStateStore(path).get_history("t")).events) == 1


@pytest.mark.parametrize(
    "content",
    [
        b"[1, 2, 3]",
        b"[1, 2, 3]\n",
        b"hello world\n",
        _event_line() + b"\n",
        b'{"kind":"event","data":{"thread_id":"t","seq',
        b'{"events":[],"checkpoints":[],"branches":[],"note":"user data"}',
        b"\n" + HEADER,
    ],
    ids=[
        "json-array",
        "json-array-line",
        "text",
        "record-without-header",
        "torn-record-without-header",
        "old-single-object-format",
        "blank-first-line",
    ],
)
def test_files_without_the_header_are_refused_and_left_alone(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "foreign.json"
    path.write_bytes(content)

    with pytest.raises(StateStoreError, match="not a nodestep state store"):
        FilesystemStateStore(path)

    assert path.read_bytes() == content


def test_an_unsupported_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "future.jsonl"
    content = b'{"kind":"nodestep-state-store","version":2}\n'
    path.write_bytes(content)

    with pytest.raises(StateStoreError, match="version 2"):
        FilesystemStateStore(path)

    assert path.read_bytes() == content


def test_a_corrupt_complete_line_raises(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.jsonl"
    path.write_bytes(HEADER + b"{not valid json\n" + _event_line() + b"\n")

    with pytest.raises(StateStoreError, match=r"corrupt\.jsonl"):
        FilesystemStateStore(path)


async def test_an_unparseable_trailing_fragment_is_truncated_and_reported(
    tmp_path: Path,
) -> None:
    path = tmp_path / "crash.jsonl"
    fragment = b'{"kind":"event","data":{"thread_id":"t","seq'
    kept = HEADER + _event_line(sequence=1) + b"\n"
    path.write_bytes(kept + fragment)

    store = FilesystemStateStore(path)

    assert store.repaired_bytes == len(fragment)
    assert path.read_bytes() == kept
    await store.append_event(_event(sequence=2))
    reopened = FilesystemStateStore(path)
    assert reopened.repaired_bytes == 0
    sequences = [event.sequence for event in (await reopened.get_history("t")).events]
    assert sequences == [1, 2]


async def test_a_complete_last_record_without_newline_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "unterminated.jsonl"
    path.write_bytes(HEADER + _event_line(sequence=1))

    store = FilesystemStateStore(path)

    assert store.repaired_bytes == 0
    assert [event.sequence for event in (await store.get_history("t")).events] == [1]
    await store.append_event(_event(sequence=2))
    reopened = FilesystemStateStore(path)
    assert [event.sequence for event in (await reopened.get_history("t")).events] == [
        1,
        2,
    ]


def test_a_torn_header_is_repaired(tmp_path: Path) -> None:
    path = tmp_path / "torn-header.jsonl"
    path.write_bytes(HEADER[:10])

    store = FilesystemStateStore(path)

    assert store.repaired_bytes == 10
    assert path.read_bytes() == HEADER


async def test_a_fragment_left_by_another_writer_is_repaired_on_the_next_write(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shared.jsonl"
    store = FilesystemStateStore(path)
    await store.append_event(_event(sequence=1))
    with path.open("ab") as handle:
        handle.write(b'{"kind":"ev')

    assert [event.sequence for event in (await store.get_history("t")).events] == [1]
    assert store.repaired_bytes == 0

    await store.append_event(_event(sequence=2))

    assert store.repaired_bytes == len(b'{"kind":"ev')
    reopened = FilesystemStateStore(path)
    assert [event.sequence for event in (await reopened.get_history("t")).events] == [
        1,
        2,
    ]


_NO_LOCK_SCRIPT = """
import sys

sys.modules["fcntl"] = None
sys.modules["msvcrt"] = None

import nodestep
from nodestep.exceptions import StateStoreError
from nodestep.state.integrations import FilesystemStateStore

try:
    FilesystemStateStore(sys.argv[1])
except StateStoreError as error:
    print("refused:", error)
"""


@pytest.mark.skipif(
    sys.platform == "win32", reason="the test fakes a runtime that is not Windows"
)
def test_a_runtime_without_file_locks_imports_but_refuses_the_store(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store.jsonl"

    completed = subprocess.run(
        [sys.executable, "-c", _NO_LOCK_SCRIPT, str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.startswith("refused:")
    assert "lock" in completed.stdout
    assert not path.exists()


_WRITER_SCRIPT = """
import asyncio
import sys

from nodestep.state import HistoryEvent
from nodestep.state.integrations import FilesystemStateStore


async def main() -> None:
    store = FilesystemStateStore(sys.argv[1])
    for _ in range(int(sys.argv[2])):
        await store.append_next_event(
            HistoryEvent(thread_id="shared", sequence=0, type="state_delta", data_json="{\\"x\\": \\"" + "y" * 2000 + "\\"}")
        )


asyncio.run(main())
"""


def test_processes_sharing_a_file_do_not_interleave(tmp_path: Path) -> None:
    path = tmp_path / "shared.jsonl"
    FilesystemStateStore(path)
    writers, per_writer = 4, 200

    processes = [
        subprocess.Popen(
            [sys.executable, "-c", _WRITER_SCRIPT, str(path), str(per_writer)],
            stderr=subprocess.PIPE,
        )
        for _ in range(writers)
    ]
    errors = [process.communicate(timeout=120)[1] for process in processes]

    assert all(process.returncode == 0 for process in processes), errors
    lines = path.read_bytes().split(b"\n")
    assert lines[0] + b"\n" == HEADER
    assert lines[-1] == b""
    records = [json.loads(line) for line in lines[1:-1]]
    sequences = [record["data"]["sequence"] for record in records]
    assert sorted(sequences) == list(range(1, writers * per_writer + 1))
    history = asyncio.run(FilesystemStateStore(path).get_history("shared"))
    assert len(history.events) == writers * per_writer


@pytest.mark.skipif(sys.platform != "win32", reason="the sidecar lock is Windows-only")
async def test_windows_locks_a_sidecar_file(tmp_path: Path) -> None:
    path = tmp_path / "store.jsonl"
    store = FilesystemStateStore(path)

    await store.append_event(_event())

    assert (tmp_path / "store.jsonl.lock").exists()


async def test_a_failed_refresh_does_not_apply_records_twice(tmp_path: Path) -> None:
    path = tmp_path / "store.jsonl"
    store = FilesystemStateStore(path)
    second = _event_line("t", 2)
    with path.open("ab") as handle:
        handle.write(_event_line("t", 1) + b"\nX" + second[1:] + b"\n")

    for _ in range(2):
        with pytest.raises(StateStoreError, match=r"store\.jsonl"):
            await store.get_history("t")
    path.write_bytes(path.read_bytes().replace(b"\nX", b"\n{"))

    events = (await store.get_history("t")).events
    assert [event.sequence for event in events] == [1, 2]
