import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

BLOCKING_APP = """
import time
from typing import TypedDict

from nodestep import END, START, Graph, node


class State(TypedDict):
    text: str


@node
def blocker(state: State) -> dict:
    time.sleep(60)
    return {"text": "woke"}


graph = Graph(State, name="blocking").flow(START >> blocker, blocker >> END)
"""


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_until_serving(client: httpx.Client, process: subprocess.Popen[str]) -> str:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        assert process.poll() is None, process.communicate()
        try:
            page = client.get("/runs/new").text
        except httpx.TransportError:
            time.sleep(0.05)
            continue
        token = re.search(r'name="csrf" value="([^"]+)"', page)
        assert token is not None
        return token.group(1)
    raise AssertionError("the sandbox did not start")


@pytest.mark.skipif(sys.platform == "win32", reason="sends SIGINT to a child process")
def test_ctrl_c_stops_the_sandbox_while_a_node_blocks_in_a_thread(
    tmp_path: Path,
) -> None:
    (tmp_path / "blocking_app.py").write_text(BLOCKING_APP, encoding="utf-8")
    port = free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from nodestep_sandbox.cli import main; main()",
            "sandbox",
            "blocking_app.py:graph",
            "--port",
            str(port),
            "--no-browser",
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
            token = wait_until_serving(client, process)
            created = client.post(
                "/runs",
                data={"csrf": token, "mode": "json", "raw": '{"text": "hi"}'},
            )
            run_id = created.headers["location"].rsplit("/", 1)[1]
            page = ""
            with client.stream("GET", created.headers["location"]) as response:
                chunks = response.iter_text()
                while "node_input" not in page:
                    page += next(chunks)
                process.send_signal(signal.SIGINT)
                stopped_at = time.monotonic()
                page += "".join(chunks)
        code = process.wait(timeout=15)
        elapsed = time.monotonic() - stopped_at
        stdout, stderr = process.communicate()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert elapsed < 10
    assert code == 130
    assert "The sandbox stopped before the run finished" in page
    assert "Ctrl+C stops it" in stdout
    assert f"exiting without waiting for unfinished runs: {run_id}" in stderr
