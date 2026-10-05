import socket
import threading
import time
from typing import Any, TypedDict

import httpx
import uvicorn

from nodestep import END, START, Graph, node
from nodestep_sandbox.session import Sandbox
from nodestep_sandbox.web import SandboxSite


class StepState(TypedDict):
    done: list[str]


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_the_run_page_streams_rows_before_the_run_finishes() -> None:
    gate = threading.Event()

    @node
    def first(state: StepState) -> dict[str, Any]:
        return {"done": ["first"]}

    @node
    def second(state: StepState) -> dict[str, Any]:
        if not gate.wait(timeout=10):
            raise TimeoutError("the test never opened the gate")
        return {"done": ["first", "second"]}

    graph = Graph(StepState, name="gated").flow(
        START >> first, first >> second, second >> END
    )
    port = free_port()
    site = SandboxSite(Sandbox(graph), host="127.0.0.1", port=port)
    server = uvicorn.Server(
        uvicorn.Config(site.app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
            created = client.post(
                "/runs", data={"csrf": site.csrf_token, "mode": "json", "raw": "{}"}
            )
            assert created.status_code == 303
            seen = ""
            with client.stream("GET", created.headers["location"]) as response:
                for chunk in response.iter_text():
                    seen += chunk
                    if "event-update" in seen and not gate.is_set():
                        assert "nodestep-card-completed" not in seen
                        gate.set()
            assert gate.is_set()
            assert seen.count('class="nodestep-list-row event-update"') == 2
            assert "nodestep-card-completed" in seen
    finally:
        gate.set()
        server.should_exit = True
        thread.join(timeout=10)
