import re
import socket
import sys
import webbrowser
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from typer.testing import CliRunner

from nodestep_sandbox import cli, commands


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    servers: list[Any] = []

    async def startup(self: uvicorn.Server, sockets: Any = None) -> None:
        self.started = True
        servers.append(self)

    async def main_loop(self: uvicorn.Server) -> None:
        return None

    async def shutdown(self: uvicorn.Server, sockets: Any = None) -> None:
        return None

    monkeypatch.setattr(uvicorn.Server, "startup", startup)
    monkeypatch.setattr(uvicorn.Server, "main_loop", main_loop)
    monkeypatch.setattr(uvicorn.Server, "shutdown", shutdown)
    return servers


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    urls: list[str] = []
    monkeypatch.setattr(webbrowser, "open", urls.append)
    return urls


ANSI_STYLE = re.compile(r"\x1b\[[0-9;]*m")


def invoke(*args: str) -> Any:
    return CliRunner().invoke(commands.app, list(args), env={"COLUMNS": "200"})


def test_help_explains_the_target_and_the_options() -> None:
    result = invoke("sandbox", "--help")
    assert result.exit_code == 0
    output = ANSI_STYLE.sub("", result.output)
    for text in (
        "TARGET",
        "package.module:attribute",
        "--context",
        "MODULE:ATTRIBUTE",
        "--host",
        "--port",
        "--no-browser",
    ):
        assert text in output


def test_sandbox_serves_the_graph_on_the_given_address(
    served: list[Any], opened: list[str]
) -> None:
    result = invoke(
        "sandbox", "nodestep_sandbox.demo:graph", "--port", "9123", "--no-browser"
    )
    assert result.exit_code == 0, result.output
    [server] = served
    assert (server.config.host, server.config.port) == ("127.0.0.1", 9123)
    site = server.config.app.state.site
    assert site.sandbox.graph.name == "support_demo"
    assert site.address == "127.0.0.1:9123"
    assert "http://127.0.0.1:9123/" in result.output
    assert opened == []


def test_sandbox_opens_the_browser_unless_told_not_to(
    served: list[Any], opened: list[str]
) -> None:
    result = invoke("sandbox", "nodestep_sandbox.demo:graph")
    assert result.exit_code == 0, result.output
    assert opened == ["http://127.0.0.1:8765/"]


def test_context_is_loaded_and_passed_to_the_sandbox(
    served: list[Any],
    opened: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "sandbox_fixture_context.py").write_text(
        "def make():\n    return {'name': 'from-cli'}\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "sandbox_fixture_context", raising=False)
    result = invoke(
        "sandbox",
        "nodestep_sandbox.demo:graph",
        "--context",
        "sandbox_fixture_context:make",
        "--no-browser",
    )
    assert result.exit_code == 0, result.output
    assert served[0].config.app.state.site.sandbox.context == {"name": "from-cli"}
    sys.modules.pop("sandbox_fixture_context", None)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["sandbox", "nodestep_sandbox.demo"], "is not package.module:attribute"),
        (["sandbox", "nodestep_sandbox.demo:missing"], "has no attribute 'missing'"),
    ],
)
def test_problems_exit_with_code_2_and_a_message(
    served: list[Any], args: list[str], message: str
) -> None:
    result = invoke(*args, "--no-browser")
    assert result.exit_code == 2
    assert message in result.output
    assert served == []


@pytest.mark.parametrize(
    ("args", "message", "frame"),
    [
        (
            ["sandbox_fixture_failing.py:graph"],
            "Calling sandbox_fixture_failing.py:graph failed: RuntimeError: "
            "factory failed on purpose",
            'sandbox_fixture_failing.py", line 2, in graph',
        ),
        (
            [
                "nodestep_sandbox.demo:graph",
                "--context",
                "sandbox_fixture_failing.py:context",
            ],
            "Calling sandbox_fixture_failing.py:context failed: RuntimeError: "
            "context failed on purpose",
            'sandbox_fixture_failing.py", line 6, in context',
        ),
    ],
)
def test_errors_in_the_loaded_code_exit_with_code_2_and_the_user_frames(
    served: list[Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    message: str,
    frame: str,
) -> None:
    (tmp_path / "sandbox_fixture_failing.py").write_text(
        "def graph():\n"
        "    raise RuntimeError('factory failed on purpose')\n"
        "\n"
        "\n"
        "def context():\n"
        "    raise RuntimeError('context failed on purpose')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "sandbox_fixture_failing", raising=False)
    result = invoke("sandbox", *args, "--no-browser")
    sys.modules.pop("sandbox_fixture_failing", None)
    assert result.exit_code == 2
    assert result.stderr.startswith("Traceback (most recent call last):\n")
    assert result.stderr.endswith(
        f"RuntimeError: {message.rsplit(': ', 1)[1]}\nError: {message}\n"
    )
    assert frame in result.stderr
    assert "asyncio" not in result.stderr
    assert "nodestep_sandbox/" not in result.stderr
    assert served == []


@pytest.mark.parametrize(
    "host",
    [
        "",
        "0.0.0.0",
        "0",
        "0.0",
        "0x0",
        "::",
        "::0",
        "0:0:0:0:0:0:0:0",
        "::ffff:0.0.0.0",
    ],
)
def test_a_wildcard_host_is_refused_however_it_is_written(
    served: list[Any], host: str
) -> None:
    result = invoke("sandbox", "nodestep_sandbox.demo:graph", "--host", host)
    assert result.exit_code == 2
    assert "is a wildcard address" in result.stderr
    assert served == []


def test_a_host_that_does_not_resolve_is_refused(
    served: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", fail)
    result = invoke(
        "sandbox", "nodestep_sandbox.demo:graph", "--host", "sandbox.invalid"
    )
    assert result.exit_code == 2
    assert (
        "Error: --host 'sandbox.invalid' cannot be resolved: "
        "Name or service not known" in result.stderr
    )
    assert served == []


@pytest.mark.parametrize(
    ("host", "address"),
    [
        ("LOCALHOST", "localhost:9123"),
        ("127.1", "127.0.0.1:9123"),
        ("0:0:0:0:0:0:0:1", "[::1]:9123"),
    ],
)
def test_the_host_is_written_the_way_a_browser_sends_it(
    served: list[Any], opened: list[str], host: str, address: str
) -> None:
    result = invoke(
        "sandbox",
        "nodestep_sandbox.demo:graph",
        "--host",
        host,
        "--port",
        "9123",
        "--no-browser",
    )
    assert result.exit_code == 0, result.output
    [server] = served
    assert server.config.app.state.site.address == address
    assert f"http://{address}/" in result.stdout
    assert result.stderr == ""


def test_a_host_other_machines_can_reach_gets_a_warning(
    served: list[Any], opened: list[str]
) -> None:
    result = invoke(
        "sandbox", "nodestep_sandbox.demo:graph", "--host", "192.0.2.10", "--no-browser"
    )
    assert result.exit_code == 0, result.output
    assert served
    assert result.stderr == (
        "Warning: 192.0.2.10 is not a loopback address, so other machines that can "
        "reach it can open the sandbox and run the graph\n"
    )


def test_main_runs_the_typer_app_when_the_extra_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(commands, "app", lambda: calls.append("app"))
    cli.main()
    assert calls == ["app"]
