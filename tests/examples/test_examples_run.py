import importlib.util
import os
import pathlib
import re
import subprocess
import sys

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples"


def _load_example(name: str):
    spec = importlib.util.spec_from_file_location(
        f"example_{name}", EXAMPLES_DIR / f"{name}.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tool_agent_example_runs_with_approval(capsys) -> None:
    module = _load_example("tool_agent")

    module.main()

    output = capsys.readouterr().out
    assert "approval requested: {'act:approve_call_1'" in output
    assert "Those are the files in the workspace root." in output
    assert len(module.chat.requests) == 2


async def test_simple_agent_example_runs(capsys) -> None:
    module = _load_example("simple_agent")

    await module.main()

    assert capsys.readouterr().out == "The capital of France is Paris.\n"


async def test_streaming_example_runs(capsys) -> None:
    module = _load_example("streaming")

    await module.main()

    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == [
        "custom {'file': 'orders.csv', 'rows': 120}",
        "custom {'file': 'returns.csv', 'rows': 8}",
        "updates count_rows ['messages', 'rows']",
    ]
    assert "tokens ' rows'" in lines
    assert lines[-1] == "final The upload has 128 rows in 2 files."


def test_durable_thread_example_resumes_in_another_process(tmp_path) -> None:
    environment = {**os.environ, "TMPDIR": str(tmp_path)}
    script = str(EXAMPLES_DIR / "durable_thread.py")

    def run(*arguments: str) -> str:
        completed = subprocess.run(
            [sys.executable, script, *arguments],
            env=environment,
            capture_output=True,
            encoding="utf-8",
            timeout=60,
            check=True,
        )
        return completed.stdout

    started = run()
    found = re.search(r"Thread (order-[0-9a-f]{8}) is paused: Ship 2 chairs\?", started)
    assert found is not None, started
    assert (tmp_path / "nodestep-examples" / "durable_thread.jsonl").is_file()

    resumed = run(found[1], "yes")

    assert resumed == f"Thread {found[1]} completed: shipped\n"


def test_send_fan_out_example_runs(capsys) -> None:
    module = _load_example("send_fan_out")

    module.main()

    assert capsys.readouterr().out.splitlines() == [
        "north: 21.0",
        "south: 17.5",
        "west: 19.9",
        "cheapest desk lamp: south at 17.5",
    ]


def test_subgraph_example_runs(capsys) -> None:
    module = _load_example("subgraph")

    module.main()

    output = capsys.readouterr().out
    assert "the title should start with a capital letter" in output
    assert "the body has 4 words, at least 12 are needed" in output
    assert "draft published: False" in output
    assert "final issues: []" in output
    assert "final published: True" in output


def test_guarded_tools_example_runs(capsys) -> None:
    module = _load_example("guarded_tools")

    module.main()

    assert capsys.readouterr().out.splitlines() == [
        "approval requested: refund {'order_id': 'A-1002', 'amount': 250.0}",
        'call_1: {"result":"refunded 30.0 on A-1001"}',
        "call_2: Tool 'refund' denied: Refunds over 100 need a manager.",
        "call_3: Tool 'refund' denied: per-tool limit exceeded (2)",
        "One refund went through; the others need a manager.",
    ]


def test_custom_middleware_example_runs(capsys) -> None:
    module = _load_example("custom_middleware")

    module.main()

    assert capsys.readouterr().out.splitlines() == [
        "step 0: take_note starts",
        "step 0: take_note returned "
        "{'note': 'Customer wrote: Please call me, or write to [email].'}",
        "step 1: answer starts",
        "step 1: answer returned {'reply': 'Thanks, we will call you back today.'}",
        "run of tickets completed",
        "stored note: Customer wrote: Please call me, or write to [email].",
    ]


def test_ask_user_example_runs(monkeypatch, capsys) -> None:
    module = _load_example("ask_user")
    monkeypatch.setattr("builtins.input", lambda prompt="": "Ada")

    module.main()
    module.main_without_middleware()

    output = capsys.readouterr().out
    assert output.count("Hello, Ada!") == 2


def test_human_approval_example_runs(capsys) -> None:
    module = _load_example("human_approval")

    module.main()

    output = capsys.readouterr().out
    assert "status: interrupted" in output
    assert "approve:confirm" in output
    assert "approved: True" in output


async def test_rewrite_message_example_runs(capsys) -> None:
    module = _load_example("rewrite_message")

    await module.main()

    output = capsys.readouterr().out
    assert "state at first event: {'messages': []}" in output
    assert "branch id: rewrite" in output
    assert "branches: ['rewrite']" in output
    assert "rewritten messages: ['hello again', 'echo: hello again']" in output
    assert "main messages: ['hi', 'echo: hi']" in output


async def test_memory_agent_example_runs(capsys) -> None:
    module = _load_example("memory_agent")

    await module.main()

    output = capsys.readouterr().out
    assert "listed 2 memories under user:42" in output
    assert "search 'python' → ['py-style']" in output
    assert "delete py-style → deleted=True" in output


async def test_memory_agent_example_runs_in_the_sandbox_with_its_context() -> None:
    from nodestep_sandbox.loader import TargetLoader
    from nodestep_sandbox.session import Sandbox

    loader = TargetLoader(EXAMPLES_DIR.parent)
    module = _load_example("memory_agent")
    memory = await loader.context("examples/memory_agent.py:temporary_memory")
    run = Sandbox(module.graph, context=memory).start({"query": "python"})
    async for _ in run.follow():
        pass

    assert run.status == "completed", run.error
