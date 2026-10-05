import json
from typing import Any

from nodestep_sandbox.runs import Run
from nodestep_sandbox.session import Sandbox


async def finish(run: Run) -> Run:
    async for _ in run.follow():
        pass
    return run


async def test_context_is_passed_to_the_run_and_to_the_resume(graphs: Any) -> None:
    sandbox = Sandbox(graphs.context(), context={"name": "ctx-1"})
    run = await finish(sandbox.start({}))
    assert run.status == "interrupted"
    assert json.loads(run.final_state or "")["before"] == "ctx-1"
    [item] = run.pending
    resumed = await finish(sandbox.resume(run, {item.key: "yes"}))
    assert resumed.status == "completed"
    assert json.loads(resumed.final_state or "") == {
        "before": "ctx-1",
        "after": "ctx-1",
    }


async def test_without_context_a_node_that_reads_it_fails(graphs: Any) -> None:
    run = await finish(Sandbox(graphs.context()).start({}))
    assert run.status == "error"
    assert "ContextNotProvidedError" in (run.error or "")


def test_the_web_pages_pass_the_context_too(graphs: Any, open_site: Any) -> None:
    site = open_site(graphs.context(), context={"name": "web"})
    paused = site.start_json({})
    run = site.last_run(paused)
    resumed = site.post(
        f"/runs/{run.id}/resume", {"answer.0": "go", "format.0": "text"}
    )
    assert "nodestep-card-completed" in resumed.text
    assert json.loads(site.last_run(resumed).final_state or "")["after"] == "web"
