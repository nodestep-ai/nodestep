import json
from typing import Any

from nodestep_sandbox.runs import Run


def position(run: Run, interrupt_id: str) -> int:
    return [item.id for item in run.pending].index(interrupt_id)


def paused_review(graphs: Any, open_site: Any) -> tuple[Any, Run, Any]:
    site = open_site(graphs.review())
    response = site.start_json({"draft": "notes"})
    return site, site.last_run(response), response


def test_the_pause_shows_one_form_section_per_interrupt_key(
    graphs: Any, open_site: Any
) -> None:
    _, run, response = paused_review(graphs, open_site)
    page = response.text
    assert "nodestep-card-waiting" in page
    for item in run.pending:
        assert f"<legend><code>{item.key}</code></legend>" in page
    approval = position(run, "approve_publish")
    for value in ("approve", "edit", "deny"):
        assert f'name="decision.{approval}" value="{value}"' in page
    assert f'name="arguments.{approval}"' in page
    assert f'name="answer.{position(run, "title")}"' in page
    assert (
        '<div class="nodestep-code nodestep-code-wrap"><pre>{\n'
        "  &#34;question&#34;: &#34;Which title?&#34;"
    ) in page


def test_the_approval_form_says_an_edit_is_merged(graphs: Any, open_site: Any) -> None:
    _, _, response = paused_review(graphs, open_site)
    assert "keys left out keep their values" in response.text


def test_no_approval_decision_is_chosen_in_advance(graphs: Any, open_site: Any) -> None:
    site, run, response = paused_review(graphs, open_site)
    title, approval = position(run, "title"), position(run, "approve_publish")
    for value in ("approve", "edit", "deny"):
        assert f'name="decision.{approval}" value="{value}">' in response.text
    refused = site.post(f"/runs/{run.id}/resume", {f"answer.{title}": "Hello"})
    assert refused.status_code == 422
    assert "Choose approve, edit or deny" in refused.text
    assert site.sandbox.next_run(run) is None
    kept = site.post(
        f"/runs/{run.id}/resume",
        {f"answer.{title}": "", f"decision.{approval}": "deny"},
    )
    assert f'name="decision.{approval}" value="deny" checked>' in kept.text


def test_answers_resume_the_thread_and_the_new_run_streams(
    graphs: Any, open_site: Any
) -> None:
    site, run, _ = paused_review(graphs, open_site)
    title, approval = position(run, "title"), position(run, "approve_publish")
    response = site.post(
        f"/runs/{run.id}/resume",
        {
            f"answer.{title}": "Hello",
            f"format.{title}": "text",
            f"decision.{approval}": "edit",
            f"arguments.{approval}": '{"draft": "final"}',
            f"message.{approval}": "",
        },
    )
    assert response.history[0].status_code == 303
    resumed = site.last_run(response)
    assert resumed.previous == run.id
    assert "nodestep-card-completed" in response.text
    state = json.loads(resumed.final_state or "")
    assert state["title"] == "Hello"
    assert json.loads(state["body"]) == {
        "action": "edit",
        "arguments": {"draft": "final"},
    }


def test_deny_sends_the_reason(graphs: Any, open_site: Any) -> None:
    site, run, _ = paused_review(graphs, open_site)
    title, approval = position(run, "title"), position(run, "approve_publish")
    response = site.post(
        f"/runs/{run.id}/resume",
        {
            f"answer.{title}": "Hi",
            f"decision.{approval}": "deny",
            f"message.{approval}": "not yet",
        },
    )
    state = json.loads(site.last_run(response).final_state or "")
    assert json.loads(state["body"]) == {"action": "deny", "message": "not yet"}


def test_a_json_answer_is_parsed(graphs: Any, open_site: Any) -> None:
    site, run, _ = paused_review(graphs, open_site)
    title, approval = position(run, "title"), position(run, "approve_publish")
    response = site.post(
        f"/runs/{run.id}/resume",
        {
            f"answer.{title}": '"Quoted"',
            f"format.{title}": "json",
            f"decision.{approval}": "approve",
        },
    )
    assert json.loads(site.last_run(response).final_state or "")["title"] == "Quoted"


def test_a_bad_answer_keeps_the_form_and_the_pause(graphs: Any, open_site: Any) -> None:
    site, run, _ = paused_review(graphs, open_site)
    title, approval = position(run, "title"), position(run, "approve_publish")
    response = site.post(
        f"/runs/{run.id}/resume",
        {
            f"answer.{title}": "Hello",
            f"decision.{approval}": "edit",
            f"arguments.{approval}": "[1]",
        },
    )
    assert response.status_code == 422
    assert "Edited arguments must be a JSON object" in response.text
    assert "[1]</textarea>" in response.text
    assert "Hello</textarea>" in response.text
    assert site.sandbox.next_run(run) is None


def test_a_pause_can_be_answered_only_once(graphs: Any, open_site: Any) -> None:
    site, run, _ = paused_review(graphs, open_site)
    title, approval = position(run, "title"), position(run, "approve_publish")
    answers = {f"answer.{title}": "Hello", f"decision.{approval}": "approve"}
    first = site.post(f"/runs/{run.id}/resume", answers)
    second = site.post(f"/runs/{run.id}/resume", answers)
    assert second.status_code == 409
    assert "already answered" in second.text
    assert f'href="/runs/{site.last_run(first).id}"' in second.text
    reloaded = site.client.get(f"/runs/{run.id}").text
    assert "This pause was answered in run" in reloaded
    assert f'action="/runs/{run.id}/resume"' not in reloaded


def test_a_second_submit_with_a_bad_form_still_says_the_pause_was_answered(
    graphs: Any, open_site: Any
) -> None:
    site, run, _ = paused_review(graphs, open_site)
    title, approval = position(run, "title"), position(run, "approve_publish")
    first = site.post(
        f"/runs/{run.id}/resume",
        {f"answer.{title}": "Hello", f"decision.{approval}": "approve"},
    )
    second = site.post(f"/runs/{run.id}/resume", {f"answer.{title}": ""})
    assert second.status_code == 409
    assert "already answered" in second.text
    assert f'href="/runs/{site.last_run(first).id}"' in second.text


def test_a_completed_run_cannot_be_resumed(graphs: Any, open_site: Any) -> None:
    site = open_site(graphs.echo())
    run = site.last_run(site.start_json({"text": "hi"}))
    response = site.post(f"/runs/{run.id}/resume", {})
    assert response.status_code == 409
    assert "nothing to answer" in response.text


def test_resuming_an_unknown_run_gives_404(graphs: Any, open_site: Any) -> None:
    assert open_site(graphs.echo()).post("/runs/nope/resume", {}).status_code == 404
