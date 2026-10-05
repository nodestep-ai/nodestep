import json
import re
from importlib.resources import files
from typing import Any, Literal, TypedDict

from nodestep import END, START, Graph, node
from nodestep_sandbox.web import MERMAID_INTEGRITY, MERMAID_URL


class PickState(TypedDict):
    choice: Literal["a", "b"]


@node
def pick(state: PickState) -> dict[str, Any]:
    return {"choice": state["choice"]}


def test_mermaid_is_pinned_by_version_and_hash() -> None:
    assert (
        MERMAID_URL == "https://cdn.jsdelivr.net/npm/mermaid@12.0.0/dist/mermaid.min.js"
    )
    assert (
        MERMAID_INTEGRITY
        == "sha384-xzghz1GQ5u9HCpVskeDPqMsdogD1yvuMQbEK53+wi+G70+6J1AG0L2cfi9PHjDWI"
    )


def test_templates_and_stylesheet_ship_inside_the_package() -> None:
    package = files("nodestep_sandbox")
    assert package.joinpath("templates", "base.html").is_file()
    assert package.joinpath("static", "sandbox.css").is_file()


def test_graph_page_shows_diagram_nodes_and_state_fields(
    graphs: Any, open_site: Any
) -> None:
    site = open_site(graphs.echo())
    response = site.client.get("/")
    assert response.status_code == 200
    page = response.text
    assert '<pre class="mermaid nodestep-mermaid">' in page
    assert "START START-shout@--&gt; shout" in page
    assert "<summary>Mermaid source</summary>" in page
    assert (
        f'<script src="{MERMAID_URL}" integrity="{MERMAID_INTEGRITY}" '
        'crossorigin="anonymous"></script>'
    ) in page
    assert '<td data-label="Node"><code>shout</code></td>' in page
    assert '<td data-label="Edge to"><code>END</code></td>' in page
    assert (
        '<td data-label="Field"><code>log</code></td>'
        '<td data-label="Type"><code>list[str]</code></td>'
        '<td data-label="Reducer"><code>add</code></td>'
    ) in page
    assert "in-memory state store" in page
    assert "The graph object you loaded is not changed." in page


def test_graph_page_lists_goto_targets(graphs: Any, open_site: Any) -> None:
    page = open_site(graphs.review()).client.get("/").text
    assert "<code>ask_title</code>, <code>ask_body</code>" in page


def test_new_run_page_has_a_control_per_field_and_a_raw_json_form(
    graphs: Any, open_site: Any
) -> None:
    site = open_site(graphs.echo())
    page = site.client.get("/runs/new").text
    assert f'name="csrf" value="{site.site.csrf_token}"' in page
    assert (
        '<input type="text" id="state.text" name="state.text" value="" '
        'class="nodestep-input">'
    ) in page
    assert '<textarea id="state.log" name="state.log"' in page
    assert '<textarea id="raw" name="raw"' in page


def test_a_literal_field_is_a_select_of_its_values(open_site: Any) -> None:
    site = open_site(Graph(PickState, name="pick").flow(START >> pick, pick >> END))
    page = site.client.get("/runs/new").text
    assert (
        '<select id="state.choice" name="state.choice" class="nodestep-select">' in page
    )
    assert '<option value="" selected>(not set)</option>' in page
    assert '<option value="1">&#39;b&#39;</option>' in page
    refused = site.post("/runs", {"mode": "form", "state.choice": "9"})
    assert refused.status_code == 422
    assert "Choose one of the listed values" in refused.text
    started = site.post("/runs", {"mode": "form", "state.choice": "1"})
    assert json.loads(site.last_run(started).final_state or "") == {"choice": "b"}


def test_starting_a_run_from_the_form_streams_the_run_page(
    graphs: Any, open_site: Any
) -> None:
    site = open_site(graphs.echo())
    response = site.post("/runs", {"mode": "form", "state.text": "hi", "state.log": ""})
    assert response.status_code == 200
    assert response.history[0].status_code == 303
    assert re.fullmatch(r"/runs/[0-9a-f]{12}", response.url.path)
    page = response.text
    assert page.count('class="nodestep-list-row event-update"') == 1
    assert 'class="nodestep-list-row event-custom"' in page
    assert 'class="nodestep-list-row event-debug"' in page
    assert "nodestep-card-completed" in page
    assert "&#34;HI&#34;" in page


def test_a_debug_row_is_one_line_that_opens_to_its_data(
    graphs: Any, open_site: Any
) -> None:
    page = open_site(graphs.echo()).start_json({"text": "hi"}).text
    assert (
        '<li class="nodestep-list-row event-debug">\n<details class="nodestep-details">'
        '<summary><span class="nodestep-tag">routes</span> <code class="node">shout'
        '</code> <span class="nodestep-hint">step 0</span> <span class="nodestep-hint">'
        "next: END</span></summary>"
    ) in page


def test_starting_a_run_from_raw_json(graphs: Any, open_site: Any) -> None:
    site = open_site(graphs.echo())
    response = site.start_json({"text": "yo"})
    assert "nodestep-card-completed" in response.text
    assert "&#34;YO&#34;" in response.text


def test_bad_form_input_shows_the_form_again_with_the_error(
    graphs: Any, open_site: Any
) -> None:
    site = open_site(graphs.echo())
    response = site.post(
        "/runs", {"mode": "form", "state.text": "hi", "state.log": "[1,"}
    )
    assert response.status_code == 422
    assert "Not valid JSON" in response.text
    assert 'value="hi"' in response.text
    assert not site.sandbox.runs


def test_bad_raw_json_shows_the_error(graphs: Any, open_site: Any) -> None:
    site = open_site(graphs.echo())
    response = site.post("/runs", {"mode": "json", "raw": "[]"})
    assert response.status_code == 422
    assert "The input must be a JSON object of state fields" in response.text
    assert "[]</textarea>" in response.text


def test_a_run_page_can_be_reloaded(graphs: Any, open_site: Any) -> None:
    site = open_site(graphs.echo())
    first = site.start_json({"text": "hi"})
    again = site.client.get(first.url.path)
    assert again.status_code == 200
    assert again.text.count('class="nodestep-list-row ') == first.text.count(
        'class="nodestep-list-row '
    )
    assert "nodestep-card-completed" in again.text


def test_step_page_shows_state_before_update_after_and_timing(
    graphs: Any, open_site: Any
) -> None:
    site = open_site(graphs.echo())
    run = site.last_run(site.start_json({"text": "hi"}))
    page = site.client.get(f"/runs/{run.id}/steps/0").text
    for heading in ("State before", "Update", "State after", "Elapsed"):
        assert heading in page
    assert "&#34;hi&#34;" in page
    assert "&#34;HI&#34;" in page


def test_threads_page_lists_the_session_threads(graphs: Any, open_site: Any) -> None:
    site = open_site(graphs.echo())
    assert "No runs yet" in site.client.get("/threads").text
    run = site.last_run(site.start_json({"text": "hi"}))
    page = site.client.get("/threads").text
    assert f'href="/threads/{run.thread_id}"' in page
    assert "nodestep-badge-completed" in page


def test_thread_page_shows_runs_and_stored_history(graphs: Any, open_site: Any) -> None:
    site = open_site(graphs.echo())
    run = site.last_run(site.start_json({"text": "hi"}))
    page = site.client.get(f"/threads/{run.thread_id}").text
    assert f'href="/runs/{run.id}"' in page
    for event_type in ("run_started", "node_completed", "run_completed"):
        assert f"<code>{event_type}</code>" in page


def test_unknown_ids_give_404(graphs: Any, open_site: Any) -> None:
    site = open_site(graphs.echo())
    run = site.last_run(site.start_json({"text": "hi"}))
    for path in (
        "/runs/nope",
        "/runs/nope/steps/0",
        f"/runs/{run.id}/steps/5",
        "/threads/nope",
    ):
        response = site.client.get(path)
        assert response.status_code == 404
        assert "does not exist in this sandbox session" in response.text


def test_an_unknown_page_gives_the_not_found_page(graphs: Any, open_site: Any) -> None:
    site = open_site(graphs.echo())
    for path in ("/nope", "/runs/nope/more", "/static/nope.css"):
        response = site.client.get(path)
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("text/html")
        assert page_title(response.text) == "Not found · nodestep sandbox"
        assert "This page does not exist in this sandbox session" in response.text


def page_title(page: str) -> str:
    found = re.search(r"<title>([^<]*)</title>", page)
    assert found
    return found.group(1)


def test_pages_are_titled_page_first_then_the_sandbox(
    graphs: Any, open_site: Any
) -> None:
    site = open_site(graphs.echo())
    run = site.last_run(site.start_json({"text": "hi"}))
    titles = {
        "/": "nodestep sandbox",
        "/runs/new": "New run · nodestep sandbox",
        "/threads": "Threads · nodestep sandbox",
        f"/threads/{run.thread_id}": f"{run.thread_id} · nodestep sandbox",
        f"/runs/{run.id}": f"{run.id} · nodestep sandbox",
        f"/runs/{run.id}/steps/0": "shout · nodestep sandbox",
        "/threads/nope": "Not found · nodestep sandbox",
    }
    assert {path: page_title(site.client.get(path).text) for path in titles} == titles


def test_the_stylesheet_is_served(graphs: Any, open_site: Any) -> None:
    response = open_site(graphs.echo()).client.get("/static/sandbox.css")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")


def test_untyped_state_offers_only_raw_json(graphs: Any, open_site: Any) -> None:
    site = open_site(graphs.untyped())
    assert 'name="mode" value="form"' not in site.client.get("/runs/new").text
    assert "Untyped state" in site.client.get("/").text
    assert "nodestep-card-completed" in site.start_json({"any": 1}).text


def test_a_failed_run_shows_the_error(graphs: Any, open_site: Any) -> None:
    response = open_site(graphs.failing()).start_json({"text": "hi"})
    assert response.status_code == 200
    assert "nodestep-card-failed" in response.text
    assert "the node failed on purpose" in response.text


def test_values_are_escaped(graphs: Any, open_site: Any) -> None:
    response = open_site(graphs.echo()).start_json(
        {"text": "<script>alert(1)</script>"}
    )
    assert "<script>alert" not in response.text
    assert "<SCRIPT>" not in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text
