from typing import Any

from starlette.routing import Mount, Route
from starlette.testclient import TestClient

from nodestep_sandbox.session import Sandbox
from nodestep_sandbox.web import (
    MERMAID_URL,
    SECURITY_HEADERS,
    SandboxSite,
    allowed_hosts,
    host_header,
)


def test_requests_for_another_host_are_refused(graphs: Any, open_site: Any) -> None:
    site = open_site(graphs.echo())
    for host in ("localhost:8765", "127.0.0.1:9999", "evil.example"):
        response = site.client.get("/", headers={"Host": host})
        assert response.status_code == 400
        assert "Host" in response.text
    refused = site.client.post(
        "/runs",
        data={"csrf": site.site.csrf_token, "mode": "json", "raw": "{}"},
        headers={"Host": "evil.example"},
    )
    assert refused.status_code == 400
    assert not site.sandbox.runs


def test_the_host_header_is_compared_without_case(graphs: Any) -> None:
    site = SandboxSite(Sandbox(graphs.echo()), host="LocalHost", port=8765)
    assert site.address == "localhost:8765"
    with TestClient(site.app, base_url="http://localhost:8765") as client:
        for host in ("localhost:8765", "LOCALHOST:8765"):
            assert client.get("/", headers={"Host": host}).status_code == 200
        assert client.get("/", headers={"Host": "localhost:9999"}).status_code == 400


def test_posts_need_the_csrf_token(graphs: Any, open_site: Any) -> None:
    site = open_site(graphs.echo())
    for extra in ({}, {"csrf": "wrong"}, {"csrf": "\u00e9"}):
        response = site.client.post(
            "/runs", data={"mode": "json", "raw": "{}", **extra}
        )
        assert response.status_code == 403
    assert not site.sandbox.runs
    run = site.last_run(site.start_json({"text": "hi"}))
    assert site.client.post(f"/runs/{run.id}/resume", data={}).status_code == 403


def test_every_site_has_its_own_random_token(graphs: Any) -> None:
    first = SandboxSite(Sandbox(graphs.echo()), host="127.0.0.1", port=8765)
    second = SandboxSite(Sandbox(graphs.echo()), host="127.0.0.1", port=8765)
    assert first.csrf_token != second.csrf_token
    assert len(first.csrf_token) >= 32


def test_scripts_may_come_only_from_the_sandbox_and_the_pinned_mermaid_folder() -> None:
    policy = SECURITY_HEADERS["Content-Security-Policy"]
    assert (
        "script-src 'self' https://cdn.jsdelivr.net/npm/mermaid@12.0.0/dist/;" in policy
    )
    assert "unsafe" not in policy.split("script-src", 1)[1].split(";", 1)[0]
    assert MERMAID_URL.startswith("https://cdn.jsdelivr.net/npm/mermaid@12.0.0/dist/")
    assert "frame-ancestors 'none'" in policy
    assert "img-src 'self';" in policy


def test_every_response_sends_the_security_headers(graphs: Any, open_site: Any) -> None:
    site = open_site(graphs.echo())
    started = site.start_json({"text": "hi"})
    responses = {
        "graph page": site.client.get("/"),
        "redirect": started.history[0],
        "streamed run page": started,
        "unknown path": site.client.get("/nope"),
        "wrong method": site.client.get("/runs"),
        "stylesheet": site.client.get("/static/sandbox.css"),
        "unknown static file": site.client.get("/static/nope.css"),
        "missing token": site.client.post("/runs", data={"mode": "json", "raw": "{}"}),
        "wrong host": site.client.get("/", headers={"Host": "evil.example"}),
    }
    for name, response in responses.items():
        for header, value in SECURITY_HEADERS.items():
            assert response.headers.get(header) == value, (name, header)


def test_allowed_host_headers() -> None:
    assert host_header("127.0.0.1", 8765) == "127.0.0.1:8765"
    assert host_header("::1", 8765) == "[::1]:8765"
    assert host_header("LocalHost", 8765) == "localhost:8765"
    assert allowed_hosts("127.0.0.1", 8765) == frozenset({"127.0.0.1:8765"})
    assert allowed_hosts("127.0.0.1", 80) == frozenset({"127.0.0.1:80", "127.0.0.1"})


def test_the_app_has_only_the_sandbox_routes(graphs: Any) -> None:
    site = SandboxSite(Sandbox(graphs.echo()), host="127.0.0.1", port=8765)
    routes = {
        (route.path, tuple(sorted(route.methods or ())))
        for route in site.app.routes
        if isinstance(route, Route)
    }
    page = ("GET", "HEAD")
    assert routes == {
        ("/", page),
        ("/runs/new", page),
        ("/runs", ("POST",)),
        ("/runs/{run_id}", page),
        ("/runs/{run_id}/resume", ("POST",)),
        ("/runs/{run_id}/steps/{index:int}", page),
        ("/threads", page),
        ("/threads/{thread_id}", page),
    }
    assert [route.path for route in site.app.routes if isinstance(route, Mount)] == [
        "/static"
    ]
