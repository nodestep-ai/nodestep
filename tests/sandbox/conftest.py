import json
import warnings
from collections.abc import Callable, Iterator
from typing import Annotated, Any, TypedDict

import httpx
import pytest

from nodestep import END, START, Command, Graph, NodeContext, add, interrupt, node
from nodestep.state import StateStore
from nodestep_sandbox.runs import Run
from nodestep_sandbox.session import Sandbox
from nodestep_sandbox.web import SandboxSite

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Using `httpx` with `starlette.testclient` is deprecated",
        category=UserWarning,
    )
    from starlette.testclient import TestClient


class EchoState(TypedDict):
    text: str
    log: Annotated[list[str], add]


class ReviewState(TypedDict):
    draft: str
    title: str
    body: str


class ContextState(TypedDict):
    before: str
    after: str


@node
def shout(state: EchoState, ctx: NodeContext) -> dict[str, Any]:
    ctx.emit({"heard": state["text"]})
    return {"text": state["text"].upper(), "log": ["shout"]}


@node
def explode(state: EchoState) -> dict[str, Any]:
    raise RuntimeError("the node failed on purpose")


@node
def ask_title(state: ReviewState) -> dict[str, Any]:
    return {"title": interrupt({"question": "Which title?"}, id="title")}


@node
def ask_body(state: ReviewState) -> dict[str, Any]:
    decision = interrupt(
        {
            "kind": "tool_interrupt",
            "tool_name": "publish",
            "tool_call_id": "call_1",
            "arguments": {"draft": state["draft"]},
        },
        id="approve_publish",
    )
    return {"body": json.dumps(decision, sort_keys=True)}


@node(goto=[ask_title, ask_body])
def split(state: ReviewState) -> Command:
    return Command(goto=[ask_title, ask_body])


@node
def read_before(state: ContextState, ctx: NodeContext) -> dict[str, Any]:
    return {"before": ctx.context["name"]}


@node
def read_after_pause(state: ContextState, ctx: NodeContext) -> dict[str, Any]:
    interrupt({"question": "Continue?"}, id="go")
    return {"after": ctx.context["name"]}


@node
def mark(state: dict[str, Any]) -> dict[str, Any]:
    return {"seen": True}


class Graphs:
    def echo(
        self, *, name: str = "echo", state_store: StateStore | None = None
    ) -> Graph[EchoState]:
        return Graph(EchoState, name=name, state_store=state_store).flow(
            START >> shout, shout >> END
        )

    def failing(self) -> Graph[EchoState]:
        return Graph(EchoState, name="failing").flow(START >> explode, explode >> END)

    def review(self) -> Graph[ReviewState]:
        return Graph(ReviewState, name="review").flow(
            START >> split, ask_title >> END, ask_body >> END
        )

    def context(self) -> Graph[ContextState]:
        return Graph(ContextState, name="context").flow(
            START >> read_before,
            read_before >> read_after_pause,
            read_after_pause >> END,
        )

    def untyped(self) -> Graph[dict[str, Any]]:
        return Graph(dict, name="untyped").flow(START >> mark, mark >> END)


@pytest.fixture
def graphs() -> Graphs:
    return Graphs()


HOST = "127.0.0.1"
PORT = 8765


class SiteClient:
    def __init__(self, graph: Graph[Any], *, context: object | None = None) -> None:
        self.site = SandboxSite(Sandbox(graph, context=context), host=HOST, port=PORT)
        self.client = TestClient(self.site.app, base_url=f"http://{HOST}:{PORT}")

    @property
    def sandbox(self) -> Sandbox:
        return self.site.sandbox

    def post(self, path: str, data: dict[str, str]) -> httpx.Response:
        return self.client.post(path, data={"csrf": self.site.csrf_token, **data})

    def start_json(self, value: dict[str, Any]) -> httpx.Response:
        return self.post("/runs", {"mode": "json", "raw": json.dumps(value)})

    def last_run(self, response: httpx.Response) -> Run:
        return self.sandbox.runs[response.url.path.split("/")[2]]


@pytest.fixture
def open_site() -> Iterator[Callable[..., SiteClient]]:
    opened: list[SiteClient] = []

    def open_site_client(
        graph: Graph[Any], *, context: object | None = None
    ) -> SiteClient:
        site = SiteClient(graph, context=context)
        site.client.__enter__()
        opened.append(site)
        return site

    yield open_site_client
    for site in opened:
        site.client.__exit__(None, None, None)
