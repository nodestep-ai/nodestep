import secrets
from collections.abc import Collection
from datetime import datetime
from typing import Any
from urllib.parse import parse_qsl

from jinja2 import Environment, PackageLoader, StrictUndefined
from starlette.applications import Starlette
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import (
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from nodestep_sandbox.errors import FormError, RunStateError
from nodestep_sandbox.interrupts import ResumeForm
from nodestep_sandbox.overview import GraphOverview
from nodestep_sandbox.records import HistoryRow
from nodestep_sandbox.runs import Run
from nodestep_sandbox.session import Sandbox
from nodestep_sandbox.state_form import StateForm

MERMAID_URL = "https://cdn.jsdelivr.net/npm/mermaid@12.0.0/dist/mermaid.min.js"
MERMAID_INTEGRITY = (
    "sha384-xzghz1GQ5u9HCpVskeDPqMsdogD1yvuMQbEK53+wi+G70+6J1AG0L2cfi9PHjDWI"
)
HTML = "text/html; charset=utf-8"
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        f"default-src 'none'; script-src 'self' {MERMAID_URL.rsplit('/', 1)[0]}/; "
        "style-src 'self' 'unsafe-inline'; img-src 'self'; "
        "form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}


def host_header(host: str, port: int) -> str:
    """Return the ``Host`` header a browser sends for ``host`` and ``port``.

    Browsers write host names in lower case, and IPv6 addresses in brackets.

    Parameters
    ----------
    host : str
    port : int

    Returns
    -------
    str
    """
    host = host.lower()
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def allowed_hosts(host: str, port: int) -> frozenset[str]:
    """Return the ``Host`` header values that name the bound address.

    Browsers leave out port 80, so it is accepted without the port too.

    Parameters
    ----------
    host : str
    port : int

    Returns
    -------
    frozenset[str]
    """
    address = host_header(host, port)
    if port == 80:
        return frozenset({address, address.removesuffix(":80")})
    return frozenset({address})


def _clock(timestamp: float | None) -> str:
    if timestamp is None:
        return ""
    return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S.%f")[:-3]


class SecurityHeaders:
    """ASGI middleware that adds ``SECURITY_HEADERS`` to every HTTP response.

    Parameters
    ----------
    app : ASGIApp
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    headers[name] = value
            await send(message)

        await self.app(scope, receive, send_with_headers)


class HostCheck:
    """ASGI middleware that refuses requests whose ``Host`` header is not allowed.

    Parameters
    ----------
    app : ASGIApp
    allowed : Collection[str]
        Accepted ``Host`` header values, compared without case.
    """

    def __init__(self, app: ASGIApp, *, allowed: Collection[str]) -> None:
        self.app = app
        self.allowed = frozenset(value.lower() for value in allowed)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] == "http"
            and Headers(scope=scope).get("host", "").lower() not in self.allowed
        ):
            response = PlainTextResponse(
                "Host not allowed: the sandbox answers only requests for "
                f"{', '.join(sorted(self.allowed))}",
                status_code=400,
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


class SandboxSite:
    """The sandbox web pages for one ``Sandbox``.

    Pages are rendered on the server with Jinja2, autoescaped; the only scripts
    are the theme button, served from ``/static``, and Mermaid on the graph
    page. Requests whose ``Host`` header does not name the bound address are
    refused, and every form carries a random token made for this process that
    each POST must send back.

    Parameters
    ----------
    sandbox : Sandbox
    host : str
        Address the server binds.
    port : int
        Port the server listens on.

    Attributes
    ----------
    app : Starlette
        The ASGI application; ``app.state.site`` is this object.
    csrf_token : str
    address : str
        The ``Host`` header value that names the bound address.
    """

    def __init__(self, sandbox: Sandbox, *, host: str, port: int) -> None:
        self.sandbox = sandbox
        self.address = host_header(host, port)
        self.csrf_token = secrets.token_urlsafe(32)
        self.overview = GraphOverview.from_graph(sandbox.graph)
        self.state_form = StateForm(sandbox.graph.state_schema)
        self.templates = Environment(
            loader=PackageLoader("nodestep_sandbox", "templates"),
            autoescape=True,
            enable_async=True,
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.templates.filters["clock"] = _clock
        self.app = Starlette(
            routes=[
                Route("/", self.graph_page, methods=["GET"]),
                Route("/runs/new", self.new_run_page, methods=["GET"]),
                Route("/runs", self.create_run, methods=["POST"]),
                Route("/runs/{run_id}", self.run_page, methods=["GET"]),
                Route("/runs/{run_id}/resume", self.resume_run, methods=["POST"]),
                Route(
                    "/runs/{run_id}/steps/{index:int}", self.step_page, methods=["GET"]
                ),
                Route("/threads", self.threads_page, methods=["GET"]),
                Route("/threads/{thread_id}", self.thread_page, methods=["GET"]),
                Mount(
                    "/static",
                    StaticFiles(packages=[("nodestep_sandbox", "static")]),
                    name="static",
                ),
            ],
            middleware=[
                Middleware(SecurityHeaders),
                Middleware(HostCheck, allowed=allowed_hosts(host, port)),
            ],
            exception_handlers={404: self.unknown_page},
        )
        self.app.state.site = self

    async def graph_page(self, request: Request) -> Response:
        """Show the diagram, the nodes and the state fields."""
        return await self._page(
            "overview.html",
            section="graph",
            overview=self.overview,
            form=self.state_form,
            mermaid_url=MERMAID_URL,
            mermaid_integrity=MERMAID_INTEGRITY,
        )

    async def new_run_page(self, request: Request) -> Response:
        """Show the new-run form."""
        return await self._page(
            "new_run.html", section="new", form=self.state_form, values={}, errors={}
        )

    async def create_run(self, request: Request) -> Response:
        """Start a run from the form or the raw JSON and redirect to its page."""
        values = await self._form(request)
        try:
            value = (
                StateForm.parse_raw(values.get("raw", ""))
                if values.get("mode") == "json"
                else self.state_form.parse(values)
            )
        except FormError as error:
            return await self._page(
                "new_run.html",
                status_code=422,
                section="new",
                form=self.state_form,
                values=values,
                errors=error.errors,
            )
        run = self.sandbox.start(value)
        return RedirectResponse(f"/runs/{run.id}", status_code=303)

    async def run_page(self, request: Request) -> Response:
        """Stream a run's rows as they arrive, then its outcome."""
        run = self.sandbox.runs.get(request.path_params["run_id"])
        if run is None:
            return await self._not_found("This run")
        return self._stream("run.html", run=run, values={}, errors={})

    async def resume_run(self, request: Request) -> Response:
        """Answer a paused run and redirect to the run that resumes it."""
        values = await self._form(request)
        run = self.sandbox.runs.get(request.path_params["run_id"])
        if run is None:
            return await self._not_found("This run")
        try:
            self.sandbox.check_answerable(run)
            resumed = self.sandbox.resume(run, ResumeForm(run.pending).parse(values))
        except FormError as error:
            return self._stream(
                "run.html", status_code=422, run=run, values=values, errors=error.errors
            )
        except RunStateError as error:
            latest = self.sandbox.latest_run(run.thread_id)
            return await self._page(
                "message.html",
                status_code=409,
                section="",
                title="Nothing to answer",
                message=str(error),
                link=f"/runs/{latest.id}",
                link_text=f"Latest run of thread {run.thread_id}",
            )
        return RedirectResponse(f"/runs/{resumed.id}", status_code=303)

    async def step_page(self, request: Request) -> Response:
        """Show one task of a run: state before, update, state after and timing."""
        run = self.sandbox.runs.get(request.path_params["run_id"])
        index = request.path_params["index"]
        if run is None or not 0 <= index < len(run.tasks):
            return await self._not_found("This step")
        return await self._page("step.html", section="", run=run, task=run.tasks[index])

    async def threads_page(self, request: Request) -> Response:
        """List the threads of this session."""
        return await self._page(
            "threads.html", section="threads", threads=self.sandbox.thread_summaries()
        )

    async def thread_page(self, request: Request) -> Response:
        """Show a thread's runs and its stored history."""
        thread_id = request.path_params["thread_id"]
        if not self.sandbox.has_thread(thread_id):
            return await self._not_found("This thread")
        history = await self.sandbox.history(thread_id)
        return await self._page(
            "thread.html",
            section="threads",
            thread_id=thread_id,
            runs=self.sandbox.thread_runs(thread_id),
            history=[HistoryRow.from_event(event) for event in history.events],
        )

    async def unknown_page(self, request: Request, error: Exception) -> Response:
        """Show the not-found page for a path no route serves."""
        return await self._not_found("This page")

    async def _form(self, request: Request) -> dict[str, str]:
        body = (await request.body()).decode("utf-8", errors="replace")
        values = dict(parse_qsl(body, keep_blank_values=True))
        if not secrets.compare_digest(
            values.get("csrf", "").encode(), self.csrf_token.encode()
        ):
            raise HTTPException(
                403,
                "The form has no sandbox token or an old one; reload the page and "
                "submit it again",
            )
        return values

    def _globals(self) -> dict[str, Any]:
        graph = self.sandbox.graph
        if self.sandbox.in_memory_store:
            store = (
                "with an in-memory state store, so threads last until the sandbox stops"
            )
        else:
            store = f"with the graph's own state store ({type(self.sandbox.store).__name__})"
        return {
            "csrf": self.csrf_token,
            "store_note": (
                f"Runs use a copy of graph '{graph.name}' {store}. "
                "The graph object you loaded is not changed."
            ),
            "next_run": self.sandbox.next_run,
        }

    async def _page(
        self, name: str, *, status_code: int = 200, **values: Any
    ) -> HTMLResponse:
        html = await self.templates.get_template(name).render_async(
            **self._globals(), **values
        )
        return HTMLResponse(html, status_code=status_code)

    def _stream(
        self, name: str, *, status_code: int = 200, run: Run, **values: Any
    ) -> StreamingResponse:
        chunks = self.templates.get_template(name).generate_async(
            **self._globals(), section="", run=run, **values
        )
        return StreamingResponse(chunks, status_code=status_code, media_type=HTML)

    async def _not_found(self, what: str) -> Response:
        return await self._page(
            "message.html",
            status_code=404,
            section="",
            title="Not found",
            message=f"{what} does not exist in this sandbox session.",
            link="/threads",
            link_text="Threads of this session",
        )
