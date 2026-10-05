import asyncio
from functools import partial
from pathlib import Path
from typing import Annotated

import typer

from nodestep_sandbox.errors import SandboxError, TargetError
from nodestep_sandbox.server import SandboxServer, ServerOptions

app = typer.Typer(
    name="nodestep",
    help="Command line tools for nodestep graphs.",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def nodestep() -> None:
    """Command line tools for nodestep graphs."""


@app.command()
def sandbox(
    target: Annotated[
        str,
        typer.Argument(
            help=(
                "Graph to load: package.module:attribute or "
                "path/to/file.py:attribute. The attribute is a Graph, or a "
                "function without arguments (sync or async) that returns one. "
                "The current directory is searched first."
            ),
            metavar="TARGET",
            show_default=False,
        ),
    ],
    context: Annotated[
        str | None,
        typer.Option(
            "--context",
            metavar="MODULE:ATTRIBUTE",
            help=(
                "Object passed as context= to every run and resume, named like "
                "TARGET. A callable is called once without arguments and its result "
                "is used."
            ),
        ),
    ] = None,
    host: Annotated[
        str,
        typer.Option(
            help="Address to bind. Requests must name it in their Host header."
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int, typer.Option(min=1, max=65535, help="Port to listen on.")
    ] = 8765,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Do not open a browser window.")
    ] = False,
) -> None:
    """Try a graph in a local web page.

    The page shows the graph, starts runs from a form built from the state
    model, streams each step and answers interrupts. Runs use a copy of the
    graph, with an in-memory state store when the graph has none.
    """
    options = ServerOptions(
        target=target,
        context=context,
        host=host,
        port=port,
        open_browser=not no_browser,
    )
    server = SandboxServer(
        options,
        cwd=Path.cwd(),
        announce=typer.echo,
        warn=partial(typer.echo, err=True),
    )
    try:
        asyncio.run(server.serve())
    except SandboxError as error:
        if isinstance(error, TargetError) and error.details:
            typer.echo(error.details, err=True, nl=False)
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from error
