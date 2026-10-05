import asyncio
import ipaddress
import os
import signal
import socket
import sys
import webbrowser
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path
from types import FrameType
from typing import Self

import uvicorn
from pydantic import BaseModel

from nodestep_sandbox.errors import SandboxError
from nodestep_sandbox.loader import TargetLoader
from nodestep_sandbox.session import Sandbox
from nodestep_sandbox.web import SandboxSite


class ServerOptions(BaseModel):
    """Options of ``nodestep sandbox``.

    Attributes
    ----------
    target : str
        ``package.module:attribute`` or ``path/to/file.py:attribute`` of the graph.
    context : str, optional
        The same form for the run context.
    host : str
    port : int
    open_browser : bool
    """

    target: str
    context: str | None = None
    host: str = "127.0.0.1"
    port: int = 8765
    open_browser: bool = True


def _ip_address(address: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    value = ipaddress.ip_address(address)
    if isinstance(value, ipaddress.IPv6Address) and value.ipv4_mapped is not None:
        return value.ipv4_mapped
    return value


def _wildcard(host: str) -> SandboxError:
    return SandboxError(
        f"--host {host!r} is a wildcard address, which listens on every network "
        "interface; bind one address, such as 127.0.0.1"
    )


class BindAddress(BaseModel):
    """The address ``--host`` names, resolved the way the server binds it.

    Attributes
    ----------
    host : str
        An IP address in its standard form (``127.1`` becomes ``127.0.0.1``,
        as browsers write it), or the name as given.
    addresses : list[str]
        The IP addresses the server listens on.
    """

    host: str
    addresses: list[str]

    @classmethod
    async def resolve(cls, host: str, port: int) -> Self:
        """Resolve ``host`` and refuse addresses that listen everywhere.

        Parameters
        ----------
        host : str
        port : int

        Returns
        -------
        BindAddress

        Raises
        ------
        SandboxError
            If ``host`` does not resolve, or resolves to an unspecified address
            however it is written (``0.0.0.0``, ``0``, ``0x0``, ``::``,
            ``::ffff:0.0.0.0`` and so on).
        """
        if not host:
            raise _wildcard(host)
        loop = asyncio.get_running_loop()
        numeric = True
        try:
            infos = await loop.getaddrinfo(
                host,
                port,
                type=socket.SOCK_STREAM,
                flags=socket.AI_PASSIVE | socket.AI_NUMERICHOST,
            )
        except socket.gaierror:
            numeric = False
            try:
                infos = await loop.getaddrinfo(
                    host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
                )
            except socket.gaierror as error:
                raise SandboxError(
                    f"--host {host!r} cannot be resolved: {error.strerror}"
                ) from error
        addresses = list(dict.fromkeys(str(info[4][0]) for info in infos))
        if any(_ip_address(address).is_unspecified for address in addresses):
            raise _wildcard(host)
        return cls(host=addresses[0] if numeric else host, addresses=addresses)

    @property
    def loopback(self) -> bool:
        """Whether every address is a loopback address."""
        return all(_ip_address(address).is_loopback for address in self.addresses)


class _Server(uvicorn.Server):
    def __init__(
        self,
        config: uvicorn.Config,
        *,
        sandbox: Sandbox,
        on_started: Callable[[], Awaitable[None]],
        warn: Callable[[str], None],
    ) -> None:
        super().__init__(config)
        self.sandbox = sandbox
        self.on_started = on_started
        self.warn = warn
        self.stop_signal: int = signal.SIGINT

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        if not self.should_exit:
            self.stop_signal = sig
        super().handle_exit(sig, frame)

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        if self.started:
            await self.on_started()

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        stopped = await self.sandbox.stop()
        await super().shutdown(sockets=sockets)
        if stopped:
            self.warn(
                "nodestep sandbox: exiting without waiting for unfinished runs: "
                + ", ".join(run.id for run in stopped)
            )
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(128 + self.stop_signal)


class SandboxServer:
    """Loads the graph and serves the sandbox pages until stopped.

    When a signal stops the server, runs that are still going are cancelled
    first, so their pages end. If there were any, the process then exits at
    once with code 128 + the signal number (130 for Ctrl+C), because a sync
    node can keep its worker thread busy and Python waits for worker threads
    before it exits.

    Parameters
    ----------
    options : ServerOptions
    cwd : Path
        Directory the targets are loaded from.
    announce : callable
        Receives the line that says where the sandbox listens.
    warn : callable
        Receives warnings.
    """

    def __init__(
        self,
        options: ServerOptions,
        *,
        cwd: Path,
        announce: Callable[[str], None],
        warn: Callable[[str], None],
    ) -> None:
        self.options = options
        self.cwd = cwd
        self.announce = announce
        self.warn = warn

    async def serve(self) -> None:
        """Load the graph and the context, then serve until uvicorn stops.

        Raises
        ------
        SandboxError
            If the host does not resolve or is a wildcard address, or a target
            cannot be loaded.
        """
        options = self.options
        address = await BindAddress.resolve(options.host, options.port)
        if not address.loopback:
            self.warn(
                f"Warning: {address.host} is not a loopback address, so other "
                "machines that can reach it can open the sandbox and run the graph"
            )
        loader = TargetLoader(self.cwd)
        graph = await loader.graph(options.target)
        context = (
            None if options.context is None else await loader.context(options.context)
        )
        sandbox = Sandbox(graph, context=context)
        site = SandboxSite(sandbox, host=address.host, port=options.port)
        url = f"http://{site.address}/"
        server = _Server(
            uvicorn.Config(
                site.app, host=address.host, port=options.port, log_level="warning"
            ),
            sandbox=sandbox,
            on_started=partial(self._started, graph.name, url),
            warn=self.warn,
        )
        await server.serve()

    async def _started(self, name: str, url: str) -> None:
        self.announce(f"nodestep sandbox: graph '{name}' at {url} (Ctrl+C stops it)")
        if self.options.open_browser:
            await asyncio.to_thread(webbrowser.open, url)
