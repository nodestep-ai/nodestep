from __future__ import annotations

import asyncio
import json as _json
from collections.abc import AsyncGenerator, Iterator, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from nodestep.core.command import EndSentinel, Interrupt, Resume
from nodestep.core.stream import FinalEventData, InterruptEventData

if TYPE_CHECKING:
    from nodestep.middleware.base import Middleware
    from nodestep.state.history import StateStore
    from nodestep.workspace.base import Workspace
from nodestep.core.flow import (
    BranchSpec,
    EdgeSpec,
    FlowBranchEdge,
    FlowEdge,
    FlowStartEdge,
    GraphNodeSpec,
    GraphSpec,
)
from nodestep.core.node import Node, get_node_metadata, get_node_name
from nodestep.core.render import mermaid
from nodestep.core.stream import StreamEvent
from nodestep.exceptions import GraphConfigError, UnknownThreadError
from nodestep.state.history import (
    BranchRecord,
    CheckpointRecord,
    History,
    HistoryEvent,
)
from nodestep.state.replay import (
    fork_history_state,
    load_history_state,
    require_state_store,
)
from nodestep.state.schema import StateSchema, StateSnapshot

StateT = TypeVar("StateT")
"""The state type of a graph: the class passed as ``Graph(state_schema)``."""


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """A registered node and its name.

    Attributes
    ----------
    goto : tuple of Node or END, optional
        Declared ``goto`` targets, names resolved to the flow's nodes.
    """

    name: str
    handler: Node
    goto: tuple[Node | EndSentinel, ...] | None = None


@dataclass
class GraphResult(Generic[StateT]):
    """Outcome of ``Graph.ainvoke``.

    Attributes
    ----------
    state : StateT
        Final state as the declared schema type.
    step : int
        Superstep in which the run finished or paused, counted from 0.
    data : dict
        Final state as a plain dict.
    thread_id : str or None
        Thread the run used.
    checkpoint_id : str or None
        Id of the run's last history event.
    status : str
        ``"completed"`` or ``"interrupted"``.
    interrupts : dict[str, Interrupt]
        Pending interrupts by ``Interrupt.key``.
    """

    state: StateT
    step: int
    data: dict[str, Any] = field(default_factory=dict)
    thread_id: str | None = None
    checkpoint_id: str | None = None
    status: str = "completed"
    interrupts: dict[str, Interrupt] = field(default_factory=dict)


@dataclass
class _FlowDefinition:
    start_node: str | None = None
    nodes: dict[str, NodeSpec] = field(default_factory=dict)
    edges: dict[str, FlowEdge] = field(default_factory=dict)
    branches: dict[str, FlowBranchEdge] = field(default_factory=dict)

    @classmethod
    def build(
        cls, items: tuple[FlowStartEdge | FlowEdge | FlowBranchEdge, ...]
    ) -> _FlowDefinition:
        definition = cls()
        for item in items:
            if isinstance(item, FlowStartEdge):
                definition._add_start(item)
            elif isinstance(item, FlowEdge):
                definition._add_edge(item)
            elif isinstance(item, FlowBranchEdge):
                definition._add_branch(item)
            else:
                raise GraphConfigError(f"Unsupported flow item: {item!r}")
        if definition.start_node is None:
            raise GraphConfigError("Graph flow must include START >> node")
        definition._register_goto_targets()
        return definition

    def _add_start(self, item: FlowStartEdge) -> None:
        if self.start_node is not None:
            raise GraphConfigError("Graph flow must have exactly one START")
        self.start_node = get_node_name(item.target)
        self._register_node(item.target)

    def _add_edge(self, edge: FlowEdge) -> None:
        source_name = self._register_source(edge.source)
        self._register_target(edge.target)
        self.edges[source_name] = edge

    def _add_branch(self, edge: FlowBranchEdge) -> None:
        source_name = self._register_source(edge.source)
        for target in edge.branch.mapping.values():
            self._register_target(target)
        self.branches[source_name] = edge

    def _register_source(self, source: Node) -> str:
        source_name = get_node_name(source)
        if source_name in self.edges or source_name in self.branches:
            raise GraphConfigError(f"Node '{source_name}' already has an outgoing flow")
        self._register_node(source)
        return source_name

    def _register_target(self, target: Node | EndSentinel) -> None:
        if isinstance(target, EndSentinel):
            return
        if not isinstance(target, Node):
            raise GraphConfigError(f"Unsupported flow target: {target!r}")
        self._register_node(target)

    def _register_node(self, handler: Node) -> None:
        metadata = get_node_metadata(handler)
        existing = self.nodes.get(metadata.name)
        if existing is not None:
            if existing.handler is handler:
                return
            raise GraphConfigError(f"Node '{metadata.name}' already registered")
        self.nodes[metadata.name] = NodeSpec(name=metadata.name, handler=handler)

    def _register_goto_targets(self) -> None:
        pending = [spec.handler for spec in self.nodes.values()]
        while pending:
            handler = pending.pop(0)
            for target in handler.goto or ():
                if isinstance(target, Node) and target.name not in self.nodes:
                    self.nodes[target.name] = NodeSpec(name=target.name, handler=target)
                    pending.append(target)

    def problems(self) -> list[str]:
        problems: list[str] = []
        for spec in self.nodes.values():
            for target in spec.handler.goto or ():
                if isinstance(target, EndSentinel):
                    continue
                if isinstance(target, str):
                    if target not in self.nodes:
                        problems.append(
                            f"Node '{spec.name}' declares goto target '{target}', "
                            f"but the flow has no node named '{target}'"
                        )
                elif self.nodes[target.name].handler is not target:
                    problems.append(
                        f"Node '{spec.name}' declares goto target '{target.name}', "
                        f"which is a different node than the '{target.name}' in the flow"
                    )
        reachable = self._reachable()
        for name, spec in self.nodes.items():
            if (
                name not in self.edges
                and name not in self.branches
                and not spec.handler.goto
            ):
                problems.append(
                    f"Node '{name}' has no outgoing edge, branch or goto; "
                    f"end it with '{name} >> END'"
                )
            if name not in reachable:
                problems.append(f"Node '{name}' is not reachable from START")
        return problems

    def _reachable(self) -> set[str]:
        reachable: set[str] = set()
        queue = [self.start_node] if self.start_node is not None else []
        while queue:
            current = queue.pop()
            if current in reachable:
                continue
            reachable.add(current)
            queue.extend(self._successors(current))
        return reachable

    def _successors(self, name: str) -> list[str]:
        targets: list[Node | EndSentinel | str] = list(
            self.nodes[name].handler.goto or ()
        )
        edge = self.edges.get(name)
        if edge is not None:
            targets.append(edge.target)
        branch_edge = self.branches.get(name)
        if branch_edge is not None:
            targets.extend(branch_edge.branch.mapping.values())
        names = [
            target if isinstance(target, str) else target.name
            for target in targets
            if not isinstance(target, EndSentinel)
        ]
        return [target for target in names if target in self.nodes]

    def resolved_nodes(self) -> dict[str, NodeSpec]:
        return {
            name: NodeSpec(
                name=name,
                handler=spec.handler,
                goto=None
                if spec.handler.goto is None
                else tuple(
                    self.nodes[target].handler if isinstance(target, str) else target
                    for target in spec.handler.goto
                ),
            )
            for name, spec in self.nodes.items()
        }


class Graph(Generic[StateT]):
    """A graph of nodes that run in supersteps over a shared state.

    Parameters
    ----------
    state_schema : type
        Pydantic model, ``TypedDict`` or ``dict``, with the reducers.
    name : str, optional
        Graph name, used in errors and in node and middleware contexts.
    max_steps : int, optional
        Superstep limit per call.
    timeout : float, optional
        Seconds one call may spend inside supersteps, summed; ``None`` means no
        limit. Time spent handling events between supersteps is not counted.
    state_store : StateStore, optional
        Store that enables threads, history, interrupts and resume.
    middleware : list[Middleware], optional
        Hooks around the graph, nodes, model calls and tools.
    workspace : Workspace, optional
        Files available to tools.

    Raises
    ------
    GraphConfigError
        If ``name`` is blank or ``state_schema`` is not a valid state type (see
        ``StateSchema.from_type``).
    TypeError
        If an item of ``middleware`` does not subclass ``Middleware``.
    """

    def __init__(
        self,
        state_schema: type[StateT],
        *,
        name: str = "graph",
        max_steps: int = 25,
        timeout: float | None = None,
        state_store: StateStore | None = None,
        middleware: list[Middleware] | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        from nodestep.middleware.base import _checked_middleware

        if not name.strip():
            raise GraphConfigError("Graph name must not be blank")
        self.name = name
        self.max_steps = max_steps
        self.timeout = timeout
        self.state_store = state_store
        self.middleware = _checked_middleware(middleware or [])
        self.workspace = workspace
        self.state_schema = StateSchema.from_type(state_schema)
        self.nodes: dict[str, NodeSpec] = {}
        self.edges: dict[str, FlowEdge] = {}
        self.branch_edges: dict[str, FlowBranchEdge] = {}
        self.start_node: str | None = None

    def __repr__(self) -> str:
        return f"Graph(name={self.name!r}, nodes={len(self.nodes)}, start={self.start_node!r})"

    def flow(self, *items: FlowStartEdge | FlowEdge | FlowBranchEdge) -> Graph[StateT]:
        """Declare the flow; a graph has one.

        The flow's nodes are those in ``items`` plus the node objects in their
        ``goto=``; a ``goto=`` name refers to the flow's node of that name.
        Every node needs an edge, a branch or ``goto`` targets, and must be
        reachable from ``START``. Nothing is assigned unless the flow is valid.

        Parameters
        ----------
        *items : FlowStartEdge, FlowEdge or FlowBranchEdge
            Transitions such as ``START >> a``, ``a >> b`` and ``b >> when(...)``.

        Returns
        -------
        Graph
            The same graph.

        Raises
        ------
        GraphConfigError
            If the graph already has a flow; the flow is empty, has no or
            several ``START``, two transitions from one node or two nodes with
            one name; or, listed together in one error, a node has no way out,
            is unreachable from ``START``, or a ``goto`` target matches no node
            of the flow or a different one.
        """
        if self.start_node is not None:
            raise GraphConfigError(f"flow already defined for graph '{self.name}'")
        if not items:
            raise GraphConfigError("Graph flow must contain at least one transition")
        definition = _FlowDefinition.build(items)
        problems = definition.problems()
        if problems:
            listed = "\n".join(f"- {problem}" for problem in problems)
            raise GraphConfigError(
                f"Graph '{self.name}' has an invalid flow:\n{listed}"
            )
        self.nodes = definition.resolved_nodes()
        self.edges = definition.edges
        self.branch_edges = definition.branches
        self.start_node = definition.start_node
        return self

    async def ainvoke(
        self,
        initial_state: StateT | dict[str, Any] | None = None,
        *,
        thread_id: str | None = None,
        branch_id: str = "main",
        resume: Resume | None = None,
        state_store: StateStore | None = None,
        context: object | None = None,
    ) -> GraphResult[StateT]:
        """Run the graph until it finishes or pauses.

        Parameters
        ----------
        initial_state : StateT or dict, optional
            Input, merged into the thread's state through the reducers and
            validated like a node's update; a new thread starts from the schema
            defaults. A state model instance replaces every field. Input starts
            a new turn from ``START`` and drops the tasks of an unfinished
            superstep. ``None`` continues the thread's unfinished superstep
            without rerunning the tasks that finished; it never starts a turn.
        thread_id : str, optional
            Required with a state store; generated when omitted without one.
        branch_id : str, optional
            History branch; one other than ``"main"`` must come from ``fork``.
        resume : Resume, optional
            Answers for every pending interrupt. Requires ``initial_state=None``.
        state_store : StateStore, optional
            Store to use instead of the graph's own.
        context : object, optional
            Run-time object read as ``ctx.context``. Never stored: pass it
            again on resume.

        Returns
        -------
        GraphResult

        Raises
        ------
        TypeError
            If ``resume`` is not a ``Resume`` or comes with ``initial_state``.
        GraphConfigError
            If the graph has no flow; with a store, if ``thread_id`` is
            missing; without one, if ``initial_state`` is ``None``, ``resume``
            is given or ``branch_id`` is not ``"main"``.
        UnknownThreadError
            If ``initial_state`` is ``None`` and the thread or branch does not
            exist, or ``branch_id`` names a branch that does not exist.
        ResumeError
            If the thread is paused and gets input or no ``resume``; there is
            nothing to continue (no unfinished superstep, or one whose writes
            cannot be merged); the answers do not match the pending interrupts;
            or the stored history does not fit the graph or its schema.
        InvalidUpdateError
            If two writes to one field in a superstep, an ``update_state`` made
            while it was paused included, cannot be merged: one has no merging
            reducer or is a ``Replace``.
        StateUpdateError
            If the input or a node's update has an unknown key, an invalid
            value, a value its reducer rejects or a secret, or leaves a required
            field of a new thread empty.
        StateStoreError
            If a written value does not survive being stored as JSON, with or
            without a store, or the stored state has undeclared fields.
        GraphExecutionError
            If the run finishes or pauses with spawned sub-agents never
            gathered; they are cancelled first.
        """
        final_event: StreamEvent | None = None
        interrupt_event: StreamEvent | None = None
        async for event in self.astream(
            initial_state,
            stream_mode=[],
            thread_id=thread_id,
            branch_id=branch_id,
            resume=resume,
            state_store=state_store,
            context=context,
        ):
            if event.mode == "final":
                final_event = event
            if event.mode == "interrupt":
                interrupt_event = event
        if interrupt_event is not None:
            data = interrupt_event.data
            assert isinstance(data, InterruptEventData)
            assert interrupt_event.step is not None
            return GraphResult(
                state=self.state_schema.to_declared(data.state),
                step=interrupt_event.step,
                data=data.state,
                thread_id=data.thread_id,
                checkpoint_id=interrupt_event.checkpoint_id,
                status="interrupted",
                interrupts=data.interrupts,
            )
        if final_event is None:
            raise GraphConfigError(f"Graph '{self.name}' produced no final event")
        data = final_event.data
        assert isinstance(data, FinalEventData)
        assert final_event.step is not None
        return GraphResult(
            state=self.state_schema.to_declared(data.state),
            step=final_event.step,
            data=data.state,
            thread_id=data.thread_id,
            checkpoint_id=final_event.checkpoint_id,
            status="completed",
        )

    def invoke(
        self,
        initial_state: StateT | dict[str, Any] | None = None,
        *,
        thread_id: str | None = None,
        branch_id: str = "main",
        resume: Resume | None = None,
        state_store: StateStore | None = None,
        context: object | None = None,
    ) -> GraphResult[StateT]:
        """Run ``ainvoke`` on a new event loop.

        It takes the arguments of ``ainvoke`` and returns its ``GraphResult``.

        Raises
        ------
        RuntimeError
            If an event loop is running; use ``ainvoke`` there.
        TypeError, GraphConfigError, UnknownThreadError, ResumeError, InvalidUpdateError, StateUpdateError, StateStoreError, GraphExecutionError
            As for ``ainvoke``.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.ainvoke(
                    initial_state,
                    thread_id=thread_id,
                    branch_id=branch_id,
                    resume=resume,
                    state_store=state_store,
                    context=context,
                )
            )
        raise RuntimeError(
            "Cannot call graph.invoke() from an async context. "
            "Use 'await graph.ainvoke()' instead."
        )

    def stream(
        self,
        initial_state: StateT | dict[str, Any] | None = None,
        *,
        stream_mode: str | Sequence[str],
        thread_id: str | None = None,
        branch_id: str = "main",
        resume: Resume | None = None,
        state_store: StateStore | None = None,
        context: object | None = None,
    ) -> Iterator[StreamEvent]:
        """Run ``astream`` on a private event loop and yield its events.

        It takes the arguments of ``astream``. The graph does not run while the
        caller handles an event: time on an event emitted during a superstep
        (``"custom"``, ``"tokens"``, the ``"node_input"`` debug event) counts
        toward the graph's and the running nodes' timeouts.

        Returns
        -------
        Iterator[StreamEvent]

        Raises
        ------
        RuntimeError
            If an event loop is running; use ``astream`` there.
        TypeError, ValueError, GraphConfigError, UnknownThreadError, ResumeError, InvalidUpdateError, StateUpdateError, StateStoreError, GraphExecutionError
            As for ``astream``, while iterating.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return _iterate_on_private_loop(
                self.astream(
                    initial_state,
                    stream_mode=stream_mode,
                    thread_id=thread_id,
                    branch_id=branch_id,
                    resume=resume,
                    state_store=state_store,
                    context=context,
                )
            )
        raise RuntimeError(
            "Cannot call graph.stream() from an async context. "
            "Use 'graph.astream()' instead."
        )

    async def astream(
        self,
        initial_state: StateT | dict[str, Any] | None = None,
        *,
        stream_mode: str | Sequence[str],
        thread_id: str | None = None,
        branch_id: str = "main",
        resume: Resume | None = None,
        state_store: StateStore | None = None,
        context: object | None = None,
    ) -> AsyncGenerator[StreamEvent]:
        """Run the graph and yield events as they happen.

        The arguments other than ``stream_mode`` are those of ``ainvoke``.

        Parameters
        ----------
        stream_mode : str or Sequence[str]
            Required. Any of ``"updates"`` (one event per task after each
            superstep), ``"values"`` (the merged state after each superstep),
            ``"custom"`` (``ctx.emit``), ``"tokens"`` (model chunks) and
            ``"debug"``. The terminal ``"final"`` or ``"interrupt"`` event
            always comes, so ``[]`` yields only that one.

        Returns
        -------
        AsyncGenerator[StreamEvent]

        Raises
        ------
        TypeError
            If ``stream_mode`` is not a mode name or a sequence of them, or as
            for ``ainvoke``.
        ValueError
            If a mode is unknown, ``"final"`` or ``"interrupt"``.
        GraphConfigError, UnknownThreadError, ResumeError, InvalidUpdateError, StateUpdateError, StateStoreError
            As for ``ainvoke``.
        GraphExecutionError
            As for ``ainvoke``, instead of the terminal event.
        """
        if self.start_node is None:
            raise GraphConfigError(f"Graph '{self.name}' has no flow")

        from nodestep.core.scheduler import execute

        events = execute(
            self,
            initial_state=initial_state,
            stream_mode=stream_mode,
            thread_id=thread_id,
            branch_id=branch_id,
            resume=resume,
            state_store=state_store,
            context=context,
        )
        async with aclosing(events):
            async for event in events:
                yield event

    def to_mermaid(self) -> str:
        """Render the flow as a Mermaid diagram."""
        return mermaid(self)

    def to_spec(self) -> GraphSpec:
        """Describe the graph as data, with the import path of each node and router.

        Returns
        -------
        GraphSpec

        Raises
        ------
        GraphConfigError
            If the graph has no flow, a node or router has no import path, or
            two branch keys of a node have the same string form.
        """
        if self.start_node is None:
            raise GraphConfigError(f"Graph '{self.name}' has no flow")
        nodes = [
            GraphNodeSpec(
                name=spec.name,
                ref=spec.handler.ref,
                goto=None
                if spec.goto is None
                else [self._target_name(target) for target in spec.goto],
            )
            for spec in self.nodes.values()
        ]
        for spec in nodes:
            if spec.ref is None:
                raise GraphConfigError(f"Node '{spec.name}' is not serializable")
        edges = [
            EdgeSpec(source=source, target=self._target_name(edge.target))
            for source, edge in self.edges.items()
        ]
        branches: list[BranchSpec] = []
        for source, edge in self.branch_edges.items():
            if edge.branch.router_ref is None:
                raise GraphConfigError(
                    f"Branch router for node '{source}' is not serializable"
                )
            mapping: dict[str, str] = {}
            keys: dict[str, Any] = {}
            for key, target in edge.branch.mapping.items():
                text = str(key)
                if text in keys:
                    raise GraphConfigError(
                        f"Branch keys {keys[text]!r} and {key!r} of node '{source}' "
                        f"both serialize to {text!r}"
                    )
                keys[text] = key
                mapping[text] = self._target_name(target)
            branches.append(
                BranchSpec(
                    source=source, router_ref=edge.branch.router_ref, mapping=mapping
                )
            )
        return GraphSpec(
            name=self.name,
            state_ref=self._state_ref(),
            start=self.start_node,
            nodes=nodes,
            edges=edges,
            branches=branches,
        )

    def to_json(self) -> str:
        """Serialize ``to_spec()`` to JSON."""
        return self.to_spec().model_dump_json()

    def _target_name(self, target: Node | EndSentinel) -> str:
        if isinstance(target, EndSentinel):
            return "END"
        return get_node_name(target)

    def _state_ref(self) -> str | None:
        schema_type = self.state_schema.schema_type
        module = getattr(schema_type, "__module__", None)
        qualname = getattr(schema_type, "__qualname__", None)
        if not module or not qualname or schema_type is dict:
            return None
        return f"{module}:{qualname}"

    def _store(self, state_store: StateStore | None) -> StateStore:
        return require_state_store(
            state_store if state_store is not None else self.state_store
        )

    async def exists(
        self,
        thread_id: str,
        *,
        branch_id: str = "main",
        state_store: StateStore | None = None,
    ) -> bool:
        """Tell whether the state store holds history for a thread branch.

        Parameters
        ----------
        thread_id : str
        branch_id : str, optional
        state_store : StateStore, optional
            Store to read instead of the graph's own.

        Returns
        -------
        bool
            True once a run, ``update_state(create=True)`` or ``fork`` has
            written the branch.

        Raises
        ------
        GraphConfigError
            If neither ``state_store`` nor the graph has a state store.
        """
        history = await self._store(state_store).get_history(thread_id, branch_id)
        return bool(history.events or history.checkpoints)

    async def history(
        self,
        thread_id: str,
        *,
        branch_id: str = "main",
        state_store: StateStore | None = None,
    ) -> History:
        """Return the recorded events and checkpoints of a thread branch.

        Parameters
        ----------
        thread_id : str
        branch_id : str, optional
        state_store : StateStore, optional
            Store to read instead of the graph's own.

        Returns
        -------
        History
            Empty for a thread that has not run yet.

        Raises
        ------
        GraphConfigError
            If neither ``state_store`` nor the graph has a state store.
        UnknownThreadError
            If ``branch_id`` is not ``"main"`` and no such branch was forked.
        """
        history = await self._store(state_store).get_history(thread_id, branch_id)
        if branch_id != "main" and not history.events and not history.checkpoints:
            raise UnknownThreadError(thread_id, branch_id=branch_id)
        return history

    async def branches(
        self, thread_id: str, *, state_store: StateStore | None = None
    ) -> list[BranchRecord]:
        """Return the branches forked from a thread, oldest first.

        Parameters
        ----------
        thread_id : str
        state_store : StateStore, optional
            Store to read instead of the graph's own.

        Returns
        -------
        list[BranchRecord]
            One record per ``fork``; ``"main"`` is not listed, and a thread
            without forks, or one that has not run yet, has none.

        Raises
        ------
        GraphConfigError
            If neither ``state_store`` nor the graph has a state store.
        """
        return await self._store(state_store).list_branches(thread_id)

    async def load(
        self,
        thread_id: str,
        *,
        branch_id: str = "main",
        at: str | None = None,
        state_store: StateStore | None = None,
    ) -> dict[str, Any]:
        """Rebuild the state of a thread from its history.

        Parameters
        ----------
        thread_id : str
        branch_id : str, optional
        at : str, optional
            Event id to load the state at; defaults to the latest event.
        state_store : StateStore, optional
            Store to read instead of the graph's own.

        Returns
        -------
        dict
            The state. Fields added to the schema after the thread was stored
            have their defaults.

        Raises
        ------
        GraphConfigError
            If neither ``state_store`` nor the graph has a state store.
        UnknownThreadError
            If the thread or the branch does not exist.
        StateUpdateError
            If ``at`` is not an event of the branch.
        StateStoreError
            If the stored state has fields the state schema does not declare.
        ResumeError
            If a stored value is not valid for its field.
        """
        return await load_history_state(
            self._store(state_store),
            self.state_schema,
            thread_id,
            branch_id=branch_id,
            at=at,
        )

    async def fork(
        self,
        thread_id: str,
        *,
        from_: str,
        name: str | None = None,
        branch_id: str = "main",
        state_store: StateStore | None = None,
    ) -> BranchRecord:
        """Create a new branch of a thread at one of its events.

        The forked branch is not changed. See [Persistence and time
        travel](../../concepts/persistence.md) for how a run continues there.

        Parameters
        ----------
        thread_id : str
        from_ : str
            Id of an event of the branch being forked.
        name : str, optional
            Id of the new branch; a random id when omitted.
        branch_id : str, optional
            Branch to fork.
        state_store : StateStore, optional
            Store to use instead of the graph's own.

        Returns
        -------
        BranchRecord
            Run on the new branch with ``branch_id=record.id``.

        Raises
        ------
        ValueError
            If ``name`` is blank.
        GraphConfigError
            If neither ``state_store`` nor the graph has a state store.
        UnknownThreadError
            If the thread or ``branch_id`` does not exist.
        StateUpdateError
            If ``from_`` is not an event of that branch.
        StateStoreError
            If the thread already has a branch named ``name``, ``"main"``
            included, or its stored state has fields the schema does not
            declare.
        ResumeError
            If a stored value is not valid for its field.
        """
        from nodestep.core.scheduler import fork_position

        if name is not None and not name.strip():
            raise ValueError("A branch name must not be blank")
        store = self._store(state_store)
        history = await store.get_history(thread_id, branch_id)
        if not history.events and not history.checkpoints:
            raise UnknownThreadError(thread_id, branch_id=branch_id)
        return await fork_history_state(
            store,
            self.state_schema,
            thread_id,
            from_=from_,
            branch_id=branch_id,
            name=name,
            position=fork_position(history, from_),
        )

    async def get_state(
        self,
        thread_id: str,
        *,
        branch_id: str = "main",
        at: str | None = None,
        state_store: StateStore | None = None,
    ) -> StateSnapshot[StateT]:
        """Return the state of a thread as the declared schema type.

        Parameters
        ----------
        thread_id : str
        branch_id : str, optional
        at : str, optional
            Event id to read the state at; defaults to the latest event.
        state_store : StateStore, optional
            Store to read instead of the graph's own.

        Returns
        -------
        StateSnapshot
            The state and the ``sequence`` of the event it was read at: ``at``,
            or the latest event of the branch.

        Raises
        ------
        GraphConfigError
            If neither ``state_store`` nor the graph has a state store.
        UnknownThreadError
            If the thread or the branch does not exist.
        StateUpdateError
            If ``at`` is not an event of the branch.
        StateStoreError
            If the stored state has fields the state schema does not declare.
        ResumeError
            If a stored value is not valid for its field.
        """
        store = self._store(state_store)
        events = (await store.get_history(thread_id, branch_id)).events
        if at is None and events:
            at = max(events, key=lambda event: event.sequence).id
        value = await load_history_state(
            store, self.state_schema, thread_id, branch_id=branch_id, at=at
        )
        return StateSnapshot(
            value=self.state_schema.to_declared(value),
            sequence=next((event.sequence for event in events if event.id == at), 0),
        )

    async def update_state(
        self,
        thread_id: str,
        values: dict[str, Any],
        *,
        branch_id: str = "main",
        create: bool = False,
        state_store: StateStore | None = None,
    ) -> str:
        """Apply a manual update to a thread, for example while it is paused.

        See [Persistence and time travel](../../concepts/persistence.md) for
        updates to a paused thread.

        Parameters
        ----------
        thread_id : str
        values : dict
            Update applied through the reducers and validated like a node's.
            ``Replace(value)`` bypasses a reducer; ``RemoveMessage(id)`` removes
            a message from an ``add_messages`` field.
        branch_id : str, optional
        create : bool, optional
            Create the thread if it does not exist, from the schema defaults.
            Branches come only from ``fork``.
        state_store : StateStore, optional
            Store to write instead of the graph's own.

        Returns
        -------
        str
            Id of the recorded event.

        Raises
        ------
        ValueError
            If ``create`` is true and ``thread_id`` is empty.
        GraphConfigError
            If neither ``state_store`` nor the graph has a state store.
        UnknownThreadError
            If the thread does not exist and ``create`` is false, or the
            branch does not exist.
        InvalidUpdateError
            If a finished task of the paused or unfinished superstep wrote a
            field of ``values`` and the two writes cannot be merged.
        StateUpdateError
            If a key is not a state field, a value is invalid or its reducer
            rejects it, a ``RemoveMessage`` is not in a list update to an
            ``add_messages`` field, a value holds a secret, or a new thread
            lacks a required field.
        StateStoreError
            If a value, or a default a new thread keeps, does not survive
            being stored as JSON.
        """
        from nodestep.core.scheduler import check_manual_update

        if create and not thread_id:
            raise ValueError("thread_id must not be empty")
        store = self._store(state_store)
        history = await store.get_history(thread_id, branch_id)
        if not history.events and not history.checkpoints:
            if not create or branch_id != "main":
                raise UnknownThreadError(thread_id, branch_id=branch_id)
            state, update = self.state_schema.merge_input(
                None, values, writer="update_state"
            )
            event = await self._append_update(store, thread_id, branch_id, update)
            await store.save_checkpoint(
                CheckpointRecord(
                    thread_id=thread_id,
                    branch_id=branch_id,
                    event_id=event.id,
                    sequence=event.sequence,
                    state_json=_json.dumps(
                        self.state_schema.dump_state(state), separators=(",", ":")
                    ),
                    status="created",
                )
            )
            return event.id
        current = await load_history_state(
            store, self.state_schema, thread_id, branch_id=branch_id
        )
        update = (
            self.state_schema.validate_update(current, values, writer="update_state")
            or {}
        )
        check_manual_update(self, history, update)
        return (await self._append_update(store, thread_id, branch_id, update)).id

    async def _append_update(
        self,
        store: StateStore,
        thread_id: str,
        branch_id: str,
        update: dict[str, Any],
    ) -> HistoryEvent:
        return await store.append_next_event(
            HistoryEvent(
                thread_id=thread_id,
                branch_id=branch_id,
                sequence=0,
                type="state_delta",
                data_json=_json.dumps(
                    {"update": self.state_schema.dump_update(update)},
                    separators=(",", ":"),
                ),
            )
        )


def _iterate_on_private_loop(
    events: AsyncGenerator[StreamEvent],
) -> Iterator[StreamEvent]:
    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                yield loop.run_until_complete(anext(events))
            except StopAsyncIteration:
                return
    finally:
        try:
            loop.run_until_complete(events.aclose())
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
            loop.close()


__all__ = ["Graph", "GraphResult", "GraphSpec", "NodeSpec"]
