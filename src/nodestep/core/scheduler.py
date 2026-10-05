from __future__ import annotations

import asyncio
import functools
import inspect
import json as _json
import time
from collections.abc import (
    AsyncGenerator,
    Callable,
    Coroutine,
    Mapping,
    Sequence,
)
from contextlib import aclosing
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import BaseModel

from nodestep.core.agent import AgentRegistry
from nodestep.core.command import (
    END,
    Command,
    EndSentinel,
    GraphInterrupt,
    GraphInterruptGroup,
    Interrupt,
    InterruptFrame,
    Resume,
    Send,
    bind_interrupt_frame,
    delegated_answer,
    unbind_interrupt_frame,
)
from nodestep.core.node import Node, get_node_name
from nodestep.core.stream import (
    TERMINAL_MODES,
    EmitFunction,
    FinalEventData,
    InterruptEventData,
    NodeContext,
    StreamEvent,
    normalize_modes,
)
from nodestep.exceptions import (
    GraphConfigError,
    GraphExecutionError,
    GraphTimeoutError,
    InvalidUpdateError,
    NodeTimeoutError,
    ResumeError,
    RunLimitExceededError,
    StateStoreError,
    StateUpdateError,
    UnknownThreadError,
)
from nodestep.middleware import (
    GraphMiddlewareContext,
    NodeMiddlewareContext,
    run_after_hooks,
    run_before_hooks,
)
from nodestep.middleware.base import (
    Middleware,
    RunOutcome,
    _resolve,
    run_end_hooks,
    run_graph_hooks,
)
from nodestep.state import CheckpointRecord, HistoryEvent
from nodestep.state.context import isolate, to_json_value
from nodestep.state.history import History, StateStore
from nodestep.state.replay import load_history_state
from nodestep.state.schema import (
    StateSchema,
    diff_states,
    merge_parallel_updates,
    states_differ,
)
from nodestep.utils.reducers import RemoveMessage, Replace, replace

if TYPE_CHECKING:
    from nodestep.core.graph import Graph, NodeSpec


@dataclass
class Task:
    """One node execution scheduled in a superstep."""

    node: str
    input_state: dict[str, Any]
    from_send: bool = False
    task_id: str = ""
    send_payload: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.task_id:
            self.task_id = self.node


@dataclass
class TaskResult:
    """Outcome of one task: its update and routes, an error, or its interrupts."""

    task: Task
    updates: dict[str, Any] | None = None
    routes: list[Any] | None = None
    error: Exception | None = None
    interrupts: list[Interrupt] | None = None


class _InvalidReturn(Exception):
    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


async def _call_node(handler: Node, state_view: Any, context: NodeContext) -> Any:
    args: tuple[Any, ...] = (state_view,)
    kwargs: dict[str, Any] = {}
    if handler.accepts_context and handler.context_is_keyword:
        kwargs[handler.context_parameter_name] = context
    elif handler.accepts_context:
        args = (state_view, context)
    if inspect.iscoroutinefunction(handler.handler):
        value = handler(*args, **kwargs)
    else:
        value = await asyncio.to_thread(handler, *args, **kwargs)
    if inspect.isawaitable(value):
        return await value
    return value


async def _call_with_timeout(
    handler: Node, state_view: Any, context: NodeContext
) -> Any:
    scope = asyncio.timeout(handler.timeout)
    try:
        async with scope:
            return await _call_node(handler, state_view, context)
    except TimeoutError as error:
        if handler.timeout is None or not scope.expired():
            raise
        raise NodeTimeoutError(handler.name, handler.timeout) from error


def _normalize_goto(goto: Any) -> list[Any]:
    if goto is None:
        return []
    if isinstance(goto, (EndSentinel, Send, Node)):
        return [goto]
    if isinstance(goto, list):
        routes: list[Any] = []
        for item in goto:
            routes.extend(_normalize_goto(item))
        return routes
    raise _InvalidReturn(GraphExecutionError(f"Unsupported goto value: {goto!r}"))


def _update_marker(value: Any) -> str | None:
    if isinstance(value, Replace):
        return "Replace"
    if isinstance(value, RemoveMessage) or (
        isinstance(value, list)
        and any(isinstance(item, RemoveMessage) for item in value)
    ):
        return "RemoveMessage"
    return None


def _normalize_send(schema: StateSchema[Any], name: str, send: Send) -> Send:
    payload = send.payload
    if isinstance(payload, BaseModel):
        payload = {
            field_name: getattr(payload, field_name)
            for field_name in [
                *type(payload).model_fields,
                *(payload.model_extra or {}),
            ]
            if field_name in payload.model_fields_set
        }
    elif payload is None:
        payload = {}
    elif not isinstance(payload, dict):
        raise _InvalidReturn(
            GraphExecutionError(
                f"Send payload must be dict or BaseModel, got {type(payload).__name__}"
            )
        )
    if not schema.is_dynamic():
        unknown = [key for key in payload if key not in schema.fields]
        if unknown:
            raise _InvalidReturn(
                StateUpdateError(
                    f"Send payload from node '{name}' to node '{send.node.name}' has "
                    "keys that are not state fields: "
                    f"{', '.join(repr(key) for key in unknown)}"
                )
            )
    for key, value in payload.items():
        marker = _update_marker(value)
        if marker is not None:
            raise _InvalidReturn(
                StateUpdateError(
                    f"Send payload from node '{name}' to node '{send.node.name}' "
                    f"holds {marker} in field '{key}'; a Send payload sets the "
                    "worker's input without reducers, so pass plain values"
                )
            )
    try:
        payload = schema.validate_payload(
            payload,
            writer=f"the Send payload from node '{name}' to node '{send.node.name}'",
        )
    except (StateUpdateError, StateStoreError) as error:
        raise _InvalidReturn(error) from error
    return Send(send.node, payload)


def _normalize_routes(
    schema: StateSchema[Any], name: str, routes: list[Any]
) -> list[Any]:
    return [
        _normalize_send(schema, name, route) if isinstance(route, Send) else route
        for route in routes
    ]


def _check_return(name: str, value: Any) -> None:
    if isinstance(value, Command):
        if value.update is not None and not isinstance(value.update, Mapping):
            raise _InvalidReturn(
                TypeError(
                    f"Node '{name}' returned "
                    f"Command(update={type(value.update).__name__}); "
                    "Command.update must be a dict update"
                )
            )
        _normalize_goto(value.goto)
    elif value is not None and not isinstance(value, (EndSentinel, Send, Mapping)):
        raise _InvalidReturn(
            TypeError(
                f"Node '{name}' returned {type(value).__name__}; return a dict update "
                "or the state object you received"
            )
        )


def _normalize_result(
    schema: StateSchema[Any], name: str, value: Any
) -> tuple[dict[str, Any] | None, list[Any] | None]:
    _check_return(name, value)
    if isinstance(value, Command):
        routes = (
            None
            if value.goto is None
            else _normalize_routes(schema, name, _normalize_goto(value.goto))
        )
        return (None if value.update is None else dict(value.update)), routes
    if isinstance(value, EndSentinel):
        return None, [END]
    if isinstance(value, Send):
        return None, [_normalize_send(schema, name, value)]
    if isinstance(value, Mapping):
        return dict(value), None
    return None, None


def _diff_input(
    schema: StateSchema[Any], name: str, baseline: dict[str, Any], value: Any
) -> dict[str, Any]:
    try:
        return diff_states(schema, baseline, value)
    except GraphExecutionError as error:
        raise _InvalidReturn(
            GraphExecutionError(
                f"Node '{name}' changed the state it received: {error}; return an "
                "update"
            )
        ) from error


def _read_return(
    schema: StateSchema[Any],
    name: str,
    value: Any,
    received: Any,
    baseline: dict[str, Any],
) -> Any:
    if value is received:
        if isinstance(value, dict):
            deleted = [key for key in baseline if key not in value]
            if deleted:
                raise _InvalidReturn(
                    GraphExecutionError(
                        f"Node '{name}' deleted "
                        f"{', '.join(repr(key) for key in deleted)} from the state "
                        "it received; state keys cannot be deleted, assign a value "
                        "instead"
                    )
                )
        return _diff_input(schema, name, baseline, value)
    if value is None:
        if states_differ(schema, baseline, received):
            raise _InvalidReturn(
                GraphExecutionError(
                    f"Node '{name}' mutated its input but returned None; return the "
                    "state object you received or a dict update"
                )
            )
        return None
    if isinstance(value, Command) and received is not None and value.update is received:
        raise _InvalidReturn(
            TypeError(
                f"Node '{name}' returned Command(update=<the state object it "
                "received>); pass a dict update"
            )
        )
    _check_return(name, value)
    if states_differ(schema, baseline, received):
        returned = "END" if isinstance(value, EndSentinel) else type(value).__name__
        raise _InvalidReturn(
            GraphExecutionError(
                f"Node '{name}' mutated its input and returned {returned}; do one or "
                "the other: return the state object you received, or return an "
                "update without changing the input"
            )
        )
    return value


def _select_flow_targets(
    graph: Graph[Any], node_name: str, state: dict[str, Any]
) -> list[Any]:
    edge = graph.edges.get(node_name)
    if edge is not None:
        return [edge.target]
    branch_edge = graph.branch_edges.get(node_name)
    if branch_edge is None:
        raise GraphExecutionError(f"Node '{node_name}' has no outgoing edge or branch")
    flow_branch = branch_edge.branch
    key = flow_branch.router(graph.state_schema.to_declared(isolate(state)))
    if flow_branch.requires_bool and not isinstance(key, bool):
        raise GraphExecutionError(
            f"when() predicate of node '{node_name}' returned "
            f"{type(key).__name__}; it must return a bool"
        )
    for mapped_key, target in flow_branch.mapping.items():
        if mapped_key == key:
            return [target]
    raise GraphExecutionError(
        f"Branch router for node '{node_name}' returned unmapped key {key!r}"
    )


def _route_error(
    graph: Graph[Any], spec: NodeSpec, routes: list[Any] | None
) -> GraphExecutionError | None:
    name = spec.name
    if routes is None:
        if name in graph.edges or name in graph.branch_edges:
            return None
        return GraphExecutionError(
            f"Node '{name}' returned no goto and has no outgoing edge or branch"
        )
    declared = spec.goto or ()
    for route in routes:
        target = route.node if isinstance(route, Send) else route
        if any(target is item for item in declared):
            continue
        if isinstance(target, EndSentinel):
            label = "END"
        else:
            label = f"'{target.name}'"
            if any(
                isinstance(item, Node) and item.name == target.name for item in declared
            ):
                return GraphExecutionError(
                    f"Node '{name}' routed to a different node named {label} than "
                    "the one declared in its goto="
                )
        return GraphExecutionError(
            f"Node '{name}' routed to {label} through Command or Send, but {label} "
            "is not declared in its goto=; add it to @node(goto=[...])"
        )
    return None


def _routes_to_tasks(
    graph: Graph[Any], routes: list[Any], state: dict[str, Any]
) -> list[Task]:
    tasks: list[Task] = []
    node_counts: dict[str, int] = {}
    pulled: set[str] = set()

    def next_task_id(node_name: str) -> str:
        index = node_counts.get(node_name, 0)
        node_counts[node_name] = index + 1
        return node_name if index == 0 else f"{node_name}[{index}]"

    for selected in routes:
        if isinstance(selected, EndSentinel):
            continue
        if isinstance(selected, Send):
            node_name = get_node_name(selected.node)
            if node_name not in graph.nodes:
                raise GraphExecutionError(f"Flow selected unknown node '{node_name}'")
            payload = dict(selected.payload or {})
            tasks.append(
                Task(
                    node=node_name,
                    input_state={**state, **payload},
                    from_send=True,
                    task_id=next_task_id(node_name),
                    send_payload=payload,
                )
            )
            continue
        if isinstance(selected, Node):
            node_name = get_node_name(selected)
            if node_name not in graph.nodes:
                raise GraphExecutionError(f"Flow selected unknown node '{node_name}'")
            if node_name in pulled:
                continue
            pulled.add(node_name)
            tasks.append(
                Task(
                    node=node_name,
                    input_state=state,
                    task_id=next_task_id(node_name),
                )
            )
            continue
        raise GraphExecutionError(f"Unsupported flow target: {selected!r}")
    return tasks


class _History:
    def __init__(
        self, store: StateStore | None, thread_id: str, branch_id: str
    ) -> None:
        self.store = store
        self.thread_id = thread_id
        self.branch_id = branch_id
        self.last_event: HistoryEvent | None = None

    async def append(
        self,
        event_type: str,
        *,
        node: str | None = None,
        data: dict[str, Any] | None = None,
        error_type: str | None = None,
        message: str | None = None,
    ) -> HistoryEvent | None:
        if self.store is None:
            return None
        event = HistoryEvent(
            thread_id=self.thread_id,
            branch_id=self.branch_id,
            sequence=0,
            type=event_type,
            node=node,
            data_json=(
                _json.dumps(data, separators=(",", ":")) if data is not None else None
            ),
            error_type=error_type,
            message=message,
        )
        self.last_event = await self.store.append_next_event(event)
        return self.last_event

    @property
    def last_event_id(self) -> str | None:
        return None if self.last_event is None else self.last_event.id


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


@dataclass
class _TimeBudget:
    limit: float | None
    spent: float = 0.0

    def remaining(self) -> float | None:
        if self.limit is None:
            return None
        return self.limit - self.spent


async def _run_superstep(
    graph: Graph[Any],
    tasks: list[Task],
    start: Callable[[Task], Coroutine[Any, Any, TaskResult]],
    budget: _TimeBudget,
) -> list[TaskResult]:
    started = time.monotonic()
    running = [asyncio.ensure_future(start(task)) for task in tasks]
    scope = asyncio.timeout(budget.remaining())
    try:
        async with scope:
            for completed in asyncio.as_completed(running):
                result = await completed
                if result.error is not None:
                    break
    except TimeoutError as error:
        if scope.expired() and budget.limit is not None:
            raise GraphTimeoutError(graph.name, budget.limit) from error
        raise
    finally:
        unfinished = [task for task in running if not task.done()]
        for task in unfinished:
            task.cancel()
        if unfinished:
            await asyncio.gather(*unfinished, return_exceptions=True)
        budget.spent += time.monotonic() - started
    return [task.result() for task in running if not task.cancelled()]


async def _live_events(
    step: asyncio.Future[Any], queue: asyncio.Queue[StreamEvent]
) -> AsyncGenerator[StreamEvent]:
    getter: asyncio.Future[StreamEvent] | None = None
    try:
        while not step.done():
            getter = asyncio.ensure_future(queue.get())
            await asyncio.wait({step, getter}, return_when=asyncio.FIRST_COMPLETED)
            if getter.done():
                event = getter.result()
                getter = None
                yield event
            else:
                getter.cancel()
                await asyncio.gather(getter, return_exceptions=True)
                getter = None
        while not queue.empty():
            yield queue.get_nowait()
    finally:
        if getter is not None:
            getter.cancel()


@dataclass(frozen=True, slots=True)
class _Write:
    node: str
    task_id: str
    update: dict[str, Any] | None


async def _commit_writes(
    schema: StateSchema[Any], history: _History, writes: list[_Write]
) -> int:
    if history.store is None:
        return 0
    committed = 0
    for write in sorted(writes, key=lambda item: item.node):
        if write.update:
            await history.append(
                "state_delta",
                node=write.node,
                data={
                    "update": schema.dump_update(write.update),
                    "task_id": write.task_id,
                },
            )
            committed += 1
    return committed


async def _checkpoint(
    schema: StateSchema[Any],
    history: _History,
    state: dict[str, Any],
    active: list[Task],
    status: str,
) -> CheckpointRecord | None:
    if history.store is None or history.last_event is None:
        return None
    record = CheckpointRecord(
        thread_id=history.thread_id,
        branch_id=history.branch_id,
        event_id=history.last_event.id,
        sequence=history.last_event.sequence,
        state_json=_json.dumps(schema.dump_state(state), separators=(",", ":")),
        active_json=_json.dumps(
            [_task_spec(schema, task) for task in active],
            separators=(",", ":"),
        ),
        status=status,
    )
    return await history.store.save_checkpoint(record)


@dataclass(frozen=True, slots=True)
class _TaskSpec:
    node: str
    task_id: str
    from_send: bool = False
    send_payload: dict[str, Any] | None = None

    def build(self, state: dict[str, Any]) -> Task:
        return Task(
            node=self.node,
            input_state={**state, **(self.send_payload or {})},
            from_send=self.from_send,
            task_id=self.task_id,
            send_payload=self.send_payload,
        )


@dataclass
class _PendingTask:
    node: str
    task_id: str
    from_send: bool = False
    send_payload: dict[str, Any] | None = None
    interrupts: dict[str, Any] = field(default_factory=dict)
    answers: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Suspension:
    tasks: dict[str, _PendingTask] = field(default_factory=dict)
    routes: list[dict[str, Any]] = field(default_factory=list)
    writes: list[dict[str, Any]] = field(default_factory=list)
    root_run_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "tasks": [asdict(task) for task in self.tasks.values()],
            "routes": self.routes,
            "writes": self.writes,
            "root_run_id": self.root_run_id,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> _Suspension:
        return cls(
            tasks={task["task_id"]: _PendingTask(**task) for task in data["tasks"]},
            routes=list(data["routes"]),
            writes=list(data["writes"]),
            root_run_id=data["root_run_id"],
        )


@dataclass(frozen=True, slots=True)
class _ThreadPosition:
    status: Literal["unfinished", "paused", "finished", "failed"]
    boundary: dict[str, Any] = field(default_factory=dict)
    completed: tuple[dict[str, Any], ...] = ()
    committed: frozenset[str] = frozenset()
    reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "boundary": self.boundary,
            "completed": list(self.completed),
            "committed": sorted(self.committed),
            "reason": self.reason,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> _ThreadPosition:
        return cls(
            status=data["status"],
            boundary=dict(data["boundary"]),
            completed=tuple(data["completed"]),
            committed=frozenset(data["committed"]),
            reason=data["reason"],
        )


@dataclass
class _Unfinished:
    tasks: list[_TaskSpec]
    routes: list[dict[str, Any]]
    writes: list[dict[str, Any]]
    order: list[str]
    committed: frozenset[str]
    answers: dict[str, dict[str, Any]]
    caches: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class _Routed:
    node: str
    task_id: str
    result: TaskResult | None
    routes: list[Any]


_SNAPSHOT_EVERY = 50


def _task_spec(schema: StateSchema[Any], task: Task) -> dict[str, Any]:
    return {
        "node": task.node,
        "task_id": task.task_id,
        "from_send": task.from_send,
        "send_payload": None
        if task.send_payload is None
        else schema.dump_state(task.send_payload),
    }


def _write_specs(
    schema: StateSchema[Any], writes: list[_Write]
) -> list[dict[str, Any]]:
    return [
        {
            "node": write.node,
            "task_id": write.task_id,
            "update": schema.dump_update(write.update),
        }
        for write in writes
    ]


def _restore_writes(graph: Graph[Any], specs: list[dict[str, Any]]) -> list[_Write]:
    return [
        _Write(
            node=spec["node"],
            task_id=spec["task_id"],
            update=None
            if spec.get("update") is None
            else graph.state_schema.restore_update(spec["update"]),
        )
        for spec in specs
    ]


def _serialize_routes(
    schema: StateSchema[Any], routes: list[Any]
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for route in routes:
        if isinstance(route, EndSentinel):
            serialized.append({"kind": "end", "node": None, "payload": None})
        elif isinstance(route, Send):
            serialized.append(
                {
                    "kind": "send",
                    "node": get_node_name(route.node),
                    "payload": schema.dump_state(route.payload or {}),
                }
            )
        else:
            serialized.append(
                {"kind": "node", "node": get_node_name(route), "payload": None}
            )
    return serialized


def _carried_entry(schema: StateSchema[Any], result: TaskResult) -> dict[str, Any]:
    return {
        "node": result.task.node,
        "task_id": result.task.task_id,
        "routes": None
        if result.routes is None
        else _serialize_routes(schema, result.routes),
    }


def _restore_routes(
    graph: Graph[Any], entry: dict[str, Any], state: dict[str, Any]
) -> list[Any]:
    if entry.get("routes") is None:
        return _select_flow_targets(graph, entry["node"], state)
    restored: list[Any] = []
    for route in entry["routes"]:
        if route["kind"] == "end":
            restored.append(END)
            continue
        name = route["node"]
        if name not in graph.nodes:
            raise ResumeError(f"Pending route targets unknown node '{name}'")
        handler = graph.nodes[name].handler
        restored.append(
            Send(
                handler, graph.state_schema.restore_payload(route.get("payload") or {})
            )
            if route["kind"] == "send"
            else handler
        )
    return restored


@dataclass
class _ResumePlan:
    tasks: list[_TaskSpec]
    answers: dict[str, dict[str, Any]]
    caches: dict[str, dict[str, Any]]


def _event_data(event: HistoryEvent) -> dict[str, Any]:
    return _json.loads(event.data_json) if event.data_json else {}


def _stored(data: dict[str, Any], key: str, event_type: str) -> Any:
    value = data.get(key)
    if value is None:
        raise ResumeError(
            f"A stored '{event_type}' record has no '{key}'; the thread history "
            "is incomplete or was not written by nodestep"
        )
    return value


def _load_suspension(events: Sequence[HistoryEvent]) -> _Suspension:
    suspension = _Suspension()
    for event in events:
        data = _event_data(event)
        if event.type == "forked":
            suspension = _Suspension.from_json(data["suspension"])
        elif event.type == "run_started" and not data.get("resume"):
            suspension = _Suspension()
        elif event.type == "interrupted":
            task_id = _stored(data, "task_id", event.type)
            suspension.tasks[task_id] = _PendingTask(
                node=_stored({"node": event.node}, "node", event.type),
                task_id=task_id,
                from_send=bool(data.get("from_send")),
                send_payload=data.get("send_payload"),
                interrupts=dict(data.get("interrupts") or {}),
                answers=dict(data.get("answers") or {}),
                cache=dict(data.get("cache") or {}),
            )
            suspension.root_run_id = _stored(data, "root_run_id", event.type)
        elif event.type == "superstep_pending":
            suspension.routes = list(data.get("routes") or [])
            suspension.writes = list(data.get("writes") or [])
        elif event.type == "superstep_failed":
            suspension = _Suspension()
        elif event.type in ("node_completed", "error"):
            suspension.tasks.pop(_stored(data, "task_id", event.type), None)
    return suspension


def _thread_position(events: Sequence[HistoryEvent]) -> _ThreadPosition:
    status: Literal["unfinished", "paused", "finished", "failed"] = "finished"
    boundary: dict[str, Any] = {}
    completed: list[dict[str, Any]] = []
    committed: set[str] = set()
    reason: str | None = None
    for event in events:
        data = _event_data(event)
        if event.type == "forked":
            forked = _ThreadPosition.from_json(data["position"])
            status, boundary, reason = forked.status, forked.boundary, forked.reason
            completed = list(forked.completed)
            committed = set(forked.committed)
        elif event.type == "run_started" and not data.get("resume"):
            status, boundary, reason = "finished", {}, None
        elif event.type == "superstep_started":
            status, boundary, completed, committed = "unfinished", data, [], set()
        elif event.type == "node_completed":
            completed.append({**data, "node": event.node})
        elif event.type == "state_delta" and "task_id" in data:
            committed.add(data["task_id"])
        elif event.type == "superstep_pending":
            status = "paused"
        elif event.type == "superstep_failed" and data.get("merge"):
            status, reason = "failed", f"{event.error_type}: {event.message}"
        elif event.type == "superstep_completed" and data.get("next"):
            status, boundary = "unfinished", data["next"]
            completed, committed = [], set()
        elif event.type in ("superstep_completed", "run_completed"):
            status, boundary = "finished", {}
    return _ThreadPosition(
        status, boundary, tuple(completed), frozenset(committed), reason
    )


def fork_position(history: History, event_id: str) -> dict[str, Any]:
    """Describe where the runs of a thread branch stood at one of its events.

    ``Graph.fork`` records the result in the first event of the new branch, so
    ``ainvoke(None)`` on the branch continues a superstep that was unfinished
    at the event and ``resume=`` answers interrupts that were pending there.

    Parameters
    ----------
    history : History
        History of the branch being forked.
    event_id : str
        Event of ``history`` to fork at.

    Returns
    -------
    dict
        JSON data with the keys ``"position"`` and ``"suspension"``.

    Raises
    ------
    StateUpdateError
        If ``event_id`` is not an event of ``history``.
    """
    index = next(
        (i for i, event in enumerate(history.events) if event.id == event_id), None
    )
    if index is None:
        raise StateUpdateError(
            f"Event id '{event_id}' not found in thread '{history.thread_id}'"
        )
    events = history.events[: index + 1]
    return {
        "position": _thread_position(events).to_json(),
        "suspension": _load_suspension(events).to_json(),
    }


def _restored_payload(
    graph: Graph[Any], payload: dict[str, Any] | None
) -> dict[str, Any] | None:
    return None if payload is None else graph.state_schema.restore_payload(payload)


def _unfinished_superstep(graph: Graph[Any], position: _ThreadPosition) -> _Unfinished:
    boundary = position.boundary
    done = {entry["task_id"] for entry in position.completed}
    tasks: list[_TaskSpec] = []
    for spec in boundary.get("tasks") or []:
        if spec.get("node") not in graph.nodes:
            raise ResumeError(
                f"Unfinished node '{spec.get('node')}' is not part of the graph"
            )
        task_id = _stored(spec, "task_id", "superstep_started")
        if task_id in done:
            continue
        tasks.append(
            _TaskSpec(
                node=spec["node"],
                task_id=task_id,
                from_send=bool(spec.get("from_send")),
                send_payload=_restored_payload(graph, spec.get("send_payload")),
            )
        )
    return _Unfinished(
        tasks=tasks,
        routes=[
            *(boundary.get("routes") or []),
            *(
                {
                    "node": entry["node"],
                    "task_id": entry["task_id"],
                    "routes": entry.get("routes"),
                }
                for entry in position.completed
            ),
        ],
        writes=[
            *(boundary.get("writes") or []),
            *(
                {
                    "node": entry["node"],
                    "task_id": entry["task_id"],
                    "update": entry.get("update"),
                }
                for entry in position.completed
            ),
        ],
        order=list(boundary["order"]),
        committed=position.committed,
        answers=dict(boundary.get("answers") or {}),
        caches=dict(boundary.get("caches") or {}),
    )


def _replaces(schema: StateSchema[Any], key: str, value: Any) -> bool:
    descriptor = schema.fields.get(key)
    return (
        isinstance(value, Replace)
        or descriptor is None
        or descriptor.reducer is replace
    )


def _manual_conflict(
    schema: StateSchema[Any],
    update: dict[str, Any],
    writes: list[_Write],
    superstep: Literal["paused", "unfinished"],
) -> InvalidUpdateError | None:
    for key, value in update.items():
        writers = [
            (write.node, write.update[key])
            for write in writes
            if write.update is not None and key in write.update
        ]
        if writers and (
            _replaces(schema, key, value)
            or any(_replaces(schema, key, written) for _, written in writers)
        ):
            descriptor = schema.fields.get(key)
            return InvalidUpdateError(
                field=key,
                nodes=sorted({node for node, _ in writers}),
                replacement=descriptor is not None
                and descriptor.reducer is not replace,
                superstep=superstep,
            )
    return None


def check_manual_update(
    graph: Graph[Any], history: History, update: dict[str, Any]
) -> None:
    """Refuse a manual update that a pending write of its thread would overwrite.

    While a superstep of the thread is paused or unfinished, the writes of its
    finished tasks are applied only when it is resumed or continued. The
    update is checked against them under the rule for parallel writes: a
    field written by both must merge through its reducer, without
    ``Replace``. Tasks that run after the update read it, so their writes
    are merged over it without a check.

    Parameters
    ----------
    graph : Graph
    history : History
        History of the thread branch the update is for.
    update : dict
        The validated update.

    Raises
    ------
    InvalidUpdateError
        If a finished task of the paused or unfinished superstep wrote a field
        of ``update`` and the two writes cannot be merged.
    """
    position = _thread_position(history.events)
    superstep: Literal["paused", "unfinished"]
    if position.status == "paused":
        superstep = "paused"
        pending = _load_suspension(history.events).writes
        committed: frozenset[str] = frozenset()
    elif position.status == "unfinished":
        superstep = "unfinished"
        unfinished = _unfinished_superstep(graph, position)
        pending, committed = unfinished.writes, unfinished.committed
    else:
        return
    writes = [
        write
        for write in _restore_writes(graph, pending)
        if write.task_id not in committed
    ]
    conflict = _manual_conflict(graph.state_schema, update, writes, superstep)
    if conflict is not None:
        raise conflict


def _snapshot_status(history: History) -> tuple[bool, int]:
    latest = max(
        (checkpoint.sequence for checkpoint in history.checkpoints), default=None
    )
    pending = sum(
        1
        for event in history.events
        if event.type == "state_delta" and (latest is None or event.sequence > latest)
    )
    return latest is not None, pending


def _pending_keys(suspension: _Suspension) -> list[str]:
    return sorted(
        key for entry in suspension.tasks.values() for key in entry.interrupts
    )


def _resume_answers(resume: Resume, pending: list[str], thread: str) -> dict[str, Any]:
    if resume.answers is None:
        if len(pending) != 1:
            raise ResumeError(
                f"thread '{thread}' is paused on {len(pending)} interrupts {pending}; "
                "Resume(value) answers only one, pass "
                "Resume(answers={key: value, ...}) with an answer for each"
            )
        return {pending[0]: resume.value}
    unknown = [key for key in resume.answers if key not in pending]
    if unknown:
        raise ResumeError(
            f"thread '{thread}' has no pending interrupt {unknown}; it is paused on "
            f"{pending}"
        )
    missing = [key for key in pending if key not in resume.answers]
    if missing:
        raise ResumeError(
            f"Resume leaves {missing} of thread '{thread}' unanswered; every pending "
            "interrupt must be answered"
        )
    return resume.answers


def _plan_resume(
    graph: Graph[Any],
    suspension: _Suspension,
    position: _ThreadPosition,
    resume: Resume,
    thread: str,
) -> _ResumePlan:
    if position.status == "unfinished":
        raise ResumeError(
            f"thread '{thread}' has no pending interrupts; continue its unfinished "
            f"superstep with ainvoke(None, thread_id='{thread}') without resume="
        )
    if position.status != "paused" or not suspension.tasks:
        raise ResumeError(
            f"thread '{thread}' has no pending interrupts; pass input to start a "
            "new turn"
        )
    given = _resume_answers(resume, _pending_keys(suspension), thread)
    plan = _ResumePlan(tasks=[], answers={}, caches={})
    for entry in suspension.tasks.values():
        if entry.node not in graph.nodes:
            raise ResumeError(f"Pending node '{entry.node}' is not part of the graph")
        answers = dict(entry.answers)
        for key in entry.interrupts:
            answers[key] = to_json_value(given[key])
        plan.answers[entry.task_id] = answers
        plan.caches[entry.task_id] = entry.cache
        plan.tasks.append(
            _TaskSpec(
                node=entry.node,
                task_id=entry.task_id,
                from_send=entry.from_send,
                send_payload=_restored_payload(graph, entry.send_payload),
            )
        )
    return plan


async def _run_single_task(
    graph: Graph[Any],
    task: Task,
    state: dict[str, Any],
    *,
    run_id: str,
    thread: str,
    branch_id: str,
    step: int,
    history: _History,
    resume_targets: dict[str, dict[str, Any]],
    caches: dict[str, dict[str, Any]],
    root_run_id: str,
    state_store: StateStore | None,
    agent_registry: AgentRegistry,
    emit: EmitFunction,
    context: Any = None,
    superstep_size: int = 1,
    stream_modes: tuple[str, ...] = (),
) -> TaskResult:
    if "debug" in stream_modes:
        emit(
            StreamEvent(
                mode="debug",
                data={
                    "type": "node_input",
                    "node": task.node,
                    "state": isolate(task.input_state),
                },
                node=task.node,
                run_id=run_id,
                step=step,
                task_id=task.task_id,
            )
        )
    spec = graph.nodes[task.node]
    state_view = graph.state_schema.to_declared(task.input_state)
    scratch: dict[str, Any] = {}

    def make_node_context(value: Any) -> NodeMiddlewareContext:
        return NodeMiddlewareContext(
            graph_name=graph.name,
            node_name=task.node,
            state=value,
            stored_state=isolate(state_view),
            run_id=run_id,
            thread_id=thread,
            step=step,
            task_id=task.task_id,
            root_run_id=root_run_id,
            superstep_size=superstep_size,
            scratch=scratch,
            state_schema=graph.state_schema,
        )

    def make_input_context(value: Any) -> NodeMiddlewareContext:
        return make_node_context(isolate(value))

    delegated = delegated_answer()
    hook_input = state_view

    def ask_middleware(payload: Any) -> Any:
        if graph.middleware:
            hook_ctx = make_node_context(isolate(hook_input))
            for middleware in graph.middleware:
                answer = middleware.on_interrupt(hook_ctx, payload)
                if inspect.isawaitable(answer):
                    if inspect.iscoroutine(answer):
                        answer.close()
                    raise TypeError(
                        f"{type(middleware).__name__}.on_interrupt returned an awaitable; "
                        "on_interrupt must be synchronous because interrupt() "
                        "calls it while the node runs"
                    )
                if answer is not None:
                    return to_json_value(answer)
        return None if delegated is None else delegated(payload)

    frame = InterruptFrame(
        task_id=task.task_id,
        resume_values=dict(resume_targets.get(task.task_id, {})),
        cache=dict(caches.get(task.task_id, {})),
        answer=ask_middleware,
    )
    node_context = NodeContext(
        graph_name=graph.name,
        run_id=run_id,
        thread_id=thread,
        branch_id=branch_id,
        root_run_id=root_run_id,
        task_id=task.task_id,
        step=step,
        node=task.node,
        middleware=graph.middleware,
        workspace=graph.workspace,
        state_store=state_store,
        state_schema=graph.state_schema,
        stream_modes=stream_modes,
        cache=frame.cache,
        resume_values=MappingProxyType(frame.resume_values),
        _agent_registry=agent_registry,
        _emit=emit,
        _context=context,
    )
    await history.append(
        "node_started", node=task.node, data={"step": step, "task_id": task.task_id}
    )

    async def failed(error: Exception) -> TaskResult:
        await history.append(
            "error",
            node=task.node,
            data={"task_id": task.task_id},
            error_type=type(error).__name__,
            message=str(error),
        )
        return TaskResult(task=task, error=error)

    async def recover(error: Exception) -> tuple[bool, Any]:
        baseline = graph.state_schema.to_dict(state_view)
        received = isolate(state_view)
        node_ctx = make_node_context(received)
        for middleware in graph.middleware:
            result = await _resolve(middleware.on_error(node_ctx, error))
            if result is not None:
                value = _read_return(
                    graph.state_schema, task.node, result, received, baseline
                )
                return True, await run_after_hooks(
                    graph.middleware, "after_node", value, make_node_context
                )
        return False, None

    async def paused(signal: GraphInterrupt | GraphInterruptGroup) -> TaskResult:
        raised = (
            signal.interrupts if isinstance(signal, GraphInterruptGroup) else [signal]
        )
        await history.append(
            "interrupted",
            node=task.node,
            data={
                "task_id": task.task_id,
                "from_send": task.from_send,
                "send_payload": None
                if task.send_payload is None
                else graph.state_schema.dump_state(task.send_payload),
                "interrupts": {
                    graph_interrupt.key: {
                        "id": graph_interrupt.id,
                        "payload": to_json_value(graph_interrupt.payload),
                    }
                    for graph_interrupt in raised
                },
                "answers": dict(frame.resume_values),
                "cache": to_json_value(frame.cache),
                "root_run_id": root_run_id,
            },
        )
        return TaskResult(
            task=task,
            interrupts=[
                Interrupt(
                    key=graph_interrupt.key,
                    id=graph_interrupt.id,
                    node=task.node,
                    task_id=task.task_id,
                    payload=graph_interrupt.payload,
                )
                for graph_interrupt in raised
            ],
        )

    token = bind_interrupt_frame(frame)
    try:
        try:
            node_input = await run_before_hooks(
                graph.middleware,
                "before_node",
                state_view,
                make_input_context,
            )
            hook_input = node_input
            baseline = graph.state_schema.to_dict(node_input)
            received = isolate(node_input)
            raw = await _call_with_timeout(spec.handler, received, node_context)
            raw = _read_return(graph.state_schema, task.node, raw, received, baseline)
            raw = await run_after_hooks(
                graph.middleware,
                "after_node",
                raw,
                make_node_context,
            )
        except _InvalidReturn:
            raise
        except Exception as error:
            recovered, raw = await recover(error)
            if not recovered:
                return await failed(error)
        update, command_routes = _normalize_result(graph.state_schema, task.node, raw)
    except _InvalidReturn as invalid:
        return await failed(invalid.error)
    except (GraphInterrupt, GraphInterruptGroup) as signal:
        return await paused(signal)
    finally:
        unbind_interrupt_frame(token)

    route_error = _route_error(graph, spec, command_routes)
    if route_error is not None:
        return await failed(route_error)
    try:
        update = graph.state_schema.validate_update(
            state, update, writer=f"node '{task.node}'"
        )
    except (StateUpdateError, StateStoreError) as error:
        return await failed(error)
    if history.store is not None:
        await history.append(
            "node_completed",
            node=task.node,
            data={
                "update": graph.state_schema.dump_update(update),
                "routes": None
                if command_routes is None
                else _serialize_routes(graph.state_schema, command_routes),
                "task_id": task.task_id,
            },
        )
    return TaskResult(task=task, updates=update, routes=command_routes)


def _superstep_events(
    schema: StateSchema[Any],
    modes: list[str],
    *,
    run_id: str,
    step: int,
    checkpoint_id: str | None,
    writes: list[_Write],
    routed: list[_Routed],
    state: dict[str, Any],
) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    if "debug" in modes:
        for item in routed:
            if item.result is not None:
                events.append(
                    StreamEvent(
                        mode="debug",
                        data={
                            "type": "node_output",
                            "node": item.node,
                            "updates": isolate(item.result.updates),
                        },
                        node=item.node,
                        run_id=run_id,
                        step=step,
                        task_id=item.task_id,
                    )
                )
            events.append(
                StreamEvent(
                    mode="debug",
                    data={
                        "type": "routes",
                        "targets": _serialize_routes(schema, item.routes),
                    },
                    node=item.node,
                    run_id=run_id,
                    step=step,
                    task_id=item.task_id,
                )
            )
        events.append(
            StreamEvent(
                mode="debug",
                data={"type": "state_merge", "state": isolate(state)},
                run_id=run_id,
                step=step,
            )
        )
    if "updates" in modes:
        events.extend(
            StreamEvent(
                mode="updates",
                data=isolate(write.update),
                node=write.node,
                run_id=run_id,
                checkpoint_id=checkpoint_id,
                step=step,
                task_id=write.task_id,
            )
            for write in writes
        )
    if "values" in modes:
        events.append(
            StreamEvent(
                mode="values",
                data=isolate(state),
                run_id=run_id,
                checkpoint_id=checkpoint_id,
                step=step,
            )
        )
    return events


@dataclass(slots=True)
class _RunRecord:
    observers: list[Middleware] = field(default_factory=list)
    state: Any = None
    make_context: Callable[[Any], Any] | None = None
    outcome: RunOutcome | None = None

    def ended(self, event: StreamEvent) -> None:
        if isinstance(event.data, InterruptEventData):
            self.outcome = RunOutcome(
                status="paused",
                interrupts=MappingProxyType(dict(event.data.interrupts)),
                error=None,
            )
        else:
            self.outcome = RunOutcome(
                status="completed", interrupts=_NO_INTERRUPTS, error=None
            )

    async def end(self, error: BaseException | None) -> None:
        if self.make_context is None or not self.observers:
            return
        outcome = self.outcome
        if outcome is None:
            cancelled = error is None or isinstance(
                error, (asyncio.CancelledError, GeneratorExit)
            )
            outcome = RunOutcome(
                status="cancelled" if cancelled else "failed",
                interrupts=_NO_INTERRUPTS,
                error=None if cancelled else error,
            )
        await run_end_hooks(self.observers, self.state, self.make_context, outcome)


_NO_INTERRUPTS: Mapping[str, Interrupt] = MappingProxyType({})


async def execute(
    graph: Graph[Any],
    *,
    initial_state: Any = None,
    thread_id: str | None = None,
    branch_id: str = "main",
    stream_mode: str | Sequence[str],
    resume: Resume | None = None,
    state_store: StateStore | None = None,
    context: Any = None,
) -> AsyncGenerator[StreamEvent]:
    """Run a graph and yield its stream events.

    This is the engine behind ``Graph.astream``; the parameters match it.

    Returns
    -------
    AsyncGenerator[StreamEvent]

    Raises
    ------
    TypeError
        If ``resume`` is not a ``Resume``, or is given together with
        ``initial_state``.
    GraphConfigError
        If a store is configured and ``thread_id`` is missing, or there is no
        store for a resume, a continuation (``initial_state=None``) or a
        ``branch_id`` other than ``"main"``.
    UnknownThreadError
        If ``initial_state`` is ``None`` and the thread or branch does not exist,
        or a store is configured and ``branch_id`` names a branch other than
        ``"main"`` that does not exist.
    ResumeError
        If ``initial_state`` is given and the thread is paused; if
        ``initial_state`` and ``resume`` are ``None`` and the thread is paused,
        has no unfinished superstep or failed to merge the writes of its last
        superstep; or if ``resume`` does not fit the thread.
    InvalidUpdateError
        If writers of one superstep conflict, including an ``update_state`` made
        while the superstep was paused and a task that is resumed.
    GraphExecutionError
        If the run finishes or pauses with sub-agents started by ``ctx.spawn``
        that were never gathered. Sub-agents still registered when the run ends
        in any way are cancelled, before the ``on_run_end`` hooks run.
    """
    agent_registry = AgentRegistry()
    record = _RunRecord()
    events = _execute(
        graph,
        initial_state=initial_state,
        thread_id=thread_id,
        branch_id=branch_id,
        stream_mode=stream_mode,
        resume=resume,
        state_store=state_store,
        context=context,
        agent_registry=agent_registry,
        record=record,
    )
    stopped_by: BaseException | None = None
    try:
        async with aclosing(events):
            async for event in events:
                if event.mode in TERMINAL_MODES:
                    await _refuse_ungathered(graph, agent_registry)
                    record.ended(event)
                yield event
    except BaseException as error:
        stopped_by = error
        raise
    finally:
        try:
            await agent_registry.cancel_all()
        finally:
            await record.end(stopped_by)


async def _refuse_ungathered(graph: Graph[Any], registry: AgentRegistry) -> None:
    handles = await registry.cancel_all()
    if not handles:
        return
    names = ", ".join(f"'{handle.name}' ({handle.id})" for handle in handles)
    raise GraphExecutionError(
        f"Graph '{graph.name}' ended with sub-agents it spawned and never "
        f"gathered: {names}; they were cancelled. Gather them with ctx.gather "
        "before the run ends or pauses, or start them with ctx.submit on an "
        "AgentExecutor to keep them running across runs"
    )


async def _execute(
    graph: Graph[Any],
    *,
    initial_state: Any,
    thread_id: str | None,
    branch_id: str,
    stream_mode: str | Sequence[str],
    resume: Resume | None,
    state_store: StateStore | None,
    context: Any,
    agent_registry: AgentRegistry,
    record: _RunRecord,
) -> AsyncGenerator[StreamEvent]:
    modes = normalize_modes(stream_mode)
    if resume is not None and not isinstance(resume, Resume):
        raise TypeError(
            f"resume= takes a Resume, got {type(resume).__name__}; pass "
            "Resume(value) or Resume(answers={key: value, ...})"
        )
    if resume is not None and initial_state is not None:
        raise TypeError("resume= answers pending interrupts; pass initial_state=None")
    store = state_store if state_store is not None else graph.state_store
    if store is not None and not thread_id:
        raise GraphConfigError("a state store is configured; pass thread_id=")
    if store is None and branch_id != "main":
        raise GraphConfigError(
            f"branch_id={branch_id!r} needs a state store: branches are made by "
            "fork in the store of a thread"
        )
    if store is None and resume is not None:
        raise GraphConfigError(
            f"Graph '{graph.name}' cannot resume without a state_store"
        )
    if store is None and initial_state is None:
        raise GraphConfigError(
            f"Graph '{graph.name}' cannot continue a thread without a state_store; "
            "pass input to start a run"
        )
    loop = asyncio.get_running_loop()
    live_events: asyncio.Queue[StreamEvent] = asyncio.Queue()

    def emit(event: StreamEvent) -> None:
        if event.mode not in modes:
            return
        if _running_loop() is loop:
            live_events.put_nowait(event)
        else:
            loop.call_soon_threadsafe(live_events.put_nowait, event)

    run_id = str(uuid4())
    thread = thread_id or str(uuid4())
    history = _History(store, thread, branch_id)
    recorded = await store.get_history(thread, branch_id) if store is not None else None
    thread_exists = recorded is not None and bool(
        recorded.events or recorded.checkpoints
    )
    if not thread_exists and (
        initial_state is None or (store is not None and branch_id != "main")
    ):
        raise UnknownThreadError(thread, branch_id=branch_id)
    if (
        initial_state is not None
        and recorded is not None
        and _thread_position(recorded.events).status == "paused"
    ):
        keys = _pending_keys(_load_suspension(recorded.events))
        raise ResumeError(
            f"thread '{thread}' is paused on {keys}; pass resume=Resume(...), "
            "or fork to start over"
        )

    base_state = (
        await load_history_state(store, graph.state_schema, thread, branch_id=branch_id)
        if store is not None and thread_exists
        else None
    )
    schema = graph.state_schema
    state, incoming_state = schema.merge_input(base_state, initial_state)

    root_run_id = run_id
    resume_targets: dict[str, dict[str, Any]] = {}
    caches: dict[str, dict[str, Any]] = {}
    restored: list[_TaskSpec] | None = None
    carried_routes: list[dict[str, Any]] = []
    carried_writes: list[_Write] = []
    order: list[str] = []
    committed: frozenset[str] = frozenset()
    if resume is not None and recorded is not None:
        position = _thread_position(recorded.events)
        suspension = _load_suspension(recorded.events)
        plan = _plan_resume(graph, suspension, position, resume, thread)
        restored = plan.tasks
        resume_targets = plan.answers
        caches = plan.caches
        carried_routes = suspension.routes
        carried_writes = _restore_writes(graph, suspension.writes)
        order = list(position.boundary["order"])
        root_run_id = _stored(
            {"root_run_id": suspension.root_run_id}, "root_run_id", "interrupted"
        )
    elif initial_state is None and recorded is not None:
        position = _thread_position(recorded.events)
        if position.status == "paused":
            keys = _pending_keys(_load_suspension(recorded.events))
            raise ResumeError(
                f"thread '{thread}' is paused on {keys}; pass resume=Resume(...)"
            )
        if position.status == "finished":
            raise ResumeError(
                f"thread '{thread}' has no unfinished run; pass input to start a "
                "new turn"
            )
        if position.status == "failed":
            raise ResumeError(
                f"thread '{thread}' cannot continue: the writes of its last "
                f"superstep cannot be merged ({position.reason}); pass input to "
                "start a new turn, or fork the thread at an earlier event"
            )
        unfinished = _unfinished_superstep(graph, position)
        restored = unfinished.tasks
        carried_routes = unfinished.routes
        carried_writes = _restore_writes(graph, unfinished.writes)
        order = unfinished.order
        committed = unfinished.committed
        resume_targets = unfinished.answers
        caches = unfinished.caches
    has_snapshot, deltas_since_snapshot = (
        _snapshot_status(recorded)
        if recorded is not None and thread_exists
        else (False, 0)
    )

    async def make_graph_context(value: Any) -> GraphMiddlewareContext:
        return GraphMiddlewareContext(
            graph_name=graph.name,
            state=value,
            run_id=run_id,
            thread_id=thread,
            branch_id=branch_id,
            root_run_id=root_run_id,
            resuming=resume is not None,
        )

    record.state = state
    record.make_context = make_graph_context
    for item in graph.middleware:
        record.observers.append(item)
        await run_graph_hooks((item,), "before_graph", state, make_graph_context)

    await history.append(
        "run_started",
        data={
            "run_id": run_id,
            "root_run_id": root_run_id,
            "resume": resume is not None,
        },
    )
    if incoming_state:
        await history.append(
            "state_delta", data={"update": schema.dump_update(incoming_state)}
        )
        deltas_since_snapshot += 1
    if base_state is None and store is not None:
        await _checkpoint(schema, history, state, [], "running")
        deltas_since_snapshot = 0

    if restored is not None:
        active = [spec.build(state) for spec in restored]
    else:
        if graph.start_node is None:
            raise GraphExecutionError(f"Graph '{graph.name}' has no flow")
        active = [Task(node=graph.start_node, input_state=state)]
        order = [task.task_id for task in active]
    step = 0
    budget = _TimeBudget(graph.timeout)
    await history.append(
        "superstep_started",
        data={
            "step": step,
            "tasks": [_task_spec(schema, task) for task in active],
            "routes": carried_routes,
            "writes": _write_specs(schema, carried_writes),
            "order": order,
            "answers": resume_targets,
            "caches": caches,
        },
    )

    while True:
        remaining = budget.remaining()
        if remaining is not None and remaining <= 0 and budget.limit is not None:
            raise GraphTimeoutError(graph.name, budget.limit)
        if step >= graph.max_steps:
            raise RunLimitExceededError("max_steps", graph.max_steps)

        start_task = functools.partial(
            _run_single_task,
            graph,
            state=state,
            run_id=run_id,
            thread=thread,
            branch_id=branch_id,
            step=step,
            history=history,
            resume_targets=resume_targets,
            caches=caches,
            root_run_id=root_run_id,
            state_store=store,
            agent_registry=agent_registry,
            emit=emit,
            context=context,
            superstep_size=len(active),
            stream_modes=tuple(modes),
        )

        step_task = asyncio.ensure_future(
            _run_superstep(graph, active, start_task, budget)
        )
        try:
            async with aclosing(_live_events(step_task, live_events)) as live:
                async for event in live:
                    yield event
        finally:
            if not step_task.done():
                step_task.cancel()
                await asyncio.gather(step_task, return_exceptions=True)
        results: list[TaskResult] = step_task.result()

        failures = [result for result in results if result.error is not None]
        if failures:
            first_error = failures[0].error
            assert first_error is not None
            await history.append("superstep_failed", data={"step": step})
            raise first_error

        pending = {
            item.key: item
            for result in results
            if result.interrupts
            for item in result.interrupts
        }
        if pending:
            done_before_interrupt = [
                result for result in results if result.interrupts is None
            ]
            pending_writes = [
                *carried_writes,
                *(
                    _Write(result.task.node, result.task.task_id, result.updates)
                    for result in done_before_interrupt
                ),
            ]
            if history.store is not None:
                await history.append(
                    "superstep_pending",
                    data={
                        "routes": [
                            *carried_routes,
                            *(
                                _carried_entry(schema, result)
                                for result in done_before_interrupt
                            ),
                        ],
                        "writes": _write_specs(schema, pending_writes),
                    },
                )
            yield StreamEvent(
                mode="interrupt",
                data=InterruptEventData(
                    state=state,
                    thread_id=thread,
                    interrupts=pending,
                ),
                run_id=run_id,
                checkpoint_id=history.last_event_id,
                step=step,
            )
            return

        rank = {task_id: index for index, task_id in enumerate(order)}
        writes = sorted(
            [
                *carried_writes,
                *(
                    _Write(result.task.node, result.task.task_id, result.updates)
                    for result in results
                ),
            ],
            key=lambda write: rank[write.task_id],
        )
        uncommitted = [write for write in writes if write.task_id not in committed]
        try:
            merged_state, _ = merge_parallel_updates(
                graph.state_schema,
                state,
                [(write.node, write.update) for write in uncommitted],
            )
        except StateUpdateError as error:
            await history.append(
                "superstep_failed",
                data={"step": step, "merge": True},
                error_type=type(error).__name__,
                message=str(error),
            )
            raise
        routed = sorted(
            [
                *(
                    _Routed(
                        result.task.node,
                        result.task.task_id,
                        result,
                        result.routes
                        if result.routes is not None
                        else _select_flow_targets(
                            graph, result.task.node, merged_state
                        ),
                    )
                    for result in results
                ),
                *(
                    _Routed(
                        entry["node"],
                        entry["task_id"],
                        None,
                        _restore_routes(graph, entry, merged_state),
                    )
                    for entry in carried_routes
                ),
            ],
            key=lambda item: rank[item.task_id],
        )
        next_active = _routes_to_tasks(
            graph, [route for item in routed for route in item.routes], merged_state
        )
        deltas_since_snapshot += await _commit_writes(schema, history, uncommitted)
        order = [task.task_id for task in next_active]
        next_superstep = (
            {
                "step": step + 1,
                "tasks": [_task_spec(schema, task) for task in next_active],
                "order": order,
            }
            if next_active
            else None
        )
        await history.append(
            "superstep_completed", data={"step": step, "next": next_superstep}
        )
        checkpoint_id = history.last_event_id
        state = merged_state
        record.state = state
        if next_superstep is not None:
            await history.append(
                "superstep_started",
                data={**next_superstep, "routes": [], "writes": []},
            )
        if store is not None and deltas_since_snapshot >= _SNAPSHOT_EVERY:
            await _checkpoint(
                schema,
                history,
                state,
                next_active,
                "running" if next_active else "completed",
            )
            has_snapshot = True
            deltas_since_snapshot = 0
        for event in _superstep_events(
            schema,
            modes,
            run_id=run_id,
            step=step,
            checkpoint_id=checkpoint_id,
            writes=writes,
            routed=routed,
            state=state,
        ):
            yield event
        carried_routes = []
        carried_writes = []
        committed = frozenset()
        if not next_active:
            break
        active = next_active
        resume_targets = {}
        caches = {}
        step += 1

    if store is not None and not has_snapshot:
        await _checkpoint(schema, history, state, [], "completed")

    await run_graph_hooks(graph.middleware, "after_graph", state, make_graph_context)

    await history.append("run_completed", data={"run_id": run_id})
    yield StreamEvent(
        mode="final",
        data=FinalEventData(state=state, thread_id=thread),
        run_id=run_id,
        checkpoint_id=history.last_event_id,
        step=step,
    )
