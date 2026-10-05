from __future__ import annotations

import json
from collections.abc import Callable, Collection, Mapping
from typing import Any, Literal
from uuid import uuid4

from nodestep.core.command import (
    GraphInterrupt,
    GraphInterruptGroup,
    Interrupt,
    Resume,
    delegate_interrupts,
)
from nodestep.core.graph import Graph
from nodestep.core.node import Node, node
from nodestep.core.stream import NodeContext
from nodestep.exceptions import GraphConfigError, GraphExecutionError
from nodestep.state.schema import StateSchema, diff_states
from nodestep.utils.reducers import Replace


def _suspend(prefix: str, interrupts: Mapping[str, Interrupt]) -> GraphInterruptGroup:
    return GraphInterruptGroup(
        [
            GraphInterrupt(item.payload, id=item.id, key=f"{prefix}{item.key}")
            for item in interrupts.values()
        ]
    )


def _check_arguments(
    graph: Graph,
    state_in: Callable[[Any], Any] | None,
    state_out: Callable[[dict[str, Any]], dict[str, Any]] | None,
    share: Collection[str] | None,
    child_thread: str,
) -> tuple[str, ...] | None:
    mapped = state_in is not None and state_out is not None and share is None
    shared = share is not None and (state_in is None and state_out is None)
    if isinstance(share, str) or not (mapped or shared) or (shared and not share):
        raise TypeError(
            "subgraph() needs state_in= and state_out=, or share= with the field "
            "names that flow both ways, e.g. share=['messages']"
        )
    if child_thread not in ("fresh", "stable"):
        raise ValueError(
            f"child_thread must be 'fresh' or 'stable', got {child_thread!r}"
        )
    if graph.state_store is not None:
        raise GraphConfigError(
            f"child graph '{graph.name}' has its own state store; a subgraph runs "
            "on the parent's store, so build the child graph without one"
        )
    if share is None:
        return None
    fields = tuple(share)
    schema = graph.state_schema
    unknown = [
        field
        for field in fields
        if not schema.is_dynamic() and field not in schema.fields
    ]
    if unknown:
        raise GraphConfigError(
            f"share= names fields child graph '{graph.name}' does not declare: "
            + ", ".join(repr(field) for field in unknown)
        )
    return fields


def subgraph(
    name: str,
    graph: Graph,
    *,
    state_in: Callable[[Any], Any] | None = None,
    state_out: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    share: Collection[str] | None = None,
    child_thread: Literal["fresh", "stable"],
) -> Node:
    """Wrap a graph as a node of another graph.

    The child runs on the parent's state store, with the parent's ``context``,
    on the ``"main"`` branch of its thread: a new thread
    ``"<parent thread>:<name>:<random hex>"`` per call with ``"fresh"``, or
    ``"<parent thread>:<name>"``, which keeps the child's state, with
    ``"stable"``. An unanswered interrupt in the child pauses the parent under
    ``"<task_id>/<child key>"``. See
    [Sub-agents](../../concepts/sub-agents.md).

    Parameters
    ----------
    name : str
        Node name.
    graph : Graph
        Child graph, without a state store of its own.
    state_in : callable, optional
        Builds the child input from the parent state.
    state_out : callable, optional
        Builds the parent update, a dict, from the child's final state dict.
    share : Collection[str], optional
        Fields both graphs declare; instead of ``state_in`` and ``state_out``.
    child_thread : {"fresh", "stable"}
        Thread policy of the child.

    Returns
    -------
    Node

    Raises
    ------
    TypeError
        Unless exactly one of these is given: ``state_in`` with ``state_out``,
        or a non-empty ``share``; when the node runs, if ``state_out`` returns
        no dict.
    ValueError
        If ``child_thread`` is not ``"fresh"`` or ``"stable"``.
    GraphConfigError
        If the child graph has a state store or does not declare a ``share``
        field; when the node runs, if the parent does not declare a ``share``
        field or ``"stable"`` runs without a state store.
    GraphExecutionError
        When the node runs: a shared field with a custom reducer changed, or a
        ``"stable"`` child starts on a forked parent branch or is already
        running.
    """
    fields = _check_arguments(graph, state_in, state_out, share, child_thread)
    schema = graph.state_schema
    running: set[str] = set()

    def check_parent(ctx: NodeContext) -> None:
        parent: StateSchema[Any] | None = ctx.state_schema
        if fields is None or parent is None or parent.is_dynamic():
            return
        unknown = [field for field in fields if field not in parent.fields]
        if unknown:
            raise GraphConfigError(
                f"share= of subgraph node '{name}' names fields parent graph "
                f"'{ctx.graph_name}' does not declare: "
                + ", ".join(repr(field) for field in unknown)
            )

    async def child_thread_id(ctx: NodeContext) -> tuple[str, dict[str, Any] | None]:
        if child_thread == "fresh":
            return f"{ctx.thread_id}:{name}:{uuid4().hex}", None
        if ctx.state_store is None:
            raise GraphConfigError(
                f"subgraph node '{name}' has child_thread='stable', which needs a "
                "state store"
            )
        if ctx.branch_id != "main":
            raise GraphExecutionError(
                f"subgraph node '{name}' has child_thread='stable', whose child "
                f"thread every branch of the parent shares; it cannot start on "
                f"branch '{ctx.branch_id}'"
            )
        thread = f"{ctx.thread_id}:{name}"
        if not await graph.exists(thread, state_store=ctx.state_store):
            return thread, None
        snapshot = await graph.get_state(thread, state_store=ctx.state_store)
        return thread, schema.to_dict(snapshot.value)

    def parent_update(result_data: dict[str, Any], cache: dict[str, Any]) -> Any:
        if state_out is not None:
            update = state_out(result_data)
            if not isinstance(update, dict):
                raise TypeError(
                    f"state_out of subgraph node '{name}' returned "
                    f"{type(update).__name__}; it must return a dict"
                )
            return update
        assert fields is not None
        baseline = schema.restore_state(json.loads(cache["baseline"]))
        before = {field: baseline[field] for field in fields if field in baseline}
        after = {field: result_data[field] for field in fields if field in result_data}
        try:
            return diff_states(schema, before, after)
        except GraphExecutionError as error:
            raise GraphExecutionError(
                f"Subgraph node '{name}' cannot derive the parent update from child "
                f"graph '{graph.name}': {error}; pass state_in= and state_out= "
                "instead of share="
            ) from error

    @node(name=name, ref=f"subgraph:{graph.name}")
    async def run_subgraph(state: Any, ctx: NodeContext) -> dict[str, Any]:
        prefix = f"{ctx.task_id or name}/"
        cache = ctx.cache
        pending: list[str] = list(cache.get("pending") or [])
        if pending:
            if cache["owner"] != ctx.branch_id:
                fork = await graph.fork(
                    cache["thread"],
                    from_=cache["event"],
                    branch_id=cache["branch"],
                    state_store=ctx.state_store,
                )
                cache["branch"], cache["owner"] = fork.id, ctx.branch_id
            with delegate_interrupts():
                result = await graph.ainvoke(
                    None,
                    thread_id=cache["thread"],
                    branch_id=cache["branch"],
                    resume=Resume(
                        answers={
                            key: ctx.resume_values[prefix + key] for key in pending
                        }
                    ),
                    state_store=ctx.state_store,
                    context=ctx._context,
                )
        else:
            check_parent(ctx)
            thread, current = await child_thread_id(ctx)
            if thread in running:
                raise GraphExecutionError(
                    f"subgraph node '{name}' has child_thread='stable' and its "
                    f"child thread '{thread}' is already running"
                )
            if state_in is not None:
                raw_input = state_in(state)
            else:
                values = ctx.state_schema.to_dict(state)
                raw_input = {
                    field: Replace(values[field])
                    for field in fields or ()
                    if field in values
                }
            child_state, child_input = schema.merge_input(current, raw_input)
            cache["thread"] = thread
            cache["branch"], cache["owner"] = "main", ctx.branch_id
            cache["baseline"] = json.dumps(schema.dump_state(child_state))
            running.add(thread)
            try:
                with delegate_interrupts():
                    result = await graph.ainvoke(
                        child_input,
                        thread_id=thread,
                        state_store=ctx.state_store,
                        context=ctx._context,
                    )
            finally:
                running.discard(thread)
        if result.status == "interrupted":
            cache["pending"] = list(result.interrupts)
            cache["event"] = result.checkpoint_id
            raise _suspend(prefix, result.interrupts)
        cache["pending"] = []
        return parent_update(result.data, cache)

    return run_subgraph


__all__ = ["subgraph"]
