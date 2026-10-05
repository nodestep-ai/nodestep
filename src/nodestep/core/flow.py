from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from nodestep.core.command import EndSentinel
from nodestep.core.node import Node
from nodestep.exceptions import GraphConfigError
from nodestep.models.base import NodestepModel
from nodestep.utils.refs import callable_ref as _callable_ref


@dataclass(frozen=True, slots=True)
class FlowStartEdge:
    """Edge from ``START`` to the first node."""

    target: Node


@dataclass(frozen=True, slots=True)
class FlowEdge:
    """Static edge between two nodes, written as ``a >> b``."""

    source: Node
    target: Node | EndSentinel


@dataclass(frozen=True, slots=True)
class FlowBranch:
    """Conditional routing table produced by ``branch`` or ``when``.

    Attributes
    ----------
    router : Callable
        Called with the state; its result picks the target in ``mapping``.
    mapping : Mapping
        Router result to the next node or ``END``.
    router_ref : str or None
        ``module:qualname`` import path of the router, for ``Graph.to_spec``.
    requires_bool : bool
        The router must return a ``bool``; set by ``when``.
    """

    router: Callable[[Any], Any]
    mapping: Mapping[Any, Node | EndSentinel]
    router_ref: str | None = None
    requires_bool: bool = False


@dataclass(frozen=True, slots=True)
class FlowBranchEdge:
    """Conditional edge, written as ``a >> branch(...)``."""

    source: Node
    branch: FlowBranch


class GraphNodeSpec(NodestepModel):
    """Serializable description of a node and its declared goto targets.

    Attributes
    ----------
    ref : str or None
        ``module:qualname`` import path of the node function.
    kind : str
        Always ``"node"``.
    goto : list[str] or None
        Names of the declared ``goto`` targets, ``"END"`` for the end.
    """

    name: str
    ref: str | None = None
    kind: str = "node"
    goto: list[str] | None = None


class EdgeSpec(NodestepModel):
    """Serializable description of a static edge.

    Attributes
    ----------
    source : str
        Name of the node the edge leaves.
    target : str
        Name of the next node, or ``"END"``.
    """

    source: str
    target: str


class BranchSpec(NodestepModel):
    """Serializable description of a conditional edge.

    Attributes
    ----------
    source : str
        Name of the node the edge leaves.
    router_ref : str
        ``module:qualname`` import path of the router.
    mapping : dict[str, str]
        Router result, as a string, to the name of the next node or ``"END"``.
    """

    source: str
    router_ref: str
    mapping: dict[str, str]


class GraphSpec(NodestepModel):
    """Serializable description of a graph.

    Attributes
    ----------
    version : int
        Version of this format.
    state_ref : str or None
        ``module:qualname`` import path of the state type; ``None`` for a dict state.
    start : str
        Name of the first node.
    """

    version: int = 1
    name: str
    state_ref: str | None = None
    start: str
    nodes: list[GraphNodeSpec]
    edges: list[EdgeSpec]
    branches: list[BranchSpec]


def branch(
    router: Callable[[Any], Any],
    mapping: Mapping[Any, Node | EndSentinel],
    *,
    ref: str | None = None,
) -> FlowBranch:
    """Route to one of several targets by a key computed from the state.

    Parameters
    ----------
    router : callable
        Receives the state and returns a key of ``mapping``, matched by
        equality. Any other key raises ``GraphExecutionError`` at run time.
    mapping : Mapping
        Router keys to target nodes or ``END``.
    ref : str, optional
        Import path of the router, for ``Graph.to_spec``.

    Returns
    -------
    FlowBranch

    Raises
    ------
    TypeError
        If ``router`` is not callable.
    GraphConfigError
        If ``mapping`` is empty.
    """
    if not callable(router):
        raise TypeError("branch router must be callable")
    if not mapping:
        raise GraphConfigError("branch mapping must not be empty")
    return FlowBranch(
        router=router, mapping=dict(mapping), router_ref=ref or _callable_ref(router)
    )


def when(
    predicate: Callable[[Any], bool],
    target: Node | EndSentinel,
    *,
    otherwise: Node | EndSentinel,
) -> FlowBranch:
    """Route to ``target`` when a predicate holds, else to ``otherwise``.

    Parameters
    ----------
    predicate : callable
        Receives the state and returns a ``bool``; any other value raises
        ``GraphExecutionError`` at run time.
    target : Node or END
        Target when the predicate is true.
    otherwise : Node or END
        Target when the predicate is false.

    Returns
    -------
    FlowBranch

    Raises
    ------
    TypeError
        If ``predicate`` is not callable.
    """
    return replace(
        branch(predicate, {True: target, False: otherwise}), requires_bool=True
    )


__all__ = [
    "BranchSpec",
    "EdgeSpec",
    "FlowBranch",
    "FlowBranchEdge",
    "FlowEdge",
    "FlowStartEdge",
    "GraphNodeSpec",
    "GraphSpec",
    "branch",
    "when",
]
