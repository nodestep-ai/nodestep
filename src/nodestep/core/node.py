from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, overload

from pydantic import ConfigDict

from nodestep.core.command import EndSentinel
from nodestep.core.stream import NodeContext
from nodestep.models.base import NodestepModel
from nodestep.utils.refs import (
    callable_name as _callable_name,
)
from nodestep.utils.refs import (
    callable_ref as _callable_ref,
)


class NodeMetadata(NodestepModel):
    """Serializable description of a node's options.

    Attributes
    ----------
    ref : str or None
        ``module:qualname`` import path of the node function.
    accepts_context : bool
        The function takes a ``NodeContext`` parameter.
    context_is_keyword : bool
        That parameter is keyword-only.
    context_parameter_name : str
        Name of that parameter.
    timeout : float or None
        Seconds before the node fails with ``NodeTimeoutError``.
    goto : tuple[str, ...] or None
        Names of the declared ``goto`` targets, ``"END"`` for the end.
    """

    name: str
    ref: str | None = None
    accepts_context: bool = False
    context_is_keyword: bool = False
    context_parameter_name: str = "ctx"
    timeout: float | None = None
    goto: tuple[str, ...] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass(frozen=True, slots=True)
class Node:
    """A graph step: a callable plus its name and options.

    Attributes
    ----------
    handler : Callable
        The wrapped function.
    ref : str or None
        ``module:qualname`` import path of the function, for ``Graph.to_spec``.
    accepts_context : bool
        The function takes a ``NodeContext`` parameter.
    context_is_keyword : bool
        That parameter is keyword-only.
    context_parameter_name : str
        Name of that parameter.
    timeout : float or None
        Seconds before the node fails with ``NodeTimeoutError``.
    goto : tuple of Node, str or END, optional
        Targets the node may reach through ``Command(goto=...)``, ``Send`` or a
        returned ``END``, as declared; names are resolved by ``Graph.flow``.
    """

    name: str
    handler: Callable[..., Any]
    ref: str | None = None
    accepts_context: bool = False
    context_is_keyword: bool = False
    context_parameter_name: str = "ctx"
    timeout: float | None = None
    goto: tuple[Node | EndSentinel | str, ...] | None = None

    def __repr__(self) -> str:
        return f"Node(name={self.name!r})"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function directly, outside any graph.

        Parameters
        ----------
        *args, **kwargs
            Passed to the function as they are.

        Returns
        -------
        Any
            What the function returns; nothing is validated or routed.
        """
        return self.handler(*args, **kwargs)

    def __rshift__(self, target: Any) -> Any:
        """Write a flow item: ``node >> other``, ``node >> END`` or ``node >> branch(...)``.

        Parameters
        ----------
        target : Node, END or FlowBranch
            The next node, the end of the flow, or a ``branch``/``when`` table.

        Returns
        -------
        FlowEdge or FlowBranchEdge
            An item for ``Graph.flow``.
        """
        from nodestep.core.flow import FlowBranch, FlowBranchEdge, FlowEdge

        if isinstance(target, FlowBranch):
            return FlowBranchEdge(source=self, branch=target)
        return FlowEdge(source=self, target=target)

    @property
    def __name__(self) -> str:
        return self.name

    @property
    def metadata(self) -> NodeMetadata:
        """The node's options as a ``NodeMetadata``."""
        return NodeMetadata(
            name=self.name,
            ref=self.ref,
            accepts_context=self.accepts_context,
            context_is_keyword=self.context_is_keyword,
            context_parameter_name=self.context_parameter_name,
            timeout=self.timeout,
            goto=None
            if self.goto is None
            else tuple(_goto_name(target) for target in self.goto),
        )


def _goto_name(target: Node | EndSentinel | str) -> str:
    if isinstance(target, EndSentinel):
        return "END"
    return target if isinstance(target, str) else target.name


_SIGNATURE_HELP = (
    "graph nodes take one state parameter, optionally followed by a parameter "
    "annotated with nodestep.NodeContext"
)


def _annotation_globals(function: Callable[..., Any]) -> dict[str, Any]:
    return getattr(inspect.unwrap(function), "__globals__", {})


def _resolve_annotation(
    function: Callable[..., Any], parameter: inspect.Parameter, node_name: str
) -> Any:
    annotation = parameter.annotation
    if not isinstance(annotation, str):
        return annotation
    try:
        return eval(annotation, _annotation_globals(function))
    except Exception as error:
        raise TypeError(
            f"Cannot resolve the annotation {annotation!r} of parameter "
            f"'{parameter.name}' of node '{node_name}': {type(error).__name__}: {error}"
        ) from error


def _accepts_context(
    function: Callable[..., Any], node_name: str
) -> tuple[bool, bool, str]:
    parameters = list(inspect.signature(function).parameters.values())
    for parameter in parameters:
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise TypeError("Graph nodes cannot use *args or **kwargs")
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    keyword_only = [
        parameter
        for parameter in parameters
        if parameter.kind == parameter.KEYWORD_ONLY
    ]
    if len(positional) == 1 and not keyword_only:
        return False, False, "ctx"
    if len(positional) == 1 and len(keyword_only) == 1:
        candidate, is_keyword = keyword_only[0], True
    elif len(positional) == 2 and not keyword_only:
        candidate, is_keyword = positional[1], False
    else:
        raise TypeError(
            f"Node '{node_name}' has an unsupported signature; {_SIGNATURE_HELP}"
        )
    if _resolve_annotation(function, candidate, node_name) is not NodeContext:
        raise TypeError(
            f"Parameter '{candidate.name}' of node '{node_name}' is not annotated "
            f"with nodestep.NodeContext; {_SIGNATURE_HELP}"
        )
    return True, is_keyword, candidate.name


def _declared_targets(
    goto: Sequence[Node | EndSentinel | str] | None, node_name: str
) -> tuple[Node | EndSentinel | str, ...] | None:
    if goto is None:
        return None
    if isinstance(goto, str) or not isinstance(goto, Sequence):
        raise TypeError(
            f"goto= of node '{node_name}' must be a list of nodes, node names or END"
        )
    targets = tuple(goto)
    if not targets:
        raise TypeError(
            f"goto= of node '{node_name}' must list at least one target; "
            "omit it when the node has no dynamic targets"
        )
    for target in targets:
        if not isinstance(target, (Node, EndSentinel, str)):
            raise TypeError(
                "goto targets must be nodes, node names or END; "
                f"node '{node_name}' got {target!r}"
            )
    return targets


def get_node_metadata(node: Node) -> NodeMetadata:
    """Return the metadata of a node.

    Parameters
    ----------
    node : Node

    Returns
    -------
    NodeMetadata

    Raises
    ------
    TypeError
        If ``node`` is not a node.
    """
    if not isinstance(node, Node):
        raise TypeError(f"{_callable_name(node)} is not a Nodestep node")
    return node.metadata


def get_node_name(node: Node) -> str:
    """Return the name of a node.

    Parameters
    ----------
    node : Node

    Returns
    -------
    str

    Raises
    ------
    TypeError
        If ``node`` is not a node.
    """
    if not isinstance(node, Node):
        raise TypeError(f"{_callable_name(node)} is not a Nodestep node")
    return node.name


@overload
def node(function: Callable[..., Any], /) -> Node: ...


@overload
def node(
    function: Callable[..., Any],
    /,
    *,
    name: str | None = None,
    ref: str | None = None,
    timeout: float | None = None,
    goto: Sequence[Node | EndSentinel | str] | None = None,
) -> Node: ...


@overload
def node(
    *,
    name: str | None = None,
    ref: str | None = None,
    timeout: float | None = None,
    goto: Sequence[Node | EndSentinel | str] | None = None,
) -> Callable[[Callable[..., Any]], Node]: ...


def node(
    function: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    ref: str | None = None,
    timeout: float | None = None,
    goto: Sequence[Node | EndSentinel | str] | None = None,
) -> Node | Callable[[Callable[..., Any]], Node]:
    """Turn a function into a graph node.

    The function takes the state, optionally followed by a parameter annotated
    ``NodeContext``. It returns a dict update, the state it received changed in
    place, ``Command``, ``Send``, ``END`` or ``None``; see [Nodes and
    context](../../concepts/nodes.md) for what each one does.

    Parameters
    ----------
    function : callable, optional
        Function to wrap; omit it to pass options.
    name : str, optional
        Node name; defaults to the function's name. Lambdas need one.
    ref : str, optional
        Import path for ``Graph.to_spec``.
    timeout : float, optional
        Seconds before the node fails with ``NodeTimeoutError``. A sync node
        fails only when it returns.
    goto : sequence of Node, str or END, optional
        Every target reached through ``Command(goto=...)``, ``Send`` or a
        returned ``END``. Listed nodes join the flow; a name refers to the
        flow's node of that name, such as the node itself or one defined later.
        Routing anywhere else raises ``GraphExecutionError``.

    Returns
    -------
    Node or callable

    Raises
    ------
    TypeError
        If ``function`` is not a function or bound method, the name is blank
        or ``<lambda>``, the signature is not supported, or ``goto`` is not a
        non-empty list of nodes, node names and ``END``.
    """

    def decorate(inner: Callable[..., Any]) -> Node:
        if not (inspect.isfunction(inner) or inspect.ismethod(inner)):
            raise TypeError(
                f"node() got {type(inner).__name__} {inner!r}; pass a function or "
                "bound method"
            )
        node_name = _callable_name(inner) if name is None else name
        if not node_name.strip():
            raise TypeError("Node name must not be empty")
        if node_name == "<lambda>":
            raise TypeError("A lambda node needs an explicit name; pass name=")
        accepts, is_keyword, parameter_name = _accepts_context(inner, node_name)
        return Node(
            name=node_name,
            handler=inner,
            ref=ref or _callable_ref(inner),
            accepts_context=accepts,
            context_is_keyword=is_keyword,
            context_parameter_name=parameter_name,
            timeout=timeout,
            goto=_declared_targets(goto, node_name),
        )

    if function is not None:
        return decorate(function)
    return decorate


__all__ = ["Node", "NodeMetadata", "get_node_metadata", "get_node_name", "node"]
