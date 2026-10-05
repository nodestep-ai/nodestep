from __future__ import annotations

import re
from typing import Any

from nodestep.core.command import EndSentinel
from nodestep.core.node import get_node_name

_UNSAFE_ID_CHARACTERS = re.compile(r"[^A-Za-z0-9_]")
_RESERVED_IDS = frozenset({"START", "END", "end"})


def mermaid_id(name: str) -> str:
    """Return the Mermaid id derived from a node name.

    Characters outside ``[A-Za-z0-9_]`` become ``_``. When a graph is rendered, an
    id already used by another node, by ``START`` or ``END``, or equal to the
    Mermaid keyword ``end`` gets the suffix ``_2``, ``_3`` and so on.

    Parameters
    ----------
    name : str
        Node name.

    Returns
    -------
    str
    """
    return _UNSAFE_ID_CHARACTERS.sub("_", name)


_LABEL_ESCAPES = {character: f"#{ord(character)};" for character in '#"&<>|\n\r'}


def _escape_label(name: str) -> str:
    return "".join(_LABEL_ESCAPES.get(character, character) for character in name)


class _IdAllocator:
    def __init__(self, names: list[str]) -> None:
        self._ids: dict[str, str] = {}
        taken = set(_RESERVED_IDS)
        for name in names:
            base = mermaid_id(name)
            candidate = base
            suffix = 1
            while candidate in taken:
                suffix += 1
                candidate = f"{base}_{suffix}"
            taken.add(candidate)
            self._ids[name] = candidate
        self._declared: set[str] = set()

    def reference(self, name: str, lines: list[str]) -> str:
        node_id = self._ids[name]
        if node_id not in self._declared:
            self._declared.add(node_id)
            lines.append(f'  {node_id}["{_escape_label(name)}"]')
        return node_id


class _EdgeIds:
    def __init__(self) -> None:
        self._counts: dict[tuple[str, str], int] = {}

    def next(self, source: str, target: str) -> str:
        count = self._counts.get((source, target), 0) + 1
        self._counts[(source, target)] = count
        edge_id = f"{source}-{target}"
        return edge_id if count == 1 else f"{edge_id}-{count}"


def mermaid(graph: Any, *, direction: str = "TD") -> str:
    """Render a graph flow as a Mermaid flowchart.

    Node ids come from ``mermaid_id``, and ``START`` and ``END`` keep theirs.
    Edge ids use Mermaid's edge id syntax, as in ``a a-b@--> b``; node ids
    never contain ``-``, so no two ids collide. Node names and branch keys are
    quoted labels in which ``#``, ``"``, ``&``, ``<``, ``>``, ``|`` and line
    breaks are Mermaid entity codes (``#34;`` for ``"``), so any name renders
    as written. See [Draw the graph](../../concepts/graphs.md#draw-the-graph).

    Parameters
    ----------
    graph : Graph
    direction : str, optional
        Mermaid direction such as ``"TD"`` or ``"LR"``.

    Returns
    -------
    str
    """
    ids = _IdAllocator(list(graph.nodes))
    edge_ids = _EdgeIds()
    lines = [f"graph {direction}", "  START([START])"]

    def node_ref(name: str) -> str:
        return ids.reference(name, lines)

    def target_ref(target: Any) -> tuple[str, str]:
        if isinstance(target, EndSentinel):
            return "END", "END([END])"
        target_id = node_ref(get_node_name(target))
        return target_id, target_id

    def add_edge(source_id: str, target: Any, arrow: str) -> None:
        target_id, reference = target_ref(target)
        edge_id = edge_ids.next(source_id, target_id)
        lines.append(f"  {source_id} {edge_id}@{arrow} {reference}")

    if graph.start_node:
        start_id = node_ref(graph.start_node)
        lines.append(f"  START {edge_ids.next('START', start_id)}@--> {start_id}")

    for source, edge in graph.edges.items():
        add_edge(node_ref(source), edge.target, "-->")

    for source, edge in graph.branch_edges.items():
        for key, target in edge.branch.mapping.items():
            add_edge(node_ref(source), target, f'-->|"{_escape_label(str(key))}"|')

    for name, spec in graph.nodes.items():
        for target in spec.goto or ():
            add_edge(node_ref(name), target, "-.->")

    return "\n".join(lines)


__all__ = ["mermaid", "mermaid_id"]
