from typing import Any, Self

from pydantic import BaseModel

from nodestep import Graph
from nodestep.core import EndSentinel


def _target_name(target: Any) -> str:
    return "END" if isinstance(target, EndSentinel) else target.name


class NodeView(BaseModel):
    """A node and where it can go next.

    Attributes
    ----------
    name : str
    edges : list[str]
        Target of the node's static edge.
    branches : list[str]
        Branch routes as ``key → target``.
    goto : list[str]
        Declared ``goto`` targets.
    """

    name: str
    edges: list[str]
    branches: list[str]
    goto: list[str]


class GraphOverview(BaseModel):
    """What the graph page shows about a graph's flow.

    Attributes
    ----------
    name : str
    start : str
        The first node.
    mermaid : str
        ``Graph.to_mermaid()``.
    nodes : list[NodeView]
    """

    name: str
    start: str
    mermaid: str
    nodes: list[NodeView]

    @classmethod
    def from_graph(cls, graph: Graph[Any]) -> Self:
        """Describe a graph that has a flow.

        Parameters
        ----------
        graph : Graph

        Returns
        -------
        GraphOverview
        """
        nodes: list[NodeView] = []
        for name, spec in graph.nodes.items():
            edge = graph.edges.get(name)
            branch = graph.branch_edges.get(name)
            nodes.append(
                NodeView(
                    name=name,
                    edges=[] if edge is None else [_target_name(edge.target)],
                    branches=[]
                    if branch is None
                    else [
                        f"{key} → {_target_name(target)}"
                        for key, target in branch.branch.mapping.items()
                    ],
                    goto=[_target_name(target) for target in spec.goto or ()],
                )
            )
        return cls(
            name=graph.name,
            start=graph.start_node or "",
            mermaid=graph.to_mermaid(),
            nodes=nodes,
        )
