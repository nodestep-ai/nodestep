"""Graphs, nodes, tools, reducers, execution and sub-agents.

The built-in nodes, such as `build_react_agent`, are in
`nodestep.core.builtin_nodes`. See [Graphs and flow](../../concepts/graphs.md).
"""

from nodestep.core.agent import (
    AgentHandle,
    AgentRegistry,
    AgentResult,
    AgentStatus,
    AgentTask,
    DurableAgentHandle,
    gather_agents,
    spawn_agents,
)
from nodestep.core.builtin_nodes import (
    AgentState,
    StructuredOutputEvent,
    ToolErrorPolicy,
    build_react_agent,
    model_node,
    run_agent,
    subgraph,
    tool_runner,
)
from nodestep.core.command import (
    END,
    START,
    Command,
    EndSentinel,
    Interrupt,
    Resume,
    Send,
    StartSentinel,
    interrupt,
)
from nodestep.core.executor import AgentExecutor, AsyncioExecutor
from nodestep.core.flow import (
    FlowBranch,
    FlowBranchEdge,
    FlowEdge,
    FlowStartEdge,
    branch,
    when,
)
from nodestep.core.graph import Graph, GraphResult, GraphSpec, NodeSpec
from nodestep.core.node import Node, NodeMetadata, node
from nodestep.core.stream import (
    FinalEventData,
    InterruptEventData,
    NodeContext,
    StreamEvent,
)
from nodestep.core.tool import (
    Tool,
    ToolCallResult,
    ToolContext,
    ToolMetadata,
    call_tool,
    tool,
)
from nodestep.middleware.base import ModelMiddlewareContext, RunOutcome
from nodestep.utils.reducers import add, add_messages, merge_dict, replace

__all__ = [
    "END",
    "START",
    "AgentExecutor",
    "AgentHandle",
    "AgentRegistry",
    "AgentResult",
    "AgentState",
    "AgentStatus",
    "AgentTask",
    "AsyncioExecutor",
    "Command",
    "DurableAgentHandle",
    "EndSentinel",
    "FinalEventData",
    "FlowBranch",
    "FlowBranchEdge",
    "FlowEdge",
    "FlowStartEdge",
    "Graph",
    "GraphResult",
    "GraphSpec",
    "Interrupt",
    "InterruptEventData",
    "ModelMiddlewareContext",
    "Node",
    "NodeContext",
    "NodeMetadata",
    "NodeSpec",
    "Resume",
    "RunOutcome",
    "Send",
    "StartSentinel",
    "StreamEvent",
    "StructuredOutputEvent",
    "Tool",
    "ToolCallResult",
    "ToolContext",
    "ToolErrorPolicy",
    "ToolMetadata",
    "add",
    "add_messages",
    "branch",
    "build_react_agent",
    "call_tool",
    "gather_agents",
    "interrupt",
    "merge_dict",
    "model_node",
    "node",
    "replace",
    "run_agent",
    "spawn_agents",
    "subgraph",
    "tool",
    "tool_runner",
    "when",
]
