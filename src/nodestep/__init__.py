"""Graph framework for LLM agents and workflows.

Exports the names most programs need; the subpackages group them by topic,
with the rest of the API. Start with the Quickstart:
https://nodestep-ai.github.io/nodestep/quickstart/
"""

from importlib.metadata import version
from typing import TYPE_CHECKING

from nodestep.chat.integrations import ScriptedChat, _openai_installed
from nodestep.core import (
    END,
    START,
    AgentHandle,
    AgentResult,
    AgentState,
    AgentStatus,
    AgentTask,
    AsyncioExecutor,
    Command,
    DurableAgentHandle,
    Graph,
    GraphResult,
    Interrupt,
    Node,
    NodeContext,
    Resume,
    Send,
    StreamEvent,
    add,
    add_messages,
    branch,
    build_react_agent,
    call_tool,
    gather_agents,
    interrupt,
    merge_dict,
    model_node,
    node,
    replace,
    run_agent,
    spawn_agents,
    subgraph,
    tool,
    tool_runner,
    when,
)
from nodestep.core.tool import Tool, ToolContext
from nodestep.exceptions import (
    AgentGroupError,
    AgentHandleLostError,
    AgentTimeoutError,
    ChatHistoryError,
    ContextNotProvidedError,
    GraphConfigError,
    GraphExecutionError,
    GraphTimeoutError,
    IntegrationNotInstalledError,
    InvalidUpdateError,
    ModelProviderError,
    NodestepError,
    NodeTimeoutError,
    PathAccessError,
    ResumeError,
    RunInterruptedError,
    RunLimitExceededError,
    StateStoreError,
    StateUpdateError,
    StructuredOutputError,
    ToolDeniedError,
    ToolExecutionError,
    ToolGroupExecutionError,
    UnknownThreadError,
    WorkspaceError,
)
from nodestep.middleware.base import Middleware, ModelMiddlewareContext, RunOutcome
from nodestep.models.base import BaseState
from nodestep.state.integrations import FilesystemStateStore, InMemoryStateStore
from nodestep.utils.reducers import RemoveMessage, Replace

__version__ = version("nodestep")

__all__ = [
    "END",
    "START",
    "AgentGroupError",
    "AgentHandle",
    "AgentHandleLostError",
    "AgentResult",
    "AgentState",
    "AgentStatus",
    "AgentTask",
    "AgentTimeoutError",
    "AsyncioExecutor",
    "BaseState",
    "ChatHistoryError",
    "Command",
    "ContextNotProvidedError",
    "DurableAgentHandle",
    "FilesystemStateStore",
    "Graph",
    "GraphConfigError",
    "GraphExecutionError",
    "GraphResult",
    "GraphTimeoutError",
    "InMemoryStateStore",
    "IntegrationNotInstalledError",
    "Interrupt",
    "InvalidUpdateError",
    "Middleware",
    "ModelMiddlewareContext",
    "ModelProviderError",
    "Node",
    "NodeContext",
    "NodeTimeoutError",
    "NodestepError",
    "PathAccessError",
    "RemoveMessage",
    "Replace",
    "Resume",
    "ResumeError",
    "RunInterruptedError",
    "RunLimitExceededError",
    "RunOutcome",
    "ScriptedChat",
    "Send",
    "StateStoreError",
    "StateUpdateError",
    "StreamEvent",
    "StructuredOutputError",
    "Tool",
    "ToolContext",
    "ToolDeniedError",
    "ToolExecutionError",
    "ToolGroupExecutionError",
    "UnknownThreadError",
    "WorkspaceError",
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


if TYPE_CHECKING:
    from nodestep.chat.integrations.openai import OpenAIChat as OpenAIChat


_LAZY_IMPORTS: dict[str, str] = {
    "OpenAIChat": "nodestep.chat.integrations.openai",
}


def __getattr__(name: str) -> object:
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is not None:
        import importlib

        return getattr(importlib.import_module(module_path), name)
    raise AttributeError(f"module 'nodestep' has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the public names, and ``OpenAIChat`` when the ``openai`` extra is installed.

    ``OpenAIChat`` is not in ``__all__``, so ``from nodestep import *`` works
    without the extra, and ``dir`` lists it only when it can be imported, so
    ``inspect.getmembers`` and ``help`` work without it too.
    """
    return [*__all__, *(_LAZY_IMPORTS if _openai_installed() else ())]
