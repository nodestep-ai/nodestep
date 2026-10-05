from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, Field, create_model

from nodestep.chat.messages import Message, ToolCall
from nodestep.core.builtin_nodes.model import model_node
from nodestep.core.builtin_nodes.tool_runner import ToolErrorPolicy, tool_runner
from nodestep.core.command import END, START
from nodestep.core.flow import when
from nodestep.exceptions import RunInterruptedError
from nodestep.middleware.base import collect_middleware_tools
from nodestep.middleware.tool_limit import ToolLimitMiddleware
from nodestep.models.base import BaseState
from nodestep.utils.reducers import add_messages

if TYPE_CHECKING:
    from nodestep.chat.base import Chat
    from nodestep.core.graph import Graph
    from nodestep.core.tool import Tool
    from nodestep.middleware.base import Middleware
    from nodestep.state.history import StateStore
    from nodestep.workspace.base import Workspace


class AgentState(BaseState):
    """State of the built-in ReAct agent.

    Attributes
    ----------
    messages : list[Message]
        The conversation; updates are merged with ``add_messages``.
    tool_calls : list[ToolCall]
        Calls of the last model turn that have not run yet.
    final_text : str or None
        Answer text of the last model turn; ``None`` while the model calls tools.
    final_output : Any
        The answer parsed into ``output_schema``, when the agent has one.
    """

    messages: Annotated[list[Message], add_messages] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_text: str | None = None
    final_output: Any = None


def build_react_agent(
    chat: Chat,
    *,
    tools: Sequence[Tool],
    name: str = "agent",
    system_prompt: str | None = None,
    middleware: Sequence[Middleware] = (),
    tool_errors: ToolErrorPolicy = "raise",
    stream: bool = False,
    output_schema: type[BaseModel] | None = None,
    max_steps: int = 25,
    max_tool_calls: int = 50,
    workspace: Workspace | None = None,
    state_store: StateStore | None = None,
) -> Graph[AgentState]:
    """Build a tool-calling agent graph.

    The model node ``think`` and the tool node ``act`` alternate until the
    model answers without tool calls. Both nodes get ``tools`` followed by the
    ``tools()`` of each middleware in ``middleware``.

    Parameters
    ----------
    chat : Chat
        Chat model to call.
    tools : Sequence[Tool]
        Tools the model may call; may be empty.
    name : str, optional
        Graph name.
    system_prompt : str, optional
        System message prepended to every model call.
    middleware : Sequence[Middleware], optional
        Extra middleware, after the built-in ``ToolLimitMiddleware``.
    tool_errors : {"raise", "return"} or callable, optional
        What a failed tool call does, as in ``tool_runner``.
    stream : bool, optional
        Stream model calls, as in ``model_node``.
    output_schema : type[BaseModel], optional
        Model the final answer is parsed into; ``final_output`` is typed
        ``output_schema | None``. An invalid answer raises
        ``StructuredOutputError``.
    max_steps : int, optional
        Superstep limit per call.
    max_tool_calls : int, optional
        Tool-call budget for one run and its resumes.
    workspace : Workspace, optional
        Workspace exposed to tools.
    state_store : StateStore, optional
        Store that enables threads, interrupts and resume.

    Returns
    -------
    Graph[AgentState]

    Raises
    ------
    GraphConfigError
        If two different tools share a name, or the same tool is listed twice,
        also through a middleware.
    """
    from nodestep.core.graph import Graph

    agent_tools = [*tools, *collect_middleware_tools(middleware)]
    think = model_node(
        "think",
        chat=chat,
        system_prompt=system_prompt,
        tools=agent_tools,
        stream=stream,
        output_schema=output_schema,
    )
    act = tool_runner("act", tools=agent_tools, tool_errors=tool_errors)

    all_middleware: list[Middleware] = [
        ToolLimitMiddleware(max_calls=max_tool_calls),
        *middleware,
    ]

    def has_tool_calls(state: AgentState) -> bool:
        return bool(state.tool_calls)

    state_type: type[AgentState] = (
        AgentState
        if output_schema is None
        else create_model(
            "AgentState",
            __base__=AgentState,
            final_output=(output_schema | None, None),
        )
    )
    graph = Graph(
        state_type,
        name=name,
        max_steps=max_steps,
        middleware=all_middleware,
        workspace=workspace,
        state_store=state_store,
    ).flow(
        START >> think,
        think >> when(has_tool_calls, act, otherwise=END),
        act >> think,
    )
    return graph


async def run_agent(graph: Graph, message: str, **kwargs: Any) -> str:
    """Send one user message to an agent and return its final text.

    Parameters
    ----------
    graph : Graph
        Agent graph, usually from ``build_react_agent``.
    message : str
        User message.
    **kwargs
        Passed to ``Graph.ainvoke``, e.g. ``thread_id``.

    Returns
    -------
    str
        The final text, or ``""`` when there is none, as after a refusal
        (see ``AIMessage.refusal``).

    Raises
    ------
    RunInterruptedError
        If the run paused for human input.
    """
    from nodestep.chat.messages import HumanMessage

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=message)]}, **kwargs
    )
    if result.status == "interrupted":
        raise RunInterruptedError(result.thread_id, result.interrupts)
    return result.data.get("final_text", "") or ""


__all__ = ["AgentState", "build_react_agent", "run_agent"]
