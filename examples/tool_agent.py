from pathlib import Path
from typing import Annotated

from pydantic import Field

from nodestep import (
    END,
    START,
    BaseState,
    Graph,
    InMemoryStateStore,
    Resume,
    ScriptedChat,
    add_messages,
    model_node,
    tool,
    tool_runner,
    when,
)
from nodestep.chat import ChatResponse, HumanMessage, Message, ToolCall
from nodestep.middleware import InterruptRule, ToolInterruptMiddleware
from nodestep.workspace import LocalWorkspace


class AgentState(BaseState):
    messages: Annotated[list[Message], add_messages] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_text: str | None = None


@tool(name="list_files", description="List files in the workspace root.")
def list_files() -> list[str]:
    return [path.name for path in Path(".").iterdir() if path.is_file()]


def has_tool_calls(state: AgentState) -> bool:
    return bool(state.tool_calls)


chat = ScriptedChat(
    [ChatResponse(tool_calls=[ToolCall(id="call_1", name="list_files", arguments={})])],
    default="Those are the files in the workspace root.",
)

think = model_node(
    "think",
    chat=chat,
    system_prompt="You are a helpful agent.",
    tools=[list_files],
)
act = tool_runner("act", tools=[list_files])

graph = Graph(
    AgentState,
    name="agent",
    workspace=LocalWorkspace(root_dir=Path(".")),
    state_store=InMemoryStateStore(),
    middleware=[
        ToolInterruptMiddleware(
            rules=[InterruptRule(tool="list_files")],
        ),
    ],
).flow(
    START >> think,
    think >> when(has_tool_calls, act, otherwise=END),
    act >> think,
)


def main() -> None:
    result = graph.invoke(
        {"messages": [HumanMessage(content="list files")]},
        thread_id="chat-123",
    )
    if result.status == "interrupted":
        print("approval requested:", result.interrupts)
        approvals = dict.fromkeys(result.interrupts, True)
        result = graph.invoke(
            None, thread_id="chat-123", resume=Resume(answers=approvals)
        )
    print(result.state.final_text)


if __name__ == "__main__":
    main()
