from typing import Annotated, TypedDict

import pytest
from pydantic import BaseModel, Field

from nodestep import (
    END,
    START,
    Graph,
    ScriptedChat,
    add,
    add_messages,
    model_node,
    tool,
    tool_runner,
)
from nodestep.chat import ToolCall
from nodestep.chat.messages import HumanMessage, Message
from nodestep.exceptions import GraphConfigError, StateUpdateError
from nodestep.utils.state import get_state_field


class Parent(BaseModel):
    log: Annotated[list[str], add] = Field(default_factory=list)


@pytest.mark.parametrize("state", [Parent(log=["x"]), {"log": ["x"]}])
def test_get_state_field_reads_a_field(state: object) -> None:
    assert get_state_field(state, "log") == ["x"]


@pytest.mark.parametrize("state", [Parent(log=["x"]), {"log": ["x"]}])
def test_get_state_field_raises_for_a_missing_field(state: object) -> None:
    with pytest.raises(StateUpdateError, match="'logs'"):
        get_state_field(state, "logs")


def test_get_state_field_says_a_dict_state_may_just_lack_a_value() -> None:
    with pytest.raises(
        StateUpdateError, match=r"no value for 'messages'.*has no value yet.*'log'"
    ):
        get_state_field({"log": ["x"]}, "messages")


class Conversation(TypedDict, total=False):
    messages: Annotated[list[Message], add_messages]
    tool_calls: list[ToolCall]
    final_text: str | None


async def test_a_misnamed_messages_field_raises_before_the_model_is_called() -> None:
    chat = ScriptedChat(["hi"])
    model = model_node("model", chat=chat, messages_field="chat_history")
    graph = Graph(Conversation).flow(START >> model, model >> END)

    with pytest.raises(GraphConfigError, match="'chat_history'"):
        await graph.ainvoke({"messages": [HumanMessage(content="hello")]})

    assert chat.requests == []


async def test_a_misnamed_tool_calls_field_raises() -> None:
    runner = tool_runner("tools", tools=[], tool_calls_field="calls")
    graph = Graph(Conversation).flow(START >> runner, runner >> END)

    with pytest.raises(GraphConfigError, match="'calls'"):
        await graph.ainvoke({"messages": [], "tool_calls": []})


tool_runs: list[str] = []


@tool
def record(text: str) -> str:
    """Record a text."""
    tool_runs.append(text)
    return text


class Calls(TypedDict, total=False):
    messages: Annotated[list[Message], add_messages]
    tool_calls: list[ToolCall]


async def test_a_misnamed_messages_field_raises_before_a_tool_runs() -> None:
    tool_runs.clear()
    runner = tool_runner("tools", tools=[record], messages_field="chat_history")
    graph = Graph(Calls).flow(START >> runner, runner >> END)
    call = ToolCall(id="c1", name="record", arguments={"text": "hi"})

    with pytest.raises(GraphConfigError, match="'chat_history'"):
        await graph.ainvoke({"messages": [], "tool_calls": [call]})

    assert tool_runs == []
