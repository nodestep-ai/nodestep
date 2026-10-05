from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field

from nodestep import END, START, Graph, InMemoryStateStore, add, add_messages, node
from nodestep.chat import AIMessage, HumanMessage, Message
from nodestep.exceptions import StateUpdateError


@pytest.mark.parametrize(
    "update", [("b", "c"), {"b"}, (item for item in "bc"), "b", None, {"b": 1}]
)
def test_add_takes_only_list_updates(update: Any) -> None:
    with pytest.raises(TypeError, match="add takes a list"):
        add(["a"], update)


def test_add_appends_the_items_of_a_list() -> None:
    assert add(["a"], ["b", "c"]) == ["a", "b", "c"]
    assert add(None, ["a"]) == ["a"]


def test_add_rejects_a_current_value_that_is_not_a_list() -> None:
    with pytest.raises(TypeError, match="add"):
        add(("a",), ["b"])


@pytest.mark.parametrize(
    "update",
    [
        "hi",
        None,
        HumanMessage(content="hi"),
        ("a", "b"),
        ["plain"],
        [{"type": "human", "content": "hi"}],
        [("user", "hi")],
    ],
)
def test_add_messages_takes_only_lists_of_messages(update: Any) -> None:
    with pytest.raises(TypeError, match="add_messages"):
        add_messages([], update)


def test_add_messages_rejects_duplicate_ids_in_one_update() -> None:
    with pytest.raises(StateUpdateError, match="'x'"):
        add_messages(
            [], [HumanMessage(id="x", content="1"), HumanMessage(id="x", content="2")]
        )


def test_add_messages_replaces_a_known_id_in_place() -> None:
    current = [AIMessage(id="m1", content="draft"), HumanMessage(id="m2", content="q")]

    merged = add_messages(current, [AIMessage(id="m1", content="final")])

    assert [(message.id, message.content) for message in merged] == [
        ("m1", "final"),
        ("m2", "q"),
    ]


def test_remove_message_removes_a_message_by_id() -> None:
    from nodestep import RemoveMessage

    current = [HumanMessage(id="a", content="1"), AIMessage(id="b", content="2")]

    merged = add_messages(
        current, [RemoveMessage("a"), HumanMessage(id="c", content="3")]
    )

    assert [message.id for message in merged] == ["b", "c"]
    assert [message.id for message in current] == ["a", "b"]


def test_remove_message_with_an_unknown_id_raises() -> None:
    from nodestep import RemoveMessage

    with pytest.raises(StateUpdateError, match="'zzz'"):
        add_messages([HumanMessage(id="a", content="1")], [RemoveMessage("zzz")])


def test_remove_message_and_a_message_with_the_same_id_raise() -> None:
    from nodestep import RemoveMessage

    with pytest.raises(StateUpdateError, match="'a'"):
        add_messages(
            [HumanMessage(id="a", content="1")],
            [RemoveMessage("a"), HumanMessage(id="a", content="again")],
        )


class Chat(BaseModel):
    messages: Annotated[list[Message], add_messages] = Field(default_factory=list)


async def test_remove_message_is_persisted_and_replayed() -> None:
    from nodestep import RemoveMessage

    @node
    def forget(state: Chat) -> dict:
        return {"messages": [RemoveMessage("q1")]}

    @node
    def greet(state: Chat) -> None:
        return None

    store = InMemoryStateStore()
    graph = Graph(Chat, state_store=store).flow(START >> forget, forget >> END)
    seed = Graph(Chat, state_store=store).flow(START >> greet, greet >> END)
    await seed.ainvoke(
        {
            "messages": [
                HumanMessage(id="q1", content="hi"),
                AIMessage(id="a1", content="hello"),
            ]
        },
        thread_id="t",
    )

    result = await graph.ainvoke({}, thread_id="t")
    loaded = await graph.load("t")

    assert [message.id for message in result.state.messages] == ["a1"]
    assert [message.id for message in loaded["messages"]] == ["a1"]


async def test_remove_message_needs_the_add_messages_reducer() -> None:
    from nodestep import RemoveMessage

    class Log(BaseModel):
        log: Annotated[list[Any], add] = Field(default_factory=list)

    @node
    def forget(state: Log) -> dict:
        return {"log": [RemoveMessage("x")]}

    graph = Graph(Log).flow(START >> forget, forget >> END)

    with pytest.raises(StateUpdateError, match=r"'log'.*add_messages"):
        await graph.ainvoke({})
