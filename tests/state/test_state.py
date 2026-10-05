import math
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field

from nodestep import END, START, BaseState, Graph, node
from nodestep.state import StateSchema, merge_parallel_updates, merge_updates
from nodestep.state.schema import diff_states, states_differ
from nodestep.utils.reducers import add, add_messages


class State(BaseModel):
    count: int = 0
    messages: Annotated[list[str], add] = Field(default_factory=list)


def test_pydantic_initial_state_fills_defaults() -> None:
    schema = StateSchema.from_type(State)

    state, update = schema.merge_input(None, {"count": 1})

    assert state == {"count": 1, "messages": []}
    assert update == {"count": 1}


def test_unknown_field_rejected() -> None:
    schema = StateSchema.from_type(State)

    with pytest.raises(Exception, match="Unknown field"):
        merge_updates(schema, {}, {"missing": True})


def test_reducer_appends_messages() -> None:
    schema = StateSchema.from_type(State)

    state, written = merge_updates(schema, {"messages": ["a"]}, {"messages": ["b"]})

    assert state["messages"] == ["a", "b"]
    assert written == {"messages"}


def test_parallel_updates_order_by_node_name() -> None:
    schema = StateSchema.from_type(State)

    state, _ = merge_parallel_updates(
        schema,
        {"messages": []},
        [("b", {"messages": ["b"]}), ("a", {"messages": ["a"]})],
    )

    assert state["messages"] == ["a", "b"]


class TypedState(BaseState):
    count: int = 0
    messages: Annotated[list[str], add] = Field(default_factory=list)


def test_base_state_model_copy_returns_new_instance() -> None:
    state = TypedState(count=1)
    updated = state.model_copy(update={"count": 2})
    assert updated.count == 2
    assert state.count == 1


def test_diff_states_detects_changes() -> None:
    schema = StateSchema.from_type(TypedState)
    before = TypedState(count=1, messages=["a"])
    after = TypedState(count=2, messages=["a"])
    diff = diff_states(schema, before, after)
    assert diff == {"count": 2}


def test_diff_states_empty_when_equal() -> None:
    schema = StateSchema.from_type(TypedState)
    state = TypedState(count=1)
    assert diff_states(schema, state, state) == {}


def test_node_returns_typed_state() -> None:
    @node
    def increment(state: TypedState) -> TypedState:
        state.count += 1
        return state

    graph = Graph(TypedState).flow(START >> increment, increment >> END)
    result = graph.invoke({"count": 0})
    assert result.data["count"] == 1


def test_node_returns_typed_state_with_reducer() -> None:
    @node
    def add_message(state: TypedState) -> TypedState:
        state.messages = [*state.messages, "hello"]
        return state

    graph = Graph(TypedState).flow(START >> add_message, add_message >> END)
    result = graph.invoke({"messages": ["existing"]})
    assert result.data["messages"] == ["existing", "hello"]


def test_node_returns_typed_state_trim_reduced_list() -> None:
    @node
    def trim(state: TypedState) -> TypedState:
        state.messages = state.messages[-1:]
        return state

    graph = Graph(TypedState).flow(START >> trim, trim >> END)
    result = graph.invoke({"messages": ["a", "b", "c"]})
    assert result.data["messages"] == ["c"]


def test_node_returns_typed_state_reorder_reduced_list() -> None:
    @node
    def reorder(state: TypedState) -> TypedState:
        state.messages = list(reversed(state.messages))
        return state

    graph = Graph(TypedState).flow(START >> reorder, reorder >> END)
    result = graph.invoke({"messages": ["a", "b", "c"]})
    assert result.data["messages"] == ["c", "b", "a"]


def test_diff_states_replace_survives_store_replay() -> None:
    from nodestep.state.integrations import InMemoryStateStore

    @node
    def trim(state: TypedState) -> TypedState:
        state.messages = state.messages[-1:]
        return state

    store = InMemoryStateStore()
    graph = Graph(TypedState, state_store=store).flow(START >> trim, trim >> END)
    result = graph.invoke({"messages": ["a", "b", "c"]}, thread_id="trim-thread")
    assert result.data["messages"] == ["c"]

    async def _load() -> dict:
        return await graph.load("trim-thread")

    import asyncio

    reloaded = asyncio.run(_load())
    assert reloaded["messages"] == ["c"]


def test_diff_states_emits_replace_for_non_append() -> None:
    from nodestep import Replace

    schema = StateSchema.from_type(TypedState)
    before = TypedState(messages=["a", "b"])
    after = TypedState(messages=["b"])
    diff = diff_states(schema, before, after)
    assert isinstance(diff["messages"], Replace)
    assert diff["messages"].value == ["b"]


def test_invalid_update_error_names_field_and_nodes() -> None:
    from nodestep.exceptions import InvalidUpdateError, StateUpdateError

    error = InvalidUpdateError(field="count", nodes=["b", "a"])
    assert isinstance(error, StateUpdateError)
    assert error.field == "count"
    assert error.nodes == ["a", "b"]
    assert "count" in str(error)
    assert "a" in str(error)
    assert "b" in str(error)


def test_parallel_replace_conflict_raises() -> None:
    from nodestep.exceptions import InvalidUpdateError

    schema = StateSchema.from_type(State)

    with pytest.raises(InvalidUpdateError) as info:
        merge_parallel_updates(
            schema,
            {"count": 0},
            [("b", {"count": 2}), ("a", {"count": 1})],
        )

    assert info.value.field == "count"
    assert info.value.nodes == ["a", "b"]


def test_parallel_reducer_field_writes_combine() -> None:
    schema = StateSchema.from_type(State)

    state, written = merge_parallel_updates(
        schema,
        {"messages": []},
        [("b", {"messages": ["b"]}), ("a", {"messages": ["a"]})],
    )

    assert state["messages"] == ["a", "b"]
    assert written == {"messages"}


def test_parallel_single_writer_replace_field_ok() -> None:
    schema = StateSchema.from_type(State)

    state, written = merge_parallel_updates(
        schema,
        {"count": 0, "messages": []},
        [("a", {"count": 5}), ("b", {"messages": ["b"]})],
    )

    assert state["count"] == 5
    assert state["messages"] == ["b"]
    assert written == {"count", "messages"}


def test_parallel_explicit_replace_conflict_raises() -> None:
    from nodestep import Replace
    from nodestep.exceptions import InvalidUpdateError

    schema = StateSchema.from_type(State)

    with pytest.raises(InvalidUpdateError) as info:
        merge_parallel_updates(
            schema,
            {"messages": []},
            [
                ("a", {"messages": Replace(["a"])}),
                ("b", {"messages": Replace(["b"])}),
            ],
        )

    assert info.value.field == "messages"
    assert info.value.nodes == ["a", "b"]


@pytest.mark.parametrize("replacer", ["a", "b"])
def test_parallel_replace_conflicts_with_an_append_in_either_order(
    replacer: str,
) -> None:
    from nodestep import Replace
    from nodestep.exceptions import InvalidUpdateError

    schema = StateSchema.from_type(State)
    updates: list[tuple[str, dict[str, Any] | None]] = [
        (name, {"messages": Replace([name]) if name == replacer else [name]})
        for name in ("a", "b")
    ]

    with pytest.raises(InvalidUpdateError, match="Replace") as info:
        merge_parallel_updates(schema, {"messages": ["old"]}, updates)

    assert info.value.field == "messages"
    assert info.value.nodes == ["a", "b"]
    assert info.value.replacement


def test_parallel_appends_and_one_node_replace_merge() -> None:
    from nodestep import Replace

    schema = StateSchema.from_type(State)

    state, _ = merge_parallel_updates(
        schema,
        {"messages": ["old"], "count": 0},
        [("a", {"messages": ["a"]}), ("b", {"messages": ["b"], "count": 1})],
    )
    replaced, _ = merge_parallel_updates(
        schema, {"messages": ["old"]}, [("a", {"messages": Replace(["a"])})]
    )

    assert state == {"messages": ["old", "a", "b"], "count": 1}
    assert replaced == {"messages": ["a"]}


def test_diff_states_dynamic_schema_captures_all_updates() -> None:
    schema = StateSchema.from_type(dict)
    before = {"count": 0, "messages": []}
    after = TypedState(count=5, messages=["hi"])
    diff = diff_states(schema, before, after)
    assert diff["count"].value == 5
    assert diff["messages"].value == ["hi"]


def test_diff_states_dynamic_schema_ignores_unchanged_keys() -> None:
    schema = StateSchema.from_type(dict)
    before = {"a": 1, "b": 2}
    after = {"a": 1, "b": 99}
    diff = diff_states(schema, before, after)
    assert "a" not in diff
    assert diff["b"].value == 99


def test_diff_states_dynamic_schema_through_dict_graph() -> None:
    from nodestep.state.integrations import InMemoryStateStore

    @node
    def bump(state: dict) -> dict:
        state["count"] = state.get("count", 0) + 1
        state["messages"] = ["x"]
        return state

    store = InMemoryStateStore()
    graph = Graph(dict, state_store=store).flow(START >> bump, bump >> END)
    result = graph.invoke({"count": 0}, thread_id="dyn-thread")
    assert result.data["count"] == 1
    assert result.data["messages"] == ["x"]

    import asyncio

    reloaded = asyncio.run(graph.load("dyn-thread"))
    assert reloaded["count"] == 1
    assert reloaded["messages"] == ["x"]


def test_typing_extensions_typeddict_keeps_reducers() -> None:
    import typing_extensions

    from nodestep.utils.reducers import add

    class Extended(typing_extensions.TypedDict, total=False):
        log: Annotated[list[str], add]

    @node
    def append_one(state: Extended) -> dict:
        return {"log": ["one"]}

    graph = Graph(Extended).flow(START >> append_one, append_one >> END)

    assert StateSchema.from_type(Extended).kind == "typeddict"
    assert graph.invoke({"log": ["seed"]}).data["log"] == ["seed", "one"]


def test_add_messages_assigns_ids_and_replaces_by_id() -> None:
    from nodestep.chat import HumanMessage

    first = add_messages([], [HumanMessage(content="a")])
    assert first[0].id is not None

    replaced = add_messages(first, [HumanMessage(id=first[0].id, content="b")])
    assert [message.content for message in replaced] == ["b"]


async def test_message_ids_are_stable_across_reloads() -> None:
    from nodestep import InMemoryStateStore
    from nodestep.chat import AIMessage, HumanMessage, Message

    class Chat(BaseModel):
        messages: Annotated[list[Message], add_messages] = Field(default_factory=list)

    @node
    def reply(state: Chat) -> dict:
        return {"messages": [AIMessage(content="pong")]}

    graph = Graph(Chat, state_store=InMemoryStateStore()).flow(
        START >> reply, reply >> END
    )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="ping")]}, thread_id="t"
    )
    first_load = await graph.load("t")
    second_load = await graph.load("t")

    live_ids = [message.id for message in result.data["messages"]]
    assert None not in live_ids
    assert [message.id for message in first_load["messages"]] == live_ids
    assert [message.id for message in second_load["messages"]] == live_ids


def test_diff_treats_a_missing_reducer_list_as_empty() -> None:
    from typing import TypedDict

    from nodestep.utils.reducers import add

    class Logged(TypedDict, total=False):
        log: Annotated[list[str], add]

    schema = StateSchema.from_type(Logged)

    assert diff_states(schema, {}, {"log": ["a"]}) == {"log": ["a"]}


def test_message_diff_matches_by_id() -> None:
    from nodestep.chat import AIMessage, HumanMessage, Message

    class Chat(BaseModel):
        messages: Annotated[list[Message], add_messages] = Field(default_factory=list)

    schema = StateSchema.from_type(Chat)
    question = HumanMessage(id="h1", content="q")
    reply = AIMessage(id="a1", content="r")

    delta = diff_states(
        schema, {"messages": [question]}, {"messages": [question, reply]}
    )

    assert delta == {"messages": [reply]}


@pytest.mark.parametrize(
    ("before", "after", "differ"),
    [
        (0.0, -0.0, True),
        (math.nan, math.nan, False),
        (1.5, 1.5, False),
        (1, 1.0, True),
    ],
)
def test_states_differ_compares_floats_by_their_stored_form(
    before: float, after: float, differ: bool
) -> None:
    schema = StateSchema.from_type(dict)

    assert states_differ(schema, {"x": before}, {"x": after}) is differ
