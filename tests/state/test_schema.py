import functools
import operator
import sys
import textwrap
import warnings
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, NotRequired, TypedDict

import pytest
import typing_extensions
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
)

from nodestep import (
    END,
    START,
    BaseState,
    Graph,
    InMemoryStateStore,
    RemoveMessage,
    Replace,
    add,
    add_messages,
    merge_dict,
    node,
    replace,
)
from nodestep.chat import AIMessage, BaseMessage, HumanMessage, Message
from nodestep.exceptions import (
    GraphConfigError,
    GraphExecutionError,
    ResumeError,
    StateStoreError,
    StateUpdateError,
)
from nodestep.state import StateSchema


@node
def noop(state: Any) -> None:
    return None


def concat(current: list[int] | None, update: list[int]) -> list[int]:
    return [*(current or []), *update]


class Concat:
    def __call__(self, current: list[int] | None, update: list[int]) -> list[int]:
        return concat(current, update)


def test_two_reducers_on_one_field_raise() -> None:
    class TwoReducers(TypedDict):
        items: Annotated[list[str], merge_dict, add]

    with pytest.raises(
        GraphConfigError, match=r"'items' of TwoReducers.*more than one reducer"
    ):
        StateSchema.from_type(TwoReducers)


def test_a_class_in_annotated_metadata_is_not_a_reducer() -> None:
    class Marker:
        pass

    class Tagged(TypedDict):
        tagged: Annotated[str, Marker]

    with pytest.raises(GraphConfigError, match=r"'tagged' of Tagged.*Marker"):
        StateSchema.from_type(Tagged)


@pytest.mark.parametrize(
    "reducer",
    [
        lambda only_one: only_one,
        lambda *values: values,
        lambda current, update, extra: update,
        lambda current, update, *, strict: update,
    ],
)
def test_a_reducer_takes_exactly_two_positional_parameters(reducer: Any) -> None:
    class Bad(TypedDict):
        values: Annotated[list[int], reducer]

    with pytest.raises(GraphConfigError, match=r"'values' of Bad.*two positional"):
        StateSchema.from_type(Bad)


def test_callables_with_two_positional_parameters_are_reducers() -> None:
    callable_instance = Concat()
    concat_partial = functools.partial(concat)

    class Reducers(TypedDict):
        plain: Annotated[list[int], concat]
        operator_add: Annotated[list[int], operator.add]
        instance: Annotated[list[int], callable_instance]
        partial: Annotated[list[int], concat_partial]
        constrained: Annotated[int, Field(ge=0)]

    fields = StateSchema.from_type(Reducers).fields

    assert fields["plain"].reducer is concat
    assert fields["operator_add"].reducer is operator.add
    assert fields["instance"].reducer is callable_instance
    assert fields["partial"].reducer is concat_partial
    assert fields["constrained"].reducer is replace


def test_a_reducer_inside_optional_raises_for_typeddict() -> None:
    class Hidden(TypedDict):
        items: Annotated[list[str], add] | None

    with pytest.raises(GraphConfigError, match="put Optional inside Annotated"):
        StateSchema.from_type(Hidden)


def test_a_reducer_inside_a_union_raises_for_pydantic() -> None:
    class Hidden(BaseModel):
        items: Annotated[list[str], add] | int = 0

    with pytest.raises(GraphConfigError, match=r"'items' of Hidden.*put Optional"):
        StateSchema.from_type(Hidden)


def test_optional_inside_annotated_keeps_the_reducer() -> None:
    class Visible(BaseModel):
        items: Annotated[list[str] | None, add] = None

    assert StateSchema.from_type(Visible).fields["items"].reducer is add


def test_typeddict_qualifiers_do_not_hide_the_reducer() -> None:
    class Qualified(TypedDict, total=False):
        outer: NotRequired[Annotated[list[str], add]]
        inner: Annotated[NotRequired[list[str]], add]

    schema = StateSchema.from_type(Qualified)

    assert schema.fields["outer"].reducer is add
    assert schema.fields["inner"].reducer is add
    assert schema.restore_state({"outer": ["a"], "inner": ["b"]}) == {
        "outer": ["a"],
        "inner": ["b"],
    }


def test_typeddict_validation_keeps_constraints_and_validators() -> None:
    class Constrained(TypedDict):
        n: Annotated[int, Field(ge=0)]
        name: Annotated[str, AfterValidator(str.upper)]
        log: Annotated[list[int], add, Field(max_length=2)]

    schema = StateSchema.from_type(Constrained)

    assert schema.fields["log"].reducer is add
    assert schema.restore_state({"n": 1, "name": "ada", "log": [1]}) == {
        "n": 1,
        "name": "ADA",
        "log": [1],
    }
    with pytest.raises(ResumeError, match="'n'") as below_zero:
        schema.restore_state({"n": -5})
    with pytest.raises(ResumeError, match="'log'") as too_long:
        schema.restore_state({"log": [1, 2, 3]})
    assert isinstance(below_zero.value.__cause__, ValidationError)
    assert isinstance(too_long.value.__cause__, ValidationError)


async def test_update_state_checks_pydantic_field_constraints() -> None:
    class NonNegative(BaseModel):
        n: Annotated[int, Field(ge=0)] = 0

    graph = Graph(NonNegative, state_store=InMemoryStateStore()).flow(
        START >> noop, noop >> END
    )
    await graph.ainvoke({}, thread_id="g")

    with pytest.raises(StateUpdateError, match="'n'"):
        await graph.update_state("g", {"n": -5})

    assert await graph.load("g") == {"n": 0}


class Cat(BaseModel):
    kind: Literal["cat"] = "cat"


class Dog(BaseModel):
    kind: Literal["dog"] = "dog"


class Pets(BaseModel):
    pet: Cat | Dog = Field(default_factory=Cat, discriminator="kind")


def test_pydantic_validation_keeps_the_field_discriminator() -> None:
    schema = StateSchema.from_type(Pets)

    assert isinstance(schema.restore_state({"pet": {"kind": "dog"}})["pet"], Dog)
    with pytest.raises(ResumeError) as info:
        schema.restore_state({"pet": {"kind": "bird"}})
    cause = info.value.__cause__
    assert isinstance(cause, ValidationError)
    assert cause.errors()[0]["type"] == "union_tag_invalid"


class Private(BaseState):
    x: int = 0
    _cache: dict[str, int] = PrivateAttr(default_factory=dict)
    VERSION: ClassVar[int] = 1


def test_pydantic_fields_skip_class_vars_and_private_attributes() -> None:
    assert list(StateSchema.from_type(Private).fields) == ["x"]


@pytest.mark.parametrize("key", ["_cache", "VERSION"])
async def test_updates_to_private_attributes_and_class_vars_raise(key: str) -> None:
    @node
    def write(state: Private) -> dict:
        return {key: 2}

    graph = Graph(Private, state_store=InMemoryStateStore()).flow(
        START >> write, write >> END
    )

    with pytest.raises(StateUpdateError, match=f"'{key}'"):
        await graph.ainvoke({}, thread_id="p")


class NotTypedDict(dict):
    messages: Annotated[list[Message], add_messages]


class EmptyTypedDict(TypedDict):
    pass


class EmptyModel(BaseModel):
    pass


class ExtraAllowed(BaseModel):
    model_config = ConfigDict(extra="allow")

    note: str = ""


@pytest.mark.parametrize(
    "schema",
    [
        None,
        Any,
        NotTypedDict,
        dict[str, Any],
        int,
        EmptyTypedDict,
        EmptyModel,
        BaseState,
    ],
)
def test_only_dict_itself_means_untyped_state(schema: Any) -> None:
    with pytest.raises(GraphConfigError):
        StateSchema.from_type(schema)


def test_a_schema_without_fields_names_the_alternative() -> None:
    with pytest.raises(GraphConfigError, match=r"EmptyModel declares no fields.*dict"):
        StateSchema.from_type(EmptyModel)


def test_a_pydantic_schema_that_allows_extra_attributes_raises() -> None:
    with pytest.raises(GraphConfigError, match=r"ExtraAllowed.*extra='allow'"):
        StateSchema.from_type(ExtraAllowed)


def test_graph_rejects_a_dict_subclass() -> None:
    with pytest.raises(GraphConfigError, match="NotTypedDict"):
        Graph(NotTypedDict)


def test_dict_is_untyped_state() -> None:
    schema = StateSchema.from_type(dict)

    assert schema.kind == "dict"
    assert schema.is_dynamic()
    assert schema.fields == {}


def wrapped_messages(current: Any, update: Any) -> list[Any]:
    return add_messages(current, update)


class Wrapped(BaseModel):
    history: Annotated[list[Message], wrapped_messages] = Field(default_factory=list)
    last: Message | None = None


@node
def reply(state: Wrapped) -> dict:
    return {"history": [AIMessage(content="pong")], "last": AIMessage(content="p")}


async def test_messages_get_ids_regardless_of_the_reducer() -> None:
    graph = Graph(Wrapped, state_store=InMemoryStateStore()).flow(
        START >> reply, reply >> END
    )

    result = await graph.ainvoke(
        {"history": [HumanMessage(content="ping")]}, thread_id="t"
    )
    live = [message.id for message in result.state.history]
    last = result.state.last

    assert None not in live
    assert last is not None
    assert last.id is not None
    for _ in range(2):
        loaded = await graph.load("t")
        assert [message.id for message in loaded["history"]] == live
        assert loaded["last"].id == last.id


async def test_messages_in_untyped_state_are_refused() -> None:
    @node
    def answer(state: dict) -> dict:
        return {"chat_history": [AIMessage(content="pong")]}

    graph = Graph(dict).flow(START >> answer, answer >> END)

    with pytest.raises(
        StateStoreError, match=r"'chat_history'.*node 'answer'.*untyped state"
    ):
        await graph.ainvoke({"other": 1})


class Reduced(TypedDict):
    log: Annotated[list[str], concat]
    meta: Annotated[dict[str, int], merge_dict]
    count: int


@node
def append_in_place(state: Reduced) -> Reduced:
    state["log"].append("x")
    return state


async def test_mutate_and_return_on_a_custom_reducer_field_raises() -> None:
    graph = Graph(Reduced).flow(START >> append_in_place, append_in_place >> END)

    with pytest.raises(
        GraphExecutionError,
        match=r"'append_in_place'.*field 'log' has a custom reducer; return an update",
    ):
        await graph.ainvoke({"log": ["a"], "meta": {}, "count": 0})


@node
def edit_meta(state: Reduced) -> Reduced:
    del state["meta"]["drop_me"]
    state["meta"]["new"] = 3
    state["count"] += 1
    return state


async def test_mutate_and_return_on_merge_dict_keeps_deleted_keys_deleted() -> None:
    graph = Graph(Reduced).flow(START >> edit_meta, edit_meta >> END)

    result = await graph.ainvoke(
        {"log": ["a"], "meta": {"keep": 1, "drop_me": 2}, "count": 0}
    )

    assert result.data == {"log": ["a"], "meta": {"keep": 1, "new": 3}, "count": 1}


class Log(BaseModel):
    log: Annotated[list[str], add] = Field(default_factory=list)
    meta: dict[str, int] = Field(default_factory=dict)


async def test_replace_bypasses_the_reducer_and_survives_a_reload() -> None:
    from nodestep import Replace

    @node
    def reset(state: Log) -> dict:
        return {"log": Replace(["fresh"])}

    graph = Graph(Log, state_store=InMemoryStateStore()).flow(
        START >> reset, reset >> END
    )

    result = await graph.ainvoke({"log": ["a", "b"]}, thread_id="r")
    assert result.data["log"] == ["fresh"]
    assert (await graph.load("r"))["log"] == ["fresh"]

    await graph.update_state("r", {"log": Replace([])})
    assert (await graph.load("r"))["log"] == []


async def test_the_stored_replace_form_is_refused_in_update_state() -> None:
    graph = Graph(Log, state_store=InMemoryStateStore()).flow(
        START >> noop, noop >> END
    )
    await graph.ainvoke({"log": ["a"]}, thread_id="m")

    with pytest.raises(StateUpdateError, match=r"__nodestep_replace__.*Replace"):
        await graph.update_state("m", {"log": {"__nodestep_replace__": []}})

    assert (await graph.load("m"))["log"] == ["a"]


async def test_the_stored_replace_form_is_refused_in_node_updates() -> None:
    @node
    def sneaky(state: Log) -> dict:
        return {"meta": {"__nodestep_replace__": 1}}

    graph = Graph(Log, state_store=InMemoryStateStore()).flow(
        START >> sneaky, sneaky >> END
    )

    with pytest.raises(StateUpdateError, match=r"'meta'.*Replace"):
        await graph.ainvoke({}, thread_id="n")


class Tagged(TypedDict):
    tags: Annotated[list[str], add]
    messages: Annotated[list[Message], add_messages]


class TaggedModel(BaseModel):
    tags: Annotated[list[str], add] = Field(default_factory=list)
    messages: Annotated[list[Message], add_messages] = Field(default_factory=list)


def _tagged(schema: Any) -> Graph:
    return Graph(schema, state_store=InMemoryStateStore()).flow(
        START >> noop, noop >> END
    )


@pytest.mark.parametrize("schema", [Tagged, TaggedModel, dict])
async def test_replace_in_the_input_of_a_new_thread_sets_the_value(
    schema: Any,
) -> None:
    graph = _tagged(schema)

    result = await graph.ainvoke({"tags": Replace(["input"])}, thread_id="new")

    assert result.data["tags"] == ["input"]
    assert (await graph.load("new"))["tags"] == ["input"]


@pytest.mark.parametrize("schema", [Tagged, TaggedModel])
async def test_remove_message_in_the_input_of_a_new_thread_raises(
    schema: Any,
) -> None:
    graph = _tagged(schema)

    with pytest.raises(
        StateUpdateError, match=r"'messages'.*RemoveMessage\('x'\) names no message"
    ):
        await graph.ainvoke({"messages": [RemoveMessage("x")]}, thread_id="new")

    assert (await graph.history("new")).events == []


@pytest.mark.parametrize("schema", [Tagged, TaggedModel, dict])
async def test_replace_in_the_input_of_an_existing_thread_bypasses_the_reducer(
    schema: Any,
) -> None:
    graph = _tagged(schema)
    await graph.ainvoke({"tags": ["a"], "messages": []}, thread_id="t")

    result = await graph.ainvoke({"tags": Replace(["b"])}, thread_id="t")

    assert result.data["tags"] == ["b"]
    assert (await graph.load("t"))["tags"] == ["b"]


@pytest.mark.parametrize("schema", [Tagged, TaggedModel])
async def test_remove_message_in_the_input_of_an_existing_thread_removes_it(
    schema: Any,
) -> None:
    graph = _tagged(schema)
    await graph.ainvoke(
        {
            "tags": [],
            "messages": [
                HumanMessage(id="q", content="hi"),
                AIMessage(id="a", content="hello"),
            ],
        },
        thread_id="t",
    )

    result = await graph.ainvoke(
        {"messages": [RemoveMessage("q"), HumanMessage(id="n", content="next")]},
        thread_id="t",
    )

    assert [message.id for message in result.data["messages"]] == ["a", "n"]
    loaded = await graph.load("t")
    assert [message.id for message in loaded["messages"]] == ["a", "n"]


@pytest.mark.parametrize("thread", ["new", "t"])
@pytest.mark.parametrize(
    ("values", "match"),
    [
        ({"tags": [RemoveMessage("q")]}, r"'tags'.*add_messages reducer"),
        ({"messages": Replace([RemoveMessage("q")])}, r"'messages'.*inside Replace"),
    ],
)
async def test_misplaced_remove_message_in_the_input_raises(
    thread: str, values: dict[str, Any], match: str
) -> None:
    graph = _tagged(TaggedModel)
    await graph.ainvoke(
        {"messages": [HumanMessage(id="q", content="hi")]}, thread_id="t"
    )

    with pytest.raises(StateUpdateError, match=match):
        await graph.ainvoke(values, thread_id=thread)

    assert [message.id for message in (await graph.load("t"))["messages"]] == ["q"]


async def test_a_remove_message_error_names_its_cause() -> None:
    graph = _tagged(TaggedModel)
    await graph.ainvoke({}, thread_id="u")

    with pytest.raises(StateUpdateError, match="add_messages reducer") as info:
        await graph.update_state("u", {"tags": [RemoveMessage("x")]})
    assert "Replace" not in str(info.value)

    with pytest.raises(StateUpdateError, match="inside Replace"):
        await graph.update_state("u", {"messages": Replace([RemoveMessage("x")])})


@dataclass
class Widget:
    kind: str


class Form(TypedDict):
    n: Annotated[int, {"ui": "slider"}]
    text: Annotated[str, Widget("textarea")]


class FormModel(BaseModel):
    n: Annotated[int, {"ui": "slider"}] = 0
    text: Annotated[str, Widget("textarea")] = ""


@node
def bump(state: Any) -> dict:
    return {"n": 2}


@pytest.mark.parametrize("schema", [Form, FormModel])
async def test_unhashable_field_metadata_keeps_a_stored_thread_usable(
    schema: Any,
) -> None:
    graph = Graph(schema, state_store=InMemoryStateStore()).flow(
        START >> bump, bump >> END
    )
    await graph.ainvoke({"n": 1, "text": "a"}, thread_id="w")

    await graph.update_state("w", {"text": "b"})
    await graph.ainvoke({"n": 5}, thread_id="w")

    assert await graph.load("w") == {"n": 2, "text": "b"}


class UntypedItems(TypedDict):
    messages: Annotated[list, add_messages]


class AnyItems(BaseModel):
    messages: Annotated[list[Any], add_messages] = Field(default_factory=list)


class DictItems(TypedDict):
    messages: Annotated[list[dict[str, Any]], add_messages]


class MixedItems(TypedDict):
    messages: Annotated[list[Message | dict[str, Any]], add_messages]


class AbstractItems(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class NotAList(TypedDict):
    messages: Annotated[Any, add_messages]


@pytest.mark.parametrize(
    "schema",
    [UntypedItems, AnyItems, DictItems, MixedItems, AbstractItems, NotAList],
)
def test_add_messages_needs_a_field_typed_as_a_list_of_chat_messages(
    schema: Any,
) -> None:
    with pytest.raises(
        GraphConfigError,
        match=rf"'messages' of {schema.__name__} uses add_messages.*list\[Message\]",
    ):
        Graph(schema)


class HumanOnly(HumanMessage):
    pass


class NarrowItems(TypedDict):
    messages: Annotated[list[HumanMessage | AIMessage], add_messages]


class OptionalItems(BaseModel):
    messages: Annotated[list[Message] | None, add_messages] = None


class SubclassItems(TypedDict):
    messages: Annotated[list[HumanOnly], add_messages]


@pytest.mark.parametrize("schema", [NarrowItems, OptionalItems, SubclassItems])
def test_add_messages_accepts_lists_of_concrete_chat_messages(schema: Any) -> None:
    assert StateSchema.from_type(schema).fields["messages"].reducer is add_messages


class ExtensionReadOnly(typing_extensions.TypedDict):
    outer: typing_extensions.ReadOnly[Annotated[list[str], add]]
    inner: Annotated[typing_extensions.ReadOnly[list[str]], add]


def test_typing_extensions_read_only_does_not_hide_the_reducer() -> None:
    schema = StateSchema.from_type(ExtensionReadOnly)

    assert schema.fields["outer"].reducer is add
    assert schema.fields["inner"].reducer is add
    assert schema.restore_state({"outer": ["a"], "inner": ["b"]}) == {
        "outer": ["a"],
        "inner": ["b"],
    }


@pytest.mark.parametrize(
    "marker", [str.strip, lambda current, update=None: update], ids=["strip", "default"]
)
def test_a_reducer_parameter_with_a_default_does_not_count(marker: Any) -> None:
    class Stripped(TypedDict):
        text: Annotated[str, marker]

    with pytest.raises(
        GraphConfigError,
        match=r"'text' of Stripped.*two positional parameters without defaults",
    ):
        StateSchema.from_type(Stripped)


def test_callable_metadata_that_is_not_a_reducer_says_it_was_read_as_one() -> None:
    class Deprecated(TypedDict):
        n: Annotated[int, typing_extensions.deprecated("old")]

    with pytest.raises(
        GraphConfigError,
        match=r"callable .* in the Annotated metadata of state field 'n' of "
        r"Deprecated is read as the field's reducer",
    ):
        StateSchema.from_type(Deprecated)


class Legacy(BaseModel):
    old: str = Field(default="", deprecated="use new")


class Renamed(BaseModel):
    old: int = Field(default=0, deprecated="use new")
    new: int = 0
    legacy: Legacy = Field(default_factory=Legacy)


@node
def bumps_new(state: Renamed) -> dict:
    return {"new": state.new + 1}


async def test_field_deprecated_is_supported_and_nodestep_does_not_read_it() -> None:
    graph = Graph(Renamed, state_store=InMemoryStateStore()).flow(
        START >> bumps_new, bumps_new >> END
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = await graph.ainvoke({}, thread_id="t")
        again = await graph.ainvoke({"new": 5}, thread_id="t")
        loaded = await graph.load("t")

    assert result.data == {"old": 0, "new": 1, "legacy": Legacy()}
    assert again.data == {"old": 0, "new": 6, "legacy": Legacy()}
    assert loaded == {"old": 0, "new": 6, "legacy": Legacy()}


@pytest.mark.skipif(
    sys.version_info < (3, 14), reason="annotations are evaluated lazily from 3.14"
)
def test_an_undefined_name_in_a_lazy_annotation_names_the_field() -> None:
    namespace: dict[str, Any] = {"__name__": "lazy_schema"}
    exec(
        textwrap.dedent(
            """
            import operator
            from typing import Annotated, TypedDict

            class Lazy(TypedDict):
                ok: int
                items: Annotated[list[Undefined], operator.add]
            """
        ),
        namespace,
    )

    with pytest.raises(GraphConfigError, match="state field 'items' of Lazy"):
        StateSchema.from_type(namespace["Lazy"])
