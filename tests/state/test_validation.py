import json
import operator
import threading
from datetime import datetime
from typing import Annotated, Any, Self, TypedDict
from uuid import uuid4

import pytest
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
    with_config,
)

from nodestep import (
    END,
    START,
    Command,
    Graph,
    InMemoryStateStore,
    RemoveMessage,
    Resume,
    Send,
    add,
    add_messages,
    interrupt,
    merge_dict,
    node,
    replace,
)
from nodestep.chat.messages import HumanMessage, Message
from nodestep.exceptions import (
    GraphConfigError,
    ResumeError,
    StateStoreError,
    StateUpdateError,
)
from nodestep.state.schema import StateSchema


class Strict(BaseModel):
    count: int = 0


class StrictDict(TypedDict, total=False):
    count: int


@node
def bad_write(state: Any) -> dict:
    return {"count": "abc"}


@node
def noop(state: Any) -> None:
    return None


def _events(history: Any) -> list[tuple[str, str | None]]:
    return [(event.type, event.node) for event in history.events]


async def test_a_node_writing_a_wrong_type_fails_naming_node_and_field() -> None:
    graph = Graph(Strict, state_store=InMemoryStateStore()).flow(
        START >> bad_write, bad_write >> END
    )

    with pytest.raises(StateUpdateError, match=r"'count'.*node 'bad_write'") as info:
        await graph.ainvoke({}, thread_id="s")

    assert isinstance(info.value.__cause__, ValidationError)
    events = _events(await graph.history("s"))
    assert ("error", "bad_write") in events
    assert ("node_completed", "bad_write") not in events
    assert await graph.load("s") == {"count": 0}
    await graph.update_state("s", {"count": 1})
    assert (await graph.get_state("s")).value.count == 1


@pytest.mark.parametrize("schema", [Strict, StrictDict])
async def test_written_values_are_stored_validated(schema: Any) -> None:
    @node
    def writes_str(state: Any) -> dict:
        return {"count": "5"}

    graph = Graph(schema, state_store=InMemoryStateStore()).flow(
        START >> writes_str, writes_str >> END
    )

    result = await graph.ainvoke({"count": 0}, thread_id="t")

    assert result.data["count"] == 5
    state = result.state
    assert (state.count if isinstance(state, BaseModel) else state["count"]) == 5
    assert (await graph.load("t"))["count"] == 5
    deltas = [
        json.loads(event.data_json)["update"]
        for event in (await graph.history("t")).events
        if event.type == "state_delta" and event.data_json is not None
    ]
    assert deltas[-1] == {"count": 5}


class Items(TypedDict, total=False):
    items: Annotated[list[str], add]
    messages: Annotated[list[Message], add_messages]


attempts: list[str] = []


@node
def flaky_add(state: Items) -> dict:
    attempts.append("flaky_add")
    if len(attempts) == 1:
        return {"items": "x"}
    return {"items": ["x"]}


@node
def remove_unknown(state: Items) -> dict:
    return {"messages": [RemoveMessage("nope")]}


async def test_a_reducer_error_fails_the_node_and_a_continuation_runs_it_again() -> (
    None
):
    attempts.clear()
    graph = Graph(Items, state_store=InMemoryStateStore()).flow(
        START >> flaky_add, flaky_add >> END
    )

    with pytest.raises(StateUpdateError, match=r"'items'.*node 'flaky_add'"):
        await graph.ainvoke({"items": []}, thread_id="t")
    events = _events(await graph.history("t"))
    assert ("error", "flaky_add") in events
    assert ("node_completed", "flaky_add") not in events

    result = await graph.ainvoke(None, thread_id="t")

    assert result.data["items"] == ["x"]
    assert attempts == ["flaky_add", "flaky_add"]


async def test_removing_an_unknown_message_names_the_node_and_field() -> None:
    graph = Graph(Items, state_store=InMemoryStateStore()).flow(
        START >> remove_unknown, remove_unknown >> END
    )

    with pytest.raises(
        StateUpdateError, match=r"'messages'.*node 'remove_unknown'.*nope"
    ):
        await graph.ainvoke({"messages": []}, thread_id="t")

    assert ("error", "remove_unknown") in _events(await graph.history("t"))


class Bounded(TypedDict, total=False):
    log: Annotated[list[int], add, Field(max_length=3)]


@node
def left(state: Bounded) -> dict:
    return {"log": [1, 2]}


@node
def right(state: Bounded) -> dict:
    return {"log": [3, 4]}


@node(goto=[left, right])
def fan(state: Bounded) -> Command:
    return Command(goto=[left, right])


async def test_parallel_writes_that_together_break_a_constraint_raise() -> None:
    graph = Graph(Bounded, state_store=InMemoryStateStore()).flow(
        START >> fan, left >> END, right >> END
    )

    with pytest.raises(StateUpdateError, match=r"'log'.*nodes 'left', 'right'"):
        await graph.ainvoke({"log": []}, thread_id="t")

    assert ("superstep_failed", None) in _events(await graph.history("t"))
    assert await graph.load("t") == {"log": []}


bang_calls: list[str] = []


def bang(value: str) -> str:
    bang_calls.append(value)
    return value if value.endswith("!") else value + "!"


class Bang(BaseModel):
    text: Annotated[str, AfterValidator(bang)]


class BangDict(TypedDict, total=False):
    text: Annotated[str, AfterValidator(bang)]


views: list[str] = []


@node
def show(state: Any) -> None:
    views.append(state.text if isinstance(state, BaseModel) else state["text"])
    return None


@pytest.mark.parametrize("schema", [Bang, BangDict])
async def test_validators_run_on_a_value_and_its_stored_form_but_not_on_reads(
    schema: Any,
) -> None:
    bang_calls.clear()
    views.clear()
    graph = Graph(schema).flow(START >> show, show >> END)

    result = await graph.ainvoke({"text": "hi"})

    assert views == ["hi!"]
    assert result.data["text"] == "hi!"
    state = result.state
    assert (state.text if isinstance(state, BaseModel) else state["text"]) == "hi!"
    assert bang_calls == ["hi", "hi!"]


upper_calls: list[str] = []


def counted_upper(value: str) -> str:
    upper_calls.append(value)
    return value.upper()


class Upper(BaseModel):
    name: Annotated[str, AfterValidator(counted_upper)] = ""


async def test_load_validates_each_stored_value_once() -> None:
    graph = Graph(Upper, state_store=InMemoryStateStore()).flow(
        START >> noop, noop >> END
    )
    await graph.ainvoke({"name": "ada"}, thread_id="t")
    upper_calls.clear()

    assert await graph.load("t") == {"name": "ADA"}
    assert upper_calls == ["ADA"]


class Known(BaseModel):
    count: int = 0


class KnownDict(TypedDict, total=False):
    count: int


@pytest.mark.parametrize("schema", [Known, KnownDict])
@pytest.mark.parametrize("existing", [False, True])
async def test_unknown_input_keys_raise_on_every_thread(
    schema: Any, existing: bool
) -> None:
    graph = Graph(schema, state_store=InMemoryStateStore()).flow(
        START >> noop, noop >> END
    )
    if existing:
        await graph.ainvoke({"count": 1}, thread_id="k")

    with pytest.raises(StateUpdateError, match="Unknown field 'cuont'"):
        await graph.ainvoke({"count": 1, "cuont": 2}, thread_id="k")


@pytest.mark.parametrize("existing", [False, True])
async def test_typeddict_input_is_coerced_on_every_thread(existing: bool) -> None:
    graph = Graph(KnownDict, state_store=InMemoryStateStore()).flow(
        START >> noop, noop >> END
    )
    if existing:
        await graph.ainvoke({"count": 1}, thread_id="t")

    result = await graph.ainvoke({"count": "5"}, thread_id="t")

    assert result.data["count"] == 5
    with pytest.raises(StateUpdateError, match=r"'count'.*the input"):
        await graph.ainvoke({"count": "not a number"}, thread_id="t")


def unique(current: list[str] | None, update: list[str]) -> list[str]:
    out = list(current or [])
    for item in update:
        if item not in out:
            out.append(item)
    return out


class Tags(BaseModel):
    tags: Annotated[list[str], unique] = Field(default_factory=list)


class TagsDict(TypedDict, total=False):
    tags: Annotated[list[str], unique]


@pytest.mark.parametrize("schema", [Tags, TagsDict])
async def test_input_on_a_new_thread_goes_through_the_reducers(schema: Any) -> None:
    graph = Graph(schema).flow(START >> noop, noop >> END)

    assert (await graph.ainvoke({"tags": ["a", "a"]})).data["tags"] == ["a"]


class Summed(TypedDict, total=False):
    log: Annotated[list[str], operator.add]


def test_a_reducer_gets_none_for_a_field_without_a_value() -> None:
    graph = Graph(Summed).flow(START >> noop, noop >> END)

    with pytest.raises(
        StateUpdateError,
        match=r"'log'.*the input.*got None as the current value",
    ):
        graph.invoke({"log": ["a"]})


def union_tags(current: set[str], update: set[str]) -> set[str]:
    return current.union(update)


class TagSet(TypedDict, total=False):
    tags: Annotated[set[str], union_tags]


@node
def tags_it(state: TagSet) -> dict:
    return {"tags": {"x"}}


@pytest.mark.parametrize(
    ("value", "writer"), [({}, "node 'tags_it'"), ({"tags": {"a"}}, "the input")]
)
def test_a_custom_reducer_that_fails_on_its_first_write_says_it_got_none(
    value: dict[str, Any], writer: str
) -> None:
    graph = Graph(TagSet).flow(START >> tags_it, tags_it >> END)

    with pytest.raises(
        StateUpdateError,
        match=(
            rf"^State field 'tags' got an invalid value from {writer}: "
            r"AttributeError: .*; the field had no value yet, so its reducer "
            r"\S+\.union_tags got None as the current value; make the reducer "
            r"handle None, or use one that does, such as nodestep\.add$"
        ),
    ):
        graph.invoke(value)


class OperatorLog(TypedDict, total=False):
    log: Annotated[list[str], operator.add]


def test_a_first_write_error_names_the_reducer_with_its_module() -> None:
    graph = Graph(OperatorLog).flow(START >> noop, noop >> END)

    with pytest.raises(StateUpdateError, match=r"its reducer _operator\.add got None"):
        graph.invoke({"log": ["a"]})


def set_once(current: list[str] | None, update: list[str]) -> list[str]:
    if current is not None:
        raise TypeError("the field is set once")
    return update


class Once(TypedDict, total=False):
    log: Annotated[list[str], set_once]


def test_a_custom_reducer_error_on_a_field_with_a_value_has_no_none_hint() -> None:
    schema = StateSchema.from_type(Once)

    with pytest.raises(
        StateUpdateError,
        match=(
            r"^State field 'log' got an invalid value from test: TypeError: the "
            r"field is set once$"
        ),
    ):
        schema.validate_update({"log": ["a"]}, {"log": ["b"]}, writer="test")


class FirstWrites(TypedDict, total=False):
    log: Annotated[list[str], add]
    messages: Annotated[list[Message], add_messages]
    meta: Annotated[dict[str, int], merge_dict]
    plain: Annotated[int, replace]


def test_nodestep_reducers_take_the_first_write_of_a_field_without_a_value() -> None:
    graph = Graph(FirstWrites).flow(START >> noop, noop >> END)

    result = graph.invoke(
        {
            "log": ["a"],
            "messages": [HumanMessage(content="hi")],
            "meta": {"k": 1},
            "plain": 2,
        }
    )

    assert result.data["log"] == ["a"]
    assert [message.content for message in result.data["messages"]] == ["hi"]
    assert result.data["meta"] == {"k": 1}
    assert result.data["plain"] == 2


class Session(BaseModel):
    token: str = Field(default_factory=lambda: uuid4().hex)
    answer: str = ""


@node
def ask_token(state: Session) -> dict:
    return {"answer": interrupt("ok?", id="ok")}


async def test_generated_defaults_of_a_new_thread_survive_a_pause() -> None:
    graph = Graph(Session, state_store=InMemoryStateStore()).flow(
        START >> ask_token, ask_token >> END
    )

    paused = await graph.ainvoke({}, thread_id="t")

    assert (await graph.load("t"))["token"] == paused.data["token"]
    resumed = await graph.ainvoke(None, thread_id="t", resume=Resume("yes"))
    assert resumed.data == {"token": paused.data["token"], "answer": "yes"}


class Counter(BaseModel):
    count: int = 0
    note: str = ""


async def test_a_model_instance_as_input_is_the_full_state() -> None:
    graph = Graph(Counter, state_store=InMemoryStateStore()).flow(
        START >> noop, noop >> END
    )
    await graph.ainvoke({"note": "keep me"}, thread_id="t")

    result = await graph.ainvoke(Counter(count=5), thread_id="t")

    assert result.data == {"count": 5, "note": ""}


class Journal(BaseModel):
    log: Annotated[list[str], add] = Field(default_factory=list)
    messages: Annotated[list[Message], add_messages] = Field(default_factory=list)
    count: int = 0


async def test_a_model_instance_as_input_replaces_reducer_fields_too() -> None:
    graph = Graph(Journal, state_store=InMemoryStateStore()).flow(
        START >> noop, noop >> END
    )
    await graph.ainvoke(
        {"log": ["old"], "messages": [HumanMessage(content="old")]}, thread_id="t"
    )

    result = await graph.ainvoke(
        Journal(log=["only"], messages=[HumanMessage(content="new")]), thread_id="t"
    )

    assert result.data["log"] == ["only"]
    assert [message.content for message in result.state.messages] == ["new"]
    stored = await graph.load("t")
    assert stored["log"] == ["only"]
    assert [message.content for message in stored["messages"]] == ["new"]
    assert stored["messages"][0].id is not None
    assert stored["messages"][0].id == result.state.messages[0].id


async def test_passing_result_state_back_as_input_changes_nothing() -> None:
    graph = Graph(Journal, state_store=InMemoryStateStore()).flow(
        START >> noop, noop >> END
    )
    first = await graph.ainvoke(
        {"log": ["a"], "messages": [HumanMessage(content="hi")], "count": 1},
        thread_id="t",
    )

    again = await graph.ainvoke(first.state, thread_id="t")

    assert again.data == first.data
    assert await graph.load("t") == first.data


class Other(BaseModel):
    count: int = 0


@pytest.mark.parametrize("schema", [Counter, KnownDict])
async def test_a_model_of_another_type_as_input_raises(schema: Any) -> None:
    graph = Graph(schema).flow(START >> noop, noop >> END)

    with pytest.raises(StateUpdateError, match="Other"):
        await graph.ainvoke(Other(count=1))


class Cat(BaseModel):
    name: str


class Dog(BaseModel):
    name: str


class Pets(BaseModel):
    pets: Annotated[list[Cat | Dog], add] = Field(default_factory=list)


@node
def rename_pet(state: Pets) -> Pets:
    state.pets[0].name = "max"
    return state


def test_the_diff_compares_the_stored_form_of_every_item() -> None:
    graph = Graph(Pets).flow(START >> rename_pet, rename_pet >> END)

    pets = graph.invoke({"pets": [Cat(name="rex")]}).data["pets"]

    assert pets == [Cat(name="max")]


class V1(BaseModel):
    a: int = 0
    legacy: str = "old"


class V2(BaseModel):
    a: int = 0
    added: list[str] = Field(default_factory=list)


@node
def bump(state: Any) -> dict:
    return {"a": 1}


async def test_loading_stored_keys_the_schema_does_not_declare_raises() -> None:
    store = InMemoryStateStore()
    await (
        Graph(V1, state_store=store)
        .flow(START >> bump, bump >> END)
        .ainvoke({"legacy": "kept?"}, thread_id="t")
    )
    graph = Graph(V2, state_store=store).flow(START >> bump, bump >> END)

    with pytest.raises(StateStoreError, match="'legacy'"):
        await graph.load("t")
    with pytest.raises(StateStoreError, match="'legacy'"):
        await graph.ainvoke({}, thread_id="t")


class Small(BaseModel):
    a: int = 0


async def test_loading_fills_the_defaults_of_new_fields() -> None:
    store = InMemoryStateStore()
    await (
        Graph(Small, state_store=store)
        .flow(START >> bump, bump >> END)
        .ainvoke({}, thread_id="t")
    )
    graph = Graph(V2, state_store=store).flow(START >> bump, bump >> END)

    assert await graph.load("t") == {"a": 1, "added": []}
    assert (await graph.ainvoke({}, thread_id="t")).data == {"a": 1, "added": []}


factory_calls: list[str] = []


def counted_token() -> str:
    factory_calls.append("token")
    return "fresh"


class Stamped(BaseModel):
    token: str = Field(default_factory=counted_token)
    added: list[str] = Field(default_factory=list)


def test_restoring_a_state_calls_only_the_default_factories_of_missing_fields() -> None:
    schema = StateSchema.from_type(Stamped)
    factory_calls.clear()

    assert schema.restore_state({"token": "kept"}) == {"token": "kept", "added": []}
    assert factory_calls == []
    assert schema.restore_state({"added": []}) == {"token": "fresh", "added": []}
    assert factory_calls == ["token"]


def test_restore_update_raises_for_a_value_that_does_not_fit() -> None:
    schema = StateSchema.from_type(V2)

    with pytest.raises(ResumeError, match="'a'"):
        schema.restore_update({"a": "not-an-int"})


class Named(BaseModel):
    a: str | None = None


class Numbered(BaseModel):
    a: int | None = None


@node
def write_text(state: Any) -> dict:
    return {"a": "text"}


@node
def interrupting(state: Any) -> None:
    interrupt("ok?", id="ok")


async def test_a_stored_update_that_no_longer_fits_raises_on_load() -> None:
    store = InMemoryStateStore()
    await (
        Graph(Named, state_store=store)
        .flow(START >> write_text, write_text >> interrupting, interrupting >> END)
        .ainvoke({}, thread_id="t")
    )
    graph = Graph(Numbered, state_store=store).flow(
        START >> write_text, write_text >> interrupting, interrupting >> END
    )

    with pytest.raises(ResumeError, match="'a'"):
        await graph.load("t")


async def test_update_state_stores_the_validated_value() -> None:
    graph = Graph(Strict, state_store=InMemoryStateStore()).flow(
        START >> noop, noop >> END
    )
    await graph.ainvoke({}, thread_id="t")

    await graph.update_state("t", {"count": "7"})
    with pytest.raises(StateUpdateError, match=r"'count'.*update_state"):
        await graph.update_state("t", {"count": "seven"})

    last = (await graph.history("t")).events[-1]
    assert json.loads(last.data_json or "{}")["update"] == {"count": 7}
    assert await graph.load("t") == {"count": 7}


class Note(BaseModel):
    text: str


class Work(TypedDict, total=False):
    note: Note | None
    count: int
    seen: Annotated[list[str], add]


@node
def worker(state: Work) -> dict:
    answer = interrupt("ok?", id="ok")
    note = state.get("note")
    return {"seen": [f"{type(note).__name__}:{state.get('count')!r}:{answer}"]}


@node
def live_worker(state: Work) -> dict:
    return {"seen": [f"{type(state.get('note')).__name__}:{state.get('count')!r}"]}


@node(goto=[worker])
def dispatch(state: Work) -> Send:
    return Send(worker, {"note": {"text": "a"}, "count": "3"})


@node(goto=[live_worker])
def dispatch_live(state: Work) -> Send:
    return Send(live_worker, {"note": {"text": "a"}, "count": "3"})


async def test_send_payloads_are_validated_as_field_values() -> None:
    graph = Graph(Work).flow(START >> dispatch_live, live_worker >> END)

    assert (await graph.ainvoke({"seen": []})).data["seen"] == ["Note:3"]


async def test_restored_send_payloads_are_read_back_as_field_values() -> None:
    graph = Graph(Work, state_store=InMemoryStateStore()).flow(
        START >> dispatch, worker >> END
    )

    paused = await graph.ainvoke({"seen": []}, thread_id="t")
    assert paused.status == "interrupted"
    result = await graph.ainvoke(None, thread_id="t", resume=Resume("yes"))

    assert result.data["seen"] == ["Note:3:yes"]


@node(goto=[worker])
def bad_send(state: Work) -> Send:
    return Send(worker, {"count": "three"})


async def test_an_invalid_send_payload_names_source_target_and_field() -> None:
    graph = Graph(Work).flow(START >> bad_send, worker >> END)

    with pytest.raises(
        StateUpdateError, match=r"'count'.*node 'bad_send' to node 'worker'"
    ):
        await graph.ainvoke({"seen": []})


def test_validator_decorators_on_a_state_schema_raise() -> None:
    class WithFieldValidator(BaseModel):
        name: str = ""

        @field_validator("name")
        @classmethod
        def up(cls, value: str) -> str:
            return value.upper()

    class WithModelValidator(BaseModel):
        name: str = ""

        @model_validator(mode="after")
        def check(self) -> Self:
            return self

    with pytest.raises(GraphConfigError, match=r"up.*AfterValidator"):
        Graph(WithFieldValidator)
    with pytest.raises(GraphConfigError, match=r"check.*AfterValidator"):
        Graph(WithModelValidator)


def test_serializer_decorators_on_a_state_schema_raise() -> None:
    class WithFieldSerializer(BaseModel):
        when: datetime | None = None

        @field_serializer("when")
        def as_timestamp(self, value: datetime | None) -> float | None:
            return None if value is None else value.timestamp()

    class WithModelSerializer(BaseModel):
        name: str = ""

        @model_serializer
        def dump(self) -> dict[str, Any]:
            return {"name": self.name}

    with pytest.raises(GraphConfigError, match=r"as_timestamp.*PlainSerializer"):
        Graph(WithFieldSerializer)
    with pytest.raises(GraphConfigError, match=r"dump.*PlainSerializer"):
        Graph(WithModelSerializer)


class Stripped(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = ""
    tags: Annotated[list[str], add] = Field(default_factory=list)


@node
def padded(state: Stripped) -> dict:
    return {"name": "  ada  ", "tags": ["  x "]}


def test_the_schema_config_applies_to_written_values() -> None:
    graph = Graph(Stripped).flow(START >> padded, padded >> END)

    assert graph.invoke({}).data == {"name": "ada", "tags": ["x"]}


class Conversation(BaseModel):
    messages: Annotated[list[Message], add_messages] = Field(default_factory=list)


def test_messages_given_as_dicts_in_the_input_become_messages() -> None:
    graph = Graph(Conversation).flow(START >> noop, noop >> END)

    result = graph.invoke({"messages": [{"type": "human", "content": "hi"}]})

    assert isinstance(result.data["messages"][0], HumanMessage)
    assert result.data["messages"][0].id is not None


@node
def writes_typo(state: Any) -> dict:
    return {"cuont": 1}


async def test_a_node_writing_an_unknown_field_fails_before_it_completes() -> None:
    graph = Graph(Known, state_store=InMemoryStateStore()).flow(
        START >> writes_typo, writes_typo >> END
    )

    with pytest.raises(StateUpdateError, match=r"Unknown field 'cuont'.*writes_typo"):
        await graph.ainvoke({}, thread_id="t")

    events = _events(await graph.history("t"))
    assert ("error", "writes_typo") in events
    assert ("node_completed", "writes_typo") not in events


class Handles(TypedDict, total=False):
    lock: Any


@node
def stores_lock(state: Handles) -> dict:
    return {"lock": threading.Lock()}


async def test_a_value_that_cannot_be_stored_fails_the_node_that_writes_it() -> None:
    graph = Graph(Handles, state_store=InMemoryStateStore()).flow(
        START >> stores_lock, stores_lock >> END
    )

    with pytest.raises(StateStoreError, match=r"'lock'.*node 'stores_lock'"):
        await graph.ainvoke({}, thread_id="t")

    assert ("error", "stores_lock") in _events(await graph.history("t"))


failures: list[str] = []


@node
def fails_once(state: Work) -> dict:
    if not failures:
        failures.append("fails_once")
        raise RuntimeError("first attempt")
    return {"seen": [type(state.get("note")).__name__]}


@node(goto=[fails_once])
def dispatch_flaky(state: Work) -> Send:
    return Send(fails_once, {"note": {"text": "a"}})


async def test_a_continued_send_task_reads_its_payload_back_as_field_values() -> None:
    failures.clear()
    graph = Graph(Work, state_store=InMemoryStateStore()).flow(
        START >> dispatch_flaky, fails_once >> END
    )
    with pytest.raises(RuntimeError, match="first attempt"):
        await graph.ainvoke({"seen": []}, thread_id="t")

    result = await graph.ainvoke(None, thread_id="t")

    assert result.data["seen"] == ["Note"]


@node
def reads_note(state: Work) -> dict:
    return {"seen": [type(state.get("note")).__name__]}


@node(goto=[reads_note])
def sends_note(state: Work) -> Send:
    return Send(reads_note, {"note": {"text": "a"}})


@node
def waits(state: Work) -> dict:
    return {"seen": [f"answer={interrupt('ok?', id='ok')}"]}


@node(goto=[sends_note, waits])
def split(state: Work) -> Command:
    return Command(goto=[sends_note, waits])


async def test_a_carried_send_route_reads_its_payload_back_as_field_values() -> None:
    graph = Graph(Work, state_store=InMemoryStateStore()).flow(
        START >> split, reads_note >> END, waits >> END
    )
    await graph.ainvoke({"seen": []}, thread_id="t")

    result = await graph.ainvoke(None, thread_id="t", resume=Resume("yes"))

    assert sorted(result.data["seen"]) == ["Note", "answer=yes"]


def test_restore_update_raises_for_a_key_the_schema_does_not_declare() -> None:
    schema = StateSchema.from_type(V2)

    with pytest.raises(StateStoreError, match="'legacy'"):
        schema.restore_update({"legacy": "x"})


class AmbiguousResult:
    def __bool__(self) -> bool:
        raise ValueError("the truth value is ambiguous")


class Ambiguous:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def __eq__(self, other: object) -> Any:
        return AmbiguousResult()

    __hash__ = object.__hash__


class Holding(TypedDict, total=False):
    item: Any


def test_a_value_with_an_ambiguous_eq_is_refused_as_state() -> None:
    graph = Graph(Holding).flow(START >> noop, noop >> END)

    with pytest.raises(StateStoreError, match=r"'item'.*cannot be stored"):
        graph.invoke({"item": Ambiguous([1])})


class Needs(BaseModel):
    need: int
    note: str = ""


def test_a_new_thread_needs_a_value_for_every_required_field() -> None:
    graph = Graph(Needs).flow(START >> noop, noop >> END)

    with pytest.raises(StateUpdateError, match="required state field 'need'"):
        graph.invoke({"note": "x"})
    assert graph.invoke({"need": 1}).data == {"need": 1, "note": ""}


@node
def takes_lock(state: Handles) -> None:
    return None


@node(goto=[takes_lock])
def sends_lock(state: Handles) -> Send:
    return Send(takes_lock, {"lock": threading.Lock()})


async def test_a_send_payload_that_cannot_be_stored_fails_the_sender() -> None:
    graph = Graph(Handles, state_store=InMemoryStateStore()).flow(
        START >> sends_lock, takes_lock >> END
    )

    with pytest.raises(StateStoreError, match=r"'lock'.*node 'sends_lock'"):
        await graph.ainvoke({}, thread_id="t")

    assert ("error", "sends_lock") in _events(await graph.history("t"))


reducer_inputs: list[str] = []


def typed_concat(current: list[Note] | None, update: list[Note]) -> list[Note]:
    reducer_inputs.extend(type(item).__name__ for item in update)
    return [*(current or []), *update]


class Notes(TypedDict, total=False):
    notes: Annotated[list[Note], typed_concat]


@node
def writes_note_dict(state: Notes) -> dict:
    return {"notes": [{"text": "n"}]}


def test_a_custom_reducer_gets_typed_values_during_a_run() -> None:
    reducer_inputs.clear()
    graph = Graph(Notes).flow(START >> writes_note_dict, writes_note_dict >> END)

    result = graph.invoke({"notes": []})

    assert [type(note) for note in result.data["notes"]] == [Note]
    assert set(reducer_inputs) == {"Note"}


def test_repeated_message_ids_in_the_input_of_a_new_thread_raise() -> None:
    graph = Graph(Conversation).flow(START >> noop, noop >> END)
    twice = [HumanMessage(id="m1", content="a"), HumanMessage(id="m1", content="b")]

    with pytest.raises(StateUpdateError, match=r"'messages'.*the input.*'m1' twice"):
        graph.invoke({"messages": twice})


def test_a_default_factory_that_reads_other_fields_raises() -> None:
    class Derived(BaseModel):
        a: int = 1
        b: int = Field(default_factory=lambda data: data["a"] + 1)

    with pytest.raises(GraphConfigError, match=r"'b'.*default_factory"):
        Graph(Derived)


def test_a_model_instance_as_input_is_not_validated_again() -> None:
    bang_calls.clear()
    graph = Graph(Bang).flow(START >> noop, noop >> END)
    built = Bang(text="hi")

    result = graph.invoke(built)

    assert result.data["text"] == built.text == "hi!"
    assert bang_calls == ["hi", "hi!"]


class Revalidated(BaseModel):
    model_config = ConfigDict(revalidate_instances="always")

    text: Annotated[str, AfterValidator(bang)]


def test_a_model_instance_is_validated_again_when_its_schema_asks_for_it() -> None:
    bang_calls.clear()
    graph = Graph(Revalidated).flow(START >> noop, noop >> END)

    assert graph.invoke(Revalidated(text="hi")).data["text"] == "hi!"
    assert bang_calls == ["hi", "hi!", "hi!"]


class Listing(BaseModel):
    groups: dict[str, list[str]] = Field(default_factory=dict)


class Letter(BaseModel):
    lines: list[str] = Field(default_factory=list)


class Mailbag(TypedDict, total=False):
    letter: Letter


def test_send_payload_values_are_copied() -> None:
    letter = Letter(lines=["a"])
    loose = {"k": ["a"]}

    typed = StateSchema.from_type(Mailbag).validate_payload(
        {"letter": letter}, writer="test"
    )
    untyped = StateSchema.from_type(dict).validate_payload(
        {"loose": loose}, writer="test"
    )
    letter.lines.append("later")
    loose["k"].append("later")

    assert typed == {"letter": Letter(lines=["a"])}
    assert untyped == {"loose": {"k": ["a"]}}


def test_input_values_are_copied_into_the_state() -> None:
    graph = Graph(Listing).flow(START >> noop, noop >> END)
    built = Listing(groups={"g": ["a"]})

    result = graph.invoke(built)
    built.groups["g"].append("later")

    assert result.data["groups"] == {"g": ["a"]}


@with_config(ConfigDict(str_strip_whitespace=True))
class StrippedDict(TypedDict, total=False):
    name: str


def test_the_pydantic_config_of_a_typeddict_applies_to_written_values() -> None:
    graph = Graph(StrippedDict).flow(START >> noop, noop >> END)

    assert graph.invoke({"name": "  ada  "}).data == {"name": "ada"}


class StrictConfigured(BaseModel):
    model_config = ConfigDict(strict=True)

    count: int = 0
    approved: bool = False


@with_config(ConfigDict(strict=True))
class StrictConfiguredDict(TypedDict, total=False):
    count: int
    approved: bool


@pytest.mark.parametrize("schema", [StrictConfigured, StrictConfiguredDict])
async def test_strict_config_refuses_strings_for_numbers_and_booleans(
    schema: Any,
) -> None:
    @node
    def writes_strings(state: Any) -> dict:
        return {"count": "7", "approved": "yes"}

    graph = Graph(schema).flow(START >> writes_strings, writes_strings >> END)

    with pytest.raises(StateUpdateError, match=r"'count'.*the input"):
        await graph.ainvoke({"count": "1"})
    with pytest.raises(StateUpdateError, match=r"'count'.*node 'writes_strings'"):
        await graph.ainvoke({"count": 1, "approved": True})


@pytest.mark.parametrize("schema", [Strict, StrictDict])
async def test_default_config_converts_numeric_strings(schema: Any) -> None:
    graph = Graph(schema).flow(START >> noop, noop >> END)

    result = await graph.ainvoke({"count": "1"})

    assert result.data["count"] == 1
