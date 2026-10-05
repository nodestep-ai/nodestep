import json
import math
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum, StrEnum
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, TypedDict
from uuid import uuid4

import pytest
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    Json,
    PlainSerializer,
    PrivateAttr,
    SecretBytes,
    SecretStr,
    computed_field,
)

from nodestep import (
    END,
    START,
    BaseState,
    Command,
    Graph,
    InMemoryStateStore,
    Replace,
    Send,
    StreamEvent,
    add,
    node,
)
from nodestep.exceptions import GraphConfigError, StateStoreError, StateUpdateError

WHEN = datetime(2026, 1, 1, tzinfo=UTC)


@node
def noop(state: Any) -> None:
    return None


def _single(schema: Any, handler: Any = noop, **options: Any) -> Graph:
    return Graph(schema, **options).flow(START >> handler, handler >> END)


def _stored(schema: Any, handler: Any = noop) -> Graph:
    return _single(schema, handler, state_store=InMemoryStateStore())


def _data(event: StreamEvent) -> Any:
    return event.data


def _events(graph: Graph) -> list[Any]:
    store = graph.state_store
    assert isinstance(store, InMemoryStateStore)
    return store.events


class Pet(BaseModel):
    name: str


class Animal(BaseModel):
    name: str


class Dog(Animal):
    barks: bool = True


class Cat(BaseModel):
    name: str


class Fox(BaseModel):
    name: str


@pytest.mark.parametrize(
    ("value", "lost"),
    [
        (WHEN, "datetime comes back as str"),
        (Path("/tmp/x"), "Path comes back as str"),
        (Pet(name="rex"), "Pet comes back as dict"),
        ((1, 2), "tuple comes back as list"),
        ({"a"}, "set comes back as list"),
        ({1: "one"}, "a int key comes back as a str key"),
        (math.inf, "float comes back as NoneType"),
        (b"abc", "bytes comes back as str"),
    ],
)
async def test_untyped_state_refuses_values_that_are_not_json(
    value: Any, lost: str
) -> None:
    graph = _stored(dict)

    with pytest.raises(StateStoreError, match=rf"'value'.*the input.*{lost}"):
        await graph.ainvoke({"value": value}, thread_id="t")

    assert _events(graph) == []


async def test_untyped_state_accepts_json_values() -> None:
    graph = _stored(dict)
    value = {"s": "x", "i": 1, "f": 1.5, "b": True, "n": None, "l": [1, {"k": "v"}]}

    result = await graph.ainvoke({"value": value}, thread_id="t")

    assert result.data == {"value": value}
    assert await graph.load("t") == {"value": value}


@node
def stamps(state: dict) -> dict:
    return {"when": WHEN}


def test_a_node_write_that_cannot_be_stored_fails_without_a_store() -> None:
    with pytest.raises(
        StateStoreError, match=r"'when'.*node 'stamps'.*datetime comes back as str"
    ):
        _single(dict, stamps).invoke({})


async def test_a_node_write_that_cannot_be_stored_is_the_node_error() -> None:
    graph = _stored(dict, stamps)

    with pytest.raises(StateStoreError, match="'when'"):
        await graph.ainvoke({"other": 1}, thread_id="t")

    events = [(event.type, event.node) for event in _events(graph)]
    assert ("error", "stamps") in events
    assert ("node_completed", "stamps") not in events
    assert await graph.load("t") == {"other": 1}


class Zoo(BaseState):
    pet: Animal | None = None
    either: Cat | Fox | None = None
    anything: Any = None


@pytest.mark.parametrize(
    ("update", "lost"),
    [
        ({"pet": Dog(name="rex", barks=False)}, "'pet'.*Dog comes back as Animal"),
        ({"either": Fox(name="f")}, "'either'.*Fox comes back as Cat"),
        ({"anything": WHEN}, "'anything'.*datetime comes back as str"),
    ],
)
async def test_typed_fields_refuse_values_their_type_cannot_rebuild(
    update: dict[str, Any], lost: str
) -> None:
    @node
    def writes(state: Zoo) -> dict:
        return update

    with pytest.raises(StateStoreError, match=rf"{lost}"):
        await _stored(Zoo, writes).ainvoke({}, thread_id="t")


class Level(StrEnum):
    LOW = "low"


@pytest.mark.parametrize(
    ("value", "lost"),
    [
        (Level.LOW, "a Level comes back as str"),
        (defaultdict(int, {"a": 1}), "a defaultdict comes back as dict"),
        (OrderedDict(a=1), "a OrderedDict comes back as dict"),
        ([Level.LOW], r"at anything\[0\], a Level comes back as str"),
        ({"k": {Level.LOW: 1}}, r"at anything\['k'\], a Level key comes back as a str"),
    ],
)
def test_subclasses_of_json_types_that_come_back_as_their_base_are_refused(
    value: Any, lost: str
) -> None:
    with pytest.raises(StateStoreError, match=rf"'anything'.*{lost}"):
        _single(Zoo).invoke({"anything": value})


def test_a_model_instance_as_input_is_checked_like_any_input() -> None:
    with pytest.raises(StateStoreError, match=r"'pet'.*the input.*Dog comes back as"):
        _single(Zoo).invoke(Zoo(pet=Dog(name="rex")))


class Loose(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str


class Aliased(BaseModel):
    item_id: int = Field(alias="itemId")


class Rich(BaseState):
    when: datetime | None = None
    pair: tuple[int, int] | None = None
    tags: set[str] = Field(default_factory=set)
    by_id: dict[int, str] = Field(default_factory=dict)
    by_level: dict[Level, int] = Field(default_factory=dict)
    where: Path | None = None
    blob: bytes = b""
    loose: Loose | None = None
    aliased: Aliased | None = None


async def test_values_their_type_can_rebuild_keep_their_types_after_a_reload() -> None:
    value = {
        "when": WHEN,
        "pair": (1, 2),
        "tags": {"a", "b"},
        "by_id": {1: "one"},
        "by_level": {Level.LOW: 1},
        "where": Path("/tmp/x"),
        "blob": b"abc",
        "loose": Loose.model_validate({"name": "n", "colour": "red"}),
        "aliased": Aliased(itemId=1),
    }
    graph = _stored(Rich)

    result = await graph.ainvoke(value, thread_id="t")
    loaded = await graph.load("t")

    assert result.data == value
    assert loaded == value
    assert isinstance(loaded["pair"], tuple)
    assert loaded["loose"].colour == "red"


async def test_models_are_stored_with_their_aliases() -> None:
    graph = _stored(Rich)

    await graph.ainvoke({"aliased": Aliased(itemId=7)}, thread_id="t")

    deltas = [
        json.loads(event.data_json)["update"]
        for event in _events(graph)
        if event.type == "state_delta"
    ]
    assert deltas == [{"aliased": {"itemId": 7}}]


class Priced(BaseModel):
    model_config = ConfigDict(extra="forbid")
    net: int

    @computed_field
    @property
    def gross(self) -> int:
        return self.net * 2

    @cached_property
    def label(self) -> str:
        return f"net {self.net}"


class Encoded(BaseModel):
    data: Json[list[int]]


class Shop(BaseState):
    priced: Priced | None = None
    encoded: Encoded | None = None


async def test_models_are_stored_so_that_they_validate_again() -> None:
    priced = Priced(net=2)
    assert priced.label == "net 2"
    graph = _stored(Shop)

    await graph.ainvoke(
        {"priced": priced, "encoded": Encoded.model_validate({"data": "[1, 2]"})},
        thread_id="t",
    )

    loaded = await graph.load("t")
    assert loaded["priced"].gross == 4
    assert loaded["encoded"].data == [1, 2]


class Shout(BaseState):
    text: Annotated[str, AfterValidator(lambda value: value + "!")]


def test_a_validator_that_changes_a_valid_value_is_refused() -> None:
    with pytest.raises(
        StateStoreError, match=r"'text'.*a str comes back with another value"
    ):
        _single(Shout).invoke({"text": "hi"})


class Masked(BaseState):
    code: Annotated[str, PlainSerializer(lambda value: "***")]


def test_a_refused_value_is_not_quoted_in_the_error() -> None:
    with pytest.raises(StateStoreError, match="'code'") as info:
        _single(Masked).invoke({"code": "hunter2"})

    assert "hunter2" not in str(info.value)


class Wrapper(BaseModel):
    value: Any = None


@dataclass
class Box:
    value: Any


class Containers(BaseState):
    tags: set[Any] = Field(default_factory=set)
    frozen: frozenset[Any] = frozenset()
    queue: deque[Any] = Field(default_factory=deque)
    box: Box | None = None
    wrapper: Wrapper | None = None


@pytest.mark.parametrize(
    ("value", "lost"),
    [
        ({"tags": {Level.LOW}}, r"at tags\{\.\.\.\}, a Level comes back as str"),
        (
            {"frozen": frozenset({Level.LOW})},
            r"at frozen\{\.\.\.\}, a Level comes back as str",
        ),
        ({"queue": deque([Level.LOW])}, r"at queue\[0\], a Level comes back as str"),
        ({"queue": deque([1], maxlen=2)}, "a deque with maxlen 2 comes back with"),
        ({"box": Box(Level.LOW)}, r"at box\.value, a Level comes back as str"),
        ({"box": Box((1, 2))}, r"at box\.value, a tuple comes back as list"),
        (
            {"wrapper": Wrapper(value=Level.LOW)},
            r"at wrapper\.value, a Level comes back as str",
        ),
    ],
)
def test_types_inside_sets_deques_dataclasses_and_models_must_survive(
    value: dict[str, Any], lost: str
) -> None:
    with pytest.raises(StateStoreError, match=rf"'{next(iter(value))}'.*{lost}"):
        _single(Containers).invoke(value)


class Tracked(BaseModel):
    title: str = ""
    _token: str = PrivateAttr(default_factory=lambda: uuid4().hex)


class Cached(BaseModel):
    title: str = ""
    _cache: dict[str, int] = PrivateAttr(default_factory=dict)


class Privates(BaseState):
    tracked: Tracked | None = None
    cached: Cached | None = None


def test_a_private_attribute_that_would_be_lost_is_named() -> None:
    filled = Cached(title="t")
    filled._cache["x"] = 1

    with pytest.raises(
        StateStoreError, match=r"at tracked\._token, a private attribute is not stored"
    ):
        _single(Privates).invoke({"tracked": Tracked(title="t")})
    with pytest.raises(
        StateStoreError, match=r"at cached\._cache, a private attribute is not stored"
    ):
        _single(Privates).invoke({"cached": filled})
    assert _single(Privates).invoke({"cached": Cached(title="t")}).data == {
        "tracked": None,
        "cached": Cached(title="t"),
    }


class Tagged(BaseState):
    tags: Annotated[list[str], add, Field(min_length=2)] = Field(
        default_factory=lambda: ["a", "b"]
    )


def test_a_reducer_update_is_checked_with_the_type_it_is_stored_as() -> None:
    @node
    def tags(state: Tagged) -> dict:
        return {"tags": ["c"]}

    assert _single(Tagged, tags).invoke({}).data == {"tags": ["a", "b", "c"]}


def as_tuple(current: Any, update: Any) -> Any:
    return (*(current or ()), *update)


class Pairs(BaseState):
    items: Annotated[Any, as_tuple] = None


def test_a_merged_value_that_cannot_be_stored_is_refused() -> None:
    @node
    def appends(state: Pairs) -> dict:
        return {"items": [1]}

    with pytest.raises(
        StateStoreError, match=r"'items'.*node 'appends'.*tuple comes back as list"
    ):
        _single(Pairs, appends).invoke({})


class Account(BaseModel):
    user: str
    api_key: str = Field(exclude=True)


class Profile(BaseModel):
    user: str
    token: str | None = Field(default=None, exclude=True)


class HasAccount(BaseState):
    account: Account | None = None
    profile: Profile | None = None


async def test_an_excluded_field_that_would_be_lost_is_refused() -> None:
    graph = _stored(HasAccount)

    with pytest.raises(StateStoreError, match=r"'account'.*api_key"):
        await graph.ainvoke(
            {"account": Account(user="u", api_key="sk-live-123")}, thread_id="t"
        )

    assert "sk-live-123" not in "".join(
        event.data_json or "" for event in _events(graph)
    )


class Login(BaseModel):
    user: str = ""
    api_key: str = Field(default="", exclude=True)


class HasLogin(BaseState):
    login: Login | None = None


@node
def logs_in(state: HasLogin) -> dict:
    return {"login": Login(user="u", api_key="sk-live-123")}


async def test_an_excluded_value_a_node_writes_stays_out_of_the_error_and_the_store() -> (
    None
):
    graph = _stored(HasLogin, logs_in)

    with pytest.raises(
        StateStoreError,
        match=r"'login'.*node 'logs_in'.*at login\.api_key, a field with exclude=True",
    ) as info:
        await graph.ainvoke({}, thread_id="t")

    assert "sk-live-123" not in str(info.value)
    events = (await graph.history("t")).events
    assert ("error", "logs_in") in [(event.type, event.node) for event in events]
    assert not any(
        "sk-live-123" in (event.message or "") + (event.data_json or "")
        for event in events
    )


async def test_an_excluded_field_at_its_default_is_not_written() -> None:
    graph = _stored(HasAccount)

    await graph.ainvoke({"profile": Profile(user="u")}, thread_id="t")

    assert (await graph.load("t"))["profile"] == Profile(user="u")
    assert "token" not in "".join(event.data_json or "" for event in _events(graph))


def test_an_excluded_state_field_is_refused_when_the_graph_is_built() -> None:
    class Excluded(BaseState):
        visible: int = 0
        hidden: str = Field(default="", exclude=True)

    class ExcludedIf(BaseState):
        visible: int = 0
        hidden: str = Field(default="", exclude_if=lambda value: not value)

    with pytest.raises(GraphConfigError, match=r"'hidden'.*exclude=True"):
        Graph(Excluded)
    with pytest.raises(GraphConfigError, match=r"'hidden'.*exclude_if"):
        Graph(ExcludedIf)


class Color(Enum):
    RED = 1


class Scores(BaseState):
    score: dict[Color, int] = Field(default_factory=dict)


def test_mapping_keys_that_cannot_be_read_back_are_refused() -> None:
    with pytest.raises(StateStoreError, match="'score'"):
        _single(Scores).invoke({"score": {Color.RED: 3}})


class Ratio(BaseState):
    ratio: float = 0.0


class Blob(BaseState):
    data: bytes = b""


@pytest.mark.parametrize(
    ("schema", "value"),
    [(Ratio, {"ratio": math.inf}), (Blob, {"data": b"\xff\x00"})],
)
def test_inf_and_non_utf8_bytes_are_refused_by_default(
    schema: Any, value: dict[str, Any]
) -> None:
    field = next(iter(value))

    with pytest.raises(StateStoreError, match=f"'{field}'.*cannot be stored"):
        _single(schema).invoke(value)


class HandledRatio(BaseState):
    model_config = ConfigDict(ser_json_inf_nan="strings")
    ratio: float = 0.0
    ratios: list[float] = Field(default_factory=list)


class HandledBlob(BaseState):
    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")
    data: bytes = b""


async def test_inf_nan_and_bytes_survive_when_the_schema_handles_them() -> None:
    ratios = _stored(HandledRatio)
    blobs = _stored(HandledBlob)

    await ratios.ainvoke({"ratio": math.inf}, thread_id="inf")
    await ratios.ainvoke(
        {"ratio": math.nan, "ratios": [1.0, math.nan]}, thread_id="nan"
    )
    await blobs.ainvoke({"data": b"\xff\x00"}, thread_id="b")

    assert (await ratios.load("inf"))["ratio"] == math.inf
    loaded = await ratios.load("nan")
    assert math.isnan(loaded["ratio"])
    assert loaded["ratios"][0] == 1.0
    assert math.isnan(loaded["ratios"][1])
    assert (await blobs.load("b"))["data"] == b"\xff\x00"


class StrictState(BaseState):
    model_config = ConfigDict(strict=True)
    when: datetime | None = None
    pet: Pet | None = None
    seen: Annotated[list[datetime], add] = Field(default_factory=list)


@node
def sees(state: StrictState) -> dict:
    return {"seen": [WHEN]}


async def test_a_strict_schema_reads_its_stored_values_back() -> None:
    graph = _stored(StrictState, sees)

    await graph.ainvoke({"when": WHEN, "pet": Pet(name="rex")}, thread_id="t")
    from_checkpoint = await graph.load("t")
    await graph.ainvoke({}, thread_id="t")
    from_deltas = await graph.load("t")

    assert from_checkpoint == {"when": WHEN, "pet": Pet(name="rex"), "seen": [WHEN]}
    assert from_deltas["seen"] == [WHEN, WHEN]


class Defaults(BaseState):
    pair: Any = Field(default_factory=lambda: (1, 2))


def test_a_default_that_cannot_be_stored_is_refused_on_a_new_thread() -> None:
    with pytest.raises(StateStoreError, match=r"'pair'.*its default.*tuple"):
        _single(Defaults).invoke({})


class Pending(BaseState):
    status: Annotated[str, AfterValidator(str.upper)] = "pending"
    log: Annotated[list[Any], add] = Field(default_factory=lambda: [(1, 2)])


def test_only_the_defaults_a_new_thread_keeps_are_checked() -> None:
    assert _single(Pending).invoke({"status": "DONE", "log": Replace(["a"])}).data == {
        "status": "DONE",
        "log": ["a"],
    }
    with pytest.raises(StateStoreError, match=r"'status'.*its default"):
        _single(Pending).invoke({"log": Replace([])})
    with pytest.raises(StateStoreError, match=r"'log'.*its default.*tuple"):
        _single(Pending).invoke({"status": "DONE", "log": ["a"]})


class Credentials(BaseModel):
    user: str = ""
    password: SecretStr | None = None


class Secrets(BaseState):
    token: SecretStr | None = None
    key: SecretBytes | None = None
    credentials: Credentials | None = None
    bundle: Any = None


@pytest.mark.parametrize(
    ("value", "secret"),
    [
        ({"token": SecretStr("s3cr3t")}, "'token'.*SecretStr"),
        ({"key": SecretBytes(b"k3y")}, "'key'.*SecretBytes"),
        (
            {"credentials": Credentials(user="u", password=SecretStr("pw"))},
            "'credentials'.*SecretStr",
        ),
        ({"bundle": [SecretStr("pw")]}, "'bundle'.*SecretStr"),
        ({"bundle": (SecretStr("pw"),)}, "'bundle'.*SecretStr"),
        ({"bundle": {SecretStr("pw")}}, "'bundle'.*SecretStr"),
        ({"bundle": frozenset({SecretStr("pw")})}, "'bundle'.*SecretStr"),
        ({"bundle": deque([SecretStr("pw")])}, "'bundle'.*SecretStr"),
        ({"bundle": Box(SecretStr("pw"))}, "'bundle'.*SecretStr"),
        ({"bundle": {"inner": Wrapper(value=SecretStr("pw"))}}, "'bundle'.*SecretStr"),
    ],
)
async def test_secrets_in_state_are_refused(value: dict[str, Any], secret: str) -> None:
    graph = _stored(Secrets)

    with pytest.raises(
        StateUpdateError, match=rf"{secret}.*pass secrets through context="
    ):
        await graph.ainvoke(value, thread_id="t")

    assert _events(graph) == []


def test_a_secret_written_by_a_node_is_refused_naming_the_node() -> None:
    @node
    def leaks(state: Secrets) -> dict:
        return {"token": "plain text becomes a SecretStr"}

    with pytest.raises(StateUpdateError, match=r"'token'.*node 'leaks'.*context="):
        _single(Secrets, leaks).invoke({})


class Plan(BaseState):
    steps: dict[str, str] = Field(default_factory=dict)


@node
def plan(state: Plan) -> dict:
    return {"steps": {"write": "draft", "review": "check", "publish": "ship"}}


async def test_key_order_is_kept_in_checkpoints_and_deltas() -> None:
    graph = _stored(Plan, plan)

    await graph.ainvoke({}, thread_id="t")
    from_checkpoint = list((await graph.load("t"))["steps"])
    await graph.ainvoke({}, thread_id="t")
    from_deltas = list((await graph.load("t"))["steps"])

    assert from_checkpoint == from_deltas == ["write", "review", "publish"]
    delta = next(
        json.loads(event.data_json)["update"]
        for event in _events(graph)
        if event.type == "state_delta" and event.node == "plan"
    )
    assert list(delta["steps"]) == ["write", "review", "publish"]


async def test_update_state_refuses_a_value_that_cannot_be_stored() -> None:
    graph = _stored(dict)
    await graph.ainvoke({"value": 1}, thread_id="t")

    with pytest.raises(StateStoreError, match=r"'value'.*update_state"):
        await graph.update_state("t", {"value": WHEN})


class Worker(TypedDict, total=False):
    when: Annotated[datetime, PlainSerializer(lambda value: value.timestamp())]
    handle: Any


@node
def uses(state: Worker) -> None:
    return None


def test_a_send_payload_that_cannot_be_stored_is_refused() -> None:
    @node(goto=[uses])
    def sends(state: Worker) -> Command:
        return Command(goto=[Send(uses, {"handle": object()})])

    graph = Graph(Worker).flow(START >> sends, uses >> END)

    with pytest.raises(
        StateStoreError, match=r"'handle'.*Send payload from node 'sends'"
    ):
        list(graph.stream({}, stream_mode="debug"))


def test_debug_routes_carry_the_json_form_of_send_payloads() -> None:
    @node(goto=[uses])
    def sends(state: Worker) -> Command:
        return Command(goto=[Send(uses, {"when": WHEN})])

    graph = Graph(Worker).flow(START >> sends, uses >> END)

    routes = [
        _data(event)["targets"]
        for event in graph.stream({}, stream_mode="debug")
        if event.mode == "debug"
        and _data(event)["type"] == "routes"
        and event.node == "sends"
    ]

    assert routes == [
        [{"kind": "send", "node": "uses", "payload": {"when": WHEN.timestamp()}}]
    ]
