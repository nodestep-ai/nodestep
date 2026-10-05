from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from typing import Annotated, Any, TypedDict

import pytest
from pydantic import BaseModel, ConfigDict, Field

from nodestep import (
    END,
    START,
    BaseState,
    Command,
    Graph,
    InMemoryStateStore,
    Send,
    add,
    node,
)
from nodestep.exceptions import GraphExecutionError, InvalidUpdateError, StateStoreError
from nodestep.middleware.base import Middleware, NodeMiddlewareContext

WHEN = datetime(2026, 1, 1, tzinfo=UTC)


class Model(BaseState):
    log: Annotated[list[str], add] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class Plain(TypedDict, total=False):
    log: Annotated[list[str], add]
    meta: dict[str, Any]


@node
def model_append_and_return(state: Model) -> Model:
    state.log.append("x")
    return state


@node
def model_nested_mutation(state: Model) -> Model:
    state.meta.setdefault("tags", []).append("t")
    return state


@node
def dict_append_no_return(state: Plain) -> None:
    state["log"].append("sneaky")
    state["meta"]["leak"] = True


@node
def dict_append_and_return(state: Plain) -> Plain:
    state["log"].append("x")
    return state


@node
def after_dict_mutation(state: Plain) -> dict:
    return {"meta": {**state["meta"], "seen_log": list(state["log"])}}


def test_pydantic_list_append_and_return_is_kept() -> None:
    graph = Graph(Model).flow(
        START >> model_append_and_return, model_append_and_return >> END
    )

    assert graph.invoke({"log": ["seed"]}).data["log"] == ["seed", "x"]


def test_pydantic_nested_dict_mutation_and_return_is_kept() -> None:
    graph = Graph(Model).flow(
        START >> model_nested_mutation, model_nested_mutation >> END
    )

    assert graph.invoke({}).data["meta"] == {"tags": ["t"]}


async def test_dict_mutation_without_return_raises_and_writes_nothing() -> None:
    store = InMemoryStateStore()
    graph = Graph(Plain, state_store=store).flow(
        START >> dict_append_no_return,
        dict_append_no_return >> after_dict_mutation,
        after_dict_mutation >> END,
    )

    with pytest.raises(GraphExecutionError, match="mutated its input"):
        await graph.ainvoke({"log": ["seed"], "meta": {}}, thread_id="t")

    assert (await graph.load("t")) == {"log": ["seed"], "meta": {}}


def test_dict_mutate_and_return_same_object_is_diffed() -> None:
    graph = Graph(Plain).flow(
        START >> dict_append_and_return, dict_append_and_return >> END
    )

    assert graph.invoke({"log": ["seed"], "meta": {}}).data["log"] == ["seed", "x"]


@node
def owner_a(state: Model) -> Model:
    state.meta["owner"] = "a"
    return state


@node
def owner_b(state: Model) -> Model:
    state.meta["owner"] = "b"
    return state


@node(goto=[owner_a, owner_b])
def fan(state: Model) -> Command:
    return Command(goto=[Send(owner_a), Send(owner_b)])


def test_parallel_in_place_writes_to_one_field_conflict() -> None:
    graph = Graph(Model).flow(START >> fan, owner_a >> END, owner_b >> END)

    with pytest.raises(InvalidUpdateError):
        graph.invoke({})


def test_after_node_receives_delta_for_state_returning_node() -> None:
    seen: list[Any] = []

    class Recorder(Middleware):
        def after_node(self, ctx: NodeMiddlewareContext) -> None:
            seen.append(ctx.state)

    graph = Graph(Model, middleware=[Recorder()]).flow(
        START >> model_append_and_return, model_append_and_return >> END
    )

    graph.invoke({"log": ["seed"]})

    assert seen == [{"log": ["x"]}]


class PlainModel(BaseModel):
    log: Annotated[list[str], add] = Field(default_factory=list)
    count: int = 0


@node
def plain_model_copy(state: PlainModel) -> PlainModel:
    return state.model_copy(update={"count": state.count + 1})


def test_plain_model_schema_returning_a_new_instance_raises() -> None:
    graph = Graph(PlainModel).flow(START >> plain_model_copy, plain_model_copy >> END)

    with pytest.raises(TypeError, match="Node 'plain_model_copy' returned PlainModel;"):
        graph.invoke({"log": ["a", "b"]})


class Holder:
    def __init__(self) -> None:
        self.items: list[str] = []


@dataclass
class Document:
    tags: list[str] = field(default_factory=list)


class Loose(TypedDict, total=False):
    log: Annotated[list[str], add]
    doc: Document
    holder: Any


@node
def tagger(state: Loose) -> None:
    state["doc"].tags.append("added-in-place")


def test_in_place_changes_to_a_dataclass_do_not_leak() -> None:
    graph = Graph(Loose).flow(START >> tagger, tagger >> END)
    doc = Document()

    with pytest.raises(GraphExecutionError, match="mutated its input"):
        graph.invoke({"log": [], "doc": doc})

    assert doc == Document()


def test_an_object_that_cannot_be_stored_is_refused_as_state() -> None:
    graph = Graph(Loose).flow(START >> tagger, tagger >> END)

    with pytest.raises(StateStoreError, match=r"'holder'.*cannot be stored"):
        graph.invoke({"holder": Holder()})


class Tagged(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = ""


class HoldsTagged(BaseState):
    tagged: Tagged = Field(default_factory=Tagged)


def _extra(model: BaseModel) -> dict[str, Any]:
    extra = model.model_extra
    assert extra is not None
    return extra


@node
def tags_in_place(state: HoldsTagged) -> None:
    _extra(state.tagged)["tags"].append("leaked")


@node
def stamps_in_place(state: HoldsTagged) -> HoldsTagged:
    _extra(state.tagged)["seen"].append(WHEN)
    return state


def test_in_place_changes_to_extra_attributes_are_caught_and_do_not_leak() -> None:
    graph = Graph(HoldsTagged).flow(START >> tags_in_place, tags_in_place >> END)
    tagged = Tagged.model_validate({"name": "n", "tags": ["orig"]})

    with pytest.raises(
        GraphExecutionError, match="mutated its input but returned None"
    ):
        graph.invoke({"tagged": tagged})

    assert tagged.model_extra == {"tags": ["orig"]}


def test_an_extra_attribute_changed_in_place_is_checked_when_it_is_written() -> None:
    graph = Graph(HoldsTagged).flow(START >> stamps_in_place, stamps_in_place >> END)

    with pytest.raises(
        StateStoreError,
        match=r"'tagged'.*node 'stamps_in_place'.*at tagged\.seen\[0\], a datetime",
    ):
        graph.invoke({"tagged": Tagged.model_validate({"name": "n", "seen": []})})


class Cat(BaseModel):
    name: str


class Dog(BaseModel):
    name: str


class Pets(BaseState):
    pets: list[Cat | Dog] = Field(default_factory=list)
    herd: Annotated[list[Cat | Dog], add] = Field(default_factory=list)


@node
def swaps_pet(state: Pets) -> Pets:
    state.pets[0] = Dog(name="rex")
    return state


@node
def swaps_herd(state: Pets) -> Pets:
    state.herd[0] = Dog(name="rex")
    return state


@pytest.mark.parametrize(
    ("handler", "key"), [(swaps_pet, "pets"), (swaps_herd, "herd")]
)
def test_an_in_place_change_with_the_same_json_form_is_not_dropped(
    handler: Any, key: str
) -> None:
    graph = Graph(Pets).flow(START >> handler, handler >> END)

    with pytest.raises(
        StateStoreError,
        match=rf"'{key}'.*node '{handler.name}'.*a Dog comes back as Cat",
    ):
        graph.invoke({key: [Cat(name="rex")]})


@node
def to_tuple(state: dict) -> dict:
    state["items"] = tuple(state["items"])
    return state


def test_a_tuple_assigned_over_an_equal_list_is_not_dropped() -> None:
    graph = Graph(dict).flow(START >> to_tuple, to_tuple >> END)

    with pytest.raises(
        StateStoreError, match=r"'items'.*node 'to_tuple'.*a tuple comes back as list"
    ):
        graph.invoke({"items": [1, 2]})


class Stamped(TypedDict, total=False):
    when: datetime


@node
def shifts_offset(state: Stamped) -> None:
    state["when"] = state["when"].astimezone(timezone(timedelta(hours=1)))


@node
def shifts_and_returns(state: Stamped) -> Stamped:
    state["when"] = state["when"].astimezone(timezone(timedelta(hours=1)))
    return state


def test_an_equal_value_with_another_json_form_counts_as_a_change() -> None:
    silent = Graph(Stamped).flow(START >> shifts_offset, shifts_offset >> END)
    returned = Graph(Stamped).flow(
        START >> shifts_and_returns, shifts_and_returns >> END
    )

    with pytest.raises(
        GraphExecutionError, match="mutated its input but returned None"
    ):
        silent.invoke({"when": WHEN})
    when = returned.invoke({"when": WHEN}).data["when"]
    assert when == WHEN
    assert when.utcoffset() == timedelta(hours=1)


class Notes(TypedDict, total=False):
    note: str | None


@node
def clears_note(state: Notes) -> Notes:
    state["note"] = None
    return state


def test_a_value_set_in_place_on_a_field_without_one_is_written() -> None:
    graph = Graph(Notes).flow(START >> clears_note, clears_note >> END)

    assert graph.invoke({}).data == {"note": None}


@dataclass(eq=False)
class Box:
    items: list[str] = field(default_factory=list)


class Boxed(TypedDict, total=False):
    box: Box
    log: Annotated[list[str], add]


@node
def reads_box(state: Boxed) -> None:
    return None


@node
def refills_box(state: Boxed) -> None:
    state["box"].items.append("b")


def test_a_value_unequal_to_its_own_copy_is_no_change_until_its_parts_change() -> None:
    reads = Graph(Boxed).flow(START >> reads_box, reads_box >> END)
    refills = Graph(Boxed).flow(START >> refills_box, refills_box >> END)

    assert reads.invoke({"box": Box(["a"])}).data["box"].items == ["a"]
    with pytest.raises(
        GraphExecutionError, match="mutated its input but returned None"
    ):
        refills.invoke({"box": Box(["a"])})
