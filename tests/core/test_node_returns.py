import math
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
    RemoveMessage,
    Replace,
    Resume,
    Send,
    add,
    branch,
    interrupt,
    node,
    when,
)
from nodestep.exceptions import GraphExecutionError, StateUpdateError
from nodestep.middleware.base import Middleware, NodeMiddlewareContext


class Plain(TypedDict):
    log: Annotated[list[str], add]
    note: str


class Model(BaseState):
    count: int = 0
    note: str = "keep-me"
    log: Annotated[list[str], add] = Field(default_factory=list)


class Other(BaseState):
    count: int | None = None


class Partial(BaseModel):
    count: int = 0
    note: str = ""


class Counter(TypedDict):
    count: int
    note: str
    log: Annotated[list[str], add]


class CounterModel(BaseState):
    count: int = 0
    note: str = "keep"
    log: Annotated[list[str], add] = Field(default_factory=list)


class WorkerInput(BaseModel):
    count: int
    note: str = "worker-default"


class Loose(TypedDict, total=False):
    temp: int
    keep: int


def _single(schema: Any, handler: Any, **options: Any) -> Graph:
    return Graph(schema, **options).flow(START >> handler, handler >> END)


@node
def boom(state: Any) -> dict:
    raise ValueError("boom")


@node
def after(state: Plain) -> dict:
    return {"log": ["after ran"]}


@node
def worker(state: Any) -> dict:
    seen = state.model_dump() if isinstance(state, BaseModel) else dict(state)
    return {"log": [f"count={seen['count']} note={seen['note']}"]}


def test_returning_a_different_base_state_raises() -> None:
    @node
    def other_state(state: Model) -> Other:
        return Other(count=7)

    with pytest.raises(
        TypeError,
        match="Node 'other_state' returned Other; return a dict update or the "
        "state object you received",
    ):
        _single(Model, other_state).invoke({"log": ["a"]})


async def test_a_rejected_return_is_recorded_and_writes_nothing() -> None:
    @node
    def other_state(state: Model) -> Other:
        return Other(count=7)

    graph = _single(Model, other_state, state_store=InMemoryStateStore())

    with pytest.raises(TypeError):
        await graph.ainvoke({"count": 7, "log": ["keep"]}, thread_id="t")

    events = (await graph.history("t")).events
    assert [
        (event.node, event.error_type) for event in events if event.type == "error"
    ] == [("other_state", "TypeError")]
    assert not [event for event in events if event.type == "state_delta" and event.node]
    assert (await graph.load("t"))["count"] == 7


def test_returning_a_plain_model_raises() -> None:
    @node
    def partial(state: Model) -> Partial:
        return Partial(count=5)

    with pytest.raises(TypeError, match="Node 'partial' returned Partial;"):
        _single(Model, partial).invoke({"note": "keep-me"})


def test_returning_a_copy_of_the_state_model_raises() -> None:
    @node
    def copied(state: Model) -> Model:
        return state.model_copy(update={"count": state.count + 1})

    with pytest.raises(TypeError, match="Node 'copied' returned Model;"):
        _single(Model, copied).invoke({})


def test_returning_an_unsupported_value_raises() -> None:
    @node
    def says_done(state: Plain) -> str:
        return "done"

    with pytest.raises(TypeError, match="Node 'says_done' returned str;"):
        _single(Plain, says_done).invoke({"log": [], "note": ""})


def test_command_update_with_the_received_dict_raises() -> None:
    @node(goto=[END])
    def cmd_state(state: Plain) -> Command:
        state["log"].append("x")
        return Command(update=state, goto=END)

    with pytest.raises(
        TypeError,
        match=r"Node 'cmd_state' returned Command\(update=<the state object it "
        r"received>\); pass a dict update",
    ):
        Graph(Plain).flow(START >> cmd_state).invoke({"log": ["a"], "note": "n"})


def test_command_update_with_the_received_model_raises() -> None:
    @node(goto=[END])
    def cmd_state(state: Any) -> Command:
        return Command(update=state, goto=END)

    with pytest.raises(
        TypeError,
        match=r"Node 'cmd_state' returned Command\(update=<the state object it "
        r"received>\)",
    ):
        Graph(Model).flow(START >> cmd_state).invoke({})


def test_command_update_with_a_model_raises() -> None:
    @node(goto=[END])
    def cmd_model(state: Model) -> Command:
        update: Any = Partial(count=1)
        return Command(update=update, goto=END)

    with pytest.raises(
        TypeError,
        match=r"Node 'cmd_model' returned Command\(update=Partial\); "
        r"Command\.update must be a dict update",
    ):
        Graph(Model).flow(START >> cmd_model).invoke({})


class ReturnsItsState(Middleware):
    def on_error(self, ctx: NodeMiddlewareContext, error: Exception) -> Any:
        return ctx.state


@pytest.mark.parametrize("schema", [Plain, Model])
def test_on_error_returning_its_state_does_not_duplicate_lists(schema: Any) -> None:
    graph = _single(schema, boom, middleware=[ReturnsItsState()])

    result = graph.invoke({"log": ["a", "b"], "note": "n"})

    assert result.data["log"] == ["a", "b"]


def test_on_error_changing_its_state_and_returning_an_update_raises() -> None:
    class Peeks(Middleware):
        def on_error(self, ctx: NodeMiddlewareContext, error: Exception) -> dict:
            ctx.state["log"].append("leaked")
            return {"note": "recovered"}

    graph = _single(Plain, boom, middleware=[Peeks()])

    with pytest.raises(GraphExecutionError, match="mutated its input and returned"):
        graph.invoke({"log": ["a"], "note": "n"})


def test_on_error_can_change_and_return_its_state() -> None:
    class Fixes(Middleware):
        def on_error(self, ctx: NodeMiddlewareContext, error: Exception) -> Any:
            ctx.state["log"].append("fixed")
            return ctx.state

    graph = _single(Plain, boom, middleware=[Fixes()])

    assert graph.invoke({"log": ["a"], "note": "n"}).data["log"] == ["a", "fixed"]


def test_on_error_returning_another_model_raises() -> None:
    class ReturnsModel(Middleware):
        def on_error(self, ctx: NodeMiddlewareContext, error: Exception) -> Any:
            return Partial(count=1)

    graph = _single(Model, boom, middleware=[ReturnsModel()])

    with pytest.raises(TypeError, match="Node 'boom' returned Partial;"):
        graph.invoke({})


def test_returning_none_after_mutating_the_input_raises() -> None:
    @node
    def forgets_return(state: Plain) -> None:
        state["note"] = "changed"
        state["log"].append("x")

    with pytest.raises(
        GraphExecutionError,
        match="Node 'forgets_return' mutated its input but returned None",
    ):
        _single(Plain, forgets_return).invoke({"log": [], "note": "orig"})


def test_returning_none_after_mutating_a_model_raises() -> None:
    @node
    def forgets_model(state: Model) -> None:
        state.count = 3

    with pytest.raises(
        GraphExecutionError,
        match="Node 'forgets_model' mutated its input but returned None",
    ):
        _single(Model, forgets_model).invoke({})


def test_returning_none_without_changes_keeps_the_state() -> None:
    @node
    def reads(state: Plain) -> None:
        assert state["log"] == ["a"]

    assert _single(Plain, reads).invoke({"log": ["a"], "note": "n"}).data == {
        "log": ["a"],
        "note": "n",
    }


class NanScore(BaseState):
    model_config = ConfigDict(ser_json_inf_nan="strings")
    score: float = 0.0


def test_returning_none_with_an_unchanged_nan_keeps_the_state() -> None:
    @node
    def reads(state: NanScore) -> None:
        assert state.score != state.score

    result = _single(NanScore, reads).invoke({"score": float("nan")})

    assert math.isnan(result.data["score"])


class Ids(TypedDict, total=False):
    ids: set[int]


def test_an_equal_set_listing_its_items_in_another_order_is_no_mutation() -> None:
    @node
    def rebuilds(state: Ids) -> None:
        state["ids"] = {1, 9}

    assert _single(Ids, rebuilds).invoke({"ids": {9, 1}}).data == {"ids": {1, 9}}


def test_a_change_that_equals_the_old_value_but_not_its_json_is_a_mutation() -> None:
    @node
    def flips(state: dict) -> None:
        state["flag"] = True

    with pytest.raises(
        GraphExecutionError, match="Node 'flips' mutated its input but returned None"
    ):
        _single(dict, flips).invoke({"flag": 1})


@pytest.mark.parametrize("kind", ["update", "command", "send", "end"])
def test_mutating_the_input_and_returning_anything_else_raises(kind: str) -> None:
    @node(goto=[after, END])
    def both(state: Plain) -> Any:
        state["log"].append("x")
        return {
            "update": {"note": "changed"},
            "command": Command(update={"note": "changed"}, goto=[after]),
            "send": Send(after, {"note": "sent"}),
            "end": END,
        }[kind]

    graph = Graph(Plain).flow(START >> both, both >> after, after >> END)

    with pytest.raises(
        GraphExecutionError,
        match=r"Node 'both' mutated its input and returned .*; do one or the other",
    ):
        graph.invoke({"log": [], "note": "orig"})


async def test_mutating_the_input_and_returning_an_update_writes_nothing() -> None:
    @node
    def both(state: Model) -> dict:
        state.count = 5
        return {"note": "changed"}

    graph = _single(Model, both, state_store=InMemoryStateStore())

    with pytest.raises(GraphExecutionError, match="mutated its input and returned"):
        await graph.ainvoke({}, thread_id="t")

    events = [(event.type, event.node) for event in (await graph.history("t")).events]
    assert ("error", "both") in events
    assert ("node_completed", "both") not in events
    assert await graph.load("t") == {"count": 0, "note": "keep-me", "log": []}


def test_empty_goto_routes_nowhere() -> None:
    @node
    def fan(state: Plain) -> Command:
        return Command(update={"note": "fan"}, goto=[])

    graph = Graph(Plain).flow(START >> fan, fan >> after, after >> END)

    assert graph.invoke({"log": [], "note": ""}).data == {"log": [], "note": "fan"}


def test_goto_none_follows_the_edges() -> None:
    @node
    def fan(state: Plain) -> Command:
        return Command(update={"note": "fan"})

    graph = Graph(Plain).flow(START >> fan, fan >> after, after >> END)

    assert graph.invoke({"log": [], "note": ""}).data == {
        "log": ["after ran"],
        "note": "fan",
    }


def test_an_empty_send_fan_out_routes_nowhere() -> None:
    items: list[str] = []

    @node(goto=[worker])
    def fan(state: Counter) -> Command:
        return Command(goto=[Send(worker, {"note": item}) for item in items])

    graph = Graph(Counter).flow(START >> fan, fan >> after, after >> END, worker >> END)

    assert graph.invoke({"count": 0, "note": "", "log": []}).data["log"] == []


async def test_empty_goto_of_a_sibling_stays_empty_after_resume() -> None:
    @node
    def quiet(state: Plain) -> Command:
        return Command(update={"note": "quiet"}, goto=[])

    @node
    def asks(state: Plain) -> dict:
        return {"log": [f"asks:{interrupt('ok?', id='ok')}"]}

    @node(goto=[quiet, asks])
    def split(state: Plain) -> Command:
        return Command(goto=[Send(quiet), Send(asks)])

    graph = Graph(Plain, state_store=InMemoryStateStore()).flow(
        START >> split, quiet >> after, after >> END, asks >> END
    )

    first = await graph.ainvoke({"log": [], "note": ""}, thread_id="t")
    final = await graph.ainvoke(resume=Resume("yes"), thread_id="t")

    assert first.status == "interrupted"
    assert final.data == {"log": ["asks:yes"], "note": "quiet"}


@pytest.mark.parametrize("schema", [dict, Loose])
def test_deleting_a_key_and_returning_the_state_raises(schema: Any) -> None:
    @node
    def delete_key(state: Any) -> Any:
        del state["temp"]
        return state

    with pytest.raises(
        GraphExecutionError,
        match="Node 'delete_key' deleted 'temp' from the state it received",
    ):
        _single(schema, delete_key).invoke({"temp": 1, "keep": 2})


def test_adding_an_unknown_key_and_returning_the_state_raises() -> None:
    @node
    def adds_typo(state: dict) -> dict:
        state["typo"] = 1
        return state

    with pytest.raises(StateUpdateError, match="Unknown field 'typo' in state update"):
        _single(Plain, adds_typo).invoke({"log": [], "note": ""})


def test_adding_an_unknown_key_and_returning_none_raises() -> None:
    @node
    def adds_typo(state: dict) -> None:
        state["typo"] = 1

    with pytest.raises(
        GraphExecutionError,
        match="Node 'adds_typo' mutated its input but returned None",
    ):
        _single(Plain, adds_typo).invoke({"log": [], "note": ""})


@pytest.mark.parametrize("schema", [Counter, CounterModel])
def test_send_payload_keys_must_be_state_fields(schema: Any) -> None:
    @node(goto=[worker])
    def dispatch(state: Any) -> Command:
        return Command(goto=[Send(worker, {"item": 3})])

    graph = Graph(schema).flow(START >> dispatch, worker >> END)

    with pytest.raises(
        StateUpdateError,
        match="Send payload from node 'dispatch' to node 'worker' has keys that "
        "are not state fields: 'item'",
    ):
        graph.invoke({"count": 0, "note": "keep", "log": []})


@pytest.mark.parametrize("schema", [Counter, CounterModel, dict])
@pytest.mark.parametrize(
    ("value", "marker"),
    [(Replace(["reset"]), "Replace"), ([RemoveMessage("m1")], "RemoveMessage")],
)
def test_update_markers_in_a_send_payload_raise(
    schema: Any, value: Any, marker: str
) -> None:
    @node(goto=[worker])
    def dispatch(state: Any) -> Command:
        return Command(goto=[Send(worker, {"log": value})])

    graph = Graph(schema).flow(START >> dispatch, worker >> END)

    with pytest.raises(
        StateUpdateError,
        match=f"Send payload from node 'dispatch' to node 'worker' holds {marker} "
        "in field 'log'",
    ):
        graph.invoke({"count": 0, "note": "keep", "log": []})


def test_untyped_state_accepts_any_send_payload_key() -> None:
    @node
    def echo(state: dict) -> dict:
        return {"seen": state["item"]}

    @node(goto=[echo])
    def dispatch(state: dict) -> Command:
        return Command(goto=[Send(echo, {"item": 3})])

    graph = Graph(dict).flow(START >> dispatch, echo >> END)

    assert graph.invoke({}).data == {"seen": 3}


@pytest.mark.parametrize("schema", [Counter, CounterModel])
def test_a_model_send_payload_sets_only_its_given_fields(schema: Any) -> None:
    @node(goto=[worker])
    def dispatch(state: Any) -> Command:
        return Command(goto=[Send(worker, WorkerInput(count=4))])

    graph = Graph(schema).flow(START >> dispatch, worker >> END)

    result = graph.invoke({"count": 0, "note": "keep", "log": []})

    assert result.data["log"] == ["count=4 note=keep"]


class OpenInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    count: int = 0


@pytest.mark.parametrize("schema", [Counter, CounterModel])
def test_extra_keys_of_a_model_send_payload_must_be_state_fields(schema: Any) -> None:
    @node(goto=[worker])
    def dispatch(state: Any) -> Command:
        return Command(
            goto=[Send(worker, OpenInput.model_validate({"count": 1, "item": 3}))]
        )

    graph = Graph(schema).flow(START >> dispatch, worker >> END)

    with pytest.raises(
        StateUpdateError, match="keys that are not state fields: 'item'"
    ):
        graph.invoke({"count": 0, "note": "keep", "log": []})


@pytest.mark.parametrize("schema", [Counter, CounterModel])
def test_extra_state_fields_of_a_model_send_payload_are_sent(schema: Any) -> None:
    @node(goto=[worker])
    def dispatch(state: Any) -> Command:
        return Command(
            goto=[Send(worker, OpenInput.model_validate({"count": 1, "note": "extra"}))]
        )

    graph = Graph(schema).flow(START >> dispatch, worker >> END)

    result = graph.invoke({"count": 0, "note": "keep", "log": []})

    assert result.data["log"] == ["count=1 note=extra"]


async def test_a_model_send_payload_sets_only_its_given_fields_after_resume() -> None:
    @node(goto=[worker])
    def sender(state: CounterModel) -> Command:
        return Command(goto=[Send(worker, WorkerInput(count=4))])

    @node
    def asks(state: CounterModel) -> dict:
        interrupt("ok?", id="ok")
        return {}

    @node(goto=[sender, asks])
    def split(state: CounterModel) -> Command:
        return Command(goto=[Send(sender), Send(asks)])

    graph = Graph(CounterModel, state_store=InMemoryStateStore()).flow(
        START >> split, asks >> END, worker >> END
    )

    await graph.ainvoke({"note": "keep"}, thread_id="t")
    final = await graph.ainvoke(resume=Resume(True), thread_id="t")

    assert final.data["log"] == ["count=4 note=keep"]


class Queue(TypedDict):
    log: Annotated[list[str], add]
    queue: list[int]


@node
def step(state: Any) -> dict:
    return {"log": ["step"]}


async def test_router_changes_to_its_state_do_not_leak() -> None:
    def popping_router(state: Queue) -> str:
        state["queue"].pop(0)
        return "done"

    graph = Graph(Queue, state_store=InMemoryStateStore()).flow(
        START >> step, step >> branch(popping_router, {"done": END})
    )

    first = await graph.ainvoke({"log": [], "queue": [1, 2, 3]}, thread_id="t")
    second = await graph.ainvoke({"log": ["turn2"]}, thread_id="t")

    assert first.data["queue"] == [1, 2, 3]
    assert second.data["queue"] == [1, 2, 3]
    assert (await graph.load("t"))["queue"] == [1, 2, 3]


class Meta(BaseState):
    log: Annotated[list[str], add] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


def test_predicate_changes_to_a_model_state_do_not_leak() -> None:
    def popping_predicate(state: Meta) -> bool:
        state.meta["queue"].pop(0)
        return True

    graph = Graph(Meta).flow(
        START >> step, step >> when(popping_predicate, END, otherwise=END)
    )

    result = graph.invoke({"meta": {"queue": [1, 2, 3]}})

    assert result.data["meta"] == {"queue": [1, 2, 3]}
