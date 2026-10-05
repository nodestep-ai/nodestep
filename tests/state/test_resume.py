import json
from collections.abc import Callable
from typing import Annotated, TypedDict

import pytest
from pydantic import BaseModel

from nodestep import (
    END,
    START,
    Command,
    FilesystemStateStore,
    Graph,
    InMemoryStateStore,
    Resume,
    Send,
    add,
    interrupt,
    node,
)
from nodestep.core.stream import InterruptEventData
from nodestep.exceptions import GraphConfigError, ResumeError
from nodestep.state import StateStore
from nodestep.state.history import History


class S(TypedDict, total=False):
    log: Annotated[list[str], add]


@node
def two_questions(state: S) -> dict:
    name = interrupt("name?", id="name")
    age = interrupt("age?", id="age")
    return {"log": [f"name={name} age={age}"]}


def _questions_graph(store: StateStore) -> Graph:
    return Graph(S, state_store=store).flow(
        START >> two_questions, two_questions >> END
    )


async def test_sequential_answers_are_all_kept() -> None:
    graph = _questions_graph(InMemoryStateStore())

    first = await graph.ainvoke({}, thread_id="q")
    second = await graph.ainvoke(resume=Resume("Alice"), thread_id="q")
    final = await graph.ainvoke(resume=Resume(30), thread_id="q")

    assert list(first.interrupts) == ["two_questions:name"]
    assert list(second.interrupts) == ["two_questions:age"]
    assert final.data["log"] == ["name=Alice age=30"]


async def test_answers_survive_a_store_reload(tmp_path) -> None:
    path = tmp_path / "history.json"

    await _questions_graph(FilesystemStateStore(path)).ainvoke({}, thread_id="q")
    await _questions_graph(FilesystemStateStore(path)).ainvoke(
        resume=Resume("Alice"), thread_id="q"
    )
    final = await _questions_graph(FilesystemStateStore(path)).ainvoke(
        resume=Resume(30), thread_id="q"
    )

    assert final.data["log"] == ["name=Alice age=30"]


@node
def worker(state: S) -> dict:
    return {"log": [f"worker={interrupt('go?', id='go')}"]}


@node(goto=[worker])
def fan_out(state: S) -> Command:
    return Command(goto=[Send(worker, {}), Send(worker, {})])


def _workers_graph() -> Graph:
    return Graph(S, state_store=InMemoryStateStore()).flow(
        START >> fan_out, worker >> END
    )


async def test_answers_reach_each_send_task_by_its_key() -> None:
    graph = _workers_graph()

    first = await graph.ainvoke({}, thread_id="w")
    final = await graph.ainvoke(
        resume=Resume(answers={"worker:go": "a", "worker[1]:go": "b"}),
        thread_id="w",
    )

    assert sorted(first.interrupts) == ["worker:go", "worker[1]:go"]
    assert sorted(final.data["log"]) == ["worker=a", "worker=b"]


async def test_unknown_resume_key_raises() -> None:
    graph = _workers_graph()
    await graph.ainvoke({}, thread_id="w")

    with pytest.raises(ResumeError, match="nope:1"):
        await graph.ainvoke(resume=Resume(answers={"nope:1": "x"}), thread_id="w")


async def test_one_value_for_two_send_tasks_raises() -> None:
    graph = _workers_graph()
    await graph.ainvoke({}, thread_id="w")

    with pytest.raises(ResumeError, match=r"worker\[1\]:go"):
        await graph.ainvoke(resume=Resume("b"), thread_id="w")


@node
def plain(state: S) -> dict:
    return {"log": ["plain"]}


async def test_resume_with_nothing_pending_raises() -> None:
    graph = Graph(S, state_store=InMemoryStateStore()).flow(
        START >> plain, plain >> END
    )
    await graph.ainvoke({}, thread_id="n")

    with pytest.raises(ResumeError):
        await graph.ainvoke(resume=Resume("x"), thread_id="n")


async def test_resume_without_a_store_raises() -> None:
    graph = Graph(S).flow(START >> two_questions, two_questions >> END)

    with pytest.raises(GraphConfigError):
        await graph.ainvoke(resume=Resume("x"), thread_id="none")


@node
def flaky_deploy(state: S) -> dict:
    answer = interrupt("deploy?", id="deploy")
    raise RuntimeError(f"deploy failed after {answer}")


async def test_interrupt_is_cleared_when_its_node_fails() -> None:
    graph = Graph(S, state_store=InMemoryStateStore()).flow(
        START >> flaky_deploy, flaky_deploy >> END
    )
    await graph.ainvoke({}, thread_id="f")
    with pytest.raises(RuntimeError):
        await graph.ainvoke(resume=Resume("yes"), thread_id="f")

    with pytest.raises(ResumeError):
        await graph.ainvoke(resume=Resume("again"), thread_id="f")


@node
def branch_a(state: S) -> dict:
    return {"log": ["A"]}


@node
def after_a(state: S) -> dict:
    return {"log": ["AFTER_A"]}


@node
def branch_b(state: S) -> dict:
    return {"log": [f"B={interrupt('approve b?', id='approve')}"]}


@node(goto=[branch_a, branch_b])
def fan_ab(state: S) -> Command:
    return Command(goto=[Send(branch_a, {}), Send(branch_b, {})])


def _saw_b(state: S) -> bool:
    return any(item.startswith("B=") for item in state.get("log", []))


async def test_sibling_branch_continues_after_resume() -> None:
    graph = Graph(S, state_store=InMemoryStateStore()).flow(
        START >> fan_ab,
        branch_a >> after_a,
        after_a >> END,
        branch_b >> END,
    )

    await graph.ainvoke({}, thread_id="p")
    final = await graph.ainvoke(resume=Resume("ok"), thread_id="p")

    assert final.data["log"] == ["A", "B=ok", "AFTER_A"]


async def test_sibling_router_sees_state_after_resume() -> None:
    from nodestep import when

    graph = Graph(S, state_store=InMemoryStateStore()).flow(
        START >> fan_ab,
        branch_a >> when(_saw_b, after_a, otherwise=END),
        after_a >> END,
        branch_b >> END,
    )

    await graph.ainvoke({}, thread_id="r")
    final = await graph.ainvoke(resume=Resume("ok"), thread_id="r")

    assert final.data["log"] == ["A", "B=ok", "AFTER_A"]


@node(goto=[after_a])
def branch_a_command(state: S) -> Command:
    return Command(update={"log": ["A"]}, goto=after_a)


async def test_sibling_command_route_survives_two_resumes() -> None:
    @node
    def branch_b_twice(state: S) -> dict:
        first = interrupt("first?", id="first")
        second = interrupt("second?", id="second")
        return {"log": [f"B={first}{second}"]}

    @node(goto=[branch_a_command, branch_b_twice])
    def fan_twice(state: S) -> Command:
        return Command(goto=[Send(branch_a_command, {}), Send(branch_b_twice, {})])

    graph = Graph(S, state_store=InMemoryStateStore()).flow(
        START >> fan_twice,
        after_a >> END,
        branch_b_twice >> END,
    )

    await graph.ainvoke({}, thread_id="c")
    await graph.ainvoke(resume=Resume("1"), thread_id="c")
    final = await graph.ainvoke(resume=Resume("2"), thread_id="c")

    assert final.data["log"] == ["A", "B=12", "AFTER_A"]


async def test_terminal_only_stream_reports_the_interrupt() -> None:
    graph = _questions_graph(InMemoryStateStore())

    events = [event async for event in graph.astream({}, thread_id="s", stream_mode=[])]

    assert events[-1].mode == "interrupt"
    assert isinstance(events[-1].data, InterruptEventData)
    assert [item.payload for item in events[-1].data.interrupts.values()] == ["name?"]


attempts: dict[str, int] = {}


@node
def step_a(state: S) -> dict:
    return {"log": ["A"]}


@node
def step_b(state: S) -> dict:
    attempts["b"] = attempts.get("b", 0) + 1
    if attempts["b"] == 1:
        raise RuntimeError("provider 503")
    return {"log": ["B"]}


def _flaky_graph(store: InMemoryStateStore) -> Graph:
    return Graph(S, state_store=store).flow(
        START >> step_a, step_a >> step_b, step_b >> END
    )


async def test_none_input_continues_a_failed_run() -> None:
    attempts.clear()
    graph = _flaky_graph(InMemoryStateStore())
    with pytest.raises(RuntimeError):
        await graph.ainvoke({}, thread_id="r")

    final = await graph.ainvoke(None, thread_id="r")

    assert final.data["log"] == ["A", "B"]


async def test_new_input_after_a_failure_starts_over() -> None:
    attempts.clear()
    graph = _flaky_graph(InMemoryStateStore())
    with pytest.raises(RuntimeError):
        await graph.ainvoke({}, thread_id="r")

    final = await graph.ainvoke({}, thread_id="r")

    assert final.data["log"] == ["A", "A", "B"]


charges: list[str] = []


@node
def charge(state: S) -> dict:
    charges.append("charged")
    return {"log": ["charged"]}


@node
def confirm(state: S) -> dict:
    return {"log": [f"confirmed={interrupt('ship it?', id='ship')}"]}


async def test_new_input_on_a_paused_thread_raises_and_a_fork_starts_over() -> None:
    charges.clear()
    graph = Graph(S, state_store=InMemoryStateStore()).flow(
        START >> charge, charge >> confirm, confirm >> END
    )
    await graph.ainvoke({}, thread_id="p")

    with pytest.raises(ResumeError, match=r"paused on \['confirm:ship'\].*fork"):
        await graph.ainvoke({"log": ["new"]}, thread_id="p")
    first_event = (await graph.history("p")).events[0]
    branch = await graph.fork("p", from_=first_event.id, name="over")
    second = await graph.ainvoke({"log": ["new"]}, thread_id="p", branch_id=branch.id)

    assert charges == ["charged", "charged"]
    assert second.status == "interrupted"
    assert second.data["log"] == ["new", "charged"]
    paused = await graph.ainvoke(resume=Resume(True), thread_id="p")
    assert paused.data["log"] == ["charged", "confirmed=True"]


async def test_checkpoints_store_tasks_without_full_state() -> None:
    import json

    store = InMemoryStateStore()
    attempts.clear()
    attempts["b"] = 1
    await _flaky_graph(store).ainvoke({"log": ["seed"]}, thread_id="k")

    for checkpoint in store.checkpoints:
        for task in json.loads(checkpoint.active_json):
            assert "input_state" not in task


class Answer(BaseModel):
    value: int


seen_answer_types: list[str] = []


@node
def ask_for_models(state: S) -> dict:
    first = interrupt("one?", id="one")
    seen_answer_types.append(type(first).__name__)
    second = interrupt("two?", id="two")
    return {"log": [f"{first['value']}+{second['value']}"]}


async def test_resume_answers_have_the_same_form_on_every_resume() -> None:
    seen_answer_types.clear()
    graph = Graph(S, state_store=InMemoryStateStore()).flow(
        START >> ask_for_models, ask_for_models >> END
    )

    await graph.ainvoke({}, thread_id="t")
    await graph.ainvoke(resume=Resume(Answer(value=1)), thread_id="t")
    final = await graph.ainvoke(resume=Resume(Answer(value=2)), thread_id="t")

    assert seen_answer_types == ["dict", "dict"]
    assert final.data["log"] == ["1+2"]


failures: dict[str, int] = {}


@node
def fails_first_time(state: S) -> dict:
    failures["a2"] = failures.get("a2", 0) + 1
    if failures["a2"] == 1:
        raise RuntimeError("a2 failed")
    return {"log": ["a2"]}


@node
def after_a2(state: S) -> dict:
    return {"log": ["after_a2"]}


@node
def b_asks(state: S) -> dict:
    return {"log": [f"b={interrupt('b?', id='b')}"]}


@node(goto=[fails_first_time, b_asks])
def fan_fail_and_ask(state: S) -> Command:
    return Command(goto=[Send(fails_first_time, {}), Send(b_asks, {})])


async def test_failed_superstep_is_continued_not_resumed() -> None:
    failures.clear()
    graph = Graph(S, state_store=InMemoryStateStore()).flow(
        START >> fan_fail_and_ask,
        fails_first_time >> after_a2,
        after_a2 >> END,
        b_asks >> END,
    )
    with pytest.raises(RuntimeError, match="a2 failed"):
        await graph.ainvoke({}, thread_id="f")

    with pytest.raises(ResumeError):
        await graph.ainvoke(resume=Resume("ok"), thread_id="f")
    paused = await graph.ainvoke(None, thread_id="f")
    final = await graph.ainvoke(resume=Resume("ok"), thread_id="f")

    assert paused.status == "interrupted"
    assert sorted(final.data["log"]) == ["a2", "after_a2", "b=ok"]


@node
def prepare(state: S) -> dict:
    return {"log": ["prepared"]}


@node
def fails_after_answer(state: S) -> dict:
    answer = interrupt("go?", id="go")
    failures["after"] = failures.get("after", 0) + 1
    if failures["after"] == 1:
        raise RuntimeError("failed after the answer")
    return {"log": [f"done {answer}"]}


async def test_failure_right_after_a_resume_can_be_continued() -> None:
    failures.clear()
    graph = Graph(S, state_store=InMemoryStateStore()).flow(
        START >> prepare, prepare >> fails_after_answer, fails_after_answer >> END
    )
    await graph.ainvoke({}, thread_id="r")
    with pytest.raises(RuntimeError, match="after the answer"):
        await graph.ainvoke(resume=Resume("yes"), thread_id="r")

    final = await graph.ainvoke(None, thread_id="r")

    assert final.status == "completed"
    assert final.data["log"] == ["prepared", "done yes"]


seen_inputs: list[list[str]] = []


@node
def records_input_then_asks(state: S) -> dict:
    seen_inputs.append(list(state.get("log", [])))
    return {"log": [f"B={interrupt('b?', id='b')}"]}


@node(goto=[branch_a, records_input_then_asks])
def fan_a_and_recorder(state: S) -> Command:
    return Command(goto=[Send(branch_a, {}), Send(records_input_then_asks, {})])


async def test_restored_task_sees_the_same_input_as_its_first_run() -> None:
    seen_inputs.clear()
    graph = Graph(S, state_store=InMemoryStateStore()).flow(
        START >> fan_a_and_recorder,
        branch_a >> END,
        records_input_then_asks >> END,
    )

    paused = await graph.ainvoke({"log": ["seed"]}, thread_id="i")
    final = await graph.ainvoke(resume=Resume("ok"), thread_id="i")

    assert seen_inputs == [["seed"], ["seed"]]
    assert paused.data["log"] == ["seed"]
    assert final.data["log"] == ["seed", "A", "B=ok"]


class Owner(TypedDict, total=False):
    owner: str


@node
def claims_owner(state: Owner) -> dict:
    return {"owner": "a"}


@node
def asks_then_claims(state: Owner) -> dict:
    interrupt("claim?", id="claim")
    return {"owner": "b"}


@node(goto=[claims_owner, asks_then_claims])
def fan_owners(state: Owner) -> Command:
    return Command(goto=[Send(claims_owner, {}), Send(asks_then_claims, {})])


async def test_conflicts_across_a_resumed_superstep_are_detected() -> None:
    from nodestep.exceptions import InvalidUpdateError

    graph = Graph(Owner, state_store=InMemoryStateStore()).flow(
        START >> fan_owners,
        claims_owner >> END,
        asks_then_claims >> END,
    )
    await graph.ainvoke({}, thread_id="o")

    with pytest.raises(InvalidUpdateError):
        await graph.ainvoke(resume=Resume(True), thread_id="o")


class _DropsStoredKey(InMemoryStateStore):
    def __init__(self, event_type: str, drop: Callable[[dict], dict]) -> None:
        super().__init__()
        self.event_type = event_type
        self.drop = drop
        self.active = False

    async def get_history(self, thread_id: str, branch_id: str = "main") -> History:
        history = await super().get_history(thread_id, branch_id)
        if not self.active:
            return history
        events = [
            event.model_copy(
                update={"data_json": json.dumps(self.drop(json.loads(event.data_json)))}
            )
            if event.type == self.event_type and event.data_json
            else event
            for event in history.events
        ]
        return history.model_copy(update={"events": events})


def _without(key: str) -> Callable[[dict], dict]:
    return lambda data: {name: value for name, value in data.items() if name != key}


@pytest.mark.parametrize("key", ["task_id", "root_run_id"])
async def test_resume_raises_when_a_stored_interrupt_lacks_a_key(key: str) -> None:
    store = _DropsStoredKey("interrupted", _without(key))
    graph = _questions_graph(store)
    await graph.ainvoke({}, thread_id="q")
    store.active = True

    with pytest.raises(ResumeError, match=f"'{key}'"):
        await graph.ainvoke(resume=Resume("Alice"), thread_id="q")


async def test_continuing_raises_when_a_stored_task_lacks_its_task_id() -> None:
    def drop_task_ids(data: dict) -> dict:
        tasks = [_without("task_id")(task) for task in data.get("tasks") or []]
        return {**data, "tasks": tasks}

    attempts.clear()
    store = _DropsStoredKey("superstep_started", drop_task_ids)
    graph = _flaky_graph(store)
    with pytest.raises(RuntimeError):
        await graph.ainvoke({}, thread_id="r")
    store.active = True

    with pytest.raises(ResumeError, match="'task_id'"):
        await graph.ainvoke(None, thread_id="r")
