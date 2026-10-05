import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from nodestep.core.command import END, START, Command, Send
from nodestep.core.flow import branch, when
from nodestep.core.graph import Graph
from nodestep.core.node import node
from nodestep.exceptions import (
    GraphExecutionError,
    GraphTimeoutError,
    RunLimitExceededError,
)


class State(BaseModel):
    count: int = 0
    done: bool = False


@node
def increment(state: State) -> dict:
    return {"count": state.count + 1, "done": True}


@node
def finish(state: State) -> dict:
    return {"count": state.count + 10}


def is_done(state: State) -> bool:
    return state.done


def test_scheduler_runs_conditional_route() -> None:
    graph = Graph(State).flow(
        START >> increment,
        increment >> when(is_done, finish, otherwise=END),
        finish >> END,
    )

    result = graph.invoke({})

    assert result.state.count == 11
    assert result.status == "completed"


@node(goto=[END])
def jump(state: State) -> Command:
    return Command(update={"count": 5}, goto=END)


def test_command_goto_overrides_blueprint() -> None:
    graph = Graph(State).flow(START >> jump, jump >> finish, finish >> END)

    result = graph.invoke({})

    assert result.state.count == 5


@node(goto=[finish])
def jump_to_finish(state: State) -> Command:
    return Command(update={"count": 1}, goto=finish)


def test_command_goto_accepts_node_function() -> None:
    graph = Graph(State).flow(START >> jump_to_finish, finish >> END)

    result = graph.invoke({})

    assert result.state.count == 11


def test_branch_unknown_key_raises() -> None:
    graph = Graph(State).flow(
        START >> increment, increment >> branch(lambda state: "missing", {"done": END})
    )

    with pytest.raises(GraphExecutionError, match="unmapped key"):
        graph.invoke({})


def test_when_false_uses_otherwise_target() -> None:
    graph = Graph(State).flow(
        START >> increment,
        increment >> when(lambda state: False, finish, otherwise=END),
        finish >> END,
    )

    result = graph.invoke({})

    assert result.state.count == 1


@node
def loop(state: State) -> dict:
    return {"count": state.count + 1}


def test_loop_stops_at_max_steps() -> None:
    graph = Graph(State, max_steps=2).flow(START >> loop, loop >> loop)

    with pytest.raises(RunLimitExceededError):
        graph.invoke({})


@pytest.mark.asyncio
async def test_graph_timeout():
    @node
    async def slow(state):
        await asyncio.sleep(10)
        return {"v": 1}

    graph = Graph(dict, timeout=0.05).flow(START >> slow, slow >> END)
    with pytest.raises(GraphTimeoutError):
        await graph.ainvoke({"v": 0})


class Ticket(BaseModel):
    amount: int = 0
    decision: str = ""
    log: list[str] = []


@node
def refund(state: Ticket) -> dict:
    return {"decision": "refund"}


@node
def deny(state: Ticket) -> dict:
    return {"decision": "deny"}


@node(goto=[refund, deny])
def decide(state: Ticket) -> Command:
    return Command(goto=refund if state.amount < 100 else deny)


def test_declared_goto_target_runs_without_a_dummy_edge() -> None:
    graph = Graph(Ticket).flow(START >> decide, refund >> END, deny >> END)

    assert graph.invoke({"amount": 10}).state.decision == "refund"
    assert graph.invoke({"amount": 500}).state.decision == "deny"


@node(goto=[refund])
def goes_to_undeclared(state: Ticket) -> Command:
    return Command(goto=deny)


def test_undeclared_command_target_raises() -> None:
    graph = Graph(Ticket).flow(START >> goes_to_undeclared, refund >> deny, deny >> END)

    with pytest.raises(
        GraphExecutionError,
        match=r"Node 'goes_to_undeclared' routed to 'deny'.*not declared in its goto=",
    ):
        graph.invoke({})


@node
def sends_undeclared(state: Ticket) -> Command:
    return Command(goto=[Send(refund, {"amount": 1})])


def test_undeclared_send_target_raises() -> None:
    graph = Graph(Ticket).flow(
        START >> sends_undeclared, sends_undeclared >> refund, refund >> END
    )

    with pytest.raises(GraphExecutionError, match="not declared in its goto="):
        graph.invoke({})


@node
def returns_end(state: Ticket) -> object:
    return END


def test_undeclared_end_raises() -> None:
    graph = Graph(Ticket).flow(START >> returns_end, returns_end >> END)

    with pytest.raises(GraphExecutionError, match="routed to END"):
        graph.invoke({})


@node(goto=[END])
def ends_explicitly(state: Ticket) -> Command:
    return Command(update={"decision": "stop"}, goto=END)


def test_declared_end_target_ends_the_run() -> None:
    graph = Graph(Ticket).flow(START >> ends_explicitly)

    assert graph.invoke({}).state.decision == "stop"


@node(goto=[refund])
def forgets_goto(state: Ticket) -> dict:
    return {"decision": "none"}


def test_goto_only_node_without_a_route_raises() -> None:
    graph = Graph(Ticket).flow(START >> forgets_goto, refund >> END)

    with pytest.raises(
        GraphExecutionError,
        match="'forgets_goto' returned no goto and has no outgoing edge",
    ):
        graph.invoke({})


@node(goto=[forgets_goto])
def hands_over(state: Ticket) -> Command:
    return Command(goto=forgets_goto)


def test_goto_target_that_returns_no_route_raises_instead_of_ending() -> None:
    graph = Graph(Ticket).flow(START >> hands_over, refund >> END)

    with pytest.raises(
        GraphExecutionError,
        match="'forgets_goto' returned no goto and has no outgoing edge",
    ):
        graph.invoke({})


@node(name="refund")
def impostor_refund(state: Ticket) -> dict:
    return {"decision": "impostor"}


@node(goto=[refund])
def sends_impostor(state: Ticket) -> Command:
    return Command(goto=[Send(impostor_refund, {})])


def test_route_to_a_different_node_with_the_same_name_raises() -> None:
    graph = Graph(Ticket).flow(START >> sends_impostor, refund >> END)

    with pytest.raises(GraphExecutionError, match="different node named 'refund'"):
        graph.invoke({})


def returns_one(state: State) -> Any:
    return 1


def test_when_predicate_must_return_a_bool() -> None:
    graph = Graph(State).flow(
        START >> increment,
        increment >> when(returns_one, finish, otherwise=END),
        finish >> END,
    )

    with pytest.raises(
        GraphExecutionError,
        match="when\\(\\) predicate of node 'increment' returned int; it must return a bool",
    ):
        graph.invoke({})


def returns_list(state: State) -> list[str]:
    return ["done"]


def test_unhashable_branch_key_is_reported_as_unmapped() -> None:
    graph = Graph(State).flow(
        START >> increment, increment >> branch(returns_list, {"done": END})
    )

    with pytest.raises(GraphExecutionError, match="'increment' returned unmapped key"):
        graph.invoke({})
