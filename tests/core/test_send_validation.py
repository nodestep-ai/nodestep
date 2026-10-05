import pytest

from nodestep import END, START, Command, Graph, Send, interrupt, node
from nodestep.exceptions import GraphConfigError, StateStoreError


@node
def worker(state: dict) -> dict:
    return state


def test_send_accepts_node() -> None:
    send = Send(worker, {"k": "v"})
    assert send.node is worker
    assert send.payload == {"k": "v"}


def test_send_accepts_node_keyword() -> None:
    send = Send(node=worker)
    assert send.node is worker
    assert send.payload is None


def test_send_rejects_string_target() -> None:
    with pytest.raises(GraphConfigError, match="Send target must be a Node"):
        Send("worker")


def test_send_rejects_plain_callable() -> None:
    def not_a_node(state: dict) -> dict:
        return state

    with pytest.raises(GraphConfigError, match="Send target must be a Node"):
        Send(not_a_node)


class Handle:
    pass


@node
def uses_handle(state: dict) -> dict:
    return {"used": isinstance(state["handle"], Handle)}


@node(goto=[uses_handle])
def sends_handle(state: dict) -> Command:
    return Command(goto=[Send(uses_handle, {"handle": Handle()})])


async def test_a_send_payload_that_cannot_be_stored_is_refused_without_a_store() -> (
    None
):
    graph = Graph(dict).flow(START >> sends_handle, uses_handle >> END)

    with pytest.raises(
        StateStoreError, match=r"'handle'.*Send payload from node 'sends_handle'"
    ):
        await graph.ainvoke({})


@node
def asks(state: dict) -> dict:
    return {"answer": interrupt("ok?", id="ok")}


@node(goto=[uses_handle])
def sends_handle_later(state: dict) -> Command:
    return Command(goto=[Send(uses_handle, {"handle": Handle()})])


@node(goto=[asks, sends_handle_later])
def fans_out(state: dict) -> Command:
    return Command(goto=[asks, sends_handle_later])


async def test_a_send_payload_that_cannot_be_stored_fails_a_pausing_superstep() -> None:
    graph = Graph(dict).flow(START >> fans_out, asks >> END, uses_handle >> END)

    with pytest.raises(StateStoreError, match="'handle'"):
        await graph.ainvoke({})
