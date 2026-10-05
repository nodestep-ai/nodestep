import sys
import tempfile
import uuid
from pathlib import Path

from nodestep import (
    END,
    START,
    BaseState,
    FilesystemStateStore,
    Graph,
    Resume,
    interrupt,
    node,
)

STORE_PATH = Path(tempfile.gettempdir()) / "nodestep-examples" / "durable_thread.jsonl"


class Order(BaseState):
    item: str = "2 chairs"
    approved: bool = False
    status: str = "new"


@node
def approve(state: Order) -> dict:
    answer = interrupt({"question": f"Ship {state.item}?"}, id="ship")
    return {"approved": answer == "yes"}


@node
def ship(state: Order) -> dict:
    return {"status": "shipped" if state.approved else "cancelled"}


graph = Graph(Order, name="durable-order").flow(
    START >> approve,
    approve >> ship,
    ship >> END,
)


def start(store: FilesystemStateStore) -> None:
    thread_id = f"order-{uuid.uuid4().hex[:8]}"
    paused = graph.invoke({}, thread_id=thread_id, state_store=store)
    pending = next(iter(paused.interrupts.values()))
    print(f"Thread {thread_id} is paused: {pending.payload['question']}")
    print(f"Stored in {store.path}")
    print(f"Resume it from another process with the arguments: {thread_id} yes")


def resume(store: FilesystemStateStore, thread_id: str, answer: str) -> None:
    result = graph.invoke(
        None, thread_id=thread_id, resume=Resume(answer), state_store=store
    )
    print(f"Thread {thread_id} {result.status}: {result.state.status}")


def main(arguments: list[str]) -> None:
    store = FilesystemStateStore(STORE_PATH)
    if arguments:
        thread_id, answer = arguments
        resume(store, thread_id, answer)
    else:
        start(store)


if __name__ == "__main__":
    main(sys.argv[1:])
