import asyncio
from typing import Annotated

from pydantic import Field

from nodestep import (
    END,
    START,
    BaseState,
    Graph,
    InMemoryStateStore,
    add,
    node,
)


class State(BaseState):
    messages: Annotated[list[str], add] = Field(default_factory=list)


@node
def reply(state: State) -> State:
    last = state.messages[-1] if state.messages else ""
    state.messages = [*state.messages, f"echo: {last}"]
    return state


graph = Graph(State, state_store=InMemoryStateStore()).flow(
    START >> reply, reply >> END
)


async def main() -> None:
    await graph.ainvoke({"messages": ["hi"]}, thread_id="chat-1")
    history = await graph.history("chat-1")

    first_event = history.events[0]
    print("first event id:", first_event.id)

    state_at_first = await graph.load("chat-1", at=first_event.id)
    print("state at first event:", state_at_first)

    branch = await graph.fork("chat-1", from_=first_event.id, name="rewrite")
    print("branch id:", branch.id)
    print("branches:", [record.id for record in await graph.branches("chat-1")])

    rewritten = await graph.ainvoke(
        {"messages": ["hello again"]},
        thread_id="chat-1",
        branch_id=branch.id,
    )
    print("rewritten messages:", rewritten.state.messages)

    main_state = await graph.get_state("chat-1")
    print("main messages:", main_state.value.messages)


if __name__ == "__main__":
    asyncio.run(main())
