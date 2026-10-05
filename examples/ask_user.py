from pydantic import BaseModel

from nodestep import (
    END,
    START,
    BaseState,
    Graph,
    InMemoryStateStore,
    Resume,
    interrupt,
    node,
)
from nodestep.middleware.base import Middleware, NodeMiddlewareContext


class AskUserRequest(BaseModel):
    inputs: dict[str, str]


class CliAskUserMiddleware(Middleware):
    def on_interrupt(
        self, ctx: NodeMiddlewareContext, payload: object
    ) -> object | None:
        if isinstance(payload, AskUserRequest):
            return {key: input(f"{label}: ") for key, label in payload.inputs.items()}
        return None


class State(BaseState):
    name: str = ""
    greeting: str = ""


@node
def ask_name(state: State) -> State:
    response = interrupt(
        AskUserRequest(inputs={"name": "What is your name?"}), id="name"
    )
    state.name = response["name"]
    return state


@node
def greet(state: State) -> State:
    state.greeting = f"Hello, {state.name}!"
    return state


graph = Graph(
    State,
    state_store=InMemoryStateStore(),
).flow(
    START >> ask_name,
    ask_name >> greet,
    greet >> END,
)

prompting_graph = Graph(
    State,
    state_store=InMemoryStateStore(),
    middleware=[CliAskUserMiddleware()],
).flow(
    START >> ask_name,
    ask_name >> greet,
    greet >> END,
)


def main() -> None:
    result = prompting_graph.invoke({}, thread_id="greeting")
    print(result.state.greeting)


def main_without_middleware() -> None:
    paused = graph.invoke({}, thread_id="chat")
    print("interrupted:", paused.status)
    print("payload:", paused.interrupts)

    name = input("What is your name? ")
    resumed = graph.invoke(None, thread_id="chat", resume=Resume({"name": name}))
    print(resumed.state.greeting)


if __name__ == "__main__":
    main()
