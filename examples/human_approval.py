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


class State(BaseState):
    approved: bool = False


@node
def approve(state: State) -> State:
    value = interrupt({"prompt": "Confirm the deployment?"}, id="confirm")
    state.approved = bool(value)
    return state


graph = Graph(
    State,
    state_store=InMemoryStateStore(),
).flow(
    START >> approve,
    approve >> END,
)


def main() -> None:
    paused = graph.invoke({}, thread_id="deploy-1")
    print("status:", paused.status)
    print("interrupts:", paused.interrupts)

    resumed = graph.invoke(None, thread_id="deploy-1", resume=Resume(True))
    print("approved:", resumed.state.approved)


if __name__ == "__main__":
    main()
