from nodestep import END, START, BaseState, Graph, branch, node


class State(BaseState):
    text: str = ""
    intent: str = ""
    answer: str = ""


@node
def classify(state: State) -> State:
    state.intent = "support" if "help" in state.text.lower() else "sales"
    return state


@node
def support(state: State) -> State:
    state.answer = "Have you tried turning it off and on again?"
    return state


@node
def sales(state: State) -> State:
    state.answer = "Want to hear about our new pricing?"
    return state


def route_after_classify(state: State) -> str:
    return state.intent


graph = Graph(State).flow(
    START >> classify,
    classify >> branch(route_after_classify, {"support": support, "sales": sales}),
    support >> END,
    sales >> END,
)


def main() -> None:
    result = graph.invoke({"text": "I need help"})
    print(result.state.intent, result.state.answer)


if __name__ == "__main__":
    main()
