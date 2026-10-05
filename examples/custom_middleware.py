import re

from nodestep import (
    END,
    START,
    BaseState,
    Graph,
    Middleware,
    RunOutcome,
    node,
)
from nodestep.middleware import (
    GraphMiddlewareContext,
    NodeMiddlewareContext,
    Replacement,
)

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


class RedactEmails(Middleware):
    def after_node(self, ctx: NodeMiddlewareContext) -> Replacement | None:
        if not isinstance(ctx.state, dict):
            return None
        redacted = {
            key: EMAIL.sub("[email]", value) if isinstance(value, str) else value
            for key, value in ctx.state.items()
        }
        return ctx.replace(redacted)


class AuditLog(Middleware):
    def __init__(self) -> None:
        self.entries: list[str] = []

    def before_node(self, ctx: NodeMiddlewareContext) -> None:
        self.entries.append(f"step {ctx.step}: {ctx.node_name} starts")

    def after_node(self, ctx: NodeMiddlewareContext) -> None:
        self.entries.append(f"step {ctx.step}: {ctx.node_name} returned {ctx.state}")

    def on_run_end(self, ctx: GraphMiddlewareContext, outcome: RunOutcome) -> None:
        self.entries.append(f"run of {ctx.graph_name} {outcome.status}")


class Ticket(BaseState):
    text: str = ""
    note: str = ""
    reply: str = ""


@node
def take_note(state: Ticket) -> dict:
    return {"note": f"Customer wrote: {state.text}"}


@node
def answer(state: Ticket) -> dict:
    return {"reply": "Thanks, we will call you back today."}


audit = AuditLog()

graph = Graph(Ticket, name="tickets", middleware=[audit, RedactEmails()]).flow(
    START >> take_note,
    take_note >> answer,
    answer >> END,
)


def main() -> None:
    result = graph.invoke({"text": "Please call me, or write to ada@example.com."})
    for entry in audit.entries:
        print(entry)
    print("stored note:", result.state.note)


if __name__ == "__main__":
    main()
