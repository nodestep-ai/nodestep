from typing import Annotated, Any

from pydantic import Field

from nodestep import END, START, BaseState, Graph, add, node, subgraph


class Proofread(BaseState):
    title: str = ""
    text: str = ""
    issues: Annotated[list[str], add] = Field(default_factory=list)


@node
def check_title(state: Proofread) -> dict:
    if not state.title[:1].isupper():
        return {"issues": ["the title should start with a capital letter"]}
    return {"issues": []}


@node
def check_length(state: Proofread) -> dict:
    words = len(state.text.split())
    if words < 12:
        return {"issues": [f"the body has {words} words, at least 12 are needed"]}
    return {"issues": []}


proofreader = Graph(Proofread, name="proofreader").flow(
    START >> check_title,
    check_title >> check_length,
    check_length >> END,
)


class Article(BaseState):
    title: str = ""
    body: str = ""
    issues: list[str] = Field(default_factory=list)
    published: bool = False


def to_proofreader(state: Article) -> dict[str, Any]:
    return {"title": state.title, "text": state.body}


def from_proofreader(child: dict[str, Any]) -> dict[str, Any]:
    return {"issues": child["issues"]}


proofread = subgraph(
    "proofread",
    proofreader,
    state_in=to_proofreader,
    state_out=from_proofreader,
    child_thread="fresh",
)


@node
def publish(state: Article) -> dict:
    return {"published": not state.issues}


graph = Graph(Article, name="article").flow(
    START >> proofread,
    proofread >> publish,
    publish >> END,
)


def main() -> None:
    draft = graph.invoke({"title": "release notes", "body": "Version 2 is out."})
    print("draft issues:", draft.state.issues)
    print("draft published:", draft.state.published)

    final = graph.invoke(
        {
            "title": "Release notes",
            "body": "Version 2 is out. It adds streaming, durable threads and "
            "a sandbox for trying graphs.",
        }
    )
    print("final issues:", final.state.issues)
    print("final published:", final.state.published)


if __name__ == "__main__":
    main()
