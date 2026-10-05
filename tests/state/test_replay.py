from typing import Annotated, Any

from pydantic import BaseModel, Field

from nodestep import END, START, Graph, InMemoryStateStore, node


class Note(BaseModel):
    text: str


async def test_replay_passes_typed_values_to_reducers() -> None:
    seen: list[str] = []

    def typed_add(current: list[Any] | None, update: list[Any]) -> list[Any]:
        seen.extend(type(item).__name__ for item in [*(current or []), *update])
        return [*(current or []), *update]

    class Notes(BaseModel):
        notes: Annotated[list[Note], typed_add] = Field(default_factory=list)

    @node
    def write_note(state: Notes) -> dict:
        return {"notes": [Note(text="n")]}

    graph = Graph(Notes, state_store=InMemoryStateStore()).flow(
        START >> write_note, write_note >> END
    )
    await graph.ainvoke({"notes": [Note(text="first")]}, thread_id="t")
    await graph.ainvoke({}, thread_id="t")
    seen.clear()

    loaded = await graph.load("t")

    assert [note.text for note in loaded["notes"]] == ["first", "n", "n"]
    assert seen
    assert set(seen) == {"Note"}
