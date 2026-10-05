from __future__ import annotations

from typing import Annotated, Any, TypedDict

import pytest
from pydantic import BaseModel, Field

from nodestep import END, START, Graph, InMemoryStateStore, add, node
from nodestep.exceptions import GraphConfigError
from nodestep.state import StateSchema


def _local_types() -> tuple[Any, Any, Any]:
    class Item(BaseModel):
        text: str

    class LocalState(BaseModel):
        items: Annotated[list[Item], add] = Field(default_factory=list)

    class LocalTypedDict(TypedDict):
        items: Annotated[list[Item], add]

    return Item, LocalState, LocalTypedDict


def test_pydantic_schema_reads_reducers_from_resolved_fields() -> None:
    item, local_state, _ = _local_types()

    field = StateSchema.from_type(local_state).fields["items"]

    assert field.reducer is add
    assert field.annotation == list[item]


async def test_pydantic_schema_with_local_types_appends_and_reloads() -> None:
    item, local_state, _ = _local_types()

    @node
    def append_item(state: Any) -> dict:
        return {"items": [item(text="new")]}

    graph = Graph(local_state, state_store=InMemoryStateStore()).flow(
        START >> append_item, append_item >> END
    )

    result = await graph.ainvoke({"items": [item(text="first")]}, thread_id="t")
    loaded = await graph.load("t")

    assert [entry.text for entry in result.state.items] == ["first", "new"]
    assert [entry.text for entry in loaded["items"]] == ["first", "new"]


def test_unresolvable_typeddict_annotation_names_the_field() -> None:
    _, _, local_typed_dict = _local_types()

    with pytest.raises(GraphConfigError, match="state field 'items' of LocalTypedDict"):
        StateSchema.from_type(local_typed_dict)


def test_incomplete_pydantic_schema_names_the_field() -> None:
    class Pending(BaseModel):
        later: Later
        count: int = 0

    class Later(BaseModel):
        value: int

    with pytest.raises(GraphConfigError, match="state field 'later' of Pending"):
        StateSchema.from_type(Pending)

    Pending.model_rebuild()

    assert StateSchema.from_type(Pending).fields["later"].annotation is Later
