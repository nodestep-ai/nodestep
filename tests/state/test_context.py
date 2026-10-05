import threading
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr
from pydantic_core import PydanticSerializationError

from nodestep.exceptions import StateUpdateError
from nodestep.state import to_json_value
from nodestep.state.context import isolate


class Account(BaseModel):
    user: str
    item_id: int = Field(alias="itemId")
    api_key: str = Field(exclude=True)


class Handle:
    pass


def test_to_json_value_dumps_models_by_alias_without_excluded_fields() -> None:
    account = Account(user="u", itemId=1, api_key="k")

    assert to_json_value(
        {"account": account, "when": datetime(2026, 1, 1, tzinfo=UTC)}
    ) == {
        "account": {"user": "u", "itemId": 1},
        "when": "2026-01-01T00:00:00Z",
    }


def test_to_json_value_masks_secrets() -> None:
    assert to_json_value({"password": SecretStr("secret")}) == {
        "password": "**********"
    }


def test_to_json_value_raises_for_values_it_does_not_know() -> None:
    with pytest.raises(PydanticSerializationError):
        to_json_value({"handle": Handle()})


def test_to_json_value_passes_unknown_values_to_the_fallback() -> None:
    assert to_json_value(
        {"handle": Handle(), "n": 1}, fallback=lambda value: type(value).__name__
    ) == {"handle": "Handle", "n": 1}


@dataclass
class Document:
    tags: list[str] = field(default_factory=list)


class Box:
    def __init__(self) -> None:
        self.items: list[str] = []


class Tags(list):
    pass


@pytest.mark.parametrize(
    ("value", "inner"),
    [
        (Document(tags=["a"]), lambda copy: copy.tags),
        (defaultdict(list, {"k": ["a"]}), lambda copy: copy["k"]),
        (OrderedDict(k=["a"]), lambda copy: copy["k"]),
        (deque([["a"]]), lambda copy: copy[0]),
        (Tags([["a"]]), lambda copy: copy[0]),
    ],
)
def test_isolate_copies_containers_and_dataclasses(value: Any, inner: Any) -> None:
    state = {"value": value}

    copied = isolate(state)["value"]
    inner(copied).append("leaked")

    assert type(copied) is type(value)
    assert inner(value) == ["a"]


def test_isolate_deep_copies_other_objects() -> None:
    box = Box()
    box.items.append("a")

    copied = isolate({"box": box})["box"]
    copied.items.append("leaked")

    assert copied is not box
    assert box.items == ["a"]


def test_isolate_names_the_field_whose_value_cannot_be_copied() -> None:
    with pytest.raises(StateUpdateError, match=r"'lock'.*cannot be copied"):
        isolate({"ok": 1, "lock": threading.Lock()})


class Loose(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = ""
    _cache: dict[str, list[str]] = PrivateAttr(default_factory=dict)


def test_isolate_copies_extra_and_private_attributes_of_models() -> None:
    loose = Loose.model_validate({"name": "n", "tags": ["a"]})
    loose._cache["k"] = ["a"]

    copied = isolate({"loose": loose})["loose"]
    assert copied.model_extra is not None
    copied.model_extra["tags"].append("leaked")
    copied._cache["k"].append("leaked")

    assert loose.model_extra == {"tags": ["a"]}
    assert loose._cache == {"k": ["a"]}
    assert copied.model_extra == {"tags": ["a", "leaked"]}
