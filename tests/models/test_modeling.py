import pytest
from pydantic import PydanticSchemaGenerationError

from nodestep.models.base import BaseState, NodestepModel
from nodestep.utils.json import dump_json_object, load_json_object


def test_json_object_round_trip() -> None:
    payload = {"b": 2, "a": 1}

    dumped = dump_json_object(payload)

    assert dumped == '{"a": 1, "b": 2}'
    assert load_json_object(dumped) == payload


def test_dump_json_object_accepts_none() -> None:
    assert dump_json_object(None) is None


def test_load_json_object_accepts_none() -> None:
    assert load_json_object(None) is None


def test_load_json_object_rejects_array() -> None:
    with pytest.raises(ValueError, match="Expected JSON object"):
        load_json_object("[]")


def test_base_state_is_mutable() -> None:
    class MyState(BaseState):
        count: int = 0

    state = MyState(count=1)
    state.count = 2
    assert state.count == 2


class Connection:
    pass


def test_base_state_refuses_arbitrary_types() -> None:
    with pytest.raises(PydanticSchemaGenerationError):

        class WithConnection(BaseState):
            connection: Connection | None = None


def test_nodestep_model_keeps_arbitrary_types() -> None:
    class Internal(NodestepModel):
        connection: Connection | None = None

    assert Internal(connection=Connection()).connection is not None
