from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, TypeAdapter

from nodestep.exceptions import StateUpdateError

REPLACE_DELTA_KEY = "__nodestep_replace__"
REMOVE_MESSAGE_KEY = "__nodestep_remove__"


_SCALARS = frozenset({str, bytes, int, float, complex, bool, type(None)})


def isolate(value: Any) -> Any:
    """Copy a state value so changes to the copy cannot reach the original.

    Dicts, lists, tuples, sets and pydantic models (their fields, extra and
    private attributes) are copied item by item; strings, numbers and None are
    kept; every other value, including dict and list subclasses, ``deque`` and
    dataclasses, is copied with ``copy.deepcopy``.

    Parameters
    ----------
    value : Any
        A state, an update or a single value; the keys of a dict or the fields
        of a model name the state fields in errors.

    Returns
    -------
    Any

    Raises
    ------
    StateUpdateError
        If a value cannot be copied, naming the state field that holds it.
    """
    return _isolate(value, None)


def _isolate(value: Any, field: str | None) -> Any:
    kind = type(value)
    if kind in _SCALARS:
        return value
    if isinstance(value, BaseModel):
        return _isolate_model(value, field)
    if kind is dict:
        return {key: _isolate(item, field or str(key)) for key, item in value.items()}
    if kind is list:
        return [_isolate(item, field) for item in value]
    if kind is tuple:
        return tuple(_isolate(item, field) for item in value)
    if kind is set:
        return {_isolate(item, field) for item in value}
    try:
        return copy.deepcopy(value)
    except Exception as error:
        subject = "A state value" if field is None else f"State field '{field}'"
        raise StateUpdateError(
            f"{subject} holds a value of type {kind.__name__}, which cannot be copied "
            f"({type(error).__name__}: {error}); a node gets its own copy of the state, "
            "so pass objects like this through context= instead"
        ) from error


def _isolate_model(model: BaseModel, field: str | None) -> BaseModel:
    parts = {name: model.__dict__[name] for name in type(model).model_fields}
    parts.update(model.__pydantic_extra__ or {})
    copied = model.model_copy(
        update={name: _isolate(item, field or name) for name, item in parts.items()}
    )
    private = model.__pydantic_private__
    if private:
        object.__setattr__(
            copied,
            "__pydantic_private__",
            {name: _isolate(item, field or name) for name, item in private.items()},
        )
    return copied


_JSON_VALUES: TypeAdapter[Any] = TypeAdapter(Any)


def to_json_value(value: Any, *, fallback: Callable[[Any], Any] | None = None) -> Any:
    """Convert a value to JSON-compatible data, encoded by its runtime type.

    Models are dumped as by ``model_dump(mode="json", by_alias=True)``,
    secrets are masked, and tuples and sets become lists. State is stored
    through the field types instead (``StateSchema.dump_state``).

    Parameters
    ----------
    value : Any
    fallback : callable, optional
        Called with each value the encoder does not know; its result is
        encoded instead.

    Returns
    -------
    Any

    Raises
    ------
    pydantic_core.PydanticSerializationError
        If a value cannot be encoded and no ``fallback`` is given.
    """
    return _JSON_VALUES.dump_python(
        value, mode="json", by_alias=True, fallback=fallback
    )


__all__ = [
    "REMOVE_MESSAGE_KEY",
    "REPLACE_DELTA_KEY",
    "isolate",
    "to_json_value",
]
