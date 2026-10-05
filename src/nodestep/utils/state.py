from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nodestep.exceptions import StateUpdateError


def get_state_field(state: object, field: str) -> Any:
    """Read a field from a dict state or a model state.

    Parameters
    ----------
    state : dict or BaseModel
        The state, e.g. a node's input.
    field : str
        Name of the field.

    Returns
    -------
    Any

    Raises
    ------
    StateUpdateError
        If the state has no value for ``field``: the field is not declared,
        for example because a configured field name is misspelled, or a dict
        state has no value for it yet.
    """
    if isinstance(state, Mapping):
        mapping: Mapping[Any, Any] = state
        if field not in mapping:
            raise StateUpdateError(
                f"The state has no value for '{field}': the field is not in the "
                "state schema, or it has no value yet; the state has values for "
                f"{', '.join(repr(key) for key in mapping) or 'no fields'}"
            )
        return mapping[field]
    try:
        return getattr(state, field)
    except AttributeError as error:
        raise StateUpdateError(
            f"The state {type(state).__name__} has no field '{field}'"
        ) from error


__all__ = ["get_state_field"]
