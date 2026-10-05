from __future__ import annotations

import json
from typing import Any


def dump_json_object(
    value: dict[str, Any] | None, *, sort_keys: bool = True
) -> str | None:
    """Serialize a dict to JSON.

    Parameters
    ----------
    value : dict or None
    sort_keys : bool, optional
        Sort the keys; ``True`` by default.

    Returns
    -------
    str or None
    """
    if value is None:
        return None
    return json.dumps(value, sort_keys=sort_keys)


def load_json_object(value: str | None) -> dict[str, Any] | None:
    """Parse a JSON object.

    Parameters
    ----------
    value : str or None

    Returns
    -------
    dict or None

    Raises
    ------
    ValueError
        If the JSON is not an object.
    """
    if value is None:
        return None
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("Expected JSON object")
    return loaded


__all__ = ["dump_json_object", "load_json_object"]
