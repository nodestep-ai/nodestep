from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from nodestep.exceptions import StateUpdateError

Reducer = Callable[[Any, Any], Any]
"""A reducer: takes the current value and the update and returns the new value."""


@dataclass(frozen=True, slots=True)
class Replace:
    """An update value that sets a field directly, bypassing its reducer.

    Works in node updates, ``Command.update``, ``Graph.update_state`` and run
    input, not in ``Send`` payloads.

    Parameters
    ----------
    value : Any
        The field's new value.

    Examples
    --------
    >>> from nodestep import Replace
    >>> update = {"messages": Replace([])}
    """

    value: Any


@dataclass(frozen=True, slots=True)
class RemoveMessage:
    """An ``add_messages`` update item that removes the message with ``id``.

    Works in node updates, ``Command.update``, ``Graph.update_state`` and
    input on an existing thread.

    Parameters
    ----------
    id : str
        Id of the message to remove.

    Examples
    --------
    >>> from nodestep import RemoveMessage
    >>> update = {"messages": [RemoveMessage("m1")]}
    """

    id: str


def replace(current: Any, update: Any) -> Any:
    """Reducer that keeps the new value."""
    return update


def add(current: Any, update: Any) -> list[Any]:
    """Reducer that appends the items of a list update to a list.

    Parameters
    ----------
    current : list or None
        The field's value, or None when the field has no value yet.
    update : list
        Items to append; a single item must be wrapped in a list.

    Returns
    -------
    list
        A new list.

    Raises
    ------
    TypeError
        If ``update`` is not a list, or ``current`` is neither a list nor None.
    """
    if not isinstance(update, list):
        raise TypeError(
            f"add takes a list update, got {type(update).__name__}; wrap a single "
            "item in a list"
        )
    if current is None:
        return list(update)
    if not isinstance(current, list):
        raise TypeError(
            f"add appends to a list, but the field holds {type(current).__name__}"
        )
    return [*current, *update]


def message_id(message: Any) -> Any:
    """Return the id of a message model or dict."""
    if isinstance(message, dict):
        return message.get("id")
    return getattr(message, "id", None)


def _with_id(message: Any) -> Any:
    from nodestep.chat.messages import BaseMessage

    if isinstance(message, BaseMessage) and message.id is None:
        return message.model_copy(update={"id": uuid4().hex})
    return message


def ensure_message_ids(value: Any) -> Any:
    """Give chat messages without an id a new unique id.

    Parameters
    ----------
    value : Any
        A chat message, a list, or any other value.

    Returns
    -------
    Any
        ``value`` with a copy of every chat message that had no id, in the
        value itself or among the items of a list, given a new id.
    """
    if isinstance(value, list):
        return [_with_id(item) for item in value]
    return _with_id(value)


def _check_message_update(update: Any) -> None:
    from nodestep.chat.messages import BaseMessage

    if not isinstance(update, list):
        raise TypeError(
            f"add_messages takes a list of messages, got {type(update).__name__}; "
            "wrap a single message in a list"
        )
    ids: set[str] = set()
    for item in update:
        if not isinstance(item, BaseMessage | RemoveMessage):
            raise TypeError(
                "add_messages takes chat messages and RemoveMessage items, got "
                f"{type(item).__name__}; a field read back from a state store holds "
                "messages only if its type names them, e.g. list[Message]"
            )
        if item.id is None:
            continue
        if item.id in ids:
            raise StateUpdateError(
                f"add_messages got the message id '{item.id}' twice in one update"
            )
        ids.add(item.id)


def add_messages(current: Any, update: Any) -> list[Any]:
    """Reducer that appends chat messages and edits or removes them by id.

    A message whose id is already in the list replaces that message in place,
    a ``RemoveMessage`` removes the message with its id, and every other
    message is appended. Messages without an id get a new one.

    Parameters
    ----------
    current : list or None
        The field's value, or None when the field has no value yet.
    update : list
        Chat messages and ``RemoveMessage`` items.

    Returns
    -------
    list
        A new list.

    Raises
    ------
    TypeError
        If ``update`` is not a list, or holds an item that is neither a chat
        message nor a ``RemoveMessage``.
    StateUpdateError
        If two items of ``update`` have the same id, or a ``RemoveMessage``
        names an id that is not in ``current``.
    """
    _check_message_update(update)
    result = list(current or [])
    positions = {
        message_id(message): position
        for position, message in enumerate(result)
        if message_id(message) is not None
    }
    removed: set[int] = set()
    for item in update:
        if isinstance(item, RemoveMessage):
            position = positions.pop(item.id, None)
            if position is None:
                raise StateUpdateError(
                    f"RemoveMessage('{item.id}') names no message in the list"
                )
            removed.add(position)
            continue
        message = _with_id(item)
        position = positions.get(message.id)
        if position is None:
            positions[message.id] = len(result)
            result.append(message)
        else:
            result[position] = message
    return [message for index, message in enumerate(result) if index not in removed]


def merge_dict(current: Any, update: Any) -> dict[Any, Any]:
    """Reducer that merges the top-level keys of a dict update into a dict."""
    if update is None:
        return dict(current or {})
    if current is None:
        return dict(update)
    if not isinstance(current, dict) or not isinstance(update, dict):
        raise TypeError("`merge_dict` reducer requires dict inputs")
    return {**current, **update}


__all__ = [
    "Reducer",
    "RemoveMessage",
    "Replace",
    "add",
    "add_messages",
    "ensure_message_ids",
    "merge_dict",
    "message_id",
    "replace",
]
