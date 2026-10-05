from __future__ import annotations

import json
from typing import Any

from nodestep.exceptions import GraphConfigError, StateUpdateError, UnknownThreadError
from nodestep.state.history import (
    BranchRecord,
    CheckpointRecord,
    HistoryEvent,
    StateStore,
)
from nodestep.state.schema import StateSchema, merge_updates


async def load_history_state(
    store: StateStore,
    schema: StateSchema[Any],
    thread_id: str,
    *,
    branch_id: str = "main",
    at: str | None = None,
) -> dict[str, Any]:
    """Rebuild a thread's state from its latest checkpoint and later deltas.

    The checkpoint is read back with ``StateSchema.restore_state`` and every
    later delta with ``StateSchema.restore_update`` before the reducers merge
    it, so reducers see the same typed values as during a run and each stored
    value is read as JSON through its field's type once.

    Parameters
    ----------
    store : StateStore
    schema : StateSchema
    thread_id : str
    branch_id : str, optional
    at : str, optional
        Event id to stop at.

    Returns
    -------
    dict
        The state.

    Raises
    ------
    UnknownThreadError
        If the store holds no history for the thread branch.
    StateUpdateError
        If ``at`` is not an event of the thread branch.
    StateStoreError
        If the stored state has a key that is not a field of ``schema``.
    ResumeError
        If a stored value is not valid for its field.
    """
    history = await store.get_history(thread_id, branch_id)
    if not history.events and not history.checkpoints:
        raise UnknownThreadError(thread_id, branch_id=branch_id)
    target_sequence: int | None = None
    if at is not None:
        matched = next(
            (event.sequence for event in history.events if event.id == at), None
        )
        if matched is None:
            raise StateUpdateError(f"Event id '{at}' not found in thread '{thread_id}'")
        target_sequence = matched
    checkpoint = await store.latest_checkpoint(thread_id, branch_id, at)
    if checkpoint is None:
        state = schema.defaults()
        checkpoint_sequence = -1
    else:
        state = schema.restore_state(json.loads(checkpoint.state_json))
        checkpoint_sequence = checkpoint.sequence
    for event in history.events:
        if event.sequence <= checkpoint_sequence:
            continue
        if target_sequence is not None and event.sequence > target_sequence:
            break
        if event.type != "state_delta" or event.data_json is None:
            continue
        try:
            payload = json.loads(event.data_json)
        except json.JSONDecodeError as error:
            raise StateUpdateError(
                f"Corrupt state_delta event '{event.id}': {error}"
            ) from error
        state, _ = merge_updates(
            schema, state, schema.restore_update(payload["update"])
        )
    return state


async def fork_history_state(
    store: StateStore,
    schema: StateSchema[Any],
    thread_id: str,
    *,
    from_: str,
    branch_id: str = "main",
    name: str | None = None,
    position: dict[str, Any],
) -> BranchRecord:
    """Create a branch at an event, with the state and the run position there.

    The state at ``from_`` is read and dumped before anything is written. The
    new branch then gets its ``BranchRecord``, a checkpoint of that state and a
    first event of type ``"forked"`` whose data holds the parent branch, the
    event and ``position``, so a run on the branch continues from ``from_``.

    Parameters
    ----------
    store : StateStore
    schema : StateSchema
    thread_id : str
    from_ : str
        Event id to branch from.
    branch_id : str, optional
        Branch to fork.
    name : str, optional
        Id of the new branch; a random id when omitted.
    position : dict
        JSON data describing where the run stood at ``from_``.

    Returns
    -------
    BranchRecord

    Raises
    ------
    UnknownThreadError
        If the store holds no history for the thread branch.
    StateUpdateError
        If ``from_`` is not an event of the thread branch.
    StateStoreError
        If the thread already has a branch named ``name``.
    """
    state = await load_history_state(
        store, schema, thread_id, branch_id=branch_id, at=from_
    )
    state_json = json.dumps(schema.dump_state(state), separators=(",", ":"))
    data_json = json.dumps(
        {"parent_branch_id": branch_id, "from_event_id": from_, **position},
        separators=(",", ":"),
    )
    branch = await store.fork(thread_id, from_=from_, branch_id=branch_id, name=name)
    await store.save_checkpoint(
        CheckpointRecord(
            thread_id=thread_id,
            branch_id=branch.id,
            event_id=from_,
            sequence=0,
            state_json=state_json,
            status="forked",
        )
    )
    await store.append_next_event(
        HistoryEvent(
            thread_id=thread_id,
            branch_id=branch.id,
            sequence=0,
            type="forked",
            data_json=data_json,
        )
    )
    return branch


def require_state_store(store: StateStore | None) -> StateStore:
    """Return the store, or raise if the graph has none.

    Parameters
    ----------
    store : StateStore or None

    Returns
    -------
    StateStore

    Raises
    ------
    GraphConfigError
    """
    if store is None:
        raise GraphConfigError("Graph has no state_store configured")
    return store


__all__ = ["fork_history_state", "load_history_state", "require_state_store"]
