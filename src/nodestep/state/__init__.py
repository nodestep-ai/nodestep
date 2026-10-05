"""State schemas, the `Reducer` type, history and the state store protocol.

The stores, `InMemoryStateStore` and `FilesystemStateStore`, are in
`nodestep.state.integrations` and exported by `nodestep`. The reducers, such as
`add` and `add_messages`, are in `nodestep.core`. See
[State](../../concepts/state.md) and
[Persistence and time travel](../../concepts/persistence.md).
"""

from nodestep.exceptions import StateUpdateError
from nodestep.models.base import NodestepModel
from nodestep.state.context import to_json_value
from nodestep.state.history import (
    BranchRecord,
    CheckpointRecord,
    History,
    HistoryEvent,
    StateStore,
)
from nodestep.state.schema import (
    FieldDescriptor,
    StateSchema,
    StateSnapshot,
    merge_parallel_updates,
    merge_updates,
    snapshot_state,
)
from nodestep.utils.json import dump_json_object, load_json_object
from nodestep.utils.reducers import Reducer

__all__ = [
    "BranchRecord",
    "CheckpointRecord",
    "FieldDescriptor",
    "History",
    "HistoryEvent",
    "NodestepModel",
    "Reducer",
    "StateSchema",
    "StateSnapshot",
    "StateStore",
    "StateUpdateError",
    "dump_json_object",
    "load_json_object",
    "merge_parallel_updates",
    "merge_updates",
    "snapshot_state",
    "to_json_value",
]
