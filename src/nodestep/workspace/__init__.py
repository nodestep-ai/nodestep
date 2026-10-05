"""Files for agents, on the local disk or in memory.

`LocalWorkspace` and `InMemoryWorkspace` implement `Workspace` and refuse
paths that leave their root. See [Workspace and memory](../../concepts/workspace.md).
"""

from nodestep.exceptions import (
    PathAccessError,
    WorkspaceError,
)
from nodestep.workspace.base import GrepResult, Workspace
from nodestep.workspace.integrations import InMemoryWorkspace, LocalWorkspace

__all__ = [
    "GrepResult",
    "InMemoryWorkspace",
    "LocalWorkspace",
    "PathAccessError",
    "Workspace",
    "WorkspaceError",
]
