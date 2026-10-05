"""`TodoListMiddleware`, which gives the model a to-do list per thread.

See [Built-in middleware](../../concepts/middleware.md#built-in-middleware).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from nodestep.core.tool import Tool, ToolContext, tool
from nodestep.exceptions import NodestepError
from nodestep.middleware.base import Middleware


class TodoItemState(StrEnum):
    """Progress state of a todo item.

    Attributes
    ----------
    PENDING
        ``"pending"``, the state of a new item.
    IN_PROGRESS
        ``"in_progress"``.
    COMPLETED
        ``"completed"``.
    BLOCKED
        ``"blocked"``.
    CANCELLED
        ``"cancelled"``.
    FAILED
        ``"failed"``.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TodoItem(BaseModel):
    """A todo item with its task and state."""

    task: str
    state: TodoItemState = TodoItemState.PENDING


class TodoListSnapshot(BaseModel):
    """The current todo list."""

    items: list[TodoItem]


class TodoListMiddlewareError(NodestepError):
    """Raised when the todo tools run without a thread id."""


class TodoListMiddleware(Middleware):
    """Give the model a todo list per thread, kept in this instance's memory.

    The list is not part of the graph state: a state store, ``history``,
    ``fork`` and other processes do not see it. Without a state store, a run
    without ``thread_id=`` gets a generated thread and so an empty list.
    """

    def __init__(self) -> None:
        self._items: dict[str, list[TodoItem]] = {}
        self._tools = (self._build_read_todos(), self._build_write_todos())

    def items_for(self, thread_id: str) -> list[TodoItem]:
        """Return the todo list of a thread.

        Parameters
        ----------
        thread_id : str

        Returns
        -------
        list[TodoItem]
            The stored list itself; changing it changes the thread's todos.
        """
        return self._items.setdefault(thread_id, [])

    def _thread_items(self, ctx: ToolContext) -> list[TodoItem]:
        if ctx.thread_id is None:
            raise TodoListMiddlewareError(
                "The todo tools keep a list per thread, but the tool context has "
                "no thread_id"
            )
        return self.items_for(ctx.thread_id)

    def _build_read_todos(self) -> Tool:
        todo = self

        @tool(name="read_todos", description="Read the current session todo list.")
        async def read_todos(ctx: ToolContext) -> TodoListSnapshot:
            """Return the todo list of the current thread."""
            return TodoListSnapshot(items=list(todo._thread_items(ctx)))

        return read_todos

    def _build_write_todos(self) -> Tool:
        todo = self

        @tool(
            name="write_todos",
            description=(
                "Replace or extend the current session todo list with typed "
                "TodoItem entries."
            ),
        )
        async def write_todos(
            ctx: ToolContext,
            items: list[TodoItem] = Field(description="Typed todo items."),
            replace: bool = True,
        ) -> TodoListSnapshot:
            """Replace or extend the todo list of the current thread."""
            current = todo._thread_items(ctx)
            if replace:
                current[:] = list(items)
            else:
                current.extend(items)
            return TodoListSnapshot(items=list(current))

        return write_todos

    def tools(self) -> tuple[Tool, ...]:
        """Return the ``read_todos`` and ``write_todos`` tools of this instance."""
        return self._tools


__all__ = [
    "TodoItem",
    "TodoItemState",
    "TodoListMiddleware",
    "TodoListMiddlewareError",
    "TodoListSnapshot",
]
