"""`ToolLimitMiddleware`, which denies tool calls over a budget per run.

See [Built-in middleware](../../concepts/middleware.md#built-in-middleware).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nodestep.exceptions import GraphConfigError, ToolDeniedError
from nodestep.middleware.base import (
    GraphMiddlewareContext,
    Middleware,
    Replacement,
    ToolMiddlewareContext,
)


@dataclass
class _Budget:
    total: int = 0
    per_tool: dict[str, int] = field(default_factory=dict)
    counted: set[str] = field(default_factory=set)


class ToolLimitMiddleware(Middleware):
    """Deny tool calls over a budget per run and its resumes; the model is told.

    Put it in the graph's ``middleware``: tool calls of a run it did not see
    start raise ``GraphConfigError``. Budgets live in this instance's memory;
    see [Built-in middleware](../../concepts/middleware.md#built-in-middleware).

    Parameters
    ----------
    max_calls : int, optional
        Total calls allowed per run.
    per_tool : dict[str, int], optional
        Limits for individual tools.
    """

    def __init__(
        self,
        *,
        max_calls: int = 50,
        per_tool: dict[str, int] | None = None,
    ) -> None:
        self.max_calls = max_calls
        self.per_tool = per_tool or {}
        self._budgets: dict[str, _Budget] = {}

    def before_graph(self, ctx: GraphMiddlewareContext) -> None:
        """Start a fresh budget, or keep the budget of a resumed run."""
        if ctx.resuming:
            self._budgets.setdefault(ctx.root_run_id, _Budget())
        else:
            self._budgets[ctx.root_run_id] = _Budget()

    def after_graph(self, ctx: GraphMiddlewareContext) -> None:
        """Drop the budget of the finished run."""
        self._budgets.pop(ctx.root_run_id, None)

    def before_tool(self, ctx: ToolMiddlewareContext) -> Replacement | None:
        """Count the call once per tool call id and deny it over the limit.

        Returns
        -------
        None

        Raises
        ------
        ToolDeniedError
            If the call is over a limit.
        GraphConfigError
            If the call belongs to no graph run this instance saw start.
        """
        budget = self._budgets.get(ctx.root_run_id or "")
        if budget is None:
            raise GraphConfigError(
                f"ToolLimitMiddleware counts tool calls per graph run, and the "
                f"call of '{ctx.tool_name}' belongs to no graph run it saw start; "
                "put the middleware in the graph's middleware"
            )
        if ctx.tool_call_id is not None:
            if ctx.tool_call_id in budget.counted:
                return None
            budget.counted.add(ctx.tool_call_id)
        budget.total += 1
        if budget.total > self.max_calls:
            raise ToolDeniedError(
                ctx.tool_name, f"total tool limit exceeded ({self.max_calls})"
            )
        if ctx.tool_name in self.per_tool:
            count = budget.per_tool.get(ctx.tool_name, 0) + 1
            budget.per_tool[ctx.tool_name] = count
            limit = self.per_tool[ctx.tool_name]
            if count > limit:
                raise ToolDeniedError(
                    ctx.tool_name,
                    f"per-tool limit exceeded ({limit})",
                )
        return None


__all__ = ["ToolLimitMiddleware"]
