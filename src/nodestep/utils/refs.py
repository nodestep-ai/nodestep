from __future__ import annotations

from collections.abc import Callable
from typing import Any


def callable_name(function: Callable[..., Any]) -> str:
    """Return the display name of a callable."""
    return getattr(function, "__name__", function.__class__.__name__)


def callable_ref(function: Callable[..., Any]) -> str | None:
    """Return the ``module:qualname`` import path of a callable, if it has one."""
    module = getattr(function, "__module__", None)
    qualname = getattr(function, "__qualname__", None)
    if not module or not qualname or "<locals>" in qualname or qualname == "<lambda>":
        return None
    return f"{module}:{qualname}"


__all__ = ["callable_name", "callable_ref"]
