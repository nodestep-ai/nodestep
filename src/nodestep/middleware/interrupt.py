"""Tool approvals: `ToolInterruptMiddleware`, its rules and argument predicates.

See [Interrupts](../../concepts/interrupts.md).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from nodestep.core.command import interrupt
from nodestep.core.tool import Tool
from nodestep.exceptions import ToolDeniedError
from nodestep.middleware.base import Middleware, Replacement, ToolMiddlewareContext
from nodestep.models.base import NodestepModel

_MISSING = object()


def _argument(arguments: Any, name: str) -> Any:
    if isinstance(arguments, dict):
        value = arguments.get(name, _MISSING)
    else:
        value = getattr(arguments, name, _MISSING)
    if value is _MISSING:
        raise LookupError(f"the tool call has no argument '{name}'")
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _typed_argument(arguments: Any, name: str, expected: Any) -> Any:
    value = _argument(arguments, name)
    if _is_number(expected):
        compatible = _is_number(value)
    else:
        compatible = isinstance(value, type(expected))
    if not compatible:
        raise TypeError(
            f"argument '{name}' is {type(value).__name__}, expected "
            f"{type(expected).__name__}"
        )
    if isinstance(value, float) and math.isnan(value):
        raise ValueError(f"argument '{name}' is NaN")
    return value


def _string_argument(arguments: Any, name: str) -> str:
    value = _argument(arguments, name)
    if not isinstance(value, str):
        raise TypeError(f"argument '{name}' is {type(value).__name__}, expected str")
    return value


def _normalize_tool(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Tool):
        return value.name
    raise TypeError(f"Expected str or Tool, got {type(value).__name__}")


class InterruptRule(NodestepModel):
    """Tool name (or ``"*"``) plus an optional argument predicate.

    Attributes
    ----------
    tool : str
        Name of the tool to review, or ``"*"`` for every tool; a ``Tool`` is
        turned into its name.
    match : callable or None
        Called with the arguments; the call needs approval when it returns
        ``True`` or raises. ``None`` reviews every call.
    """

    tool: str
    match: Callable[[Any], bool] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("tool", mode="before")
    @classmethod
    def _coerce_tool(cls, value: Any) -> str:
        return _normalize_tool(value)


class ToolDecision(NodestepModel):
    """A reviewer's answer to a tool approval.

    Attributes
    ----------
    action : {"approve", "deny", "edit"}
        Run the call, refuse it, or run it with ``arguments``.
    arguments : dict[str, Any] or None
        With ``action="edit"``, keys that replace the call's arguments.
    message : str or None
        Reason given to ``ToolDeniedError`` when the call is denied.
    """

    action: Literal["approve", "deny", "edit"]
    arguments: dict[str, Any] | None = None
    message: str | None = None


class ToolInterrupt(NodestepModel):
    """Interrupt payload asking a reviewer to approve a tool call.

    Attributes
    ----------
    kind : str
        Always ``"tool_interrupt"``, to tell the payload apart.
    tool_call_id : str or None
        Id of the call under review.
    arguments : dict
        Arguments the tool would run with.
    """

    kind: str = "tool_interrupt"
    tool_name: str
    tool_call_id: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)


def _decision(response: Any) -> ToolDecision:
    if response is True:
        return ToolDecision(action="approve")
    if response is False:
        return ToolDecision(action="deny")
    if isinstance(response, ToolDecision):
        return response
    if isinstance(response, dict):
        try:
            return ToolDecision.model_validate(response)
        except ValidationError:
            pass
    return ToolDecision(
        action="deny", message=f"invalid approval response {response!r}"
    )


_PLAIN_ID_CHARACTER = re.compile(r"[A-Za-z0-9_-]")


def _approval_id(tool_name: str, tool_call_id: str | None) -> str:
    if tool_call_id is None:
        raise ValueError(
            f"ToolInterruptMiddleware keys the approval of tool '{tool_name}' by its "
            "tool call id, but the call has none; pass tool_call_id= to ToolContext"
        )
    encoded = "".join(
        character
        if _PLAIN_ID_CHARACTER.fullmatch(character)
        else "".join(f".{byte:02x}" for byte in character.encode())
        for character in tool_call_id
    )
    return f"approve_{encoded}"


def _arguments(value: Any, *, mode: Literal["python", "json"]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode=mode)
    return dict(value or {})


def should_interrupt(
    rules: Iterable[InterruptRule],
    tool: Tool | str,
    raw_input: Any = None,
) -> bool:
    """Check whether any rule requires approval for a tool call.

    A predicate that raises counts as a match.

    Parameters
    ----------
    rules : Iterable[InterruptRule]
    tool : Tool or str
        The tool, or its name.
    raw_input : dict, optional
        Tool arguments passed to the rule predicates.

    Returns
    -------
    bool
    """
    tool_name = _normalize_tool(tool)
    for rule in rules:
        tool_matches = rule.tool == "*" or rule.tool == tool_name
        if not tool_matches:
            continue
        if rule.match is None or _fails_closed(rule.match, raw_input):
            return True
    return False


def _fails_closed(predicate: Callable[[Any], bool], arguments: Any) -> bool:
    try:
        return bool(predicate(arguments))
    except Exception:
        return True


def gt(field: str, value: Any) -> Callable[[Any], bool]:
    """Predicate that matches when argument ``field`` ``> value``.

    Parameters
    ----------
    field : str
        Argument name.
    value : Any
        A number, compared with number arguments but not ``bool``; any other
        value needs an argument of its type.

    Returns
    -------
    callable
        Raises ``LookupError``, ``TypeError`` or ``ValueError`` for a missing,
        mistyped or NaN argument, which ``should_interrupt`` counts as a match.
    """
    return lambda arguments: _typed_argument(arguments, field, value) > value


def ge(field: str, value: Any) -> Callable[[Any], bool]:
    """Predicate that matches when argument ``field`` ``>= value``.

    Parameters
    ----------
    field : str
        Argument name.
    value : Any
        A number, compared with number arguments but not ``bool``; any other
        value needs an argument of its type.

    Returns
    -------
    callable
        Raises ``LookupError``, ``TypeError`` or ``ValueError`` for a missing,
        mistyped or NaN argument, which ``should_interrupt`` counts as a match.
    """
    return lambda arguments: _typed_argument(arguments, field, value) >= value


def lt(field: str, value: Any) -> Callable[[Any], bool]:
    """Predicate that matches when argument ``field`` ``< value``.

    Parameters
    ----------
    field : str
        Argument name.
    value : Any
        A number, compared with number arguments but not ``bool``; any other
        value needs an argument of its type.

    Returns
    -------
    callable
        Raises ``LookupError``, ``TypeError`` or ``ValueError`` for a missing,
        mistyped or NaN argument, which ``should_interrupt`` counts as a match.
    """
    return lambda arguments: _typed_argument(arguments, field, value) < value


def le(field: str, value: Any) -> Callable[[Any], bool]:
    """Predicate that matches when argument ``field`` ``<= value``.

    Parameters
    ----------
    field : str
        Argument name.
    value : Any
        A number, compared with number arguments but not ``bool``; any other
        value needs an argument of its type.

    Returns
    -------
    callable
        Raises ``LookupError``, ``TypeError`` or ``ValueError`` for a missing,
        mistyped or NaN argument, which ``should_interrupt`` counts as a match.
    """
    return lambda arguments: _typed_argument(arguments, field, value) <= value


def eq(field: str, value: Any) -> Callable[[Any], bool]:
    """Predicate that matches when argument ``field`` ``== value``.

    Parameters
    ----------
    field : str
        Argument name.
    value : Any
        A number, compared with number arguments but not ``bool``; any other
        value needs an argument of its type.

    Returns
    -------
    callable
        Raises ``LookupError``, ``TypeError`` or ``ValueError`` for a missing,
        mistyped or NaN argument, which ``should_interrupt`` counts as a match.
    """
    return lambda arguments: _typed_argument(arguments, field, value) == value


def matches(field: str, pattern: str) -> Callable[[Any], bool]:
    """Predicate that matches a string argument against a regular expression.

    Parameters
    ----------
    field : str
        Argument name.
    pattern : str
        Regular expression, searched with ``re.search``.

    Returns
    -------
    callable
        Raises ``LookupError`` or ``TypeError`` for a missing or non-string
        argument, which ``should_interrupt`` counts as a match.
    """
    compiled = re.compile(pattern)
    return lambda arguments: (
        compiled.search(_string_argument(arguments, field)) is not None
    )


def _field_checks(patterns: dict[str, str]) -> list[Callable[[Any], bool]]:
    if not patterns:
        raise ValueError("give at least one argument pattern")
    return [matches(name, pattern) for name, pattern in patterns.items()]


def all_fields(**patterns: str) -> Callable[[Any], bool]:
    """Predicate that matches when every given string argument matches its pattern.

    Parameters
    ----------
    **patterns : str
        Regular expressions keyed by argument name.

    Returns
    -------
    callable
        Checks every argument, and raises like ``matches`` for a missing or
        non-string one.

    Raises
    ------
    ValueError
        If no pattern is given.
    """
    checks = _field_checks(patterns)

    def predicate(arguments: Any) -> bool:
        results = [check(arguments) for check in checks]
        return all(results)

    return predicate


def any_field(**patterns: str) -> Callable[[Any], bool]:
    """Predicate that matches when at least one string argument matches its pattern.

    Parameters
    ----------
    **patterns : str
        Regular expressions keyed by argument name.

    Returns
    -------
    callable
        Checks every argument, and raises like ``matches`` for a missing or
        non-string one.

    Raises
    ------
    ValueError
        If no pattern is given.
    """
    checks = _field_checks(patterns)

    def predicate(arguments: Any) -> bool:
        results = [check(arguments) for check in checks]
        return any(results)

    return predicate


class ToolInterruptMiddleware(Middleware):
    """Pause tool calls that match a rule until a person answers.

    - Interrupt id: ``"approve_"`` plus the tool call id, with each character
      other than a letter, digit, ``_`` or ``-`` written as ``.`` and two hex
      digits per UTF-8 byte (``"fc:1"`` becomes ``"approve_fc.3a1"``).
    - Payload: a ``ToolInterrupt``.
    - Answer: ``True`` or ``ToolDecision(action="approve")`` runs the call;
      ``ToolDecision(action="edit", arguments=...)`` runs it with those keys
      replacing the call's; anything else denies. A dict works like a
      ``ToolDecision``.

    Parameters
    ----------
    rules : list[InterruptRule]
        Which tool calls need approval.
    """

    def __init__(self, *, rules: list[InterruptRule]) -> None:
        self.rules = list(rules)

    def before_tool(self, ctx: ToolMiddlewareContext) -> Replacement | None:
        """Ask for approval and apply the reviewer's decision.

        Returns
        -------
        Replacement or None
            The edited arguments, or ``None`` to run the call as it is.

        Raises
        ------
        ValueError
            If the call needs approval and has no ``tool_call_id``.
        ToolDeniedError
            If the reviewer denies the call.
        """
        arguments = _arguments(ctx.value, mode="python")
        if not should_interrupt(self.rules, ctx.tool_name, arguments):
            return None
        decision = _decision(
            interrupt(
                ToolInterrupt(
                    tool_name=ctx.tool_name,
                    tool_call_id=ctx.tool_call_id,
                    arguments=_arguments(ctx.value, mode="json"),
                ),
                id=_approval_id(ctx.tool_name, ctx.tool_call_id),
            )
        )
        if decision.action == "approve":
            return None
        if decision.action == "edit" and decision.arguments is not None:
            return ctx.replace({**arguments, **decision.arguments})
        raise ToolDeniedError(ctx.tool_name, decision.message or "denied by reviewer")


__all__ = [
    "InterruptRule",
    "ToolDecision",
    "ToolInterrupt",
    "ToolInterruptMiddleware",
    "all_fields",
    "any_field",
    "eq",
    "ge",
    "gt",
    "le",
    "lt",
    "matches",
    "should_interrupt",
]
