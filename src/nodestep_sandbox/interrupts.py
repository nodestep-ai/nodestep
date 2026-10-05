from collections.abc import Mapping, Sequence
from typing import Any, Self

from pydantic import BaseModel, ValidationError

from nodestep import Interrupt
from nodestep.middleware import ToolDecision, ToolInterrupt
from nodestep.state import to_json_value
from nodestep_sandbox.errors import FormError
from nodestep_sandbox.records import json_text, parse_json


class ApprovalRequest(BaseModel):
    """A tool call that waits for approval.

    It is read from an interrupt payload that validates as the
    ``ToolInterrupt`` of ``ToolInterruptMiddleware``: ``{"kind":
    "tool_interrupt", "tool_name": ..., "tool_call_id": ..., "arguments":
    {...}}``.

    Attributes
    ----------
    tool_name : str
    tool_call_id : str, optional
    arguments_json : str
        The proposed arguments as JSON.
    """

    tool_name: str
    tool_call_id: str | None = None
    arguments_json: str


class PendingInterrupt(BaseModel):
    """A pending interrupt as the resume form shows it.

    Attributes
    ----------
    key : str
        The key answered in ``Resume(answers={key: ...})``.
    id : str
        The id passed to ``interrupt``.
    node : str
        The node that paused.
    payload_json : str
        The payload as JSON.
    approval : ApprovalRequest, optional
        Set when the payload is a tool approval.
    """

    key: str
    id: str
    node: str
    payload_json: str
    approval: ApprovalRequest | None = None

    @classmethod
    def from_interrupt(cls, item: Interrupt) -> Self:
        """Describe a pending ``Interrupt``.

        Parameters
        ----------
        item : Interrupt

        Returns
        -------
        PendingInterrupt
        """
        payload = to_json_value(item.payload, fallback=repr)
        return cls(
            key=item.key,
            id=item.id,
            node=item.node,
            payload_json=json_text(payload),
            approval=_approval(payload),
        )


def _approval(payload: Any) -> ApprovalRequest | None:
    if not isinstance(payload, dict) or payload.get("kind") != "tool_interrupt":
        return None
    try:
        request = ToolInterrupt.model_validate(payload)
    except ValidationError:
        return None
    return ApprovalRequest(
        tool_name=request.tool_name,
        tool_call_id=request.tool_call_id,
        arguments_json=json_text(request.arguments),
    )


class ResumeForm:
    """Turns a submitted resume form into the answers of ``Resume(answers=...)``.

    Control names carry the position of the interrupt in ``pending``:
    ``answer.N`` and ``format.N`` (``text`` or ``json``) for a plain interrupt;
    ``decision.N`` (``approve``, ``edit`` or ``deny``), ``arguments.N`` and
    ``message.N`` for a tool approval.

    Parameters
    ----------
    pending : Sequence[PendingInterrupt]
    """

    def __init__(self, pending: Sequence[PendingInterrupt]) -> None:
        self.pending = list(pending)

    def parse(self, values: Mapping[str, str]) -> dict[str, Any]:
        """Return one answer per pending interrupt key.

        A plain answer is the text as a string, or the parsed value when its
        format is JSON. A tool approval gives a ``ToolDecision`` as a dict:
        ``{"action": "approve"}``, ``{"action": "edit", "arguments": {...}}``,
        or ``{"action": "deny"}`` with an optional ``"message"``.

        Parameters
        ----------
        values : Mapping[str, str]
            Submitted form values by control name.

        Returns
        -------
        dict[str, Any]

        Raises
        ------
        FormError
            If a text answer is empty, JSON does not parse, edited arguments are
            not a JSON object, or no decision or format is chosen.
        """
        answers: dict[str, Any] = {}
        errors: dict[str, str] = {}
        for index, item in enumerate(self.pending):
            try:
                answers[item.key] = (
                    _approval_answer(index, values)
                    if item.approval is not None
                    else _plain_answer(index, values)
                )
            except FormError as error:
                errors.update(error.errors)
        if errors:
            raise FormError(errors)
        return answers


def _plain_answer(index: int, values: Mapping[str, str]) -> Any:
    control = f"answer.{index}"
    text = values.get(control, "")
    kind = values.get(f"format.{index}", "text")
    if kind == "json":
        try:
            return parse_json(text)
        except ValueError as error:
            raise FormError({control: str(error)}) from None
    if kind != "text":
        raise FormError({f"format.{index}": "Choose text or JSON"})
    if not text:
        raise FormError({control: "Enter an answer"})
    return text


def _approval_answer(index: int, values: Mapping[str, str]) -> dict[str, Any]:
    decision = values.get(f"decision.{index}", "")
    if decision == "approve":
        return _answer(ToolDecision(action="approve"))
    if decision == "edit":
        control = f"arguments.{index}"
        try:
            arguments = parse_json(values.get(control, ""))
        except ValueError as error:
            raise FormError({control: str(error)}) from None
        if not isinstance(arguments, dict):
            raise FormError({control: "Edited arguments must be a JSON object"})
        return _answer(ToolDecision(action="edit", arguments=arguments))
    if decision == "deny":
        message = values.get(f"message.{index}", "").strip() or None
        return _answer(ToolDecision(action="deny", message=message))
    raise FormError({f"decision.{index}": "Choose approve, edit or deny"})


def _answer(decision: ToolDecision) -> dict[str, Any]:
    return decision.model_dump(mode="json", exclude_none=True)
