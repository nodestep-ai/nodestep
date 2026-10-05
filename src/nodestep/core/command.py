from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, overload

from nodestep.exceptions import GraphExecutionError, ResumeError


class StartSentinel:
    """Marker for the entry of a graph flow, used as ``START >> node``."""

    _instance: StartSentinel | None = None

    def __new__(cls) -> StartSentinel:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "START"

    def __str__(self) -> str:
        return "__start__"

    def __rshift__(self, target: Any) -> Any:
        from nodestep.core.flow import FlowStartEdge

        return FlowStartEdge(target=target)


START = StartSentinel()
"""The entry of a flow: ``START >> node`` names the first node of a graph."""


class EndSentinel:
    """Marker for the end of a graph flow."""

    _instance: EndSentinel | None = None

    def __new__(cls) -> EndSentinel:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "END"

    def __str__(self) -> str:
        return "__end__"


END = EndSentinel()
"""The end of a flow: ``node >> END``, a branch target, or a ``goto`` target."""


@dataclass(frozen=True, slots=True)
class Send:
    """Route to a node with its own input payload.

    Attributes
    ----------
    node : Node
        Target node.
    payload : dict or BaseModel, optional
        State fields laid over the current state for that task only, without
        reducers. Keys must be state fields, unless the state is a plain
        ``dict``; a model gives only the fields set on it. When the node
        returns the ``Send``, a ``Replace``, a ``RemoveMessage`` or an invalid
        value raises ``StateUpdateError``, and a value that does not survive
        being stored as JSON raises ``StateStoreError``.
    """

    node: Any
    payload: Any = None

    def __post_init__(self) -> None:
        from nodestep.core.node import Node as _Node
        from nodestep.exceptions import GraphConfigError

        if not isinstance(self.node, _Node):
            raise GraphConfigError(
                f"Send target must be a Node, got {type(self.node).__name__}"
            )


@dataclass(frozen=True, slots=True)
class Command:
    """Node return value that updates state and routes.

    Attributes
    ----------
    update : Mapping, optional
        Update applied through the reducers; not the state object the node
        received.
    goto : Node, Send, END or list, optional
        Next targets, replacing the node's edges. ``[]`` routes nowhere;
        ``None`` follows the edges.
    """

    update: Mapping[str, Any] | None = None
    goto: Any = None


@dataclass(frozen=True, slots=True)
class Interrupt:
    """A pending interrupt of a paused run.

    Attributes
    ----------
    key : str
        Key for ``Resume(answers={key: ...})``: ``"<task_id>:<id>"``; inside a
        child graph, prefixed with the subgraph node's ``"<task_id>/"``.
    id : str
        The ``id`` passed to ``interrupt``.
    node : str
        Node that paused; the subgraph node for a child graph's interrupt.
    task_id : str
        Task of that node.
    payload : Any
        Payload passed to ``interrupt``.
    """

    key: str
    id: str
    node: str
    task_id: str
    payload: Any


class _NoValue:
    def __repr__(self) -> str:
        return "<no value>"


_NO_VALUE: Any = _NoValue()


@dataclass(frozen=True, slots=True, init=False)
class Resume:
    """Answers for every pending interrupt of a paused thread.

    Parameters
    ----------
    value : Any, default: omitted
        Answer for the only pending interrupt.
    answers : Mapping[str, Any], optional
        Answers by ``Interrupt.key``, one for each pending interrupt.

    Attributes
    ----------
    value : Any
        The single answer; ``None`` when ``answers`` is given.
    answers : dict or None
        The answers by key; ``None`` when ``value`` is given.

    Raises
    ------
    TypeError
        If ``answers`` is not a mapping.
    ResumeError
        If both or neither of ``value`` and ``answers`` are given, or
        ``answers`` is empty.
    """

    value: Any
    answers: dict[str, Any] | None

    @overload
    def __init__(self, value: Any) -> None: ...

    @overload
    def __init__(self, *, answers: Mapping[str, Any]) -> None: ...

    def __init__(
        self, value: Any = _NO_VALUE, *, answers: Mapping[str, Any] | None = None
    ) -> None:
        if answers is None and value is _NO_VALUE:
            raise ResumeError(
                "Resume needs an answer: pass Resume(value) for the only pending "
                "interrupt or Resume(answers={key: value, ...})"
            )
        if answers is not None and value is not _NO_VALUE:
            raise ResumeError("Resume takes either a value or answers=, not both")
        if answers is not None and not isinstance(answers, Mapping):
            raise TypeError(
                "Resume answers= takes a mapping of interrupt keys to answers, got "
                f"{type(answers).__name__}"
            )
        if answers is not None and not answers:
            raise ResumeError(
                "Resume(answers={}) answers nothing; give an answer for every "
                "pending interrupt key"
            )
        object.__setattr__(self, "value", None if value is _NO_VALUE else value)
        object.__setattr__(self, "answers", None if answers is None else dict(answers))


class GraphInterrupt(BaseException):
    """Raised by ``interrupt`` to pause the run.

    It derives from ``BaseException``, so ``except Exception`` in a node does not
    swallow the pause.

    Parameters
    ----------
    payload : Any
        Payload passed to ``interrupt``.
    id : str
        The interrupt's id.
    key : str
        The interrupt's key.
    """

    def __init__(self, payload: Any, *, id: str, key: str) -> None:
        self.payload = payload
        self.id = id
        self.key = key
        super().__init__(f"Graph interrupted at {key}")


class GraphInterruptGroup(BaseException):
    """Several interrupts raised by one task; a ``BaseException`` like them.

    Parameters
    ----------
    interrupts : list[GraphInterrupt]
    """

    def __init__(self, interrupts: list[GraphInterrupt]) -> None:
        self.interrupts = interrupts
        keys = ", ".join(graph_interrupt.key for graph_interrupt in interrupts)
        super().__init__(f"Graph interrupted at {keys}")


@dataclass
class InterruptFrame:
    """Per-task bookkeeping for interrupt answers, ids and cache.

    Attributes
    ----------
    task_id : str
        Task whose node is running; the prefix of its interrupt keys.
    resume_values : dict
        Answers by interrupt key.
    cache : dict
        The task's ``NodeContext.cache``.
    answer : callable, optional
        Asks the middleware for an answer to a payload; returns ``None`` when
        none answers.
    used_ids : set[str]
        Ids already passed to ``interrupt`` in this run of the node.
    """

    task_id: str
    resume_values: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)
    answer: Callable[[Any], Any] | None = None
    used_ids: set[str] = field(default_factory=set)


_interrupt_frame: ContextVar[InterruptFrame | None] = ContextVar(
    "_interrupt_frame", default=None
)

_delegated_answer: ContextVar[Callable[[Any], Any] | None] = ContextVar(
    "_delegated_answer", default=None
)

_INTERRUPT_ID = re.compile(r"[A-Za-z0-9_.-]+")


def interrupt(payload: Any, *, id: str) -> Any:
    """Pause the current node until the interrupt is answered.

    An ``on_interrupt`` hook may answer at once; otherwise the run pauses. On
    resume the node runs again from the top, and this call returns the
    answer. See [Interrupts](../../concepts/interrupts.md).

    Parameters
    ----------
    payload : Any
        Question or data for whoever answers.
    id : str
        Unique within one run of the node; letters, digits, ``_``, ``.`` and
        ``-``.

    Returns
    -------
    Any
        The answer, normalized to JSON values.

    Raises
    ------
    ValueError
        If ``id`` is empty or has other characters.
    GraphExecutionError
        If the node already used ``id`` in this run.
    RuntimeError
        If called outside a running node.
    TypeError
        If an ``on_interrupt`` hook returns an awaitable.
    """
    if _INTERRUPT_ID.fullmatch(id) is None:
        raise ValueError(f"Interrupt id {id!r} must match [A-Za-z0-9_.-]+")
    frame = _interrupt_frame.get()
    if frame is None:
        raise RuntimeError("interrupt() called outside a graph node")
    if id in frame.used_ids:
        raise GraphExecutionError(
            f"Task '{frame.task_id}' called interrupt() twice with id '{id}'; "
            "every interrupt of a node run needs its own id"
        )
    frame.used_ids.add(id)
    key = f"{frame.task_id}:{id}"
    if key in frame.resume_values:
        return frame.resume_values[key]
    if frame.answer is not None:
        answer = frame.answer(payload)
        if answer is not None:
            frame.resume_values[key] = answer
            return answer
    raise GraphInterrupt(payload, id=id, key=key)


def bind_interrupt_frame(frame: InterruptFrame) -> Any:
    """Make a frame the active interrupt frame.

    Graphs run inside the frame's node do not see an answerer delegated to the
    node's own graph; ``delegate_interrupts`` hands one on explicitly.

    Parameters
    ----------
    frame : InterruptFrame

    Returns
    -------
    Any
        Token for ``unbind_interrupt_frame``.
    """
    return _interrupt_frame.set(frame), _delegated_answer.set(None)


def unbind_interrupt_frame(token: Any) -> None:
    """Restore the previously active interrupt frame.

    Parameters
    ----------
    token : Any
        Token returned by ``bind_interrupt_frame``.
    """
    frame_token, delegate_token = token
    _delegated_answer.reset(delegate_token)
    _interrupt_frame.reset(frame_token)


@contextmanager
def delegate_interrupts() -> Iterator[None]:
    """Offer interrupts of a graph run inside this block to the running node.

    A graph started inside the block asks the running node's ``answer`` for
    every interrupt its own ``on_interrupt`` hooks leave unanswered.
    ``subgraph`` runs its child graph this way.

    Yields
    ------
    None
    """
    frame = _interrupt_frame.get()
    token = _delegated_answer.set(None if frame is None else frame.answer)
    try:
        yield
    finally:
        _delegated_answer.reset(token)


def delegated_answer() -> Callable[[Any], Any] | None:
    """Return the answer function handed on by ``delegate_interrupts``.

    Returns
    -------
    callable or None
        ``None`` outside a ``delegate_interrupts`` block and inside a node
        that has not delegated.
    """
    return _delegated_answer.get()


__all__ = [
    "END",
    "START",
    "Command",
    "EndSentinel",
    "GraphInterrupt",
    "GraphInterruptGroup",
    "Interrupt",
    "InterruptFrame",
    "Resume",
    "Send",
    "StartSentinel",
    "bind_interrupt_frame",
    "delegate_interrupts",
    "delegated_answer",
    "interrupt",
    "unbind_interrupt_frame",
]
