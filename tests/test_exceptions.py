import copy
import inspect
import pickle
from collections.abc import Callable
from typing import Any

import pytest

import nodestep.exceptions
from nodestep.core.agent import AgentResult
from nodestep.exceptions import (
    AgentGroupError,
    AgentHandleLostError,
    AgentTimeoutError,
    ChatHistoryError,
    ContextNotProvidedError,
    GraphConfigError,
    GraphExecutionError,
    GraphTimeoutError,
    IntegrationNotInstalledError,
    InvalidUpdateError,
    ModelProviderError,
    NodestepError,
    NodeTimeoutError,
    PathAccessError,
    ResumeError,
    RunInterruptedError,
    RunLimitExceededError,
    StateStoreError,
    StateUpdateError,
    StructuredOutputError,
    ToolDeniedError,
    ToolExecutionError,
    ToolGroupExecutionError,
    UnknownThreadError,
    WorkspaceError,
)

_DONE = AgentResult(id="a1", name="research", state=None, data={}, status="completed")
_FAILED = AgentResult(
    id="a2",
    name="write",
    state=None,
    data={},
    status="error",
    error=RuntimeError("provider down"),
)

_ERRORS: list[NodestepError] = [
    NodestepError("base"),
    ModelProviderError("provider down"),
    ChatHistoryError("orphan tool message"),
    StructuredOutputError("Answer", "x: field required", "not json"),
    IntegrationNotInstalledError("openai", "openai"),
    GraphConfigError("no flow"),
    GraphExecutionError("node failed"),
    RunLimitExceededError("max_steps", 25),
    NodeTimeoutError("fetch", 1.5),
    ResumeError("thread 't1' is paused"),
    RunInterruptedError("t1", {"approve": {"amount": 5}}),
    GraphTimeoutError("support", 2.0),
    WorkspaceError("read failed"),
    PathAccessError("outside the root"),
    StateUpdateError("unknown key"),
    StateStoreError("corrupt file"),
    UnknownThreadError("t1", branch_id="draft"),
    ContextNotProvidedError(),
    InvalidUpdateError("messages", ["b", "a"], replacement=True),
    ToolExecutionError("echo", {"value": 1}, "boom"),
    ToolGroupExecutionError([RuntimeError("a"), ValueError("b")]),
    ToolDeniedError("echo", "not allowed"),
    AgentTimeoutError([_DONE], []),
    AgentGroupError([_DONE, _FAILED]),
    AgentHandleLostError("h1", "this executor does not know it"),
]


def _fingerprint(error: BaseException) -> tuple[Any, ...]:
    return (
        type(error),
        error.args,
        str(error),
        repr(vars(error)),
        getattr(error, "msg", None),
    )


def _pickle_roundtrip(error: NodestepError) -> NodestepError:
    return pickle.loads(pickle.dumps(error))


@pytest.mark.parametrize(
    ("error", "base"),
    [
        (UnknownThreadError, StateStoreError),
        (ContextNotProvidedError, NodestepError),
        (ChatHistoryError, ModelProviderError),
        (AgentGroupError, NodestepError),
        (StructuredOutputError, NodestepError),
    ],
)
def test_errors_extend_their_base(error: type, base: type) -> None:
    assert issubclass(error, base)


def test_unknown_thread_error_names_the_thread() -> None:
    error = UnknownThreadError("t1")

    assert (error.thread_id, error.branch_id) == ("t1", "main")
    assert str(error) == "Thread 't1' does not exist in the state store"


def test_unknown_thread_error_names_a_missing_branch() -> None:
    error = UnknownThreadError("t1", branch_id="draft")

    assert error.branch_id == "draft"
    assert str(error) == (
        "Branch 'draft' of thread 't1' does not exist in the state store"
    )


def test_context_not_provided_error_says_how_to_pass_context() -> None:
    message = str(ContextNotProvidedError())

    assert "context=" in message
    assert "resum" in message


def test_agent_group_error_holds_every_result_and_names_the_failures() -> None:
    done = AgentResult(
        id="a1", name="research", state=None, data={}, status="completed"
    )
    failed = AgentResult(
        id="a2",
        name="write",
        state=None,
        data={},
        status="error",
        error=RuntimeError("provider down"),
    )

    error = AgentGroupError([done, failed])

    assert error.results == [done, failed]
    assert str(error) == (
        "1 of 2 sub-agents failed: 'write' (a2): RuntimeError: provider down"
    )


def test_structured_output_error_keeps_the_event_fields() -> None:
    error = StructuredOutputError(
        schema_name="Answer", message="x: field required", raw_content="not json"
    )

    assert error.schema_name == "Answer"
    assert error.message == "x: field required"
    assert error.raw_content == "not json"
    assert error.kind == "structured_output_error"
    assert str(error) == (
        "Structured output does not match 'Answer': x: field required"
    )


def test_every_error_class_is_covered_by_the_copy_cases() -> None:
    defined = {
        cls
        for _, cls in inspect.getmembers(nodestep.exceptions, inspect.isclass)
        if issubclass(cls, NodestepError) and cls.__module__ == "nodestep.exceptions"
    }

    assert {type(error) for error in _ERRORS} == defined


@pytest.mark.parametrize("error", _ERRORS, ids=lambda error: type(error).__name__)
@pytest.mark.parametrize(
    "roundtrip",
    [copy.copy, copy.deepcopy, _pickle_roundtrip],
    ids=["copy", "deepcopy", "pickle"],
)
def test_errors_survive_copy_and_pickle(
    error: NodestepError, roundtrip: Callable[[NodestepError], NodestepError]
) -> None:
    restored = roundtrip(error)

    assert _fingerprint(restored) == _fingerprint(error)
