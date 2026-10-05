"""The errors nodestep raises, all derived from `NodestepError`.

An error keeps its arguments as attributes of the same name. The concept and
guide pages, starting at [Graphs and flow](../concepts/graphs.md), say when
each one is raised.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from nodestep.core.agent import AgentHandle, AgentResult


_SOURCE = "git+https://github.com/nodestep-ai/nodestep"


def _rebuild_error(cls: type[NodestepError], args: tuple[Any, ...]) -> NodestepError:
    error = cls.__new__(cls)
    super(NodestepError, error).__init__(*args)
    return error


class NodestepError(Exception):
    """Base class of all nodestep errors.

    Copying or pickling restores the message and attributes without calling
    ``__init__`` again.
    """

    def __reduce__(self) -> tuple[Any, ...]:
        """Rebuild the error from ``args`` and its instance attributes."""
        return (_rebuild_error, (type(self), self.args), self.__dict__)


class ModelProviderError(NodestepError):
    """A call to a chat model failed."""


class ChatHistoryError(ModelProviderError):
    """The messages sent to a chat model do not form a valid conversation."""


class StructuredOutputError(NodestepError):
    """A model's structured output does not validate against its schema.

    Parameters
    ----------
    schema_name : str
        Name of the output schema.
    message : str
        Validation error reported for the output.
    raw_content : str, optional
        Text the model returned.

    Attributes
    ----------
    kind : str
        Always ``"structured_output_error"``, as on ``StructuredOutputEvent``.
    """

    kind = "structured_output_error"

    def __init__(
        self, schema_name: str, message: str, raw_content: str | None = None
    ) -> None:
        self.schema_name = schema_name
        self.message = message
        self.raw_content = raw_content
        super().__init__(f"Structured output does not match '{schema_name}': {message}")


class IntegrationNotInstalledError(NodestepError, ImportError):
    """An optional dependency of a chat integration is not installed.

    Parameters
    ----------
    integration : str
        Name of the chat integration, such as ``"openai"``.
    extra : str
        Name of the nodestep extra that installs the dependency.
    """

    def __init__(self, integration: str, extra: str) -> None:
        self.integration = integration
        self.extra = extra
        super().__init__(
            f"The {integration} chat integration needs the optional dependency. "
            f'Install it with: uv add "nodestep[{extra}] @ {_SOURCE}" '
            f'(or pip install "nodestep[{extra}] @ {_SOURCE}")'
        )


class GraphConfigError(NodestepError):
    """A graph is configured incorrectly."""


class GraphExecutionError(NodestepError):
    """A graph run failed."""


class RunLimitExceededError(NodestepError):
    """A run exceeded a limit such as ``max_steps``.

    Parameters
    ----------
    limit_name : str
        Name of the limit, such as ``"max_steps"``.
    limit_value : int
        Value of the limit.
    """

    def __init__(self, limit_name: str, limit_value: int) -> None:
        self.limit_name = limit_name
        self.limit_value = limit_value
        super().__init__(f"Run limit exceeded for {limit_name}: {limit_value}")


class NodeTimeoutError(GraphExecutionError):
    """A node exceeded its ``timeout``.

    Parameters
    ----------
    node_name : str
        Node that timed out.
    timeout : float
        The node's ``timeout`` in seconds.
    """

    def __init__(self, node_name: str, timeout: float) -> None:
        self.node_name = node_name
        self.timeout = timeout
        super().__init__(f"Node '{node_name}' timed out after {timeout}s")


class ResumeError(GraphExecutionError):
    """A run cannot be resumed as requested."""


class RunInterruptedError(NodestepError):
    """A run paused for input; resume it with ``Resume``.

    Parameters
    ----------
    thread_id : str or None
        Thread of the paused run.
    interrupts : dict[str, Interrupt]
        Pending interrupts by ``Interrupt.key``, as in ``GraphResult.interrupts``.
    """

    def __init__(self, thread_id: str | None, interrupts: dict[str, Any]) -> None:
        self.thread_id = thread_id
        self.interrupts = interrupts
        super().__init__(
            f"Run on thread '{thread_id}' is waiting on {len(interrupts)} interrupt(s)"
        )


class GraphTimeoutError(GraphExecutionError):
    """A run exceeded the graph ``timeout``.

    Parameters
    ----------
    graph_name : str
        Name of the graph.
    timeout : float
        The graph's ``timeout`` in seconds.
    """

    def __init__(self, graph_name: str, timeout: float) -> None:
        self.graph_name = graph_name
        self.timeout = timeout
        super().__init__(f"Graph '{graph_name}' timed out after {timeout}s")


class WorkspaceError(NodestepError):
    """A workspace operation failed."""


class PathAccessError(WorkspaceError):
    """A path points outside the workspace root."""


class StateUpdateError(NodestepError):
    """A state update is invalid."""


class StateStoreError(NodestepError):
    """A state store cannot be used, for example a corrupt file."""


class UnknownThreadError(StateStoreError):
    """A thread, or a branch of it, does not exist in the state store.

    Parameters
    ----------
    thread_id : str
        Thread that was requested.
    branch_id : str, optional
        Branch that was requested.
    """

    def __init__(self, thread_id: str, *, branch_id: str = "main") -> None:
        self.thread_id = thread_id
        self.branch_id = branch_id
        subject = (
            f"Thread '{thread_id}'"
            if branch_id == "main"
            else f"Branch '{branch_id}' of thread '{thread_id}'"
        )
        super().__init__(f"{subject} does not exist in the state store")


class ContextNotProvidedError(NodestepError):
    """A node or tool read ``context``, but the run was started without one."""

    def __init__(self) -> None:
        super().__init__(
            "This run has no context; pass context= to ainvoke, astream, invoke or "
            "stream, and pass it again when resuming, because context is never "
            "persisted"
        )


class InvalidUpdateError(StateUpdateError):
    """Writes to one field in one superstep cannot be merged.

    Parameters
    ----------
    field : str
        Field that was written.
    nodes : list[str]
        Nodes whose writes conflict.
    replacement : bool, optional
        True when the conflict comes from a ``Replace`` of a field that has a
        merging reducer.
    superstep : {"paused", "unfinished"}, optional
        Set when ``update_state`` is the other writer: the update was made
        while the superstep of ``nodes`` was paused or unfinished.
    """

    def __init__(
        self,
        field: str,
        nodes: list[str],
        *,
        replacement: bool = False,
        superstep: Literal["paused", "unfinished"] | None = None,
    ) -> None:
        self.field = field
        self.nodes = sorted(nodes)
        self.replacement = replacement
        self.superstep = superstep
        joined = ", ".join(repr(name) for name in self.nodes)
        if superstep is not None:
            reason = (
                "one of the writes replaces the whole value (Replace), so they "
                "cannot be merged."
                if replacement
                else "the field has no merging reducer (replace/default), so that "
                "write would silently overwrite the update."
            )
            message = (
                f"update_state conflicts with the write of "
                f"node{'s' if len(self.nodes) > 1 else ''} {joined} to field "
                f"'{field}' in the {superstep} superstep: {reason} Update the field "
                "after the superstep has finished."
            )
        else:
            reason = (
                "at least one write replaces the whole value (Replace), which cannot "
                "be merged with other writes in the same super-step."
                if replacement
                else "field has no merging reducer (replace/default), so parallel "
                "updates would silently overwrite. Use a reducer (e.g. add / "
                "add_messages / merge_dict) on this field, or ensure only one branch "
                "writes it per super-step."
            )
            message = (
                f"Conflicting concurrent writes to field '{field}' from nodes "
                f"{joined}: {reason}"
            )
        super().__init__(message)


class ToolExecutionError(NodestepError):
    """A tool call failed.

    Parameters
    ----------
    tool_name : str
        Tool that was called.
    input_data : Any
        Arguments of the call.
    message : str
        Why the call failed; it goes into the error text, not into an attribute.
    """

    def __init__(self, tool_name: str, input_data: Any, message: str) -> None:
        self.tool_name = tool_name
        self.input_data = input_data
        super().__init__(f"Tool '{tool_name}' execution failed: {message}")


class ToolGroupExecutionError(NodestepError):
    """Several parallel tool calls failed.

    Parameters
    ----------
    errors : list[BaseException]
        The error of each failed call.
    """

    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = errors
        details = ", ".join(f"{type(error).__name__}: {error}" for error in errors)
        super().__init__(f"{len(errors)} concurrent tool calls failed: {details}")


class ToolDeniedError(NodestepError):
    """A tool call was denied by a reviewer or a limit.

    Parameters
    ----------
    tool_name : str
        Tool whose call was denied.
    reason : str, optional
        Why it was denied, such as the reviewer's message or the limit.
    """

    def __init__(self, tool_name: str, reason: str | None = None) -> None:
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(
            f"Tool '{tool_name}' denied" + (f": {reason}" if reason else "")
        )


class AgentTimeoutError(NodestepError):
    """Waiting for sub-agents timed out.

    Parameters
    ----------
    completed : list[AgentResult]
        Results of the sub-agents that finished in time.
    pending : list[AgentHandle]
        Handles of the sub-agents that did not finish; they were cancelled.
    """

    def __init__(
        self,
        completed: list[AgentResult],
        pending: list[AgentHandle],
    ) -> None:
        self.completed = completed
        self.pending = pending
        super().__init__(
            f"Agent gather timed out: {len(completed)} completed, {len(pending)} pending"
        )


class AgentGroupError(NodestepError):
    """One or more sub-agents in a group failed.

    Parameters
    ----------
    results : list[AgentResult]
        Results of every sub-agent in the group, failed or not.
    """

    def __init__(self, results: list[AgentResult]) -> None:
        self.results = results
        failed = [result for result in results if result.error is not None]
        details = "; ".join(
            f"'{result.name}' ({result.id}): {type(result.error).__name__}: "
            f"{result.error}"
            for result in failed
        )
        super().__init__(
            f"{len(failed)} of {len(results)} sub-agents failed: {details}"
        )


class AgentHandleLostError(NodestepError):
    """A sub-agent handle is not known where it was looked up.

    Parameters
    ----------
    handle_id : str
        The id that was looked up.
    reason : str
        Why the id is not known there.
    """

    def __init__(self, handle_id: str, reason: str) -> None:
        self.handle_id = handle_id
        self.reason = reason
        super().__init__(f"Agent handle '{handle_id}' is not available: {reason}")
