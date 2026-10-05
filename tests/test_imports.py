import nodestep
from nodestep.core.command import Interrupt, Resume
from nodestep.utils.reducers import RemoveMessage, Replace


def test_top_level_imports() -> None:
    assert nodestep.Graph is not None
    assert nodestep.END is not None
    assert callable(nodestep.node)
    assert callable(nodestep.branch)
    assert callable(nodestep.when)
    assert callable(nodestep.tool)
    assert nodestep.Resume is Resume
    assert nodestep.Interrupt is Interrupt
    assert nodestep.Replace is Replace
    assert nodestep.RemoveMessage is RemoveMessage
    assert nodestep.OpenAIChat is not None


def test_tool_is_class() -> None:
    from nodestep.core.tool import Tool

    assert isinstance(Tool, type)


def test_chat_no_streaming_chat() -> None:
    import nodestep.chat as chat_module

    assert not hasattr(chat_module, "StreamingChat")


def test_assemble_stream_importable() -> None:
    from nodestep.chat import assemble_stream

    assert callable(assemble_stream)


def test_agent_export() -> None:
    assert hasattr(nodestep, "build_react_agent")
    assert hasattr(nodestep, "AgentState")


def test_run_agent_export() -> None:
    assert callable(nodestep.run_agent)


def test_store_primitives_top_level() -> None:
    assert nodestep.InMemoryStateStore is not None
    assert nodestep.FilesystemStateStore is not None
    assert nodestep.interrupt is not None


def test_store_top_level_matches_integration_module() -> None:
    from nodestep.state.integrations import (
        FilesystemStateStore,
        InMemoryStateStore,
    )

    assert nodestep.InMemoryStateStore is InMemoryStateStore
    assert nodestep.FilesystemStateStore is FilesystemStateStore


def test_cleanup_public_imports() -> None:
    from nodestep.state.integrations import InMemoryStateStore

    assert "TransientState" not in nodestep.__all__
    assert "SecretState" not in nodestep.__all__
    assert InMemoryStateStore is not None


def test_middleware_imports() -> None:
    from nodestep import Middleware
    from nodestep.middleware import FilesystemSkills, TodoListMiddleware

    assert Middleware is not None
    assert FilesystemSkills is not None
    assert TodoListMiddleware is not None


def test_subgraph_and_summarization_imports() -> None:
    from nodestep import subgraph
    from nodestep.middleware import SummarizationMiddleware

    assert SummarizationMiddleware is not None
    assert callable(subgraph)


def test_summarization_trigger_classes_import() -> None:
    from nodestep.middleware import (
        ApproximateTokenCounter,
        FixedTokenTrigger,
        MessageCountTrigger,
        RecentMessagesPolicy,
        RecentTokensPolicy,
        SummarizationDecision,
        SummarizationResult,
        SummarizationTrigger,
        TokenCounter,
        TokenLimitTrigger,
    )

    assert ApproximateTokenCounter is not None
    assert TokenLimitTrigger is not None
    assert FixedTokenTrigger is not None
    assert MessageCountTrigger is not None
    assert RecentMessagesPolicy is not None
    assert RecentTokensPolicy is not None
    assert SummarizationDecision is not None
    assert SummarizationResult is not None
    assert TokenCounter is not None
    assert SummarizationTrigger is not None


def test_todo_state_enum_imports() -> None:
    from nodestep.middleware import TodoItem, TodoItemState

    assert TodoItem is not None
    assert {state.value for state in TodoItemState} == {
        "pending",
        "in_progress",
        "completed",
        "blocked",
        "cancelled",
        "failed",
    }


def test_memory_imports() -> None:
    from nodestep.middleware.memory import (
        BaseMemory,
        FilesystemMemory,
        Memory,
        bm25_search,
    )

    assert BaseMemory is not None
    assert FilesystemMemory is not None
    assert Memory is not None
    assert callable(bm25_search)


def test_validation_warning_is_removed() -> None:
    import nodestep.core
    import nodestep.core.flow

    assert not hasattr(nodestep.core, "ValidationWarning")
    assert not hasattr(nodestep.core.flow, "ValidationWarning")


def test_mermaid_id_import() -> None:
    from nodestep.core.render import mermaid_id

    assert mermaid_id("a b") == "a_b"


def test_tool_denied_error_import() -> None:
    from nodestep.exceptions import ToolDeniedError

    assert ToolDeniedError is not None


def test_timeout_error_imports() -> None:
    from nodestep.exceptions import GraphTimeoutError, NodeTimeoutError

    assert GraphTimeoutError is not None
    assert NodeTimeoutError is not None


def test_state_models_imports() -> None:
    from nodestep.models.base import NodestepModel
    from nodestep.utils.json import dump_json_object, load_json_object

    assert NodestepModel is not None
    assert dump_json_object({"b": 2, "a": 1}) == '{"a": 1, "b": 2}'
    assert load_json_object('{"a": 1}') == {"a": 1}


def test_state_schema_imports() -> None:
    from nodestep.state import (
        FieldDescriptor,
        StateSchema,
        StateUpdateError,
        to_json_value,
    )

    assert FieldDescriptor is not None
    assert StateSchema is not None
    assert StateUpdateError is not None
    assert to_json_value({"a": (1, 2)}) == {"a": [1, 2]}


def test_state_history_imports() -> None:
    from nodestep.state import (
        BranchRecord,
        CheckpointRecord,
        History,
        HistoryEvent,
        StateStore,
    )
    from nodestep.state.integrations import FilesystemStateStore, InMemoryStateStore

    assert History is not None
    assert HistoryEvent is not None
    assert CheckpointRecord is not None
    assert BranchRecord is not None
    assert StateStore is not None
    assert InMemoryStateStore is not None
    assert FilesystemStateStore is not None


def test_state_history_module_exists() -> None:
    import importlib.util

    assert importlib.util.find_spec("nodestep.state.history") is not None


def test_core_import_has_no_httpx_or_tenacity() -> None:
    import subprocess
    import sys

    code = (
        "import sys; import nodestep; "
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] in {'httpx', 'tenacity'}); "
        "assert not leaked, leaked"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8"
    )
    assert result.returncode == 0, result.stderr


def test_openaichat_is_lazy_attribute() -> None:
    import nodestep

    assert "OpenAIChat" not in nodestep.__all__
    assert "OpenAIChat" in dir(nodestep)
    cls = nodestep.OpenAIChat
    assert cls.__name__ == "OpenAIChat"


def test_openaichat_not_eager_module_global() -> None:
    import nodestep

    assert "OpenAIChat" not in vars(nodestep), (
        "OpenAIChat must be resolved lazily via __getattr__, not bound at module load"
    )


def test_import_nodestep_does_not_import_openai_eagerly() -> None:
    import subprocess
    import sys

    code = (
        "import sys; import nodestep; "
        "assert 'openai' not in sys.modules, sorted(m for m in sys.modules if 'openai' in m)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8"
    )
    assert result.returncode == 0, result.stderr


def test_workspace_and_errors_imports() -> None:
    from nodestep.exceptions import WorkspaceError
    from nodestep.workspace import LocalWorkspace

    assert LocalWorkspace is not None
    assert WorkspaceError is not None


def test_version_comes_from_package_metadata() -> None:
    from importlib.metadata import version

    assert nodestep.__version__ == version("nodestep")


def test_dir_lists_the_public_names() -> None:
    assert dir(nodestep) == sorted([*nodestep.__all__, "OpenAIChat"])


def test_star_import_works_without_the_openai_extra() -> None:
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['pydantic_settings'] = None; sys.modules['openai'] = None\n"
        "namespace = {}\n"
        "exec('from nodestep import *', namespace)\n"
        "exec('from nodestep.chat.integrations import *', namespace)\n"
        "assert 'ScriptedChat' in namespace and 'Graph' in namespace\n"
        "assert 'OpenAIChat' not in namespace and 'OpenAISettings' not in namespace\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8"
    )
    assert result.returncode == 0, result.stderr


def test_every_public_name_resolves() -> None:
    for name in nodestep.__all__:
        assert getattr(nodestep, name) is not None, name


def test_top_level_exports_the_contract_names() -> None:
    from nodestep import exceptions
    from nodestep.core.agent import AgentStatus
    from nodestep.core.builtin_nodes.model import StructuredOutputEvent
    from nodestep.middleware.base import ModelMiddlewareContext

    assert nodestep.ScriptedChat is not None
    assert nodestep.AgentStatus is AgentStatus
    assert nodestep.ModelMiddlewareContext is ModelMiddlewareContext
    assert nodestep.core.ModelMiddlewareContext is ModelMiddlewareContext
    assert nodestep.core.StructuredOutputEvent is StructuredOutputEvent
    for name in (
        "GraphConfigError",
        "GraphExecutionError",
        "GraphTimeoutError",
        "NodeTimeoutError",
        "RunLimitExceededError",
        "ResumeError",
        "InvalidUpdateError",
        "StateStoreError",
        "ModelProviderError",
        "IntegrationNotInstalledError",
        "UnknownThreadError",
        "ContextNotProvidedError",
        "ChatHistoryError",
        "AgentGroupError",
        "StructuredOutputError",
    ):
        assert name in nodestep.__all__, name
        assert getattr(nodestep, name) is getattr(exceptions, name), name


def test_openaichat_without_the_extra_raises_integration_not_installed() -> None:
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['pydantic_settings'] = None; sys.modules['openai'] = None; "
        "import nodestep\n"
        "try:\n"
        "    nodestep.OpenAIChat\n"
        "except nodestep.IntegrationNotInstalledError as error:\n"
        "    assert 'uv add \"nodestep[openai] @ git+' in str(error), error\n"
        "else:\n"
        "    raise AssertionError('no error')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8"
    )
    assert result.returncode == 0, result.stderr
