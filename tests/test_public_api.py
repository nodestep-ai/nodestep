import importlib
import inspect
import pkgutil
import re
import subprocess
import sys
from typing import Any

import pytest

import nodestep
import nodestep.core
import nodestep_sandbox
from nodestep import exceptions
from nodestep.core.agent import AgentHandle, AgentStatus, gather_agents
from nodestep.core.builtin_nodes.tool_runner import ToolErrorPolicy
from nodestep.core.executor import AsyncioExecutor
from nodestep.core.graph import Graph
from nodestep.core.node import Node
from nodestep.core.tool import Tool, ToolCallResult, ToolContext, call_tool
from nodestep.middleware import SkillNotFoundError
from nodestep.middleware.base import Middleware, ToolProvider
from nodestep.state import StateStore
from nodestep.state.integrations import FilesystemStateStore, InMemoryStateStore
from nodestep.workspace import InMemoryWorkspace, LocalWorkspace, Workspace

_WITHOUT_OPENAI = "import sys\nsys.modules['openai'] = None\nsys.modules['pydantic_settings'] = None\n"


def _run_without_openai(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _WITHOUT_OPENAI + code],
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "code",
    [
        "from nodestep import *",
        "from nodestep.chat.integrations import *",
        "import inspect, nodestep\ninspect.getmembers(nodestep)",
        "import inspect, nodestep.chat.integrations as integrations\n"
        "inspect.getmembers(integrations)",
        "import pydoc, nodestep\npydoc.render_doc(nodestep)",
        "import pydoc, nodestep.chat.integrations as integrations\n"
        "pydoc.render_doc(integrations)",
    ],
)
def test_introspection_works_without_the_openai_extra(code: str) -> None:
    result = _run_without_openai(code)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module", ["nodestep", "nodestep.chat.integrations"])
def test_explicit_openai_import_without_the_extra_names_the_install_command(
    module: str,
) -> None:
    code = (
        "from nodestep.exceptions import IntegrationNotInstalledError\n"
        "try:\n"
        f"    from {module} import OpenAIChat\n"
        "except IntegrationNotInstalledError as error:\n"
        "    print(error)\n"
    )
    result = _run_without_openai(code)

    assert result.returncode == 0, result.stderr
    assert "git+https://github.com/nodestep-ai/nodestep" in result.stdout


@pytest.mark.parametrize("module", ["nodestep", "nodestep.chat.integrations"])
def test_hasattr_openai_chat_without_the_extra_raises_integration_not_installed(
    module: str,
) -> None:
    code = (
        "import importlib\n"
        "from nodestep.exceptions import IntegrationNotInstalledError\n"
        f"loaded = importlib.import_module({module!r})\n"
        "try:\n"
        "    hasattr(loaded, 'OpenAIChat')\n"
        "except IntegrationNotInstalledError:\n"
        "    print('raised')\n"
    )
    result = _run_without_openai(code)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "raised"


def test_optional_integrations_stay_out_of_the_star_exports() -> None:
    import nodestep.chat.integrations as integrations

    assert "OpenAIChat" not in nodestep.__all__
    assert {"OpenAIChat", "OpenAISettings"}.isdisjoint(integrations.__all__)


def test_integration_hint_installs_from_github() -> None:
    error = exceptions.IntegrationNotInstalledError("openai", "openai")

    assert str(error) == (
        "The openai chat integration needs the optional dependency. Install it with: "
        'uv add "nodestep[openai] @ git+https://github.com/nodestep-ai/nodestep" '
        '(or pip install "nodestep[openai] @ git+https://github.com/nodestep-ai/nodestep")'
    )


def _exception_classes(module: object) -> set[str]:
    return {
        name
        for name, value in vars(module).items()
        if inspect.isclass(value)
        and issubclass(value, BaseException)
        and value.__module__ == "nodestep.exceptions"
    }


def test_top_level_exports_every_nodestep_exception() -> None:
    names = _exception_classes(exceptions)

    assert names <= set(nodestep.__all__), sorted(names - set(nodestep.__all__))
    for name in names:
        assert getattr(nodestep, name) is getattr(exceptions, name), name


def test_core_exports_no_exceptions() -> None:
    exported = [
        name
        for name in nodestep.core.__all__
        if inspect.isclass(getattr(nodestep.core, name))
        and issubclass(getattr(nodestep.core, name), BaseException)
    ]

    assert exported == []


def test_core_exports_the_types_of_its_signatures() -> None:
    assert nodestep.core.AgentStatus is AgentStatus
    assert nodestep.core.ToolCallResult is ToolCallResult
    assert nodestep.core.ToolErrorPolicy is ToolErrorPolicy
    for name in ("AgentStatus", "ToolCallResult", "ToolErrorPolicy"):
        assert name in nodestep.core.__all__, name


def test_skill_not_found_error_is_a_nodestep_error() -> None:
    assert issubclass(SkillNotFoundError, exceptions.NodestepError)
    assert issubclass(SkillNotFoundError, LookupError)


def test_tool_names_are_written_out() -> None:
    assert list(inspect.signature(call_tool).parameters) == [
        "tool",
        "arguments",
        "ctx",
    ]
    assert list(inspect.signature(Tool.__init__).parameters) == [
        "self",
        "function",
        "metadata",
    ]
    assert "context_parameter_name" in Node.__dataclass_fields__
    assert "context_param_name" not in Node.__dataclass_fields__


ABBREVIATED_NAMES = {
    "exc",
    "func",
    "fn",
    "params",
    "param",
    "attr",
    "arg",
    "err",
    "msg",
}


def _public_callables() -> list[Any]:
    import nodestep.chat.integrations.openai.mapping as openai_mapping
    import nodestep.middleware
    import nodestep.state
    import nodestep.workspace
    from nodestep.chat.integrations.openai import OpenAIChat
    from nodestep_sandbox.loader import ObjectPath

    found: list[Any] = [OpenAIChat, ObjectPath]
    for module in (
        nodestep,
        nodestep.core,
        nodestep.middleware,
        nodestep.state,
        nodestep.workspace,
        openai_mapping,
    ):
        for name in module.__all__:
            value = getattr(module, name)
            if callable(value):
                found.append(value)
            if inspect.isclass(value):
                found.extend(
                    member
                    for member_name, member in vars(value).items()
                    if callable(member) and not member_name.startswith("_")
                )
    return found


def test_public_signatures_write_names_out() -> None:
    abbreviated: list[str] = []
    for value in _public_callables():
        try:
            parameters = inspect.signature(value).parameters
        except (TypeError, ValueError):
            continue
        abbreviated.extend(
            f"{getattr(value, '__qualname__', value)}({name})"
            for name in parameters
            if name in ABBREVIATED_NAMES or name == "request_params"
        )
    assert abbreviated == []


NAME_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def _exported_names() -> list[str]:
    names: list[str] = []
    for package in (nodestep, nodestep_sandbox):
        for info in pkgutil.walk_packages(package.__path__, f"{package.__name__}."):
            module = importlib.import_module(info.name)
            names.extend(
                f"{info.name}.{name}" for name in getattr(module, "__all__", ())
            )
    return names


def test_public_names_write_words_out() -> None:
    abbreviated = [
        path
        for path in _exported_names()
        if any(
            word.lower() in ABBREVIATED_NAMES
            for word in NAME_WORD.findall(path.rpartition(".")[2])
        )
    ]
    assert abbreviated == []


def test_decorators_take_a_function() -> None:
    assert next(iter(inspect.signature(nodestep.node).parameters)) == "function"
    assert next(iter(inspect.signature(nodestep.tool).parameters)) == "function"


def test_openai_request_parameters_are_written_out() -> None:
    from nodestep.chat.integrations.openai import OpenAIChat
    from nodestep_sandbox.loader import ObjectPath

    assert "request_parameters" in inspect.signature(OpenAIChat).parameters
    assert "attribute" in ObjectPath.model_fields


def test_agent_status_and_forget_are_awaitable() -> None:
    assert inspect.iscoroutinefunction(AsyncioExecutor.forget)
    assert inspect.iscoroutinefunction(AgentHandle.status)


def test_agent_handle_takes_only_public_fields() -> None:
    assert list(inspect.signature(AgentHandle).parameters) == [
        "id",
        "name",
        "task",
        "graph_name",
        "thread_id",
    ]


def test_gather_agents_takes_handle_ids() -> None:
    parameter = inspect.signature(gather_agents).parameters["handles"]

    assert parameter.annotation == "Sequence[AgentHandle | str]"


@pytest.mark.parametrize(
    "store", [StateStore, InMemoryStateStore, FilesystemStateStore]
)
def test_state_store_fork_takes_keywords_like_graph_fork(store: Any) -> None:
    parameters = inspect.signature(store.fork).parameters

    assert parameters["from_"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["branch_id"].kind is inspect.Parameter.KEYWORD_ONLY


def test_invoke_spells_out_the_ainvoke_keywords() -> None:
    invoke = inspect.signature(Graph.invoke).parameters
    ainvoke = inspect.signature(Graph.ainvoke).parameters

    assert list(invoke) == list(ainvoke)
    for name, parameter in ainvoke.items():
        assert invoke[name].kind is parameter.kind, name
        assert invoke[name].default == parameter.default, name
        assert invoke[name].annotation == parameter.annotation, name


def test_tool_annotations_are_precise() -> None:
    assert Middleware.tools.__annotations__["return"] == "Iterable[Tool]"
    assert ToolProvider.tools.__annotations__["return"] == "Iterable[Tool]"
    assert ToolContext.from_node_context.__annotations__["ctx"] == "NodeContext"


def _documented_errors(docstring: str) -> set[str]:
    section = inspect.cleandoc(docstring).partition("Raises\n------\n")[2]
    return {
        part.strip()
        for line in section.splitlines()
        if line and not line.startswith(" ") and not line.endswith("-")
        for part in line.split(",")
    }


@pytest.mark.parametrize(
    ("base", "concrete"),
    [
        (Workspace, LocalWorkspace),
        (Workspace, InMemoryWorkspace),
        (StateStore, InMemoryStateStore),
        (StateStore, FilesystemStateStore),
    ],
)
def test_concrete_methods_document_what_the_abstract_ones_do(
    base: type, concrete: type
) -> None:
    names = [
        name
        for name, value in vars(base).items()
        if inspect.isfunction(value) and not name.startswith("_")
    ]
    assert names
    for name in names:
        abstract_doc = inspect.getdoc(getattr(base, name)) or ""
        doc = getattr(concrete, name).__doc__ or ""
        for section in ("Parameters", "Returns"):
            if f"{section}\n---" in abstract_doc:
                assert f"{section}\n" in inspect.cleandoc(doc), (
                    f"{concrete.__name__}.{name} has no {section} section"
                )
        missing = _documented_errors(abstract_doc) - _documented_errors(doc)
        assert not missing, f"{concrete.__name__}.{name} does not list {missing}"


def test_astream_lists_the_errors_ainvoke_lists() -> None:
    missing = _documented_errors(Graph.ainvoke.__doc__ or "") - _documented_errors(
        Graph.astream.__doc__ or ""
    )

    assert not missing, missing
