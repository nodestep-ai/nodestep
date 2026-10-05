import re
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from nodestep import Graph
from nodestep_sandbox.errors import TargetError
from nodestep_sandbox.loader import ObjectPath, TargetLoader

TARGET_MODULE = """
from typing import TypedDict

from nodestep import END, START, Graph, node


class State(TypedDict):
    text: str


@node
def echo(state: State) -> dict:
    return {"text": state["text"]}


def build() -> Graph:
    return Graph(State, name="GRAPH_NAME").flow(START >> echo, echo >> END)


graph = build()


async def build_async() -> Graph:
    return build()


def needs_argument(name):
    return build()


not_a_graph = 42


def returns_number():
    return 7


unflowed = Graph(State, name="unflowed")


class Deps:
    def __init__(self) -> None:
        self.db = "handle"


deps = Deps()


async def make_deps() -> Deps:
    return Deps()


def needs_config(config):
    return Deps()


def factory_raises():
    raise RuntimeError("factory failed on purpose")


async def async_factory_raises():
    raise RuntimeError("async factory failed on purpose")
"""


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setattr(sys, "path", list(sys.path))
    before = set(sys.modules)
    package = tmp_path / "sandbox_fixture_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "app.py").write_text(
        TARGET_MODULE.replace("GRAPH_NAME", "module_graph"), encoding="utf-8"
    )
    (tmp_path / "files").mkdir()
    (tmp_path / "files" / "sandbox_fixture_file.py").write_text(
        TARGET_MODULE.replace("GRAPH_NAME", "file_graph"), encoding="utf-8"
    )
    (tmp_path / "files" / "sandbox_fixture_syntax.py").write_text(
        "def (:\n", encoding="utf-8"
    )
    (tmp_path / "sandbox_fixture_raises.py").write_text(
        'raise ValueError("module fails at import")\n', encoding="utf-8"
    )
    yield tmp_path
    for name in set(sys.modules) - before:
        del sys.modules[name]


async def test_a_module_target_is_imported_with_cwd_first_on_sys_path(
    project: Path,
) -> None:
    graph = await TargetLoader(project).graph("sandbox_fixture_pkg.app:graph")
    assert isinstance(graph, Graph)
    assert graph.name == "module_graph"
    assert sys.path[0] == str(project)


async def test_a_file_target_is_imported_under_its_file_name(project: Path) -> None:
    graph = await TargetLoader(project).graph("files/sandbox_fixture_file.py:graph")
    assert graph.name == "file_graph"
    assert sys.modules["sandbox_fixture_file"].graph is graph


async def test_an_absolute_file_path_works_from_any_directory(project: Path) -> None:
    target = f"{project / 'files' / 'sandbox_fixture_file.py'}:graph"
    graph = await TargetLoader(project / "files").graph(target)
    assert graph.name == "file_graph"


async def test_loading_the_same_file_twice_reuses_the_module(project: Path) -> None:
    loader = TargetLoader(project)
    first = await loader.graph("files/sandbox_fixture_file.py:graph")
    second = await loader.graph("files/sandbox_fixture_file.py:graph")
    assert first is second


@pytest.mark.parametrize("attribute", ["build", "build_async"])
async def test_sync_and_async_factories_are_called(
    project: Path, attribute: str
) -> None:
    graph = await TargetLoader(project).graph(f"sandbox_fixture_pkg.app:{attribute}")
    assert graph.name == "module_graph"


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("sandbox_fixture_pkg.app", "is not package.module:attribute"),
        ("sandbox_fixture_pkg.app:", "is not package.module:attribute"),
        (":graph", "is not package.module:attribute"),
        (
            "sandbox_fixture_pkg.app:not-an-identifier",
            "is not package.module:attribute",
        ),
        ("sandbox_fixture_missing:graph", "No module named 'sandbox_fixture_missing'"),
        (
            "sandbox_fixture_pkg.nope:graph",
            "No module named 'sandbox_fixture_pkg.nope'",
        ),
        ("files/missing.py:graph", "does not exist"),
        ("sandbox_fixture_pkg.app:missing", "has no attribute 'missing'"),
        ("sandbox_fixture_pkg.app:not_a_graph", "is of type int, not a Graph"),
        ("sandbox_fixture_pkg.app:returns_number", "returned a value of type int"),
        ("sandbox_fixture_pkg.app:needs_argument", "takes arguments (name)"),
        ("sandbox_fixture_pkg.app:unflowed", "has no flow"),
    ],
)
async def test_bad_targets_raise_target_error(
    project: Path, target: str, message: str
) -> None:
    with pytest.raises(TargetError, match=re.escape(message)):
        await TargetLoader(project).graph(target)


async def test_an_import_error_inside_the_target_module_is_not_hidden(
    project: Path,
) -> None:
    (project / "sandbox_fixture_broken.py").write_text(
        "import sandbox_fixture_dependency_that_is_missing\n", encoding="utf-8"
    )
    with pytest.raises(TargetError) as caught:
        await TargetLoader(project).graph("sandbox_fixture_broken:graph")
    assert str(caught.value) == (
        "Importing sandbox_fixture_broken failed: ModuleNotFoundError: "
        "No module named 'sandbox_fixture_dependency_that_is_missing'"
    )
    assert isinstance(caught.value.__cause__, ModuleNotFoundError)


@pytest.mark.parametrize(
    ("target", "message", "frame"),
    [
        (
            "sandbox_fixture_pkg.app:factory_raises",
            "Calling sandbox_fixture_pkg.app:factory_raises failed: RuntimeError: "
            "factory failed on purpose",
            'in factory_raises\n    raise RuntimeError("factory failed on purpose")',
        ),
        (
            "sandbox_fixture_pkg.app:async_factory_raises",
            "Calling sandbox_fixture_pkg.app:async_factory_raises failed: "
            "RuntimeError: async factory failed on purpose",
            "in async_factory_raises\n    raise RuntimeError(",
        ),
        (
            "sandbox_fixture_raises:graph",
            "Importing sandbox_fixture_raises failed: ValueError: "
            "module fails at import",
            'sandbox_fixture_raises.py", line 1, in <module>',
        ),
        (
            "files/sandbox_fixture_syntax.py:graph",
            "Importing sandbox_fixture_syntax.py failed: SyntaxError: "
            "invalid syntax (sandbox_fixture_syntax.py, line 1)",
            'sandbox_fixture_syntax.py", line 1',
        ),
    ],
)
async def test_errors_raised_by_the_loaded_code_become_target_errors(
    project: Path, target: str, message: str, frame: str
) -> None:
    with pytest.raises(TargetError) as caught:
        await TargetLoader(project).graph(target)
    assert str(caught.value) == message
    assert caught.value.__cause__ is not None
    details = caught.value.details or ""
    assert frame in details
    assert (
        details.rstrip()
        .splitlines()[-1]
        .startswith(type(caught.value.__cause__).__name__)
    )
    assert "nodestep_sandbox" not in details
    assert "importlib" not in details


async def test_a_context_object_is_used_as_it_is(project: Path) -> None:
    context = await TargetLoader(project).context("sandbox_fixture_pkg.app:deps")
    assert context.db == "handle"


@pytest.mark.parametrize("attribute", ["Deps", "make_deps"])
async def test_a_context_factory_is_called(project: Path, attribute: str) -> None:
    context = await TargetLoader(project).context(
        f"sandbox_fixture_pkg.app:{attribute}"
    )
    assert context.db == "handle"


async def test_a_context_factory_that_raises_gives_a_target_error(
    project: Path,
) -> None:
    with pytest.raises(TargetError) as caught:
        await TargetLoader(project).context("sandbox_fixture_pkg.app:factory_raises")
    assert str(caught.value) == (
        "Calling sandbox_fixture_pkg.app:factory_raises failed: RuntimeError: "
        "factory failed on purpose"
    )


async def test_a_context_factory_with_arguments_is_refused(project: Path) -> None:
    with pytest.raises(TargetError, match=re.escape("takes arguments (config)")):
        await TargetLoader(project).context("sandbox_fixture_pkg.app:needs_config")


def test_object_path_splits_at_the_last_colon() -> None:
    path = ObjectPath.parse("C:\\work\\app.py:graph")
    assert (path.source, path.attribute, path.is_file) == (
        "C:\\work\\app.py",
        "graph",
        True,
    )
    assert str(path) == "C:\\work\\app.py:graph"
    assert ObjectPath.parse("pkg.mod:graph").is_file is False
