import ast
import importlib.util
import pathlib
import sys

import pytest

from nodestep import Graph
from nodestep_sandbox.loader import TargetLoader

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXAMPLES = sorted(
    path for path in (ROOT / "examples").glob("*.py") if path.name != "__init__.py"
)


def _uses_openai(path: pathlib.Path) -> bool:
    return any(
        isinstance(statement, ast.alias) and statement.name == "OpenAIChat"
        for statement in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
    )


@pytest.mark.parametrize("path", EXAMPLES, ids=[path.stem for path in EXAMPLES])
async def test_each_example_exposes_a_graph_the_sandbox_loads(
    path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if _uses_openai(path):
        if importlib.util.find_spec("openai") is None:
            pytest.skip("needs the openai extra")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, path.stem, raising=False)

    graph = await TargetLoader(ROOT).graph(f"examples/{path.name}:graph")

    assert isinstance(graph, Graph)
    assert graph.start_node is not None
    sys.modules.pop(path.stem, None)
