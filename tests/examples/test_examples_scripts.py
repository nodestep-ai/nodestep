import ast
import pathlib

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples"
EXAMPLES = sorted(
    path for path in EXAMPLES_DIR.glob("*.py") if path.name != "__init__.py"
)


def imported_names(path: pathlib.Path) -> set[str]:
    return {
        alias.name
        for statement in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(statement, ast.ImportFrom)
        for alias in statement.names
    }


@pytest.mark.parametrize("path", EXAMPLES, ids=[path.stem for path in EXAMPLES])
def test_example_starts_with_its_imports(path: pathlib.Path) -> None:
    text = path.read_text(encoding="utf-8")

    assert ast.get_docstring(ast.parse(text)) is None
    assert "# ///" not in text
    assert text.startswith(("import ", "from "))


@pytest.mark.parametrize("path", EXAMPLES, ids=[path.stem for path in EXAMPLES])
def test_example_runs_without_a_key(path: pathlib.Path) -> None:
    assert "OpenAIChat" not in imported_names(path)
