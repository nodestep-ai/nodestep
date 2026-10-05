import ast
import pathlib

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples"

FORBIDDEN = {
    ("nodestep.core.command", "interrupt"),
    ("nodestep.state.integrations", "InMemoryStateStore"),
    ("nodestep.state.integrations", "FilesystemStateStore"),
}

PLACEHOLDER_MODELS = {"gpt-5-nano", "gpt-4o-mini"}


def _example_files() -> list[pathlib.Path]:
    return sorted(
        path for path in EXAMPLES_DIR.glob("*.py") if path.name != "__init__.py"
    )


def test_examples_dir_holds_the_examples() -> None:
    assert _example_files(), f"no examples found in {EXAMPLES_DIR}"


def test_examples_use_public_interrupt_and_stores() -> None:
    offenders: list[str] = []
    for path in _example_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for statement in ast.walk(tree):
            if isinstance(statement, ast.ImportFrom):
                for alias in statement.names:
                    if (statement.module, alias.name) in FORBIDDEN:
                        offenders.append(
                            f"{path.name}: from {statement.module} import {alias.name}"
                        )
    assert not offenders, offenders


def test_examples_have_no_placeholder_models() -> None:
    offenders: list[str] = []
    for path in _example_files():
        text = path.read_text(encoding="utf-8")
        for placeholder in PLACEHOLDER_MODELS:
            if placeholder in text:
                offenders.append(f"{path.name}: uses placeholder model {placeholder!r}")
    assert not offenders, offenders
