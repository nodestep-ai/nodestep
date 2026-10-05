import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
    "project"
]


def _imported_top_level_modules(directory: Path) -> set[str]:
    modules: set[str] = set()
    for path in directory.rglob("*.py"):
        for statement in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(statement, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in statement.names)
            elif (
                isinstance(statement, ast.ImportFrom)
                and statement.module
                and statement.level == 0
            ):
                modules.add(statement.module.split(".")[0])
    return modules


def test_the_license_is_given_only_as_an_spdx_expression() -> None:
    assert PROJECT["license"] == "MIT"
    assert not [item for item in PROJECT["classifiers"] if item.startswith("License")]


def test_typing_extensions_is_a_declared_dependency() -> None:
    assert "typing-extensions>=4.14.1,<5" in PROJECT["dependencies"]


def test_the_packages_do_not_import_pydantic_core_directly() -> None:
    for package in ("nodestep", "nodestep_sandbox"):
        assert "pydantic_core" not in _imported_top_level_modules(
            ROOT / "src" / package
        )


def test_the_version_is_the_first_alpha() -> None:
    assert PROJECT["version"] == "0.1.0a1"
    assert "Development Status :: 3 - Alpha" in PROJECT["classifiers"]


def test_the_description_says_alpha() -> None:
    assert "alpha" in PROJECT["description"].lower()


def test_the_all_extra_installs_every_other_extra() -> None:
    extras = PROJECT["optional-dependencies"]
    others = sorted(name for name in extras if name != "all")
    assert extras["all"] == [f"nodestep[{','.join(others)}]"]
