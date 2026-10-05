import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from nodestep_sandbox import cli

ROOT = Path(__file__).resolve().parents[2]
HINT = (
    "nodestep sandbox needs the sandbox extra: "
    'uv add --dev "nodestep[sandbox] @ git+https://github.com/nodestep-ai/nodestep"\n'
)


@pytest.mark.parametrize("missing", ["typer", "starlette"])
def test_main_without_the_extra_prints_the_install_hint_and_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], missing: str
) -> None:
    monkeypatch.setitem(sys.modules, missing, None)
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2
    assert capsys.readouterr().err == HINT


def test_missing_extra_modules_names_what_cannot_be_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "jinja2", None)
    assert cli.missing_extra_modules() == ["jinja2"]


def test_the_entry_module_imports_only_the_standard_library() -> None:
    code = (
        "import sys, nodestep_sandbox.cli\n"
        "print(sorted(name for name in ('typer', 'starlette', 'jinja2', 'uvicorn', "
        "'nodestep', 'pydantic') if name in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    )
    assert result.stdout.strip() == "[]"


def test_pyproject_declares_the_console_script_and_ships_the_package() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"] == {"nodestep": "nodestep_sandbox.cli:main"}
    packages = project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert packages == ["src/nodestep", "src/nodestep_sandbox"]
