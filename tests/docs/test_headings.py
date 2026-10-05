import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "overrides" / "hooks" / "headings.py"
MODULE_NAME = '<span class="doc doc-object-name doc-module-name">{}</span>'


@pytest.fixture(scope="module")
def hook() -> ModuleType:
    pytest.importorskip("mkdocs")
    spec = importlib.util.spec_from_file_location("headings", HOOK)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mkdocs_runs_the_hook() -> None:
    config = pytest.importorskip("mkdocs.config").load_config(str(ROOT / "mkdocs.yml"))
    assert "overrides/hooks/headings.py" in config["plugins"]


def test_dotted_names_may_break_after_each_dot(hook: ModuleType) -> None:
    html = MODULE_NAME.format("nodestep.chat.integrations.openai")
    assert hook.break_after_dots(html) == MODULE_NAME.format(
        "nodestep.<wbr>chat.<wbr>integrations.<wbr>openai"
    )


def test_other_text_is_left_alone(hook: ModuleType) -> None:
    html = (
        "<p>See <code>nodestep.core.Graph</code>.</p>"
        '<span class="doc doc-object-name doc-function-name">from_env</span>'
    )
    assert hook.break_after_dots(html) == html


def test_reference_headings_break_at_the_dots(site: Path) -> None:
    html = (
        site / "reference" / "chat" / "integrations" / "openai" / "index.html"
    ).read_text(encoding="utf-8")
    assert (
        MODULE_NAME.format("nodestep.<wbr>chat.<wbr>integrations.<wbr>openai") in html
    )
    assert '<a href="#nodestep.chat.integrations.openai" class="headerlink"' in html
