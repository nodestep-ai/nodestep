import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "overrides" / "hooks" / "reference.py"
PAGE = (
    '<h1 id="graphs">Graphs</h1><p>Intro.</p>'
    '<h2 id="graph">Graph<a class="headerlink" href="#graph">&para;</a></h2>'
    '<div class="doc ref-entry" data-ref-id="nodestep.core.Graph" data-ref-name="Graph">'
    '<p class="ref-summary">A graph.</p>'
    '<details class="ref-method"><summary><span class="ref-method-summary">'
    "Run it.</span></summary></details></div>"
    '<div class="doc ref-entry" data-ref-id="nodestep.core.END" data-ref-name="END">'
    "</div>"
    '<h2 id="flow">Flow<a class="headerlink" href="#flow">&para;</a></h2>'
    '<div class="doc ref-entry" data-ref-id="nodestep.core.when" data-ref-name="when">'
    '<p class="ref-summary">Route on a predicate.</p></div>'
)


@pytest.fixture(scope="module")
def hook() -> ModuleType:
    pytest.importorskip("mkdocs")
    spec = importlib.util.spec_from_file_location("reference", HOOK)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mkdocs_runs_the_hook_with_the_reference_templates() -> None:
    config = pytest.importorskip("mkdocs.config").load_config(str(ROOT / "mkdocs.yml"))
    assert "overrides/hooks/reference.py" in config["plugins"]
    plugin = config["plugins"]["mkdocstrings"].config
    assert plugin["custom_templates"] == str(ROOT / "templates")
    for name in ("module", "class", "function", "attribute"):
        assert (
            ROOT / "templates" / "python" / "material" / f"{name}.html.jinja"
        ).is_file()


def test_the_overview_groups_entries_under_the_heading_before_them(
    hook: ModuleType,
) -> None:
    groups = hook.Overview(PAGE).groups()
    assert [(group.title, group.anchor) for group in groups] == [
        ("Graph", "graph"),
        ("Flow", "flow"),
    ]
    assert [
        [(row.name, row.anchor, row.summary) for row in group.rows] for group in groups
    ] == [
        [
            ("Graph", "nodestep.core.Graph", "A graph."),
            ("END", "nodestep.core.END", ""),
        ],
        [("when", "nodestep.core.when", "Route on a predicate.")],
    ]


def test_a_page_without_group_headings_has_no_overview_groups(
    hook: ModuleType,
) -> None:
    flat = PAGE.replace('<h2 id="graph">', "<h3>").replace('<h2 id="flow">', "<h3>")
    assert hook.Overview(flat).groups() == []


def test_the_table_of_contents_keeps_groups_and_entries_but_not_members(
    hook: ModuleType,
) -> None:
    from mkdocs.structure.toc import AnchorLink

    title = AnchorLink("Graphs", "graphs", 1)
    group = AnchorLink("Graph", "graph", 2)
    entry = AnchorLink("Graph", "nodestep.core.Graph", 3)
    entry.children = [AnchorLink("invoke", "nodestep.core.Graph.invoke", 4)]
    group.children = [entry]
    title.children = [group]
    module = AnchorLink("nodestep.chat", "nodestep.chat", 1)
    flat_entry = AnchorLink("Message", "nodestep.chat.Message", 2)
    flat_entry.children = [AnchorLink("text", "nodestep.chat.Message.text", 3)]
    module.children = [flat_entry]
    hook.Contents([title, module]).without_members()
    assert title.children == [group]
    assert group.children == [entry]
    assert entry.children == []
    assert module.children == [flat_entry]
    assert flat_entry.children == []
