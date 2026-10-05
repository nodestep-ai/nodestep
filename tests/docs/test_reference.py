import importlib
import importlib.util
import inspect
import re
from collections import Counter
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import nodestep

pytest.importorskip("mkdocstrings", reason="needs the docs dependency group")

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"
REFERENCE_DIR = DOCS_DIR / "reference"
DIRECTIVE = re.compile(r"^::: ([\w.]+)$", re.MULTILINE)
TITLE = re.compile(r"^# (.+)$", re.MULTILINE)
HEADING_ID = re.compile(r'<h[1-6] id="([^"]+)"')
MISSING = object()
API_REFERENCE = [
    {
        "core": [
            {"core": "reference/core/index.md"},
            {"Graphs": "reference/core/graphs.md"},
            {"Flow and branches": "reference/core/flow.md"},
            {"Nodes and tools": "reference/core/nodes.md"},
            {"Running and results": "reference/core/running.md"},
            {"Reducers": "reference/core/reducers.md"},
            {"Sub-agents": "reference/core/sub-agents.md"},
            {"builtin_nodes": "reference/core/builtin_nodes.md"},
        ]
    },
    {
        "state": [
            {"state": "reference/state/index.md"},
            {
                "integrations": [
                    {"filesystem": "reference/state/integrations/filesystem.md"},
                    {"inmemory": "reference/state/integrations/inmemory.md"},
                ]
            },
        ]
    },
    {
        "chat": [
            {"chat": "reference/chat/index.md"},
            {
                "integrations": [
                    {"integrations": "reference/chat/integrations/index.md"},
                    {"openai": "reference/chat/integrations/openai.md"},
                ]
            },
        ]
    },
    {
        "middleware": [
            {"middleware": "reference/middleware/index.md"},
            {"interrupt": "reference/middleware/interrupt.md"},
            {"memory": "reference/middleware/memory.md"},
            {"skills": "reference/middleware/skills.md"},
            {"summarization": "reference/middleware/summarization.md"},
            {"todolist": "reference/middleware/todolist.md"},
            {"tool_limit": "reference/middleware/tool_limit.md"},
        ]
    },
    {
        "workspace": [
            {"workspace": "reference/workspace/index.md"},
            {
                "integrations": [
                    {"local": "reference/workspace/integrations/local.md"},
                    {"virtual": "reference/workspace/integrations/virtual.md"},
                ]
            },
        ]
    },
    {"exceptions": "reference/exceptions.md"},
    {"Sandbox CLI": "reference/sandbox-cli.md"},
]


def is_module(identifier: str) -> bool:
    try:
        return importlib.util.find_spec(identifier) is not None
    except ModuleNotFoundError:
        return False


class ReferencePage:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.uri = source.relative_to(DOCS_DIR).as_posix()
        self.text = source.read_text(encoding="utf-8")
        first = DIRECTIVE.findall(self.text)[0]
        self.is_package_page = is_module(first)
        self.module = first if self.is_package_page else self.folder_package()

    def folder_package(self) -> str:
        folder = Path(self.uri).parent.relative_to("reference")
        return ".".join(("nodestep", *folder.parts))

    @property
    def title(self) -> str:
        return TITLE.findall(self.text)[0]

    @property
    def url(self) -> str:
        path = self.uri.removesuffix(".md")
        return path.removesuffix("index") if path.endswith("/index") else f"{path}/"

    def html(self, site: Path) -> str:
        return (site / self.url / "index.html").read_text(encoding="utf-8")

    def headings(self, site: Path) -> list[str]:
        return HEADING_ID.findall(self.html(site))

    def documented(self, site: Path) -> list[object]:
        return [
            value
            for heading in self.headings(site)
            if (value := documented_object(heading)) is not MISSING
        ]


def documented_object(heading: str) -> object:
    module_name, _, name = heading.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ValueError):
        return MISSING
    return getattr(module, name, MISSING)


@cache
def reference_pages() -> tuple[ReferencePage, ...]:
    return tuple(
        ReferencePage(source)
        for source in sorted(REFERENCE_DIR.rglob("*.md"))
        if DIRECTIVE.search(source.read_text(encoding="utf-8"))
    )


def public_names(module: ModuleType) -> list[str]:
    if hasattr(module, "__all__"):
        return list(module.__all__)
    return [
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and getattr(value, "__module__", None) == module.__name__
    ]


@cache
def public_objects() -> dict[str, object]:
    modules = [nodestep] + [
        importlib.import_module(page.module) for page in reference_pages()
    ]
    return {
        f"{module.__name__}.{name}": getattr(module, name)
        for module in modules
        for name in public_names(module)
    }


def nav_files(items: Any) -> list[str]:
    if isinstance(items, str):
        return [items]
    if isinstance(items, dict):
        return [page for value in items.values() for page in nav_files(value)]
    return [page for item in items for page in nav_files(item)]


def api_reference_nav() -> list[Any]:
    config = pytest.importorskip("mkdocs.config")
    nav = config.load_config(str(ROOT / "mkdocs.yml"))["nav"]
    docs = next(item["nodestep"] for item in nav if "nodestep" in item)
    return next(item["API reference"] for item in docs if "API reference" in item)


def page_paths(
    items: list[Any], path: tuple[str, ...]
) -> list[tuple[tuple[str, ...], str]]:
    found = []
    for item in items:
        for label, value in item.items():
            if isinstance(value, list):
                found.extend(page_paths(value, (*path, label)))
            elif value.endswith("/index.md"):
                assert label == path[-1], value
                found.append((path, value))
            else:
                found.append(((*path, label), value))
    return found


def test_the_api_reference_is_the_package_tree() -> None:
    assert api_reference_nav() == API_REFERENCE


def test_the_package_tree_has_no_page_for_the_root_package() -> None:
    assert not (REFERENCE_DIR / "nodestep.md").exists()
    assert "nodestep" not in {page.module for page in reference_pages()}


def test_each_label_is_the_name_of_its_package_or_module() -> None:
    pages = {page.uri: page for page in reference_pages()}
    labelled = page_paths(api_reference_nav(), ())
    for labels, uri in labelled:
        page = pages.get(uri)
        if page and page.is_package_page:
            assert ("nodestep", *labels) == tuple(page.module.split(".")), uri
        elif page:
            assert labels[-1] == page.title, uri
            assert ("nodestep", *labels[:-1]) == tuple(page.module.split(".")), uri
    assert {uri for _, uri in labelled} >= set(pages)


def test_a_page_split_by_purpose_belongs_to_the_package_of_its_folder() -> None:
    split = [page for page in reference_pages() if not page.is_package_page]
    assert {page.uri for page in split} >= {
        "reference/core/graphs.md",
        "reference/core/flow.md",
        "reference/core/nodes.md",
        "reference/core/running.md",
        "reference/core/reducers.md",
        "reference/core/sub-agents.md",
    }
    for page in split:
        index = DOCS_DIR / Path(page.uri).parent / "index.md"
        package = DIRECTIVE.findall(index.read_text(encoding="utf-8"))[0]
        assert page.module == package, page.uri


def test_no_page_is_titled_after_its_file_name(site: Path) -> None:
    offenders = [
        str(page.relative_to(site))
        for page in sorted(site.rglob("index.html"))
        if re.search(
            r'aria-label="(?:Previous|Next): Index"|<title>Index ',
            page.read_text(encoding="utf-8"),
        )
    ]
    assert not offenders


def test_every_reference_page_is_in_the_package_tree() -> None:
    listed = nav_files(api_reference_nav())
    on_disk = sorted(
        source.relative_to(DOCS_DIR).as_posix()
        for source in REFERENCE_DIR.rglob("*.md")
    )
    assert sorted(listed) == on_disk


def test_every_public_name_has_exactly_one_reference_page(site: Path) -> None:
    pages = {
        page.uri: {id(value) for value in page.documented(site)}
        for page in reference_pages()
    }
    wrong = {
        name: sorted(
            uri for uri, documented in pages.items() if id(value) in documented
        )
        for name, value in public_objects().items()
    }
    wrong = {name: uris for name, uris in wrong.items() if len(uris) != 1}
    assert not wrong, (
        f"each public name needs exactly one reference page: {wrong}. A name needs "
        "a docstring to be rendered; leave a name out of a package page with "
        "'filters' or 'members' when its own module has a page, and add a "
        "'::: module.path.name' entry for a name no page renders."
    )


def test_every_name_is_on_the_page_of_the_package_that_defines_it(
    site: Path,
) -> None:
    modules = {page.module for page in reference_pages()}
    misplaced = []
    for page in reference_pages():
        for value in page.documented(site):
            defined_in = getattr(value, "__module__", "") or ""
            homes = [
                module
                for module in modules
                if defined_in == module or defined_in.startswith(f"{module}.")
            ]
            if homes and max(homes, key=len) != page.module:
                misplaced.append(f"{value!r} on {page.uri}")
    assert not misplaced, misplaced


@pytest.mark.parametrize("page", reference_pages(), ids=lambda page: page.uri)
def test_reference_page_renders_each_name_once(site: Path, page: ReferencePage) -> None:
    counts = Counter(page.headings(site))
    repeated = sorted(heading for heading, count in counts.items() if count > 1)
    assert not repeated, (
        f"{page.uri} renders {', '.join(repeated)} more than once; remove the "
        "explicit '::: ...' entry of a name the package entry now renders."
    )


def test_package_groups_open_with_their_own_page(site: Path) -> None:
    html = (
        site / "reference" / "chat" / "integrations" / "openai" / "index.html"
    ).read_text(encoding="utf-8")
    for href, label in (
        ("../../../core/", "core"),
        ("../../../state/", "state"),
        ("../../", "chat"),
        ("../", "integrations"),
        ("../../../middleware/", "middleware"),
        ("../../../workspace/", "workspace"),
    ):
        assert re.search(
            r'<div class="md-nav__link md-nav__container">\s*'
            rf'<a href="{re.escape(href)}" class="md-nav__link ">\s*'
            rf'<span class="md-ellipsis">\s*{label}\s*</span>',
            html,
        ), label
    assert re.search(
        r'<label class="md-nav__link" for="\w+" id="\w+_label" tabindex="0">\s*'
        r'<span class="md-ellipsis">\s*integrations\s*</span>',
        html,
    )


def test_only_groups_inside_a_sidebar_section_open_with_their_own_page(
    site: Path,
) -> None:
    html = (site / "index.html").read_text(encoding="utf-8")
    assert re.search(r'<a href="\." class="md-nav__link md-nav__link--active">', html)
    assert "Get started" in html
    assert not re.search(
        r'md-nav__container">\s*<a href="[^"]*"[^>]*>\s*<span class="md-ellipsis">'
        r"\s*Get started",
        html,
    )


UNRESOLVED = re.compile(
    r'(?<!<td><code>)(?<!<b><code>)<span title="(nodestep\.[^"]+)">'
)


def reference_html(site: Path) -> list[str]:
    return [
        page.read_text(encoding="utf-8")
        for page in (site / "reference").rglob("index.html")
    ]


def test_public_types_in_signatures_resolve(site: Path) -> None:
    unresolved = sorted(
        {
            name
            for html in reference_html(site)
            for name in UNRESOLVED.findall(html)
            if not name.rpartition(".")[2].startswith("_")
        }
    )
    assert not unresolved, (
        f"the reference links to {', '.join(unresolved)}, which no page renders; "
        "export the name from its package or add a '::: module.path.name' entry"
    )


def test_public_signatures_show_no_private_names(site: Path) -> None:
    private = sorted(
        {
            name
            for html in reference_html(site)
            for name in UNRESOLVED.findall(html)
            if name.rpartition(".")[2].startswith("_")
        }
    )
    assert not private, (
        f"the reference shows {', '.join(private)} in a public signature; a "
        "caller can neither pass nor read a private name there"
    )


def test_the_openai_integration_has_its_own_page_under_chat_integrations(
    site: Path,
) -> None:
    page = next(
        page
        for page in reference_pages()
        if page.module == "nodestep.chat.integrations.openai"
    )
    assert page.uri == "reference/chat/integrations/openai.md"
    assert {
        "nodestep.chat.integrations.openai.OpenAIChat",
        "nodestep.chat.integrations.openai.OpenAISettings",
    } <= set(page.headings(site))


def test_the_built_in_nodes_have_their_own_page_under_core(site: Path) -> None:
    page = next(
        page
        for page in reference_pages()
        if page.module == "nodestep.core.builtin_nodes"
    )
    assert page.uri == "reference/core/builtin_nodes.md"
    documented = {id(value) for value in page.documented(site)}
    assert {id(nodestep.build_react_agent), id(nodestep.subgraph)} <= documented


def test_classes_and_functions_are_found_by_identity() -> None:
    assert documented_object("nodestep.core.Graph") is nodestep.Graph
    assert documented_object("nodestep.core.node.node") is nodestep.node
    assert documented_object("nodestep.core.Graph.invoke") is MISSING
    assert inspect.isclass(
        public_objects()["nodestep.chat.integrations.openai.OpenAIChat"]
    )
