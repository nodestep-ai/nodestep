import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

import nodestep

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "overrides" / "hooks" / "autolink.py"
OPENAI_CHAT = "nodestep.chat.integrations.openai.OpenAIChat"


@pytest.fixture(scope="module")
def hook() -> ModuleType:
    pytest.importorskip("mkdocs")
    spec = importlib.util.spec_from_file_location("autolink", HOOK)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def link(hook: ModuleType, markdown: str, page: str = "concepts/graphs.md") -> str:
    return hook.LINKS.page(markdown, page)


def test_mkdocs_runs_the_hook_and_watches_the_sources() -> None:
    config = pytest.importorskip("mkdocs.config").load_config(str(ROOT / "mkdocs.yml"))
    assert "overrides/hooks/autolink.py" in config["plugins"]
    assert str(ROOT / "src") in config["watch"]


def test_every_export_has_a_reference_identifier(hook: ModuleType) -> None:
    identifiers = hook.LINKS.identifiers
    assert set(identifiers) == {*nodestep.__all__, "OpenAIChat"}
    assert identifiers["Graph"] == "nodestep.core.graph.Graph"
    assert identifiers["OpenAIChat"] == OPENAI_CHAT
    assert identifiers["node"] == "nodestep.core.node.node"
    assert identifiers["FilesystemStateStore"] == (
        "nodestep.state.integrations.filesystem.FilesystemStateStore"
    )
    assert identifiers["START"] == "nodestep.core.START"


def test_no_identifier_points_at_the_root_package(hook: ModuleType) -> None:
    assert not [
        identifier
        for identifier in hook.LINKS.identifiers.values()
        if identifier.count(".") == 1
    ]


def test_inline_code_naming_an_export_becomes_a_reference_link(
    hook: ModuleType,
) -> None:
    assert link(hook, "A `Graph` holds `START` and `END`.\n") == (
        "A [`Graph`][nodestep.core.graph.Graph] holds [`START`][nodestep.core.START]"
        " and [`END`][nodestep.core.END].\n"
    )
    assert link(hook, "`OpenAIChat` calls the API.") == (
        f"[`OpenAIChat`][{OPENAI_CHAT}] calls the API."
    )
    assert link(hook, "- with `@node` or `node`") == (
        "- with `@node` or [`node`][nodestep.core.node.node]"
    )


@pytest.mark.parametrize(
    "text",
    [
        "`Graph(...)` takes settings",
        "`OpenAIChat.from_env` reads them",
        "`Graphs` and `graph` and `GraphX`",
        "``Graph`` with two backticks",
        "``a `Graph` b`` inside a longer span",
    ],
)
def test_only_exact_names_are_linked(hook: ModuleType, text: str) -> None:
    assert link(hook, text) == text


@pytest.mark.parametrize(
    "text",
    [
        "## The `Graph` class\n",
        "    # `Graph` in an indented heading\n",
        "[`Graph`](concepts/graphs.md) is linked already\n",
        "[`Graph`][nodestep.core.graph.Graph] is linked already\n",
        "[the `Graph` class](concepts/graphs.md)\n",
        "![`Graph` diagram](assets/graph.svg)\n",
        '!!! note "About `Graph`"\n',
    ],
)
def test_headings_links_and_titles_are_left_alone(hook: ModuleType, text: str) -> None:
    assert link(hook, text) == text


def test_code_blocks_are_left_alone(hook: ModuleType) -> None:
    markdown = (
        "```python\nfrom nodestep import Graph\n`Graph`\n```\n"
        "    ~~~text\n    `Graph`\n    ~~~\n"
        "````markdown\n```\n`Graph`\n```\n````\n"
        "After `Graph`.\n"
    )
    assert link(hook, markdown) == markdown.replace(
        "After `Graph`.", "After [`Graph`][nodestep.core.graph.Graph]."
    )


@pytest.mark.parametrize("page", ["reference/core/index.md", "changelog.md"])
def test_reference_pages_and_the_changelog_are_left_alone(
    hook: ModuleType, page: str
) -> None:
    assert link(hook, "A `Graph`.", page) == "A `Graph`."


@pytest.mark.parametrize("page", ["nodeartifact/tracing.md"])
def test_the_sibling_sections_are_left_alone(hook: ModuleType, page: str) -> None:
    assert link(hook, "A `tool` event or `ScriptedChat`.", page) == (
        "A `tool` event or `ScriptedChat`."
    )


def prose_pages(site: Path) -> list[Path]:
    return [
        page
        for page in sorted(site.rglob("index.html"))
        if page.parent.relative_to(site).parts[:1]
        not in {("reference",), ("changelog",)}
    ]


def test_prose_links_openai_chat_to_its_reference_entry(site: Path) -> None:
    anchor = re.compile(
        r'<a [^>]*href="(?:\.\./)*reference/chat/integrations/openai/#'
        + re.escape(OPENAI_CHAT)
        + r'"[^>]*><code>OpenAIChat</code></a>'
    )
    linked = [
        page
        for page in prose_pages(site)
        if anchor.search(page.read_text(encoding="utf-8"))
    ]
    assert linked


def test_code_blocks_render_without_links(site: Path) -> None:
    blocks = [
        block
        for page in prose_pages(site)
        for block in re.findall(
            r'<div class="(?:language-\w+ )?highlight">.*?</div>',
            page.read_text(encoding="utf-8"),
            re.S,
        )
    ]
    assert blocks
    assert not [block for block in blocks if "<a " in block]


def test_the_changelog_keeps_plain_inline_code(site: Path) -> None:
    html = (site / "changelog" / "index.html").read_text(encoding="utf-8")
    article = html.split("<article", 1)[1].split("</article>", 1)[0]
    assert "<code>Send</code>" in article
    assert "reference/" not in article


def test_prose_links_graph_to_its_core_page(site: Path) -> None:
    html = (site / "concepts" / "graphs" / "index.html").read_text(encoding="utf-8")
    assert re.search(
        r'<a [^>]*href="\.\./\.\./reference/core/graphs/#nodestep\.core\.Graph"'
        r"[^>]*><code>Graph</code></a>",
        html,
    )


def test_prose_links_a_split_core_name_to_the_page_of_its_purpose(
    site: Path,
) -> None:
    html = (site / "concepts" / "sub-agents" / "index.html").read_text(encoding="utf-8")
    assert re.search(
        r'href="\.\./\.\./reference/core/sub-agents/#nodestep\.core\.spawn_agents"',
        html,
    )


def test_prose_links_a_state_store_to_its_module_page(site: Path) -> None:
    html = (site / "concepts" / "persistence" / "index.html").read_text(
        encoding="utf-8"
    )
    assert re.search(
        r'href="\.\./\.\./reference/state/integrations/filesystem/'
        r'#nodestep\.state\.integrations\.filesystem\.FilesystemStateStore"',
        html,
    )


def test_every_generated_link_resolves(site: Path) -> None:
    unresolved = [
        str(page.relative_to(site))
        for page in sorted(site.rglob("*.html"))
        if "][nodestep." in page.read_text(encoding="utf-8")
    ]
    assert not unresolved
