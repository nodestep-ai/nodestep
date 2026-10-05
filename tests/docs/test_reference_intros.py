import importlib
import importlib.util
import posixpath
import re
from pathlib import Path

import pytest

import nodestep

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"
REFERENCE_DIR = DOCS_DIR / "reference"
DIRECTIVE = re.compile(r"^::: ([\w.]+)$", re.MULTILINE)
DOC_LINK = re.compile(r"\]\((?P<page>(?:\.\./)+[\w/-]+\.md)(?:#[\w-]+)?\)")
GUIDE_PAGE = re.compile(r"(?:concepts/|guides/)?[\w-]+\.md")
TITLE = re.compile(r"^# (?P<title>.+)\n\n(?P<intro>[^\n]+)\n", re.MULTILINE)


def is_module(identifier: str) -> bool:
    try:
        return importlib.util.find_spec(identifier) is not None
    except ModuleNotFoundError:
        return False


DIRECTIVES = {
    source.relative_to(DOCS_DIR).as_posix(): match
    for source in sorted(REFERENCE_DIR.rglob("*.md"))
    if (match := DIRECTIVE.findall(source.read_text(encoding="utf-8")))
}
REFERENCE_PAGES = {
    match[0]: page for page, match in DIRECTIVES.items() if is_module(match[0])
}
PURPOSE_PAGES = [page for page, match in DIRECTIVES.items() if not is_module(match[0])]


def linked_pages(package: str) -> list[str]:
    docstring = importlib.import_module(package).__doc__ or ""
    folder = posixpath.dirname(REFERENCE_PAGES[package])
    return [
        posixpath.normpath(posixpath.join(folder, link["page"]))
        for link in DOC_LINK.finditer(docstring)
    ]


def test_every_package_page_is_checked() -> None:
    assert {
        "nodestep.core",
        "nodestep.core.builtin_nodes",
        "nodestep.state",
        "nodestep.state.integrations.filesystem",
        "nodestep.chat.integrations.openai",
        "nodestep.middleware.interrupt",
        "nodestep.workspace.integrations.virtual",
        "nodestep.exceptions",
    } <= set(REFERENCE_PAGES)
    assert "nodestep" not in REFERENCE_PAGES


@pytest.mark.parametrize("package", REFERENCE_PAGES)
def test_each_reference_package_introduces_itself(package: str) -> None:
    docstring = importlib.import_module(package).__doc__ or ""
    summary, _, body = docstring.strip().partition("\n\n")
    assert summary.endswith(".")
    assert "\n" not in summary
    assert any(GUIDE_PAGE.fullmatch(page) for page in linked_pages(package)), (
        f"{package} links no guide or concept page"
    )
    assert body


@pytest.mark.parametrize("package", REFERENCE_PAGES)
def test_each_introduction_links_existing_pages(package: str) -> None:
    missing = [
        page for page in linked_pages(package) if not (DOCS_DIR / page).is_file()
    ]
    assert not missing, f"{package} links missing pages: {', '.join(missing)}"


@pytest.mark.parametrize(("package", "page"), REFERENCE_PAGES.items())
def test_the_reference_page_shows_the_introduction(
    site: Path, package: str, page: str
) -> None:
    summary = (importlib.import_module(package).__doc__ or "").strip().split("\n")[0]
    url = page.removesuffix(".md").removesuffix("/index")
    html = (site / url / "index.html").read_text(encoding="utf-8")
    text = " ".join(re.sub(r"<[^>]+>", "", html).split())
    depth = url.count("/") + 1
    assert summary
    assert summary.replace("`", "") in text
    assert re.search(rf'href="(?:\.\./){{{depth}}}(?:concepts/|guides/|[\w-]+/")', html)


def test_the_root_package_docstring_links_no_docs_page() -> None:
    assert "](" not in (nodestep.__doc__ or "")


def test_the_reducers_are_introduced_on_the_page_that_documents_them() -> None:
    core = importlib.import_module("nodestep.core").__doc__ or ""
    state = importlib.import_module("nodestep.state").__doc__ or ""
    assert "reducers" in core.strip().split("\n")[0]
    assert "reducers" not in state.strip().split("\n")[0]
    assert "`nodestep.core`" in state


def test_the_core_pages_split_by_purpose_are_checked() -> None:
    assert {
        "reference/core/graphs.md",
        "reference/core/flow.md",
        "reference/core/nodes.md",
        "reference/core/running.md",
        "reference/core/reducers.md",
        "reference/core/sub-agents.md",
    } <= set(PURPOSE_PAGES)


@pytest.mark.parametrize("page", PURPOSE_PAGES)
def test_each_purpose_page_opens_with_its_title_and_one_sentence(page: str) -> None:
    text = (DOCS_DIR / page).read_text(encoding="utf-8")
    opening = TITLE.match(text)
    assert opening, f"{page} must start with '# Title' and an intro paragraph"
    sentence, _, links = opening["intro"].partition(" See ")
    assert sentence.endswith(".")
    assert not re.search(r"\.\s+[A-Z]", sentence), f"{page}: one sentence"
    folder = posixpath.dirname(page)
    linked = [
        posixpath.normpath(posixpath.join(folder, link["page"]))
        for link in DOC_LINK.finditer(links)
    ]
    assert any(GUIDE_PAGE.fullmatch(linked_page) for linked_page in linked), page
    assert all((DOCS_DIR / linked_page).is_file() for linked_page in linked), page


@pytest.mark.parametrize("page", PURPOSE_PAGES)
def test_the_purpose_page_shows_its_introduction(site: Path, page: str) -> None:
    opening = TITLE.match((DOCS_DIR / page).read_text(encoding="utf-8"))
    assert opening
    url = page.removesuffix(".md")
    html = (site / url / "index.html").read_text(encoding="utf-8")
    text = " ".join(re.sub(r"<[^>]+>", "", html).split())
    assert opening["intro"].partition(" See ")[0] in text
