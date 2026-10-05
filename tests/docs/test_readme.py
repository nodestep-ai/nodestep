import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
README_LOGO = ROOT / "docs" / "assets" / "nodestep.svg"
README_DARK_LOGO = ROOT / "docs" / "assets" / "nodestep-dark.svg"
README_HEADER = (
    "<picture>\n"
    '  <source media="(prefers-color-scheme: dark)"'
    ' srcset="docs/assets/nodestep-dark.svg">\n'
    '  <img src="docs/assets/nodestep.svg" alt="" width="56">\n'
    "</picture>\n\n# nodestep\n"
)
HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}\b")
SRC_DIR = ROOT / "src"
DOCS_DIR = ROOT / "docs"
DOCS_SITE = "https://nodestep-ai.github.io/nodestep/"
DOCS_SITE_LINK = re.compile(r"https://nodestep-ai\.github\.io/nodestep/[^\s)\"'<>]*")
DOCS_SITE_PAGE = re.compile(
    r"https://nodestep-ai\.github\.io/nodestep/(?:(?P<path>[\w-]+(?:/[\w-]+)*)/)?"
    r"(?:#[\w-]+)?"
)
SIBLINGS = ("nodeartifact",)
QUICK_START_LINES = range(25, 36)
TIMEOUT_SECONDS = 60
STATUS_NOTE = (
    "> [!WARNING]\n"
    "> Alpha (0.1.0a1). Anything may change between releases without a "
    "deprecation period, so pin a tag or a commit."
)


class ReadmeBlock(BaseModel):
    model_config = ConfigDict(frozen=True)

    line: int
    code: str


class ReadmeParser:
    PYTHON_BLOCK = re.compile(
        r"^```python[ \t]*\n(?P<code>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL
    )

    def __init__(self, path: Path) -> None:
        self.path = path

    def python_blocks(self) -> list[ReadmeBlock]:
        text = self.path.read_text(encoding="utf-8")
        return [
            ReadmeBlock(line=text.count("\n", 0, match.start()) + 1, code=match["code"])
            for match in self.PYTHON_BLOCK.finditer(text)
        ]


BLOCKS = ReadmeParser(README).python_blocks()


def keyless_environment() -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("OPENAI_")
    }
    environment["PYTHONPATH"] = str(SRC_DIR)
    return environment


def test_readme_has_python_blocks() -> None:
    assert BLOCKS


@pytest.mark.parametrize(
    "block", BLOCKS, ids=[f"README.md:{block.line}" for block in BLOCKS]
)
def test_readme_block_runs(block: ReadmeBlock, tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", block.code],
        cwd=tmp_path,
        env=keyless_environment(),
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        check=False,
        encoding="utf-8",
    )
    assert completed.returncode == 0, (
        f"README.md:{block.line} exited with {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def test_readme_starts_with_the_logo_and_the_title() -> None:
    assert README.read_text(encoding="utf-8").startswith(README_HEADER)


@pytest.mark.parametrize(
    ("logo", "ink"), [(README_LOGO, "#1f1b17"), (README_DARK_LOGO, "#ede9e4")]
)
def test_each_readme_logo_has_one_ink_and_the_amber_accent(
    logo: Path, ink: str
) -> None:
    text = logo.read_text(encoding="utf-8")
    assert 'd="M14 50V14L50 50V14"' in text
    assert set(HEX_COLOR.findall(text)) == {ink, "#f5a524"}
    assert "prefers-color-scheme" not in text


def test_readme_starts_with_the_development_warning() -> None:
    text = README.read_text(encoding="utf-8")
    assert text.split("\n\n")[2] == STATUS_NOTE


def readme_sections() -> dict[str, str]:
    parts = README.read_text(encoding="utf-8").split("\n## ")[1:]
    return {
        part.split("\n", 1)[0]: part.split("\n", 1)[1] if "\n" in part else ""
        for part in parts
    }


def page_exists(path: str) -> bool | None:
    section, _, rest = path.partition("/")
    if section in SIBLINGS:
        folder = ROOT.parent / section / "docs"
        if not folder.is_dir():
            return None
        return (folder / f"{rest or 'index'}.md").is_file()
    return (DOCS_DIR / f"{path or 'index'}.md").is_file()


def test_readme_sections_follow_the_shared_order() -> None:
    headings = list(readme_sections())
    assert headings[:2] == ["Install", "Quick start"]
    assert headings[-3:] == ["Documentation", "Development", "License"]


def test_the_quick_start_is_one_small_example_that_links_the_docs() -> None:
    quick_start = readme_sections()["Quick start"]
    blocks = ReadmeParser.PYTHON_BLOCK.findall(quick_start)
    assert len(blocks) == 1
    assert len(BLOCKS) == 1
    assert len(blocks[0].strip().splitlines()) in QUICK_START_LINES
    assert f"{DOCS_SITE}quickstart/" in quick_start


def test_readme_documentation_links_the_docs_site() -> None:
    documentation = readme_sections()["Documentation"]
    assert DOCS_SITE in documentation
    assert f"({DOCS_SITE}nodeartifact/)" in documentation
    assert "(https://github.com/nodestep-ai/text-to-sql-demo)" in documentation


def test_readme_has_no_feature_list_or_related_projects() -> None:
    assert not {"Approach", "Features", "Related projects"} & set(readme_sections())


def test_readme_links_existing_docs_pages() -> None:
    links = DOCS_SITE_LINK.findall(README.read_text(encoding="utf-8"))
    pages = {link: DOCS_SITE_PAGE.fullmatch(link) for link in links}
    missing = [
        link
        for link, page in pages.items()
        if page is None or page_exists(page["path"] or "") is False
    ]
    assert links
    assert not missing, f"README.md links missing docs pages: {', '.join(missing)}"
