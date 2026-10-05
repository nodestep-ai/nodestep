from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "overrides" / "hooks" / "siblings.py"
SIBLINGS = ("nodeartifact",)
SETTINGS = f"""site_name: test
hooks:
  - {HOOK}
validation:
  omitted_files: warn
  unrecognized_links: warn
  anchors: warn
nav:
  - nodestep:
      - Home: index.md
  - nodeartifact:
      - Overview: nodeartifact/index.md
      - Guide: nodeartifact/guide.md
"""


@pytest.fixture(scope="module")
def hook() -> ModuleType:
    pytest.importorskip("mkdocs")
    spec = importlib.util.spec_from_file_location("siblings", HOOK)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def checkout(folder: Path, *siblings: str) -> Path:
    project = folder / "nodestep"
    write(project / "mkdocs.yml", SETTINGS)
    write(project / "docs" / "index.md", "# nodestep\n")
    if "nodeartifact" in siblings:
        docs = folder / "nodeartifact" / "docs"
        write(docs / "index.md", "# nodeartifact\n\nSee the [guide](guide.md#usage).\n")
        write(docs / "guide.md", "# Guide\n\n## Usage\n\n![Mark](assets/mark.svg)\n")
        write(docs / "assets" / "mark.svg", '<svg xmlns="http://www.w3.org/2000/svg"/>')
    return project / "mkdocs.yml"


def load(settings: Path) -> MkDocsConfig:
    config = pytest.importorskip("mkdocs.config")
    return config.load_config(str(settings))


def configured(config: MkDocsConfig) -> MkDocsConfig:
    return config.plugins.on_config(config)


def sections(nav: list[dict[str, object]]) -> list[str]:
    return [next(iter(entry)) for entry in nav]


def build(settings: Path, site: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(site)],
        cwd=settings.parent,
        capture_output=True,
        encoding="utf-8",
        check=False,
    )


def test_mkdocs_runs_the_sibling_hook() -> None:
    config = pytest.importorskip("mkdocs.config").load_config(str(ROOT / "mkdocs.yml"))
    assert "overrides/hooks/siblings.py" in config["plugins"]


def test_the_only_sibling_is_nodeartifact(hook: ModuleType) -> None:
    assert hook.SIBLINGS == SIBLINGS


def test_without_the_siblings_their_sections_are_left_out(tmp_path: Path) -> None:
    config = configured(load(checkout(tmp_path)))
    assert sections(config["nav"]) == ["nodestep"]
    assert not [path for path in config["watch"] if "docs" in Path(path).parts]


def test_with_the_siblings_every_section_stays_and_their_docs_are_watched(
    tmp_path: Path,
) -> None:
    config = configured(load(checkout(tmp_path, *SIBLINGS)))
    assert sections(config["nav"]) == ["nodestep", "nodeartifact"]
    for name in SIBLINGS:
        assert config["watch"].count(str(tmp_path / name / "docs")) == 1
    configured(config)
    for name in SIBLINGS:
        assert config["watch"].count(str(tmp_path / name / "docs")) == 1


def test_sibling_files_are_added_under_the_sibling_name(tmp_path: Path) -> None:
    files_module = pytest.importorskip("mkdocs.structure.files")
    config = configured(load(checkout(tmp_path, *SIBLINGS)))
    files = config.plugins.on_files(files_module.Files([]), config=config)
    added = {file.src_uri: file for file in files}
    assert sorted(added) == [
        "nodeartifact/assets/mark.svg",
        "nodeartifact/guide.md",
        "nodeartifact/index.md",
    ]
    guide = added["nodeartifact/guide.md"]
    assert guide.abs_src_path == str(tmp_path / "nodeartifact" / "docs" / "guide.md")
    assert guide.dest_uri == "nodeartifact/guide/index.html"
    assert guide.is_documentation_page()
    assert added["nodeartifact/assets/mark.svg"].is_media_file()


def test_without_the_siblings_no_files_are_added(tmp_path: Path) -> None:
    files_module = pytest.importorskip("mkdocs.structure.files")
    config = configured(load(checkout(tmp_path)))
    assert list(config.plugins.on_files(files_module.Files([]), config=config)) == []


def test_a_strict_build_without_the_siblings_passes(tmp_path: Path) -> None:
    pytest.importorskip("mkdocs")
    site = tmp_path / "site"
    result = build(checkout(tmp_path), site)
    assert result.returncode == 0, result.stderr
    assert (site / "index.html").is_file()
    assert not (site / "nodeartifact").exists()


def test_a_strict_build_with_the_siblings_has_their_pages(tmp_path: Path) -> None:
    pytest.importorskip("mkdocs")
    site = tmp_path / "site"
    result = build(checkout(tmp_path, *SIBLINGS), site)
    assert result.returncode == 0, result.stderr
    guide = (site / "nodeartifact" / "guide" / "index.html").read_text(encoding="utf-8")
    assert 'src="../assets/mark.svg"' in guide
    index = (site / "nodeartifact" / "index.html").read_text(encoding="utf-8")
    assert 'href="guide/#usage"' in index
    assert (site / "nodeartifact" / "assets" / "mark.svg").is_file()
    assert "not included in the" not in result.stderr


def test_the_site_excludes_no_sibling_pages() -> None:
    config = pytest.importorskip("mkdocs.config").load_config(str(ROOT / "mkdocs.yml"))
    assert config["exclude_docs"] is None


def test_the_site_has_the_sibling_pages_that_are_checked_out(site: Path) -> None:
    for name in SIBLINGS:
        checked_out = (ROOT.parent / name / "docs").is_dir()
        assert (site / name / "index.html").is_file() == checked_out, name
