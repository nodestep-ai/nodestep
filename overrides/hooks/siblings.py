from pathlib import Path
from typing import Self

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import File, Files
from mkdocs.structure.nav import Navigation
from mkdocs.structure.pages import Page
from mkdocs.utils.templates import TemplateContext

SIBLINGS = ("nodeartifact",)


class SiblingDocs:
    def __init__(self, root: Path, names: tuple[str, ...]) -> None:
        self.root = root
        self.names = names

    @classmethod
    def for_config(cls, config: MkDocsConfig) -> Self:
        return cls(Path(config.config_file_path).parent, SIBLINGS)

    def folder(self, name: str) -> Path:
        return self.root.parent / name / "docs"

    def present(self) -> list[str]:
        return [name for name in self.names if self.folder(name).is_dir()]

    def configure(self, config: MkDocsConfig) -> MkDocsConfig:
        present = self.present()
        absent = set(self.names) - set(present)
        config.nav = [
            entry
            for entry in config.nav or []
            if not (isinstance(entry, dict) and entry.keys() & absent)
        ]
        for name in present:
            folder = str(self.folder(name))
            if folder not in config.watch:
                config.watch.append(folder)
        return config

    def repository(self, url: str, default: str) -> str:
        section = url.split("/", 1)[0]
        if section not in self.names:
            return default
        owner = default.split("/", 1)[0]
        return f"{owner}/{section}"

    def add_files(self, files: Files, config: MkDocsConfig) -> Files:
        for name in self.present():
            folder = self.folder(name)
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    uri = f"{name}/{path.relative_to(folder).as_posix()}"
                    files.append(File.generated(config, uri, abs_src_path=str(path)))
        return files


def on_config(config: MkDocsConfig) -> MkDocsConfig:
    return SiblingDocs.for_config(config).configure(config)


def on_files(files: Files, config: MkDocsConfig) -> Files:
    return SiblingDocs.for_config(config).add_files(files, config)


def on_page_context(
    context: TemplateContext, page: Page, config: MkDocsConfig, nav: Navigation
) -> TemplateContext:
    if config.repo_name:
        siblings = SiblingDocs.for_config(config)
        page.meta["repository"] = siblings.repository(page.url, config.repo_name)
    return context
