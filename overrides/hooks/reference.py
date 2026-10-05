import re
from pathlib import Path

from mkdocs.structure import StructureItem
from mkdocs.structure.nav import Section
from mkdocs.structure.pages import Page
from mkdocs.structure.toc import AnchorLink
from mkdocs.utils import get_relative_url
from pydantic import BaseModel

GROUP = re.compile(r'<h2 id="(?P<anchor>[^"]+)">(?P<title>.*?)<a class="headerlink"')
ENTRY = re.compile(r'data-ref-id="(?P<anchor>[^"]+)" data-ref-name="(?P<name>[^"]+)"')
SUMMARY = re.compile(r'<p class="ref-summary">(?P<text>.*?)</p>', re.DOTALL)
DIRECTIVE = re.compile(
    r"^::: (?P<identifier>[\w.]+)\n(?P<options>(?:[ \t]+\S.*\n?)*)", re.MULTILINE
)
NO_MEMBERS = re.compile(r"^ +members: false$", re.MULTILINE)


class Row(BaseModel):
    name: str
    anchor: str
    summary: str


class Group(BaseModel):
    title: str
    anchor: str
    rows: list[Row] = []


class PageRow(BaseModel):
    title: str
    url: str
    identifiers: list[str]


class Overview:
    def __init__(self, html: str) -> None:
        self.html = html

    def groups(self) -> list[Group]:
        marks = sorted(
            [*GROUP.finditer(self.html), *ENTRY.finditer(self.html)],
            key=lambda mark: mark.start(),
        )
        groups: list[Group] = []
        for index, mark in enumerate(marks):
            if "title" in mark.groupdict():
                groups.append(Group(title=mark["title"], anchor=mark["anchor"]))
            elif groups:
                end = (
                    marks[index + 1].start()
                    if index + 1 < len(marks)
                    else len(self.html)
                )
                summary = SUMMARY.search(self.html, mark.end(), end)
                groups[-1].rows.append(
                    Row(
                        name=mark["name"],
                        anchor=mark["anchor"],
                        summary=summary["text"].strip() if summary else "",
                    )
                )
        return [group for group in groups if group.rows]

    def table(self, groups: list[Group]) -> str:
        bodies = "".join(
            '<tbody><tr class="ref-overview-group">'
            f'<th colspan="2" scope="rowgroup"><a href="#{group.anchor}">'
            f"{group.title}</a></th></tr>"
            + "".join(
                f'<tr><td><a href="#{row.anchor}"><code>{row.name}</code></a></td>'
                f"<td>{row.summary}</td></tr>"
                for row in group.rows
            )
            + "</tbody>"
            for group in groups
        )
        return (
            '<table><thead><tr><th scope="col">Name</th>'
            f'<th scope="col">What it is</th></tr></thead>{bodies}</table>'
        )


class PageIndex:
    def __init__(self, page: Page) -> None:
        self.page = page

    def pages(self, items: list[StructureItem]) -> list[Page]:
        found: list[Page] = []
        for item in items:
            if isinstance(item, Section):
                found.extend(self.pages(item.children))
            elif isinstance(item, Page) and item is not self.page:
                found.append(item)
        return found

    def rows(self) -> list[PageRow]:
        if not (
            self.page.file.src_uri.endswith("/index.md")
            and isinstance(self.page.parent, Section)
            and NO_MEMBERS.search(self.page.markdown or "")
        ):
            return []
        rows = []
        for page in self.pages(self.page.parent.children):
            source = Path(page.file.abs_src_path or "").read_text(encoding="utf-8")
            rows.append(
                PageRow(
                    title=str(page.title or ""),
                    url=get_relative_url(page.url, self.page.url),
                    identifiers=[
                        directive["identifier"]
                        for directive in DIRECTIVE.finditer(source)
                        if not NO_MEMBERS.search(directive["options"])
                    ],
                )
            )
        return rows

    def table(self, rows: list[PageRow]) -> str:
        cells = "".join(
            f'<tr><td><a href="{row.url}">{row.title}</a></td><td>'
            + ", ".join(
                f'<autoref identifier="{identifier}"><code>'
                f"{identifier.rpartition('.')[2]}</code></autoref>"
                for identifier in row.identifiers
            )
            + "</td></tr>"
            for row in rows
        )
        return (
            '<table><thead><tr><th scope="col">Page</th>'
            f'<th scope="col">Names</th></tr></thead><tbody>{cells}</tbody></table>'
        )


class ReferencePage:
    def __init__(self, page: Page) -> None:
        self.page = page

    def applies(self) -> bool:
        return self.page.file.src_uri.startswith("reference/")

    def with_overview(self, html: str) -> str:
        overview = Overview(html)
        index = PageIndex(self.page)
        groups = overview.groups()
        rows = index.rows()
        if not groups and not rows:
            return html
        tables = (overview.table(groups) if groups else "") + (
            index.table(rows) if rows else ""
        )
        block = (
            f'<nav class="ref-overview" aria-label="Names on this page">{tables}</nav>'
        )
        first_group = GROUP.search(html)
        if first_group is None:
            return html + block
        return html[: first_group.start()] + block + html[first_group.start() :]


class Contents:
    def __init__(self, items: list[AnchorLink]) -> None:
        self.items = items

    def without_members(self) -> None:
        for item in self.items:
            if item.level > 1 and "." in item.id:
                item.children = []
            else:
                Contents(item.children).without_members()


def on_page_content(html: str, page: Page, **_: object) -> str:
    reference = ReferencePage(page)
    if not reference.applies():
        return html
    Contents(page.toc.items).without_members()
    return reference.with_overview(html)
