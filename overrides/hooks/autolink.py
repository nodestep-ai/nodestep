import inspect
import re
from typing import Self

from mkdocs.structure.pages import Page

import nodestep

DOCUMENTED_AT = {
    "START": "nodestep.core.START",
    "END": "nodestep.core.END",
    "OpenAIChat": "nodestep.chat.integrations.openai.OpenAIChat",
}
UNLINKED_PAGES = re.compile(r"reference/|changelog\.md$|nodeartifact/")
FENCE = re.compile(r"\s*(`{3,}|~{3,})")
UNLINKED_LINE = re.compile(r"\s*(?:#|!!!|\?\?\?|===)")
BRACKETS = re.compile(r"!?\[(?:[^\[\]]|\[[^\[\]]*\])*\]")
CODE_SPAN = re.compile(r"(`+)(.+?)(?<!`)\1(?!`)")


class ApiLinks:
    def __init__(self, identifiers: dict[str, str]) -> None:
        self.identifiers = identifiers

    @classmethod
    def for_package(cls) -> Self:
        return cls(
            {
                name: cls.defined_at(getattr(nodestep, name))
                for name in nodestep.__all__
                if name not in DOCUMENTED_AT
            }
            | DOCUMENTED_AT
        )

    @staticmethod
    def defined_at(value: object) -> str:
        if not (inspect.isclass(value) or inspect.isfunction(value)):
            raise TypeError(f"add {value!r} to DOCUMENTED_AT")
        return f"{value.__module__}.{value.__qualname__}"

    def page(self, markdown: str, src_uri: str) -> str:
        if UNLINKED_PAGES.match(src_uri):
            return markdown
        lines = []
        fence = ""
        for line in markdown.splitlines(keepends=True):
            if fence:
                stripped = line.strip()
                if set(stripped) == {fence[0]} and len(stripped) >= len(fence):
                    fence = ""
                lines.append(line)
            elif opening := FENCE.match(line):
                fence = opening.group(1)
                lines.append(line)
            elif UNLINKED_LINE.match(line):
                lines.append(line)
            else:
                lines.append(self.line(line))
        return "".join(lines)

    def line(self, line: str) -> str:
        bracketed = [match.span() for match in BRACKETS.finditer(line)]

        def link(span: re.Match[str]) -> str:
            name = span.group(2)
            inside = any(start <= span.start() < end for start, end in bracketed)
            if span.group(1) != "`" or name not in self.identifiers or inside:
                return span.group(0)
            return f"[`{name}`][{self.identifiers[name]}]"

        return CODE_SPAN.sub(link, line)


LINKS = ApiLinks.for_package()


def on_page_markdown(markdown: str, page: Page, **_: object) -> str:
    return LINKS.page(markdown, page.file.src_uri)
