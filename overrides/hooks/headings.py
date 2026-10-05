import re

from mkdocs.structure.pages import Page

OBJECT_NAME = re.compile(r'(<span class="doc doc-object-name [\w-]+">)([^<]+)(</span>)')


def break_after_dots(html: str) -> str:
    return OBJECT_NAME.sub(
        lambda match: (
            match.group(1) + match.group(2).replace(".", ".<wbr>") + match.group(3)
        ),
        html,
    )


def on_page_content(html: str, page: Page, **_: object) -> str:
    return break_after_dots(html)
