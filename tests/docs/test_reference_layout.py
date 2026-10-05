import html
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"
REFERENCE_DIR = DOCS_DIR / "reference"
DIRECTIVE = re.compile(r"^::: ([\w.]+)$", re.MULTILINE)
DIRECTIVE_BLOCK = re.compile(
    r"^::: (?P<identifier>[\w.]+)\n(?P<options>(?:[ \t]+\S.*\n?)*)", re.MULTILINE
)
GROUP_HEADING = re.compile(r"^## (.+)$", re.MULTILINE)
TAG = re.compile(r"<[^>]+>")
WORD = re.compile(r"[^\W\d_][\w'-]*")
WORD_LIMIT = 2500
TABLE_ROW = re.compile(
    r'<td class="ref-table-name"><code>(?P<name>[^<]*)</code></td>\s*'
    r'<td class="ref-table-type"(?P<span> colspan="2")?>(?P<type>.*?)</td>\s*'
    r'(?:<td class="ref-table-text">(?P<text>.*?)</td>)?',
    re.DOTALL,
)
ENTRY_PAGES = sorted(
    source.relative_to(DOCS_DIR).as_posix()
    for source in REFERENCE_DIR.rglob("*.md")
    if DIRECTIVE.search(source.read_text(encoding="utf-8"))
)
GROUPED = sorted(
    source.relative_to(DOCS_DIR).as_posix()
    for source in REFERENCE_DIR.rglob("*.md")
    if GROUP_HEADING.search(text := source.read_text(encoding="utf-8"))
    and DIRECTIVE.search(text)
)
CORE_PAGES = [
    "reference/core/graphs.md",
    "reference/core/flow.md",
    "reference/core/nodes.md",
    "reference/core/running.md",
    "reference/core/reducers.md",
    "reference/core/sub-agents.md",
]


def url(page: str) -> str:
    path = page.removesuffix(".md")
    return path.removesuffix("/index") if path.endswith("/index") else path


def built(site: Path, page: str) -> str:
    return (site / url(page) / "index.html").read_text(encoding="utf-8")


def article(site: Path, page: str) -> str:
    text = built(site, page)
    return text[text.index("<article") : text.index("</article>")]


def plain(fragment: str) -> str:
    return html.unescape(" ".join(TAG.sub("", fragment).split()))


def words(fragment: str) -> list[str]:
    return WORD.findall(html.unescape(TAG.sub(" ", fragment)))


def declared_groups(page: str) -> list[tuple[str, list[str]]]:
    text = (DOCS_DIR / page).read_text(encoding="utf-8")
    parts = GROUP_HEADING.split(text)[1:]
    return [
        (title, DIRECTIVE.findall(body))
        for title, body in zip(parts[::2], parts[1::2], strict=True)
    ]


def overview(site: Path, page: str) -> str:
    found = re.search(
        r'<nav class="ref-overview"[^>]*>(.*?)</nav>', article(site, page)
    )
    assert found, f"{page} has no overview"
    return found[1]


def entry_ids(page: str) -> list[str]:
    return [
        block["identifier"]
        for block in DIRECTIVE_BLOCK.finditer(
            (DOCS_DIR / page).read_text(encoding="utf-8")
        )
        if "members: false" not in block["options"]
    ]


SINGLE_ENTRY_PAGES = [
    "reference/state/integrations/filesystem.md",
    "reference/state/integrations/inmemory.md",
    "reference/middleware/tool_limit.md",
    "reference/workspace/integrations/local.md",
    "reference/workspace/integrations/virtual.md",
]


WITH_ENTRIES = [*GROUPED, *SINGLE_ENTRY_PAGES]


def test_a_page_with_several_entries_groups_them_and_one_with_one_does_not() -> None:
    for page in ENTRY_PAGES:
        if page.endswith("/index.md") or page in SINGLE_ENTRY_PAGES:
            continue
        assert len(entry_ids(page)) > 1, page
        assert page in GROUPED, page
    for page in SINGLE_ENTRY_PAGES:
        assert len(entry_ids(page)) == 1, page
        assert page not in GROUPED, page


@pytest.mark.parametrize("page", SINGLE_ENTRY_PAGES)
def test_a_page_with_one_entry_shows_it_right_after_the_intro(
    site: Path, page: str
) -> None:
    text = article(site, page)
    assert '<nav class="ref-overview"' not in text
    assert not re.search(r'<h2 id="[^"]+">', text)
    (identifier,) = entry_ids(page)
    heading = re.search(
        r'<h(\d) id="([^"]+)" class="doc doc-heading ref-entry-heading"', text
    )
    assert heading
    assert heading.groups() == ("2", identifier)
    assert text.index("<p", text.index("</h1>")) < heading.start()
    methods = re.findall(r'<h3 id="([^"]+)"', text)
    folded = re.findall(
        r'<details class="ref-method">\s*<summary><h3 id="([^"]+)"', text
    )
    assert methods == folded
    sidebar = built(site, page)
    sidebar = sidebar[
        sidebar.index('class="md-sidebar md-sidebar--secondary"') : sidebar.index(
            '<div class="md-content"'
        )
    ]
    assert re.findall(r'<a href="#([^"]+)" class="md-nav__link">', sidebar) == [
        identifier
    ]


@pytest.mark.parametrize("page", GROUPED)
def test_a_group_heading_names_a_purpose_not_its_page_or_an_entry(page: str) -> None:
    text = (DOCS_DIR / page).read_text(encoding="utf-8")
    title = re.match(r"# (.+)", text)
    names = {identifier.rpartition(".")[2].lower() for identifier in entry_ids(page)}
    for group, _ in declared_groups(page):
        assert group.lower() not in names, f"{page}: {group}"
        assert not title or group.lower() != title[1].lower(), f"{page}: {group}"


def test_the_core_pages_and_the_built_in_nodes_declare_groups() -> None:
    assert {*CORE_PAGES, "reference/core/builtin_nodes.md"} <= set(GROUPED)


@pytest.mark.parametrize("page", GROUPED)
def test_a_grouped_page_shows_title_intro_overview_then_groups(
    site: Path, page: str
) -> None:
    text = article(site, page)
    title = text.index("<h1")
    intro = text.index("<p", title)
    nav = text.index('<nav class="ref-overview"')
    first_group = re.search(r'<h2 id="[^"]+">', text)
    assert first_group
    assert title < intro < nav < first_group.start()
    assert "doc-symbol" not in text[title : text.index("</h1>")]


@pytest.mark.parametrize("page", GROUPED)
def test_the_overview_lists_each_entry_once_under_its_group(
    site: Path, page: str
) -> None:
    table = overview(site, page)
    assert '<th scope="col">Name</th><th scope="col">What it is</th>' in table
    groups = re.findall(
        r'<tr class="ref-overview-group"><th colspan="2" scope="rowgroup">'
        r'<a href="#[^"]+">(.*?)</a></th></tr>(.*?)</tbody>',
        table,
    )
    listed = [
        (plain(title), re.findall(r'<tr><td><a href="#([^"]+)">', rows))
        for title, rows in groups
    ]
    assert listed == declared_groups(page)
    summaries = re.findall(r"<tr><td><a [^>]+>.*?</a></td><td>(.*?)</td></tr>", table)
    assert summaries
    assert all(plain(summary) for summary in summaries)


@pytest.mark.parametrize("page", GROUPED)
def test_the_table_of_contents_lists_only_groups_and_entries(
    site: Path, page: str
) -> None:
    text = built(site, page)
    sidebar = text[
        text.index('class="md-sidebar md-sidebar--secondary"') : text.index(
            '<div class="md-content"'
        )
    ]
    links = re.findall(r'<a href="#([^"]+)" class="md-nav__link">', sidebar)
    group_ids = re.findall(r'<h2 id="([^"]+)">', article(site, page))
    expected = []
    for group_id, (_, identifiers) in zip(
        group_ids, declared_groups(page), strict=True
    ):
        expected += [group_id, *identifiers]
    assert links == expected


@pytest.mark.parametrize("page", GROUPED)
def test_methods_are_closed_folds_inside_their_class_entry(
    site: Path, page: str
) -> None:
    text = article(site, page)
    method_ids = re.findall(r'<h4 id="([^"]+)"', text)
    folded = re.findall(
        r'<details class="ref-method">\s*<summary><h4 id="([^"]+)"', text
    )
    assert method_ids == folded
    assert '<details class="ref-method" open' not in text
    entries = entry_ids(page)
    for method_id in method_ids:
        owner, _, name = method_id.rpartition(".")
        assert owner in entries, method_id
        assert not name.startswith("_"), method_id


@pytest.mark.parametrize("page", WITH_ENTRIES)
def test_entries_leave_out_dunders_and_model_settings(site: Path, page: str) -> None:
    text = article(site, page)
    assert not re.search(r'id="[^"]*\.__\w+__"', text)
    assert "<code>model_config</code>" not in text


@pytest.mark.parametrize("page", WITH_ENTRIES)
def test_signatures_are_inline_lines_not_code_blocks(site: Path, page: str) -> None:
    text = article(site, page)
    assert "doc-signature" not in text
    for entry in re.split(
        r'(?=<div class="doc doc-object doc-function ref-entry")', text
    )[1:]:
        assert entry.count('<p class="ref-signature">') >= 1


@pytest.mark.parametrize("page", WITH_ENTRIES)
def test_every_summary_is_one_sentence_on_one_line(site: Path, page: str) -> None:
    text = article(site, page)
    summaries = re.findall(r'<p class="ref-summary">(.*?)</p>', text, re.DOTALL)
    summaries += re.findall(
        r'<span class="ref-method-summary">(.*?)</span>\s*</summary>', text, re.DOTALL
    )
    assert summaries
    for summary in summaries:
        sentence = plain(summary)
        assert "\n" not in summary, sentence
        assert sentence.endswith("."), sentence
        assert not re.search(r"\.\s+[A-Z]", sentence), sentence


@pytest.mark.parametrize("page", WITH_ENTRIES)
def test_a_page_stays_short_enough_to_read(site: Path, page: str) -> None:
    assert len(words(article(site, page))) <= WORD_LIMIT


def test_a_type_alias_reads_as_one_line_of_choices(site: Path) -> None:
    entry = entry_html(
        site,
        "reference/core/builtin_nodes.md",
        "nodestep.core.builtin_nodes.ToolErrorPolicy",
    )
    assert "type_alias" in entry
    line = re.search(r'<p class="ref-signature ref-signature-alias">(.*?)</p>', entry)
    assert line
    assert plain(line[1]).startswith(
        'ToolErrorPolicy: one of "raise", "return" or Callable'
    )
    assert "<pre" not in entry


def test_a_constant_shows_its_value_inline(site: Path) -> None:
    text = article(site, "reference/core/flow.md")
    entry = text[text.index('data-ref-id="nodestep.core.START"') :]
    line = re.search(r'<p class="ref-signature">(.*?)</p>', entry, re.DOTALL)
    assert line
    assert plain(line[1]) == "START = StartSentinel()"
    assert 'href="#nodestep.core.StartSentinel"' in line[1]


def entry_html(site: Path, page: str, identifier: str) -> str:
    text = article(site, page)
    entry = text[text.index(f'data-ref-id="{identifier}"') :]
    end = entry.find('<div class="doc doc-object')
    return entry if end == -1 else entry[:end]


def test_a_data_class_lists_its_fields_and_methods(site: Path) -> None:
    entry = entry_html(
        site, "reference/core/sub-agents.md", "nodestep.core.AgentHandle"
    )
    fields = entry[entry.index("<caption>Attributes</caption>") :]
    fields = fields[: fields.index("</table>")]
    assert re.findall(r'<td class="ref-table-name"><code>(\w+)</code>', fields) == [
        "id",
        "name",
        "task",
        "graph_name",
        "thread_id",
    ]
    assert '<p class="ref-methods-title">Methods</p>' in entry
    assert re.findall(r'<h4 id="([^"]+)"', entry) == [
        "nodestep.core.AgentHandle.done",
        "nodestep.core.AgentHandle.status",
    ]


def test_documented_parameters_replace_the_attributes_they_set(site: Path) -> None:
    entry = entry_html(
        site, "reference/core/sub-agents.md", "nodestep.core.AsyncioExecutor"
    )
    assert "<caption>Parameters</caption>" in entry
    assert "<caption>Attributes</caption>" not in entry


def test_a_method_row_shows_positional_names_then_dots(site: Path) -> None:
    entry = entry_html(
        site, "reference/core/sub-agents.md", "nodestep.core.AsyncioExecutor"
    )
    assert re.search(
        r'<h4 id="nodestep\.core\.AsyncioExecutor\.submit" [^>]*><code>'
        r'<span class="ref-signature-keyword">async </span>'
        r'<span class="ref-signature-name">submit</span>\(task, \.\.\.\)</code>',
        entry,
    )


def test_the_core_index_lists_each_page_with_its_names(site: Path) -> None:
    table = overview(site, "reference/core/index.md")
    rows = re.findall(
        r'<tr><td><a href="([^"]+)">(.*?)</a></td><td>(.*?)</td></tr>', table
    )
    pages = [*CORE_PAGES, "reference/core/builtin_nodes.md"]
    assert [href for href, _, _ in rows] == [f"{Path(page).stem}/" for page in pages]
    for (_, _, names), page in zip(rows, pages, strict=True):
        expected = [identifier.rpartition(".")[2] for identifier in entry_ids(page)]
        assert re.findall(r"<code>(\w+)</code>", names) == expected, page
        assert "<autoref" not in names
        assert len(re.findall(r'<a [^>]*href="[^"]+/#nodestep\.', names)) == len(
            expected
        )


MIDDLEWARE_PAGES = [
    "reference/middleware/interrupt.md",
    "reference/middleware/memory.md",
    "reference/middleware/skills.md",
    "reference/middleware/summarization.md",
    "reference/middleware/todolist.md",
    "reference/middleware/tool_limit.md",
]
WORKSPACE_PAGES = [
    "reference/workspace/integrations/local.md",
    "reference/workspace/integrations/virtual.md",
]


def test_the_middleware_and_workspace_pages_declare_groups() -> None:
    assert {
        "reference/middleware/index.md",
        *MIDDLEWARE_PAGES,
        "reference/workspace/index.md",
        *WORKSPACE_PAGES,
    } - set(SINGLE_ENTRY_PAGES) <= set(GROUPED)


@pytest.mark.parametrize(
    ("index", "pages"),
    [
        ("reference/middleware/index.md", MIDDLEWARE_PAGES),
        ("reference/workspace/index.md", WORKSPACE_PAGES),
    ],
)
def test_the_middleware_and_workspace_indexes_list_their_groups_then_their_pages(
    site: Path, index: str, pages: list[str]
) -> None:
    table = overview(site, index)
    assert table.index('<tr class="ref-overview-group">') < table.index(
        '<th scope="col">Page</th>'
    )
    rows = re.findall(
        r'<tr><td><a href="([^"#]+)">(.*?)</a></td><td>(.*?)</td></tr>', table
    )
    folder = Path(index).parent
    assert [href for href, _, _ in rows] == [
        f"{Path(page).relative_to(folder).with_suffix('').as_posix()}/"
        for page in pages
    ]
    for (_, _, names), page in zip(rows, pages, strict=True):
        expected = [identifier.rpartition(".")[2] for identifier in entry_ids(page)]
        assert re.findall(r"<code>(\w+)</code>", names) == expected, page


def test_the_hook_names_read_as_one_line_of_choices(site: Path) -> None:
    entry = entry_html(
        site, "reference/middleware/index.md", "nodestep.middleware.HookName"
    )
    line = re.search(r'<p class="ref-signature ref-signature-alias">(.*?)</p>', entry)
    assert line
    assert plain(line[1]).startswith(
        'HookName: one of "before_graph", "after_graph", "before_node",'
    )
    assert "<pre" not in entry


@pytest.mark.parametrize(
    ("identifier", "described"),
    [
        (
            "nodestep.middleware.NodeMiddlewareContext",
            {"state", "stored_state", "state_schema"},
        ),
        ("nodestep.middleware.ToolMiddlewareContext", {"value", "task_id"}),
        ("nodestep.middleware.ModelMiddlewareContext", {"model", "response"}),
    ],
)
def test_a_hook_context_describes_its_fields_in_its_table(
    site: Path, identifier: str, described: set[str]
) -> None:
    entry = entry_html(site, "reference/middleware/index.md", identifier)
    fields = entry[entry.index("<caption>Attributes</caption>") :]
    fields = fields[: fields.index("</table>")]
    assert described <= {
        row["name"] for row in TABLE_ROW.finditer(fields) if plain(row["text"] or "")
    }
    assert "ref-description" not in entry


def test_a_memory_entry_shows_its_fields_not_the_model_hook(site: Path) -> None:
    entry = entry_html(
        site, "reference/middleware/memory.md", "nodestep.middleware.memory.Memory"
    )
    assert "model_post_init" not in entry
    fields = entry[entry.index("<caption>Attributes</caption>") :]
    fields = fields[: fields.index("</table>")]
    assert re.findall(r'<td class="ref-table-name"><code>(\w+)</code>', fields) == [
        "key",
        "title",
        "content",
        "memory_type",
        "scope",
        "created_at",
        "updated_at",
    ]


def test_only_a_long_parameter_may_wrap_inside_a_signature(site: Path) -> None:
    entry = entry_html(
        site,
        "reference/middleware/summarization.md",
        "nodestep.middleware.summarization.SummarizationMiddleware",
    )
    signature = re.search(r'<p class="ref-signature">(.*?)</p>', entry, re.DOTALL)
    assert signature
    classes = {
        plain(text).partition("=")[0]: kind
        for kind, text in re.findall(
            r'<span class="(ref-param[\w -]*)">(.*?)</span>', signature[1], re.DOTALL
        )
    }
    assert classes.pop("summary_prompt") == "ref-param ref-param-long"
    assert set(classes.values()) == {"ref-param"}


STATE_CHAT_AND_EXCEPTION_PAGES = [
    "reference/state/index.md",
    "reference/state/integrations/filesystem.md",
    "reference/state/integrations/inmemory.md",
    "reference/chat/index.md",
    "reference/chat/integrations/index.md",
    "reference/chat/integrations/openai.md",
    "reference/exceptions.md",
]


def test_the_state_chat_and_exception_pages_declare_groups() -> None:
    assert set(STATE_CHAT_AND_EXCEPTION_PAGES) - set(SINGLE_ENTRY_PAGES) <= set(GROUPED)


@pytest.mark.parametrize(
    ("index", "pages"),
    [
        (
            "reference/state/index.md",
            [
                (
                    "integrations/filesystem/",
                    "reference/state/integrations/filesystem.md",
                ),
                ("integrations/inmemory/", "reference/state/integrations/inmemory.md"),
            ],
        ),
        (
            "reference/chat/index.md",
            [
                ("integrations/", "reference/chat/integrations/index.md"),
                ("integrations/openai/", "reference/chat/integrations/openai.md"),
            ],
        ),
        (
            "reference/chat/integrations/index.md",
            [("openai/", "reference/chat/integrations/openai.md")],
        ),
    ],
)
def test_the_state_and_chat_indexes_list_their_groups_then_the_pages_under_them(
    site: Path, index: str, pages: list[tuple[str, str]]
) -> None:
    own, found, listed = overview(site, index).partition('<th scope="col">Page</th>')
    assert found
    assert '<tr class="ref-overview-group">' in own
    rows = re.findall(
        r'<tr><td><a href="([^"]+)">(.*?)</a></td><td>(.*?)</td></tr>', listed
    )
    assert [href for href, _, _ in rows] == [href for href, _ in pages]
    for (_, _, names), (_, page) in zip(rows, pages, strict=True):
        expected = [identifier.rpartition(".")[2] for identifier in entry_ids(page)]
        assert expected, page
        assert re.findall(r"<code>(\w+)</code>", names) == expected, page


def test_every_page_with_entries_is_checked_for_blank_rows() -> None:
    assert {
        *CORE_PAGES,
        "reference/core/builtin_nodes.md",
        *STATE_CHAT_AND_EXCEPTION_PAGES,
        "reference/middleware/index.md",
        *MIDDLEWARE_PAGES,
        "reference/workspace/index.md",
        *WORKSPACE_PAGES,
    } <= set(ENTRY_PAGES)


@pytest.mark.parametrize("page", ENTRY_PAGES)
def test_every_table_row_gives_a_type_or_a_description(site: Path, page: str) -> None:
    blank = [
        row["name"]
        for row in TABLE_ROW.finditer(article(site, page))
        if not plain(row["type"]) and not plain(row["text"] or "")
    ]
    assert not blank, f"{page}: {', '.join(blank)}"


@pytest.mark.parametrize("page", ENTRY_PAGES)
def test_a_row_without_a_description_lets_its_type_fill_the_row(
    site: Path, page: str
) -> None:
    rows = list(TABLE_ROW.finditer(article(site, page)))
    assert len(rows) == article(site, page).count('<td class="ref-table-name">')
    for row in rows:
        if row["text"] is None:
            assert row["span"], f"{page}: {row['name']}"
        else:
            assert not row["span"], f"{page}: {row['name']}"
            assert plain(row["text"]), f"{page}: {row['name']} has an empty cell"


@pytest.mark.parametrize(
    ("page", "identifier", "bases"),
    [
        (
            "reference/exceptions.md",
            "nodestep.exceptions.NodeTimeoutError",
            "GraphExecutionError",
        ),
        (
            "reference/exceptions.md",
            "nodestep.exceptions.IntegrationNotInstalledError",
            "NodestepError and ImportError",
        ),
        ("reference/exceptions.md", "nodestep.exceptions.NodestepError", "Exception"),
        ("reference/chat/index.md", "nodestep.chat.AIMessage", "BaseMessage"),
        (
            "reference/state/integrations/filesystem.md",
            "nodestep.state.integrations.filesystem.FilesystemStateStore",
            "InMemoryStateStore",
        ),
    ],
)
def test_a_subclass_names_its_bases_after_its_summary(
    site: Path, page: str, identifier: str, bases: str
) -> None:
    line = re.search(
        r'<p class="ref-summary">.*?</p>\s*<p class="ref-bases">(.*?)</p>',
        entry_html(site, page, identifier),
        re.DOTALL,
    )
    assert line
    assert plain(line[1]) == f"Subclass of {bases}."


def test_a_base_that_has_an_entry_links_to_it(site: Path) -> None:
    for page, identifier, href in (
        (
            "reference/exceptions.md",
            "nodestep.exceptions.NodeTimeoutError",
            "#nodestep.exceptions.GraphExecutionError",
        ),
        (
            "reference/state/integrations/filesystem.md",
            "nodestep.state.integrations.filesystem.FilesystemStateStore",
            "../inmemory/#nodestep.state.integrations.inmemory.InMemoryStateStore",
        ),
    ):
        line = re.search(
            r'<p class="ref-bases">(.*?)</p>',
            entry_html(site, page, identifier),
            re.DOTALL,
        )
        assert line
        assert f'href="{href}"' in line[1], identifier


@pytest.mark.parametrize(
    ("page", "identifier"),
    [
        ("reference/state/index.md", "nodestep.state.StateSnapshot"),
        ("reference/state/index.md", "nodestep.state.StateStore"),
        ("reference/chat/index.md", "nodestep.chat.BaseMessage"),
        (
            "reference/chat/integrations/openai.md",
            "nodestep.chat.integrations.openai.OpenAISettings",
        ),
    ],
)
def test_the_shared_model_base_and_library_bases_are_left_out(
    site: Path, page: str, identifier: str
) -> None:
    assert '<p class="ref-bases">' not in entry_html(site, page, identifier)


def test_an_annotated_alias_reads_as_the_type_it_annotates(site: Path) -> None:
    entry = entry_html(site, "reference/chat/index.md", "nodestep.chat.Message")
    line = re.search(r'<p class="ref-signature ref-signature-alias">(.*?)</p>', entry)
    assert line
    assert plain(line[1]) == (
        "Message: SystemMessage or HumanMessage or AIMessage or ToolMessage"
    )
    assert len(re.findall(r'<a [^>]*href="#nodestep\.chat\.\w+Message"', line[1])) == 4


@pytest.mark.parametrize(
    ("page", "identifier", "concept"),
    [
        (
            "reference/state/integrations/filesystem.md",
            "nodestep.state.integrations.filesystem.FilesystemStateStore",
            "concepts/persistence/#state-stores",
        ),
        (
            "reference/state/index.md",
            "nodestep.state.StateSchema",
            "concepts/state/#validation",
        ),
    ],
)
def test_a_description_a_concept_page_explains_is_short_and_links_it(
    site: Path, page: str, identifier: str, concept: str
) -> None:
    description = re.search(
        r'<div class="ref-description">(.*?)</div>',
        entry_html(site, page, identifier),
        re.DOTALL,
    )
    assert description
    assert re.search(rf'href="(?:\.\./)+{re.escape(concept)}"', description[1])
    assert len(words(description[1])) <= 60


@pytest.mark.parametrize(
    ("page", "identifier", "kind"),
    [
        ("reference/core/flow.md", "nodestep.core.START", "constant"),
        ("reference/core/flow.md", "nodestep.core.END", "constant"),
        ("reference/core/graphs.md", "nodestep.core.graph.StateT", "type_variable"),
        ("reference/state/index.md", "nodestep.state.schema.StateT", "type_variable"),
        (
            "reference/middleware/index.md",
            "nodestep.middleware.base.S",
            "type_variable",
        ),
        ("reference/state/index.md", "nodestep.state.Reducer", "type_alias"),
    ],
)
def test_a_module_value_is_labelled_by_what_it_is(
    site: Path, page: str, identifier: str, kind: str
) -> None:
    label = re.search(
        r'<code class="doc-symbol doc-symbol-heading doc-symbol-(\w+)">',
        entry_html(site, page, identifier),
    )
    assert label
    assert label[1] == kind


def signature(entry: str) -> str:
    line = re.search(
        r'<p class="ref-signature"><code>(.*?)</code></p>', entry, re.DOTALL
    )
    assert line
    return plain(line[1])


@pytest.mark.parametrize(
    ("page", "identifier", "expected"),
    [
        (
            "reference/middleware/summarization.md",
            "nodestep.middleware.summarization.TokenLimitTrigger",
            "TokenLimitTrigger(*, max_input_tokens, threshold_ratio=0.8)",
        ),
        (
            "reference/middleware/summarization.md",
            "nodestep.middleware.summarization.RecentMessagesPolicy",
            "RecentMessagesPolicy(*, keep_recent=20)",
        ),
        (
            "reference/middleware/memory.md",
            "nodestep.middleware.memory.Memory",
            "Memory(*, key, title, content, memory_type='fact', scope='global',"
            " created_at=None, updated_at=None)",
        ),
        (
            "reference/chat/index.md",
            "nodestep.chat.ChatRequest",
            "ChatRequest(*, messages, tools=[], tool_choice=None, output_schema=None)",
        ),
        (
            "reference/chat/index.md",
            "nodestep.chat.HumanMessage",
            "HumanMessage(*, id=None, content=None, type='human')",
        ),
        (
            "reference/state/index.md",
            "nodestep.state.HistoryEvent",
            "HistoryEvent(*, id=str(uuid4()), thread_id, branch_id='main', sequence,"
            " type, node=None, parent_id=None, data_json=None, error_type=None,"
            " message=None, created_at=time.time())",
        ),
    ],
)
def test_a_model_entry_shows_its_keyword_constructor_with_defaults(
    site: Path, page: str, identifier: str, expected: str
) -> None:
    assert signature(entry_html(site, page, identifier)) == expected


def test_settings_read_from_the_environment_show_their_field_descriptions(
    site: Path,
) -> None:
    entry = entry_html(
        site,
        "reference/chat/integrations/openai.md",
        "nodestep.chat.integrations.openai.OpenAISettings",
    )
    assert '<p class="ref-signature">' not in entry
    rows = {row["name"]: plain(row["text"]) for row in TABLE_ROW.finditer(entry)}
    assert rows["api_key"] == "OpenAI API key."
    assert rows["timeout"] == "HTTP request timeout in seconds."


TABLE = re.compile(
    r'<table class="ref-table [\w-]+">\s*<caption>(?P<caption>\w+)</caption>'
    r"(?P<body>.*?)</table>",
    re.DOTALL,
)


@pytest.mark.parametrize("page", ENTRY_PAGES)
def test_rows_of_one_table_do_not_repeat_the_same_text(site: Path, page: str) -> None:
    repeated = []
    for table in TABLE.finditer(article(site, page)):
        texts = [
            plain(text)
            for text in re.findall(
                r'<td class="ref-table-text">(.*?)</td>', table["body"], re.DOTALL
            )
        ]
        repeated += sorted({text for text in texts if texts.count(text) > 1})
    assert not repeated, f"{page}: {repeated}"


def test_a_raises_row_for_several_errors_lists_them_without_brackets(
    site: Path,
) -> None:
    entry = entry_html(site, "reference/core/graphs.md", "nodestep.core.Graph")
    fold = entry[entry.index('id="nodestep.core.Graph.invoke"') :]
    fold = fold[: fold.index("</details>")]
    raises = fold[fold.index("<caption>Raises</caption>") :]
    types = [
        plain(cell)
        for cell in re.findall(r'<td class="ref-table-type">(.*?)</td>', raises)
    ]
    assert any(
        cell.startswith("TypeError, GraphConfigError, UnknownThreadError, ResumeError")
        for cell in types
    )
    assert not [cell for cell in types if cell.startswith("(")]


@pytest.mark.parametrize(
    "identifier",
    [
        "nodestep.core.builtin_nodes.model_node",
        "nodestep.core.builtin_nodes.tool_runner.tool_runner",
    ],
)
def test_errors_a_node_raises_when_it_runs_are_in_its_raises_table(
    site: Path, identifier: str
) -> None:
    entry = entry_html(site, "reference/core/builtin_nodes.md", identifier)
    assert "<details" not in entry
    raises = entry[entry.index("<caption>Raises</caption>") :]
    texts = [
        plain(text)
        for text in re.findall(
            r'<td class="ref-table-text">(.*?)</td>', raises, re.DOTALL
        )
    ]
    assert any(text.startswith("When the node runs") for text in texts)


@pytest.mark.parametrize("page", ENTRY_PAGES)
def test_every_docstring_section_uses_the_entry_layout(site: Path, page: str) -> None:
    text = article(site, page)
    assert "doc-section-title" not in text
    assert '<details class="note"' not in text


def test_examples_have_a_caption_like_the_tables(site: Path) -> None:
    entry = entry_html(
        site, "reference/core/reducers.md", "nodestep.utils.reducers.Replace"
    )
    assert re.search(
        r'<div class="ref-examples">\s*<p class="ref-section-title">Examples</p>', entry
    )
