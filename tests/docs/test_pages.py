import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from nodestep import exceptions
from nodestep.chat import integrations
from reference_docstrings import ReferenceDocstrings

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"
EXAMPLES = sorted(
    path for path in (ROOT / "examples").glob("*.py") if path.name != "__init__.py"
)
CORE_CONCEPT_PAGES = [
    "concepts/graphs.md",
    "concepts/state.md",
    "concepts/nodes.md",
    "concepts/execution.md",
]
CAPABILITY_PAGES = [
    "concepts/streaming.md",
    "concepts/persistence.md",
    "concepts/interrupts.md",
    "concepts/agents.md",
    "concepts/middleware.md",
    "concepts/sub-agents.md",
    "concepts/workspace.md",
]
CONCEPT_PAGES = [*CORE_CONCEPT_PAGES, *CAPABILITY_PAGES]

REQUIRED_TERMS = {
    "index.md": ["uv add", "quickstart.md", *CONCEPT_PAGES],
    "install.md": ["`openai`", "`sandbox`", "IntegrationNotInstalledError", "uvx"],
    "quickstart.md": [
        "BaseState",
        "@node",
        "flow(",
        "when(",
        "otherwise=",
        "to_mermaid",
        "invoke(",
        "stream_mode",
        "InMemoryStateStore",
        "thread_id",
        "interrupt(",
        "Resume(",
    ],
    "concepts/graphs.md": [
        "START",
        "END",
        "flow(",
        "when(",
        "otherwise=",
        "branch(",
        "goto=",
        "to_mermaid",
        "GraphConfigError",
    ],
    "concepts/state.md": [
        "BaseState",
        "TypedDict",
        "add",
        "add_messages",
        "merge_dict",
        "Replace(",
        "RemoveMessage",
        "StateUpdateError",
    ],
    "concepts/nodes.md": [
        "NodeContext",
        "ctx.context",
        "ctx.emit",
        "ctx.cache",
        "Command(",
        "Send(",
        "timeout=",
        "ContextNotProvidedError",
    ],
    "concepts/execution.md": [
        "superstep",
        "max_steps",
        "timeout",
        "InvalidUpdateError",
        "RunLimitExceededError",
        "ainvoke",
        "GraphResult",
    ],
    "concepts/streaming.md": [
        "stream_mode",
        '"updates"',
        '"values"',
        '"custom"',
        '"tokens"',
        '"debug"',
        '"final"',
        '"interrupt"',
    ],
    "concepts/persistence.md": [
        "thread_id",
        "InMemoryStateStore",
        "FilesystemStateStore",
        "history(",
        "load(",
        "fork(",
        "get_state(",
        "update_state(",
        "branch_id",
        "InvalidUpdateError",
    ],
    "concepts/interrupts.md": [
        "interrupt(",
        "Resume(",
        "answers=",
        "ToolInterruptMiddleware",
        "InterruptRule",
        "ToolDecision",
        "on_interrupt",
    ],
    "concepts/agents.md": [
        "@tool",
        "ToolContext",
        "model_node",
        "tool_runner",
        "build_react_agent",
        "tool_errors",
        "output_schema",
        "ScriptedChat",
        "OpenAIChat",
        "## Chat models",
    ],
    "concepts/middleware.md": [
        "before_graph",
        "after_graph",
        "before_node",
        "after_node",
        "before_tool",
        "after_tool",
        "before_model",
        "after_model",
        "on_interrupt",
        "on_error",
        "on_run_end",
        "RunOutcome",
        "on_tool_error",
        "task_id",
        "ctx.model",
        "ctx.replace",
        "not checked",
    ],
    "concepts/sub-agents.md": [
        "ctx.spawn",
        "ctx.gather",
        "AgentTask",
        "AsyncioExecutor",
        "AgentStatus",
        "on_progress",
        "subgraph(",
    ],
    "concepts/workspace.md": [
        "LocalWorkspace",
        "InMemoryWorkspace",
        "PathAccessError",
        "FilesystemMemory",
        "scope=",
    ],
    "guides/branching.md": ["when(", "otherwise=", "branch(", "to_mermaid"],
    "guides/fan-out.md": ["Command(", "Send(", "goto=", "add"],
    "guides/approval.md": [
        "InMemoryStateStore",
        "thread_id",
        "ToolInterruptMiddleware",
        "InterruptRule",
        "ToolDecision",
        "Resume(",
    ],
    "guides/sandbox.md": ["nodestep sandbox", "--context", "uvx"],
    "guides/testing.md": ["ScriptedChat", "chat.requests", "default="],
    "guides/tracing.md": [
        "nodeartifact.configure",
        "nodeartifact.instrument",
        "nodeartifact serve",
    ],
    "guides/chat-models.md": [
        "Chat",
        "ChatRequest",
        "ChatResponse",
        "ChatStreamChunk",
        "assemble_stream",
        "tool_call_delta",
        "`refusal`",
    ],
    "reference/chat/integrations/openai.md": [
        "OpenAIChat.from_env",
        "gpt-6-luna",
        "--env-file",
        "client=",
        "OPENAI_CUSTOM_HEADERS",
        "IntegrationNotInstalledError",
        "async with",
    ],
    "examples.md": ["nodestep sandbox"],
    "changelog.md": ['--8<-- "CHANGELOG.md"'],
}

EXPLICIT_RULES = {
    "environment read only by from_env": (
        "reference/chat/integrations/openai.md",
        "constructor reads no environment variable",
    ),
    "no .env loading": (
        "reference/chat/integrations/openai.md",
        "nothing loads a .env file",
    ),
    "a store needs a thread_id": (
        "concepts/persistence.md",
        "run with a store | needs a thread_id",
    ),
    "context is never stored": ("concepts/nodes.md", "it is never stored"),
    "tools= is the complete list": ("concepts/agents.md", "complete tool list"),
    "middleware tools are listed by hand": ("concepts/middleware.md", "yourself"),
    "the flow is checked": ("concepts/graphs.md", "graphconfigerror"),
    "unknown return values": ("concepts/nodes.md", "any other value raises typeerror"),
    "changing the input and returning an update": (
        "concepts/nodes.md",
        "changing the input and returning",
    ),
    "invoke(None) never restarts": (
        "concepts/persistence.md",
        "never starts a new turn",
    ),
    "pending interrupts are not dropped": (
        "concepts/interrupts.md",
        "new input raises resumeerror",
    ),
    "resume takes only Resume": ("concepts/interrupts.md", "takes only a resume"),
    "GraphInterrupt is a BaseException": ("concepts/interrupts.md", "baseexception"),
    "unchecked approval rules ask": (
        "concepts/interrupts.md",
        "a predicate that raises",
    ),
    "tool failures fail the run": ("concepts/agents.md", "fails the run by default"),
    "no history repair by default": ("concepts/agents.md", "repair_history=true"),
    "invalid structured output raises": ("concepts/agents.md", "structuredoutputerror"),
    "memory scope set in code": ("concepts/workspace.md", "never by the model"),
    "subgraph state wired explicitly": ("concepts/sub-agents.md", "wired explicitly"),
    "ungathered sub-agents": ("concepts/sub-agents.md", "never gathered"),
    "misspelled hooks": ("concepts/middleware.md", "misspelled"),
    "stream_mode is required": ("concepts/streaming.md", "is required"),
    "ctx.emit sends custom events": ("concepts/streaming.md", "sends only"),
    "replace is the default reducer": ("concepts/state.md", "the default reducer"),
    "a known message id edits the message": (
        "concepts/state.md",
        "a message with the same id replaces",
    ),
    "merge_dict is shallow": ("concepts/state.md", "cannot remove a key"),
    "parallel writes in node-name order": ("concepts/execution.md", "node-name order"),
    "parallel replace conflicts": ("concepts/execution.md", "last one wins"),
    "message ids": ("concepts/state.md", "uuid4"),
    "resume answers as JSON": ("concepts/interrupts.md", "normalized to json"),
    "sparse snapshots": ("concepts/persistence.md", "every 50"),
    "update_state applies reducers": (
        "concepts/persistence.md",
        "applies the update through the reducers",
    ),
    "editing a paused thread": (
        "concepts/persistence.md",
        "a finished task of the paused superstep already wrote",
    ),
    "parent directories are created": ("concepts/persistence.md", "parent directories"),
    "file locks": ("concepts/persistence.md", "msvcrt"),
    "node names are the stored identity": (
        "concepts/persistence.md",
        "renaming a node",
    ),
    "dump_json_object sorts keys": (
        "concepts/persistence.md",
        "dump_json_object sorts",
    ),
    "history ids and timestamps": ("concepts/persistence.md", "time.time()"),
    "both node styles": ("concepts/nodes.md", "both styles end up as an update"),
    "reducers receive None": ("concepts/state.md", "none on the first write"),
    "lax validation": ("concepts/state.md", "strict=true"),
    "OpenAI timeout and retries": (
        "reference/chat/integrations/openai.md",
        "60 seconds",
    ),
    "OpenAI strict schema rewrite": (
        "reference/chat/integrations/openai.md",
        "strict mode",
    ),
    "lenient provider parsing": ("guides/chat-models.md", "lenient"),
    "docstrings become descriptions": (
        "concepts/agents.md",
        "docstring becomes the description",
    ),
    "tool results wrapped": ("concepts/agents.md", 'wrapped as {"result":'),
    "tool calls run concurrently": ("concepts/agents.md", "run concurrently"),
    "build_react_agent tool limit": ("concepts/agents.md", "max_tool_calls | 50"),
    "lazy OpenAIChat export": ("install.md", "lazily"),
    "approval decisions and edit merging": (
        "concepts/interrupts.md",
        "an edit is merged",
    ),
    "Command.goto replaces edges": ("concepts/graphs.md", "replaces the node's edges"),
    "routers see the superstep's writes": ("concepts/execution.md", "routers see"),
    "no join barrier": ("concepts/execution.md", "join barrier"),
    "max_steps per call": ("concepts/execution.md", "max_steps=25"),
    "input on an existing thread is a patch": ("concepts/state.md", "patch"),
    "stored models keep aliases and drop computed fields": (
        "concepts/state.md",
        "keep their aliases and leave out computed fields",
    ),
    "a storage refusal shows no values": ("concepts/state.md", "never the values"),
    "resumed nodes run from the top": ("concepts/interrupts.md", "from the top"),
    "ctx.cache lifetime": ("concepts/nodes.md", "empty in every other task"),
    "generated thread ids": ("concepts/persistence.md", "generated thread id"),
    "task ids": ("concepts/nodes.md", "name[1]"),
    "branch ids": ("concepts/persistence.md", "or a random id"),
    "sub-agent threads": ("concepts/sub-agents.md", ":sub:<handle id>"),
    "approval interrupt ids": ("concepts/interrupts.md", '"approve_" plus'),
    "sync wrappers": ("concepts/execution.md", "private event loop"),
    "sync-node timeouts": ("concepts/nodes.md", "cannot be stopped"),
    "fail fast": ("concepts/execution.md", "first failure"),
    "continue an unfinished superstep": (
        "concepts/persistence.md",
        "unfinished superstep",
    ),
    "resume after a failed superstep": ("concepts/persistence.md", "failed superstep"),
    "paused result without finished siblings": (
        "concepts/interrupts.md",
        "state before the paused superstep",
    ),
    "sync handlers in threads": ("concepts/execution.md", "worker thread"),
    "KeyboardInterrupt propagates": ("concepts/execution.md", "keyboardinterrupt"),
    "ToolLimitMiddleware budgets stay in memory": (
        "concepts/middleware.md",
        "budgets live in the instance's memory",
    ),
    "unchecked FilesystemSkills node names": ("concepts/middleware.md", "not checked"),
    "OpenAI SDK environment variables": (
        "reference/chat/integrations/openai.md",
        "openai_log",
    ),
    "a client you build reads the connection variables": (
        "reference/chat/integrations/openai.md",
        "for a client you build, openai_api_key, openai_base_url",
    ),
    "ChatHistoryError after a stopped tool turn": (
        "concepts/agents.md",
        "after a stopped tool turn",
    ),
}

INSTALL_COMMAND = re.compile(
    r"(?:uv add(?: --dev)?|uv pip install|pip install|uvx --from)\s+"
    r"(?P<quote>[\"']?)(?P<requirement>[^\"'\s]+(?: @ [^\"'\s]+)?)(?P=quote)"
)
OWN_PACKAGE = re.compile(r"^(?P<name>nodestep|nodeartifact|text-to-sql-demo)\b")
MODEL_NAME = re.compile(r"\bgpt-[A-Za-z0-9.-]+")
PLANNED = "anthropic (claude) and ollama are planned."
PLANNED_ROWS = ("| Anthropic (Claude) |", "| Ollama |")
STATUS_NOTE = (
    "alpha (0.1.0a1). anything may change between releases without a deprecation "
    "period, so pin a tag or a commit, as the [install](install.md#pin-a-version) "
    "page shows. nodestep is not on pypi yet."
)
EMOJI = re.compile("[\U0001f000-\U0001faff☀-➿⬀-⯿️]")
FENCED_BLOCK = re.compile(r"^```.*?^```[ \t]*$", re.MULTILINE | re.DOTALL)
RULES_HEADER = "| Rule or setting | What happens |"


def pages() -> list[Path]:
    return sorted(DOCS_DIR.rglob("*.md"))


def normalized(text: str) -> str:
    return " ".join(text.replace("`", "").lower().split())


def nav_files(items: Any) -> Iterator[str]:
    if isinstance(items, str):
        yield items
    elif isinstance(items, list):
        for item in items:
            yield from nav_files(item)
    elif isinstance(items, dict):
        for value in items.values():
            yield from nav_files(value)


@pytest.fixture(scope="module")
def navigation() -> list[str]:
    config = pytest.importorskip("mkdocs.config")
    return list(nav_files(config.load_config(str(ROOT / "mkdocs.yml"))["nav"]))


@pytest.fixture(scope="module")
def sections() -> dict[str, dict[str, str]]:
    config = pytest.importorskip("mkdocs.config")
    nav = config.load_config(str(ROOT / "mkdocs.yml"))["nav"]
    docs = next(item["nodestep"] for item in nav if "nodestep" in item)
    return {
        label: {title: page for entry in entries for title, page in entry.items()}
        for item in docs
        for label, entries in item.items()
        if isinstance(entries, list)
    }


@pytest.mark.parametrize("page", list(REQUIRED_TERMS))
def test_content_page_is_in_the_navigation(page: str, navigation: list[str]) -> None:
    assert (DOCS_DIR / page).is_file(), f"docs/{page} is missing"
    assert page in navigation, f"docs/{page} is not in the mkdocs.yml nav"


def page_source(page: str) -> str:
    path = DOCS_DIR / page
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    if page.startswith("reference/"):
        return f"{text}\n\n{ReferenceDocstrings(path).text()}"
    return text


@pytest.mark.parametrize("page", list(REQUIRED_TERMS))
def test_content_page_covers_its_topics(page: str) -> None:
    text = page_source(page)
    missing = [term for term in REQUIRED_TERMS[page] if term not in text]
    assert not missing, f"docs/{page} does not mention {', '.join(missing)}"


def page_text(page: str) -> str:
    return normalized(page_source(page))


@pytest.mark.parametrize("rule", list(EXPLICIT_RULES))
def test_every_default_and_ruled_item_is_documented(rule: str) -> None:
    page, phrase = EXPLICIT_RULES[rule]
    assert phrase in page_text(page), (
        f"docs/{page} does not document {rule!r} (looked for {phrase!r})"
    )


def section(text: str, heading: str) -> str:
    _, found, rest = text.partition(f"\n{heading}\n")
    assert found, f"no {heading!r} section"
    return FENCED_BLOCK.sub("", rest.split("\n## ", 1)[0])


def page_title(page: str) -> str:
    first_line = (DOCS_DIR / page).read_text(encoding="utf-8").splitlines()[0]
    return first_line.removeprefix("# ")


@pytest.mark.parametrize(
    "group", ["Get started", "Core concepts", "Capabilities", "How-to"]
)
def test_pages_are_headed_by_their_navigation_title(
    group: str, sections: dict[str, dict[str, str]]
) -> None:
    wrong = [
        f"{page}: {page_title(page)!r} is listed as {title!r}"
        for title, page in sections[group].items()
        if page != "index.md" and page_title(page) != title
    ]
    assert not wrong, wrong


def test_there_is_no_provider_guide(sections: dict[str, dict[str, str]]) -> None:
    assert not (DOCS_DIR / "providers").exists()
    titles = [
        title
        for group, entries in sections.items()
        if group != "API reference"
        for title in entries
    ]
    assert [title for title in titles if "OpenAI" in title] == []


def test_the_chat_page_lists_the_chat_integrations() -> None:
    text = (DOCS_DIR / "concepts" / "agents.md").read_text(encoding="utf-8")
    rows = [
        line for line in section(text, "## Chat models").splitlines() if line[:1] == "|"
    ]
    for link in (
        "](../reference/chat/integrations/openai.md)",
        "](../reference/chat/integrations/index.md)",
    ):
        assert any(link in row for row in rows), link


def test_the_integrations_package_names_the_planned_integrations() -> None:
    assert PLANNED in normalized(integrations.__doc__ or "")
    text = (DOCS_DIR / "concepts" / "agents.md").read_text(encoding="utf-8")
    rows = section(text, "## Chat models").splitlines()
    for start in PLANNED_ROWS:
        assert any(row.startswith(start) and "Planned" in row for row in rows), start


@pytest.mark.parametrize("page", CONCEPT_PAGES)
def test_concept_pages_follow_the_template(page: str) -> None:
    lines = (DOCS_DIR / page).read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# ")
    assert lines[1] == ""
    assert lines[2].endswith(".")
    assert not re.search(r"\.\s+[A-Z]", lines[2]), f"{page}: one sentence"
    fences = [line for line in lines if line.startswith("```")]
    assert fences[0] == "```python"
    assert fences[2] == "```text"
    headings = [line for line in lines if line.startswith("## ")]
    assert headings[-1] == "## Rules and defaults"
    rules = lines[lines.index("## Rules and defaults") + 1 :]
    assert next(line for line in rules if line.strip()) == RULES_HEADER


def test_every_exception_is_explained_on_a_page() -> None:
    text = "".join(
        page.read_text(encoding="utf-8")
        for page in pages()
        if not page.relative_to(DOCS_DIR).as_posix().startswith("reference/")
    )
    missing = [
        name
        for name, value in vars(exceptions).items()
        if isinstance(value, type)
        and issubclass(value, Exception)
        and value.__module__ == exceptions.__name__
        and value is not exceptions.NodestepError
        and f"`{name}`" not in text
    ]
    assert not missing, f"no docs page names {', '.join(missing)}"


@pytest.mark.parametrize("example", EXAMPLES, ids=[path.stem for path in EXAMPLES])
def test_examples_gallery_opens_each_example_in_the_sandbox(example: Path) -> None:
    path = DOCS_DIR / "examples.md"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    command = f"nodestep sandbox examples/{example.name}:graph"
    assert command in text, f"docs/examples.md does not show '{command}'"


@pytest.mark.parametrize("example", EXAMPLES, ids=[path.stem for path in EXAMPLES])
def test_examples_gallery_runs_each_example_from_a_clone(example: Path) -> None:
    path = DOCS_DIR / "examples.md"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    command = f"uv run --no-sync python examples/{example.name}"
    assert command in text, f"docs/examples.md does not show '{command}'"


def test_install_commands_use_github() -> None:
    offenders = [
        f"{page.relative_to(ROOT)}: {match['requirement']}"
        for page in pages()
        for match in INSTALL_COMMAND.finditer(page.read_text(encoding="utf-8"))
        if (own := OWN_PACKAGE.match(match["requirement"])) is not None
        and f" @ git+https://github.com/nodestep-ai/{own['name']}"
        not in match["requirement"]
    ]
    assert not offenders, "install from GitHub: " + ", ".join(offenders)


def test_docs_name_only_gpt_6_luna() -> None:
    offenders = sorted(
        {
            f"{page.relative_to(ROOT)}: {name}"
            for page in pages()
            for name in MODEL_NAME.findall(page.read_text(encoding="utf-8"))
            if name != "gpt-6-luna"
        }
    )
    assert not offenders, offenders


def test_docs_have_no_emoji() -> None:
    offenders = [
        str(page.relative_to(ROOT))
        for page in pages()
        if EMOJI.search(page.read_text(encoding="utf-8"))
    ]
    assert not offenders, offenders


def test_home_page_shows_the_development_warning() -> None:
    text = (DOCS_DIR / "index.md").read_text(encoding="utf-8")
    assert '!!! warning "In development"' in text
    assert STATUS_NOTE in normalized(text)


def test_no_built_page_has_an_announcement_bar(site: Path) -> None:
    pages_with_a_bar = [
        str(page.relative_to(site))
        for page in sorted(site.rglob("index.html"))
        if "md-banner" in page.read_text(encoding="utf-8")
    ]
    assert not pages_with_a_bar


def test_docs_home_has_no_approach_or_related_projects() -> None:
    text = (DOCS_DIR / "index.md").read_text(encoding="utf-8")
    assert "## Approach" not in text
    assert "## Related projects" not in text


def test_the_tracing_guide_is_short_and_links_the_nodeartifact_section() -> None:
    text = (DOCS_DIR / "guides" / "tracing.md").read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 70
    assert "https://nodestep-ai.github.io/nodestep/nodeartifact/" in text
    assert "github.com/nodestep-ai/nodeartifact#readme" not in text


def test_changelog_page_renders_the_changelog(site: Path) -> None:
    html = (site / "changelog" / "index.html").read_text(encoding="utf-8")
    assert "0.1.0a1" in html
    assert "Unreleased" not in html
    assert "Semantic Versioning" not in html


def test_the_first_release_is_a_short_summary() -> None:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert text.startswith("# Changelog\n\n## [0.1.0a1] - 2026-10-05\n")
    assert "[Unreleased]" not in text
    release = section(text, "## [0.1.0a1] - 2026-10-05")
    summary = release.strip().split("\n\n", 1)[0]
    assert summary.startswith("First alpha.")
    assert summary.endswith(".")
    assert len(re.findall(r"^- ", release, flags=re.MULTILINE)) <= 8
    assert [line for line in text.splitlines() if line != line.rstrip()] == []
