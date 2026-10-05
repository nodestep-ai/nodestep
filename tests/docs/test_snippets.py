import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from reference_docstrings import ReferenceDocstrings

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"
REFERENCE_DIR = DOCS_DIR / "reference"
SRC_DIR = ROOT / "src"
TIMEOUT_SECONDS = 60
NOT_RUN = "<!-- not-run -->"
CONTINUE = "<!-- continue -->"
SNIPPET_INCLUDE = re.compile(r"^[ \t]*-+8<-+")


class ContinuationError(ValueError):
    pass


class DocBlock(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    line: int
    code: str
    runnable: bool = True
    continued: bool = False
    output: str | None = None

    @property
    def id(self) -> str:
        return f"{self.path}:{self.line}"

    @property
    def includes_file(self) -> bool:
        return any(SNIPPET_INCLUDE.match(line) for line in self.code.splitlines())


class FencedCodeParser:
    OPENING = re.compile(
        r"^(?P<indent>[ \t]*)(?P<fence>`{3,}(?!.*`)|~{3,})(?P<info>.*)$"
    )
    PYTHON = frozenset({"python", "py", "python3", "py3"})
    OUTPUT = frozenset({"text", "mermaid"})
    COMMENT_START = "<!--"
    COMMENT_END = "-->"

    def __init__(self, root: Path) -> None:
        self.root = root

    def collect(self, docs_dir: Path) -> list[DocBlock]:
        return [
            block
            for page in sorted(docs_dir.rglob("*.md"))
            for block in self.parse(page)
        ]

    def parse(self, page: Path) -> list[DocBlock]:
        text = page.read_text(encoding="utf-8")
        return self.parse_text(text, page.relative_to(self.root).as_posix())

    def parse_text(self, text: str, path: str) -> list[DocBlock]:
        lines = text.splitlines()
        blocks: list[DocBlock] = []
        marker: str | None = None
        awaiting_output: int | None = None
        index = 0
        while index < len(lines):
            opening = self.OPENING.match(lines[index])
            if opening is None:
                stripped = lines[index].strip()
                if stripped in (NOT_RUN, CONTINUE):
                    marker = stripped
                elif stripped:
                    marker = None
                index = self._comment_end(lines, index) + 1
                continue
            end = self._closing_line(lines, index + 1, opening["fence"])
            language = self._language(opening["info"])
            indent = opening["indent"]
            body = "\n".join(
                self._dedent(line, indent) for line in lines[index + 1 : end]
            )
            if language in self.PYTHON:
                code = body
                if marker == CONTINUE:
                    code = f"{self._continued(blocks, path, index + 1)}\n\n{code}"
                blocks.append(
                    DocBlock(
                        path=path,
                        line=index + 1,
                        code=code,
                        runnable=marker != NOT_RUN,
                        continued=marker == CONTINUE,
                    )
                )
                awaiting_output = len(blocks) - 1
            else:
                if awaiting_output is not None and language in self.OUTPUT:
                    blocks[awaiting_output] = blocks[awaiting_output].model_copy(
                        update={"output": body}
                    )
                awaiting_output = None
            marker = None
            index = end + 1
        return blocks

    @staticmethod
    def _continued(blocks: list[DocBlock], path: str, line: int) -> str:
        if not blocks:
            raise ContinuationError(
                f"{path}:{line} continues, but no block is before it"
            )
        if not blocks[-1].runnable:
            raise ContinuationError(
                f"{path}:{line} continues {blocks[-1].id}, which is {NOT_RUN}"
            )
        return blocks[-1].code

    @classmethod
    def _comment_end(cls, lines: list[str], start: int) -> int:
        line = lines[start]
        if not line.lstrip().startswith(cls.COMMENT_START):
            return start
        last_start = line.rfind(cls.COMMENT_START) + len(cls.COMMENT_START)
        if cls.COMMENT_END in line[last_start:]:
            return start
        for index in range(start + 1, len(lines)):
            if cls.COMMENT_END in lines[index]:
                return index
        return len(lines)

    @staticmethod
    def _closing_line(lines: list[str], start: int, fence: str) -> int:
        for index in range(start, len(lines)):
            stripped = lines[index].strip()
            if len(stripped) >= len(fence) and set(stripped) == {fence[0]}:
                return index
        return len(lines)

    @staticmethod
    def _language(info: str) -> str:
        info = info.strip()
        if info.startswith("{"):
            words = [word for word in info.strip("{}").split() if word[0] == "."]
            return words[0][1:].lower() if words else ""
        words = info.split()
        return words[0].lower() if words else ""

    @staticmethod
    def _dedent(line: str, indent: str) -> str:
        return line[len(indent) :] if line.startswith(indent) else line.lstrip()


PARSER = FencedCodeParser(ROOT)
DOCSTRING_BLOCKS = [
    block
    for page in sorted(REFERENCE_DIR.rglob("*.md"))
    for identifier, docstring in ReferenceDocstrings(page)
    for block in PARSER.parse_text(docstring, identifier)
]
BLOCKS = [*PARSER.collect(DOCS_DIR), *DOCSTRING_BLOCKS]
BLOCK_PARAMETERS = [
    pytest.param(
        block,
        id=block.id,
        marks=() if block.runnable else pytest.mark.skip(reason=NOT_RUN),
    )
    for block in BLOCKS
] or [
    pytest.param(
        None,
        id="no-blocks",
        marks=pytest.mark.skip(reason="no python blocks in docs/"),
    )
]


def snippet_environment() -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("OPENAI_")
    }
    environment["PYTHONPATH"] = str(SRC_DIR)
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def test_marker_before_a_fence_skips_the_block(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text(
        f"```python\nprint(1)\n```\n\n{NOT_RUN}\n\n```python\nprint(2)\n```\n",
        encoding="utf-8",
    )

    blocks = FencedCodeParser(tmp_path).parse(page)

    assert [(block.code, block.runnable) for block in blocks] == [
        ("print(1)", True),
        ("print(2)", False),
    ]


def test_continue_marker_runs_the_block_after_the_one_before(
    tmp_path: Path,
) -> None:
    page = tmp_path / "page.md"
    page.write_text(
        f"```python\nx = 1\n```\n\n{CONTINUE}\n\n```python\ny = x + 1\n```\n\n"
        f"{CONTINUE}\n\n```python\nprint(y)\n```\n",
        encoding="utf-8",
    )

    blocks = FencedCodeParser(tmp_path).parse(page)

    assert [block.code for block in blocks] == [
        "x = 1",
        "x = 1\n\ny = x + 1",
        "x = 1\n\ny = x + 1\n\nprint(y)",
    ]


@pytest.mark.parametrize(
    "text",
    [
        f"{CONTINUE}\n\n```python\nprint(1)\n```\n",
        f"{NOT_RUN}\n\n```python\nx = 1\n```\n\n{CONTINUE}\n\n```python\nprint(x)\n```\n",
    ],
    ids=["first-block", "after-a-not-run-block"],
)
def test_continue_marker_needs_a_run_block_before_it(text: str) -> None:
    with pytest.raises(ContinuationError):
        FencedCodeParser(ROOT).parse_text(text, "page.md")


def test_the_fence_after_a_block_is_its_output(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text(
        "```python\nprint(1)\n```\n\nIt prints:\n\n```text\n1\n```\n\n"
        "```python\nx = 2\n```\n\n```bash\necho 2\n```\n\n```text\n2\n```\n",
        encoding="utf-8",
    )

    blocks = FencedCodeParser(tmp_path).parse(page)

    assert [block.output for block in blocks] == ["1", None]


def test_docstring_blocks_of_the_reference_pages_are_collected() -> None:
    owners = {block.path for block in DOCSTRING_BLOCKS}
    assert {
        "nodestep.chat.integrations.openai",
        "nodestep.chat.integrations.openai.OpenAIChat",
    } <= owners


def test_marker_is_never_shown_in_rendered_code() -> None:
    offenders = [
        block.id
        for block in BLOCKS
        if "not-run" in block.code or CONTINUE in block.code
    ]
    assert not offenders, offenders


def test_doc_blocks_run_without_openai_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1")

    environment = snippet_environment()

    assert [name for name in environment if name.startswith("OPENAI_")] == []
    assert environment["PYTHONPATH"] == str(SRC_DIR)


@pytest.mark.parametrize("block", BLOCK_PARAMETERS)
def test_docs_python_block_runs(block: DocBlock, tmp_path: Path) -> None:
    assert not block.includes_file, (
        f"{block.id} includes a file with --8<--, which this test does not "
        f"expand; mark the block {NOT_RUN} or write the code in the page"
    )
    script = tmp_path / "snippet.py"
    script.write_text(block.code + "\n", encoding="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            env=snippet_environment(),
            capture_output=True,
            encoding="utf-8",
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{block.id} did not finish within {TIMEOUT_SECONDS} s")
    assert result.returncode == 0, (
        f"{block.id} exited with code {result.returncode}\n"
        f"{result.stdout}{result.stderr}"
    )
    if block.output is not None:
        shown = block.output.strip("\n")
        printed = result.stdout.strip("\n")
        matches = printed.endswith(shown) if block.continued else printed == shown
        assert matches, f"{block.id} printed\n{printed}\nbut the page shows\n{shown}"
