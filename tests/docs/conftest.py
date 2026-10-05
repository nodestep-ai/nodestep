import subprocess
import sys
from pathlib import Path

import pytest

BLOCK_TEST = "test_snippets.py::test_docs_python_block_runs["
ROOT = Path(__file__).resolve().parents[2]


class DocBlockSummary:
    def __init__(self) -> None:
        self.outcomes: dict[str, str] = {}

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        _, found, rest = report.nodeid.partition(BLOCK_TEST)
        if found and (report.when == "call" or report.outcome != "passed"):
            self.outcomes.setdefault(rest.removesuffix("]"), report.outcome)

    def pytest_terminal_summary(
        self, terminalreporter: pytest.TerminalReporter
    ) -> None:
        if not self.outcomes:
            return
        terminalreporter.write_sep("-", "python blocks in docs/")
        for block_id, outcome in self.outcomes.items():
            terminalreporter.write_line(f"{outcome:<8} {block_id}")


def pytest_configure(config: pytest.Config) -> None:
    config.pluginmanager.register(DocBlockSummary(), "docs-block-summary")


@pytest.fixture(scope="session")
def site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("mkdocstrings", reason="needs the docs dependency group")
    site = tmp_path_factory.mktemp("site")
    result = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--quiet", "--site-dir", str(site)],
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return site
