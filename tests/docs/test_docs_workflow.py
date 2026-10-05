from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "docs.yml"
CHECKOUT = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
SIBLING_REF = "${{ github.base_ref || github.ref_name }}"
SIBLING_TOKEN = "${{ secrets.SIBLING_CHECKOUT_TOKEN || github.token }}"
ON_MAIN = (
    "(github.event_name == 'push' || github.event_name == 'workflow_dispatch')"
    " && github.ref == 'refs/heads/main'"
)


@pytest.fixture(scope="module")
def workflow() -> dict[Any, Any]:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def build_steps(workflow: dict[Any, Any]) -> list[dict[str, Any]]:
    return workflow["jobs"]["build"]["steps"]


def test_the_docs_can_be_rebuilt_by_hand(workflow: dict[Any, Any]) -> None:
    triggers = workflow[True]
    assert "workflow_dispatch" in triggers
    assert triggers["push"] == {"branches": ["main"]}
    assert "pull_request" in triggers


def test_the_build_checks_out_nodestep_nodeartifact_and_the_stylesheet(
    workflow: dict[Any, Any],
) -> None:
    checkouts = [
        step["with"]
        for step in build_steps(workflow)
        if step.get("uses", "").startswith(CHECKOUT)
    ]
    assert checkouts == [
        {"path": "nodestep", "persist-credentials": False},
        {
            "repository": "nodestep-ai/nodeartifact",
            "ref": SIBLING_REF,
            "token": SIBLING_TOKEN,
            "path": "nodeartifact",
            "persist-credentials": False,
        },
        {
            "repository": "nodestep-ai/nodestep-stylesheet",
            "ref": SIBLING_REF,
            "token": SIBLING_TOKEN,
            "path": "nodestep-stylesheet",
            "persist-credentials": False,
        },
    ]


def test_the_build_runs_in_the_nodestep_folder_and_uploads_its_site(
    workflow: dict[Any, Any],
) -> None:
    build = workflow["jobs"]["build"]
    assert build["defaults"] == {"run": {"working-directory": "nodestep"}}
    runs = [step["run"] for step in build_steps(workflow) if "run" in step]
    assert runs == [
        "uv sync --locked --all-extras --group docs",
        "uv run --no-sync ty check overrides tests/docs",
        "uv run --no-sync mkdocs build --strict",
        "uv run --no-sync pytest -q tests/docs tests/design",
        "node --test tests/docs/theme.test.ts",
    ]
    upload = next(
        step
        for step in build_steps(workflow)
        if step.get("uses", "").startswith("actions/upload-pages-artifact@")
    )
    assert upload["with"] == {"path": "nodestep/site"}
    assert upload["if"] == ON_MAIN


def test_a_push_or_a_manual_run_on_main_deploys(workflow: dict[Any, Any]) -> None:
    assert workflow["jobs"]["deploy"]["if"] == ON_MAIN
