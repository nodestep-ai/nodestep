from pathlib import Path

import pytest

from nodestep.exceptions import PathAccessError
from nodestep.middleware.interrupt import (
    InterruptRule,
    all_fields,
    any_field,
    matches,
    should_interrupt,
)
from nodestep.workspace import LocalWorkspace


def test_interrupt_first_match_wins() -> None:
    rules = [
        InterruptRule(tool="delete_file", match=matches("path", r"\.sqlite$")),
        InterruptRule(tool="delete_file"),
    ]

    assert should_interrupt(rules, "delete_file", {"path": "data/cache.sqlite"})
    assert should_interrupt(rules, "delete_file", {"path": "data/other.json"})


def test_no_rules_no_interrupt() -> None:
    assert not should_interrupt([], "run_command")


def test_wildcard_matches_everything() -> None:
    rules = [InterruptRule(tool="*")]
    assert should_interrupt(rules, "anything", {"foo": "bar"})


def test_no_match_function_always_interrupts() -> None:
    rules = [InterruptRule(tool="run_command")]
    assert should_interrupt(rules, "run_command", {"command": "anything"})


def test_match_callable() -> None:
    rules = [
        InterruptRule(
            tool="run_command",
            match=lambda arguments: "&&" in arguments.get("command", ""),
        ),
    ]

    assert should_interrupt(rules, "run_command", {"command": "cd /tmp && rm -rf /"})
    assert not should_interrupt(rules, "run_command", {"command": "ls"})


def test_matches_helper() -> None:
    rules = [
        InterruptRule(
            tool="run_command",
            match=matches("command", r"(&&|\|\||;)\s*rm\b"),
        ),
    ]

    assert should_interrupt(rules, "run_command", {"command": "cd /tmp && rm -rf /"})
    assert not should_interrupt(rules, "run_command", {"command": "rm file.txt"})


def test_all_fields_helper() -> None:
    rules = [
        InterruptRule(
            tool="send_email",
            match=all_fields(to=r"@external\.com$", subject=r"(?i)urgent"),
        ),
    ]

    assert should_interrupt(
        rules, "send_email", {"to": "user@external.com", "subject": "URGENT: review"}
    )
    assert not should_interrupt(
        rules, "send_email", {"to": "user@external.com", "subject": "hello"}
    )


def test_any_field_helper() -> None:
    rules = [
        InterruptRule(
            tool="send_email",
            match=any_field(to=r"@evil\.com$", subject=r"(?i)spam"),
        ),
    ]

    assert should_interrupt(
        rules, "send_email", {"to": "x@evil.com", "subject": "hello"}
    )
    assert should_interrupt(
        rules, "send_email", {"to": "x@good.com", "subject": "SPAM offer"}
    )
    assert not should_interrupt(
        rules, "send_email", {"to": "x@good.com", "subject": "hello"}
    )


def test_local_workspace_rejects_parent_path(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)
    with pytest.raises(PathAccessError):
        workspace.resolve_path("../secret.txt")
