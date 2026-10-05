from pathlib import Path

import pytest

from nodestep.exceptions import PathAccessError, WorkspaceError
from nodestep.workspace import LocalWorkspace


@pytest.fixture
def layout(tmp_path: Path, symlinks: None) -> tuple[LocalWorkspace, Path]:
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside_secret.txt").write_text("TOP SECRET outside", encoding="utf-8")
    (tmp_path / "root2").mkdir()
    (tmp_path / "root2" / "neighbour.txt").write_text(
        "SECRET neighbour", encoding="utf-8"
    )
    (root / "inside.txt").write_text("inside SECRET", encoding="utf-8")
    (root / "notes.txt").symlink_to(tmp_path / "outside_secret.txt")
    return LocalWorkspace(root), tmp_path


@pytest.mark.parametrize("pattern", ["../*", "../root2/*", "/etc/*", "sub/../../*"])
def test_glob_patterns_cannot_leave_the_root(layout, pattern: str) -> None:
    workspace, _ = layout

    with pytest.raises(PathAccessError):
        workspace.glob(pattern)


def test_grep_glob_patterns_cannot_leave_the_root(layout) -> None:
    workspace, _ = layout

    with pytest.raises(PathAccessError):
        workspace.grep("SECRET", glob_pattern="../**/*.txt")


def test_symlinks_pointing_outside_are_not_listed_or_searched(layout) -> None:
    workspace, _ = layout

    assert workspace.glob("*.txt") == ["inside.txt"]
    assert workspace.grep("SECRET").matches == {"inside.txt": ["inside SECRET"]}


@pytest.mark.parametrize("glob_pattern", [None, "*.txt"])
def test_grep_reports_symlinks_pointing_outside_as_skipped(
    layout, glob_pattern: str | None
) -> None:
    workspace, _ = layout

    result = workspace.grep("SECRET", glob_pattern=glob_pattern)

    assert result.skipped == ["notes.txt"]


def test_delete_of_a_symlink_removes_only_the_link(layout) -> None:
    workspace, base = layout
    important = base / "root" / "important.db"
    important.write_text("data", encoding="utf-8")
    (base / "root" / "shortcut").symlink_to(important)

    assert workspace.delete("shortcut") is True

    assert important.exists()
    assert not (base / "root" / "shortcut").exists()


@pytest.mark.parametrize("path", [".", "", "sub/.."])
def test_deleting_the_root_is_refused(layout, path: str) -> None:
    workspace, base = layout

    with pytest.raises(WorkspaceError):
        workspace.delete(path)

    assert (base / "root").exists()


def test_delete_through_a_symlinked_directory_is_refused(layout) -> None:
    workspace, base = layout
    (base / "root" / "escape").symlink_to(base / "root2", target_is_directory=True)

    with pytest.raises(PathAccessError):
        workspace.delete("escape/neighbour.txt")

    assert (base / "root2" / "neighbour.txt").exists()


def test_crlf_line_endings_are_preserved(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)
    (tmp_path / "win.txt").write_bytes(b"line1\r\nline2\r\nline3\r\n")

    assert workspace.read_file("win.txt") == "line1\r\nline2\r\nline3\r\n"
    assert workspace.edit("win.txt", "line1\r\nline2", "LINE1\r\nLINE2") == 1
    assert workspace.edit("win.txt", "line3\n", "LINE3\n") == 1
    assert (tmp_path / "win.txt").read_bytes() == b"LINE1\r\nLINE2\r\nLINE3\r\n"


def test_write_file_does_not_translate_newlines(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)

    workspace.write_file("mixed.txt", "a\r\nb\n")

    assert (tmp_path / "mixed.txt").read_bytes() == b"a\r\nb\n"


def test_oversize_read_is_refused(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path, max_file_read_bytes=1000)
    (tmp_path / "big.txt").write_text("x" * 5000, encoding="utf-8")

    with pytest.raises(WorkspaceError, match="1000"):
        workspace.read_file("big.txt")


def test_binary_read_raises_a_workspace_error(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)
    (tmp_path / "img.bin").write_bytes(bytes(range(256)))

    with pytest.raises(WorkspaceError, match="UTF-8"):
        workspace.read_file("img.bin")
