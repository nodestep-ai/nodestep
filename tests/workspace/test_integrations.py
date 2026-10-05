from pathlib import Path

import pytest

from nodestep.exceptions import PathAccessError, WorkspaceError
from nodestep.workspace import GrepResult, InMemoryWorkspace, LocalWorkspace


def test_local_workspace_imports_from_workspace_package(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)
    assert workspace.root_dir == tmp_path.resolve()


def test_local_workspace_rejects_parent_path(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)
    with pytest.raises(PathAccessError):
        workspace.resolve_path("../secret.txt")


def test_local_workspace_resolves_valid_path(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)
    resolved, relative = workspace.resolve_path("subdir/file.txt")
    assert resolved == tmp_path / "subdir" / "file.txt"
    assert relative == "subdir/file.txt"


def test_local_workspace_resolves_dot_path(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)
    resolved, _ = workspace.resolve_path(".")
    assert resolved == tmp_path.resolve()


def test_local_workspace_resolves_none_path(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)
    resolved, _ = workspace.resolve_path(None)
    assert resolved == tmp_path.resolve()


def test_local_workspace_read_write(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)
    workspace.write_file("hello.txt", "world")
    assert workspace.read_file("hello.txt") == "world"
    assert workspace.exists("hello.txt")


def test_local_workspace_list_dir(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)
    workspace.write_file("a.txt", "a")
    workspace.write_file("sub/b.txt", "b")
    entries = workspace.list_dir()
    assert "a.txt" in entries
    assert "sub/" in entries


def test_local_workspace_exists_missing(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)
    assert not workspace.exists("nope.txt")


def test_inmemory_workspace_read_write() -> None:
    workspace = InMemoryWorkspace()
    workspace.write_file("hello.txt", "world")
    assert workspace.read_file("hello.txt") == "world"


def test_inmemory_workspace_exists() -> None:
    workspace = InMemoryWorkspace()
    assert not workspace.exists("missing.txt")
    workspace.write_file("doc.md", "# hi")
    assert workspace.exists("doc.md")


def test_inmemory_workspace_list_dir() -> None:
    workspace = InMemoryWorkspace()
    workspace.write_file("a.txt", "a")
    workspace.write_file("b.txt", "b")
    workspace.write_file("sub/c.txt", "c")
    entries = workspace.list_dir()
    assert entries == ["a.txt", "b.txt", "sub/"]


def test_inmemory_workspace_list_subdir() -> None:
    workspace = InMemoryWorkspace()
    workspace.write_file("sub/c.txt", "c")
    workspace.write_file("sub/d.txt", "d")
    entries = workspace.list_dir("sub")
    assert entries == ["c.txt", "d.txt"]


def test_inmemory_workspace_read_missing_raises() -> None:
    workspace = InMemoryWorkspace()
    with pytest.raises(FileNotFoundError):
        workspace.read_file("nope.txt")


def test_inmemory_workspace_rejects_parent_path() -> None:
    workspace = InMemoryWorkspace()
    with pytest.raises(PathAccessError):
        workspace.resolve_path("../secret")


def test_inmemory_workspace_exists_directory() -> None:
    workspace = InMemoryWorkspace()
    workspace.write_file("dir/file.txt", "x")
    assert workspace.exists("dir")


def test_local_workspace_delete_file(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)
    workspace.write_file("f.txt", "data")
    assert workspace.delete("f.txt") is True
    assert not workspace.exists("f.txt")


def test_local_workspace_delete_missing(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)
    assert workspace.delete("nope.txt") is False


def test_inmemory_workspace_delete_file() -> None:
    workspace = InMemoryWorkspace()
    workspace.write_file("f.txt", "data")
    assert workspace.delete("f.txt") is True
    assert not workspace.exists("f.txt")


def test_inmemory_workspace_delete_directory() -> None:
    workspace = InMemoryWorkspace()
    workspace.write_file("d/a.txt", "a")
    workspace.write_file("d/b.txt", "b")
    assert workspace.delete("d") is True
    assert not workspace.exists("d/a.txt")


def test_inmemory_workspace_delete_missing() -> None:
    workspace = InMemoryWorkspace()
    assert workspace.delete("nope.txt") is False


def test_local_workspace_edit_single(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)
    workspace.write_file("f.txt", "foo bar foo")
    assert workspace.edit("f.txt", "foo", "baz") == 1
    assert workspace.read_file("f.txt") == "baz bar foo"


def test_local_workspace_edit_replace_all(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)
    workspace.write_file("f.txt", "foo bar foo")
    assert workspace.edit("f.txt", "foo", "baz", replace_all=True) == 2
    assert workspace.read_file("f.txt") == "baz bar baz"


def test_local_workspace_edit_no_match(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)
    workspace.write_file("f.txt", "hello")
    assert workspace.edit("f.txt", "missing", "x") == 0
    assert workspace.read_file("f.txt") == "hello"


def test_inmemory_workspace_edit_single() -> None:
    workspace = InMemoryWorkspace()
    workspace.write_file("f.txt", "foo bar foo")
    assert workspace.edit("f.txt", "foo", "baz") == 1
    assert workspace.read_file("f.txt") == "baz bar foo"


def test_inmemory_workspace_edit_replace_all() -> None:
    workspace = InMemoryWorkspace()
    workspace.write_file("f.txt", "foo bar foo")
    assert workspace.edit("f.txt", "foo", "baz", replace_all=True) == 2
    assert workspace.read_file("f.txt") == "baz bar baz"


def test_inmemory_workspace_edit_missing_file() -> None:
    workspace = InMemoryWorkspace()
    with pytest.raises(FileNotFoundError):
        workspace.edit("nope.txt", "a", "b")


def test_local_workspace_glob(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)
    workspace.write_file("a.py", "x")
    workspace.write_file("b.txt", "x")
    workspace.write_file("sub/c.py", "x")
    assert workspace.glob("*.py") == ["a.py"]
    assert workspace.glob("**/*.py") == ["a.py", "sub/c.py"]


def test_inmemory_workspace_glob() -> None:
    workspace = InMemoryWorkspace()
    workspace.write_file("a.py", "x")
    workspace.write_file("b.txt", "x")
    workspace.write_file("sub/c.py", "x")
    assert workspace.glob("*.py") == ["a.py"]


def test_inmemory_workspace_glob_subdir() -> None:
    workspace = InMemoryWorkspace()
    workspace.write_file("src/a.py", "x")
    workspace.write_file("src/b.txt", "x")
    assert workspace.glob("*.py", "src") == ["src/a.py"]


def test_local_workspace_grep(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)
    workspace.write_file("a.py", "import os\nimport sys\n")
    workspace.write_file("b.txt", "no match\n")
    result = workspace.grep("import").matches
    assert "a.py" in result
    assert len(result["a.py"]) == 2
    assert "b.txt" not in result


def test_local_workspace_grep_with_glob(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)
    workspace.write_file("a.py", "import os\n")
    workspace.write_file("b.txt", "import sys\n")
    result = workspace.grep("import", glob_pattern="*.py").matches
    assert "a.py" in result
    assert "b.txt" not in result


def test_inmemory_workspace_grep() -> None:
    workspace = InMemoryWorkspace()
    workspace.write_file("a.py", "import os\nimport sys\n")
    workspace.write_file("b.txt", "no match\n")
    result = workspace.grep("import").matches
    assert "a.py" in result
    assert len(result["a.py"]) == 2
    assert "b.txt" not in result


def test_inmemory_workspace_grep_with_glob() -> None:
    workspace = InMemoryWorkspace()
    workspace.write_file("a.py", "import os\n")
    workspace.write_file("b.txt", "import sys\n")
    result = workspace.grep("import", glob_pattern="*.py").matches
    assert "a.py" in result
    assert "b.txt" not in result


def test_local_workspace_requires_a_root() -> None:
    with pytest.raises(TypeError):
        LocalWorkspace()  # ty: ignore[missing-argument]


def test_local_workspace_missing_root_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(WorkspaceError, match="create=True"):
        LocalWorkspace(missing)

    assert not missing.exists()


def test_local_workspace_creates_the_root_when_asked(tmp_path: Path) -> None:
    root = tmp_path / "a" / "b"

    workspace = LocalWorkspace(root, create=True)

    assert root.is_dir()
    assert workspace.root_dir == root.resolve()


def test_local_workspace_root_must_be_a_directory(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")

    with pytest.raises(WorkspaceError, match="not a directory"):
        LocalWorkspace(tmp_path / "file.txt", create=True)


def test_local_workspace_has_no_unused_settings(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        LocalWorkspace(tmp_path, python_executable="python")  # ty: ignore[unknown-argument]


def test_workspace_context_is_removed() -> None:
    import nodestep.workspace

    assert not hasattr(nodestep.workspace, "WorkspaceContext")


def test_local_grep_reports_files_it_could_not_read(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path, max_file_read_bytes=100)
    workspace.write_file("ok.txt", "needle\n")
    (tmp_path / "big.txt").write_text("needle\n" * 100, encoding="utf-8")
    (tmp_path / "latin1.txt").write_bytes("needle \xe9\n".encode("latin-1"))

    result = workspace.grep("needle")

    assert result == GrepResult(
        matches={"ok.txt": ["needle"]}, skipped=["big.txt", "latin1.txt"]
    )


def test_inmemory_grep_skips_nothing() -> None:
    workspace = InMemoryWorkspace()
    workspace.write_file("a.txt", "needle\n")

    assert workspace.grep("needle") == GrepResult(
        matches={"a.txt": ["needle"]}, skipped=[]
    )


def _workspaces(tmp_path: Path) -> list[LocalWorkspace | InMemoryWorkspace]:
    local = LocalWorkspace(tmp_path)
    memory = InMemoryWorkspace()
    for workspace in (local, memory):
        workspace.write_file("src/a.py", "needle\n")
    return [local, memory]


def test_glob_on_a_missing_path_raises(tmp_path: Path) -> None:
    for workspace in _workspaces(tmp_path):
        with pytest.raises(FileNotFoundError, match="scr"):
            workspace.glob("*.py", "scr")


def test_grep_on_a_missing_path_raises(tmp_path: Path) -> None:
    for workspace in _workspaces(tmp_path):
        with pytest.raises(FileNotFoundError, match="scr"):
            workspace.grep("needle", "scr")
        with pytest.raises(FileNotFoundError, match="scr"):
            workspace.grep("needle", "scr", glob_pattern="*.py")


def test_glob_and_grep_on_an_empty_root_find_nothing(tmp_path: Path) -> None:
    for workspace in (LocalWorkspace(tmp_path), InMemoryWorkspace()):
        assert workspace.glob("*.py") == []
        assert workspace.grep("needle") == GrepResult()


def test_glob_and_grep_on_existing_paths_still_work(tmp_path: Path) -> None:
    for workspace in _workspaces(tmp_path):
        assert workspace.glob("*.py", "src") == ["src/a.py"]
        assert workspace.grep("needle", "src").matches == {"src/a.py": ["needle"]}
        assert workspace.grep("needle", "src/a.py").matches == {"src/a.py": ["needle"]}


@pytest.mark.parametrize("kind", ["local", "memory"])
def test_grep_on_a_file_searches_it_whatever_the_glob(
    tmp_path: Path, kind: str
) -> None:
    workspace = LocalWorkspace(tmp_path) if kind == "local" else InMemoryWorkspace()
    workspace.write_file("a.py", "hello\n")

    for glob_pattern in (None, "*.py", "*.md"):
        result = workspace.grep("hello", "a.py", glob_pattern=glob_pattern)
        assert result.matches == {"a.py": ["hello"]}


def test_local_workspace_root_is_not_expanded_with_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "home" / "project").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(WorkspaceError, match="does not exist"):
        LocalWorkspace("~/project")
