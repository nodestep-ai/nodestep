from pathlib import Path

import pytest

from nodestep.exceptions import PathAccessError, WorkspaceError
from nodestep.state.integrations import FilesystemStateStore
from nodestep.workspace import GrepResult, InMemoryWorkspace, LocalWorkspace, Workspace


@pytest.fixture(params=["local", "virtual"])
def workspace(request: pytest.FixtureRequest, tmp_path: Path) -> Workspace:
    if request.param == "local":
        return LocalWorkspace(tmp_path)
    return InMemoryWorkspace()


def _seed(workspace: Workspace, files: dict[str, str]) -> None:
    for path, content in files.items():
        workspace.write_file(path, content)


def test_double_star_matches_whole_directory_segments(workspace: Workspace) -> None:
    _seed(
        workspace,
        {"src/pkg/test_a.py": "", "src/footest_x.py": "", "src/test_top.py": ""},
    )

    assert workspace.glob("src/**/test_*.py") == [
        "src/pkg/test_a.py",
        "src/test_top.py",
    ]


def test_character_classes_are_supported(workspace: Workspace) -> None:
    _seed(workspace, {"a1.py": "", "b1.py": ""})

    assert workspace.glob("[a]1.py") == ["a1.py"]
    assert workspace.glob("[!a]1.py") == ["b1.py"]


def test_grep_accepts_a_file_path(workspace: Workspace) -> None:
    _seed(workspace, {"main.py": "import os\nx = 1\n", "other.py": "import sys\n"})

    assert workspace.grep("import", path="main.py") == GrepResult(
        matches={"main.py": ["import os"]}, skipped=[]
    )


def test_the_root_exists(workspace: Workspace) -> None:
    assert workspace.exists(".")


def test_writing_over_a_directory_fails(workspace: Workspace) -> None:
    _seed(workspace, {"dir/inner.txt": "x"})

    with pytest.raises(IsADirectoryError):
        workspace.write_file("dir", "y")


def test_writing_below_a_file_fails(workspace: Workspace) -> None:
    _seed(workspace, {"f.txt": "x"})

    with pytest.raises(FileExistsError):
        workspace.write_file("f.txt/child", "y")


def test_dot_dot_that_stays_inside_is_allowed(workspace: Workspace) -> None:
    _seed(workspace, {"a/b/c.txt": "x"})

    assert workspace.read_file("a/b/../b/c.txt") == "x"


@pytest.mark.parametrize("path", ["../x.txt", "a/../../x.txt", "/etc/passwd"])
def test_paths_leaving_the_root_are_rejected(workspace: Workspace, path: str) -> None:
    with pytest.raises(PathAccessError):
        workspace.read_file(path)


def test_deleting_the_root_is_refused(workspace: Workspace) -> None:
    with pytest.raises(WorkspaceError):
        workspace.delete(".")


def test_crlf_round_trips(workspace: Workspace) -> None:
    workspace.write_file("w.txt", "a\r\nb")

    assert workspace.read_file("w.txt") == "a\r\nb"


def test_in_memory_workspace_has_no_filesystem_root() -> None:
    with pytest.raises(WorkspaceError):
        _ = InMemoryWorkspace().root_dir


def test_filesystem_store_takes_only_a_path() -> None:
    with pytest.raises(TypeError):
        FilesystemStateStore(InMemoryWorkspace())  # ty: ignore[invalid-argument-type]


def test_list_dir_of_a_missing_directory_raises(workspace: Workspace) -> None:
    with pytest.raises(FileNotFoundError):
        workspace.list_dir("typo")


def test_list_dir_of_a_file_raises(workspace: Workspace) -> None:
    _seed(workspace, {"f.txt": "x"})

    with pytest.raises(NotADirectoryError):
        workspace.list_dir("f.txt")


@pytest.mark.parametrize("replace_all", [False, True])
def test_edit_with_empty_old_text_raises(
    workspace: Workspace, replace_all: bool
) -> None:
    _seed(workspace, {"f.txt": "abc"})

    with pytest.raises(ValueError, match="old"):
        workspace.edit("f.txt", "", "X", replace_all=replace_all)

    assert workspace.read_file("f.txt") == "abc"
