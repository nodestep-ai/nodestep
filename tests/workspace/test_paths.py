from pathlib import Path

import pytest

from nodestep.exceptions import PathAccessError
from nodestep.workspace import InMemoryWorkspace, LocalWorkspace
from nodestep.workspace.paths import check_glob_pattern, check_path

REFUSED_PATHS = [
    "C:foo",
    "c:foo/bar.txt",
    "C:\\x",
    "C:/x",
    "\\\\server\\share\\x.txt",
    "//server/share/x.txt",
    "/etc/passwd",
    "\\windows\\system32",
    "notes.txt:secret",
    "dir/file.txt::$DATA",
    "a/b:c/d.txt",
    "CON",
    "con",
    "Prn.txt",
    "aux.tar.gz",
    "NUL",
    "nul .txt",
    "dir/COM1",
    "com9.log",
    "LPT1",
    "lpt9.txt",
    "name.",
    "name ",
    "dir./x.txt",
    "dir /x.txt",
    "...",
    "COM\u00b9",
    "lpt\u00b3.txt",
    "CONIN$",
    "conout$.log",
    "a<b.txt",
    "a>b",
    'say"hi".txt',
    "a|b",
    "what?.txt",
    "star*.txt",
    "tab\there.txt",
    "bell\x07",
    "dir/\x1f.txt",
]

ALLOWED_PATHS = [
    "notes.txt",
    "src/pkg/module.py",
    "a\\b.txt",
    "a/../b.txt",
    "./x.txt",
    "..",
    ".hidden",
    "a b/c d.txt",
    "console.txt",
    "nullable",
    "COM",
    "COM10",
    "LPT0",
    "auxiliary/x.txt",
    "COM\u2074",
    "CONIN",
    "price$.txt",
    "caf\u00e9.txt",
]


@pytest.mark.parametrize("path", REFUSED_PATHS)
def test_windows_unsafe_paths_are_refused(path: str) -> None:
    with pytest.raises(PathAccessError):
        check_path(path)


@pytest.mark.parametrize("path", ALLOWED_PATHS)
def test_portable_relative_paths_are_allowed(path: str) -> None:
    check_path(path)


@pytest.mark.parametrize(
    "pattern",
    ["C:*", "C:\\*.txt", "c:/**/*.py", "/etc/*", "\\*", "//server/share/*", "../*"],
)
def test_glob_patterns_with_a_drive_root_or_parent_are_refused(pattern: str) -> None:
    with pytest.raises(PathAccessError):
        check_glob_pattern(pattern)


@pytest.mark.parametrize("pattern", ["*.py", "src/**/*.py", "[!a]1.py", "a\\*.txt"])
def test_relative_glob_patterns_are_allowed(pattern: str) -> None:
    check_glob_pattern(pattern)


@pytest.mark.parametrize("path", ["C:foo", "notes.txt:secret", "nul.txt", "name."])
def test_workspaces_refuse_windows_unsafe_paths(tmp_path: Path, path: str) -> None:
    for workspace in (LocalWorkspace(tmp_path), InMemoryWorkspace()):
        with pytest.raises(PathAccessError):
            workspace.write_file(path, "x")
        with pytest.raises(PathAccessError):
            workspace.read_file(path)
        with pytest.raises(PathAccessError):
            workspace.delete(path)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("pattern", ["C:*", "/etc/*"])
def test_workspaces_refuse_rooted_glob_patterns(tmp_path: Path, pattern: str) -> None:
    for workspace in (LocalWorkspace(tmp_path), InMemoryWorkspace()):
        with pytest.raises(PathAccessError):
            workspace.glob(pattern)
        with pytest.raises(PathAccessError):
            workspace.grep("x", glob_pattern=pattern)
