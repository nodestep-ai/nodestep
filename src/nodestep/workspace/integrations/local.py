"""`LocalWorkspace`, which keeps an agent's files in a folder on disk.

See [Workspaces](../../../concepts/workspace.md#workspaces).
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from nodestep.exceptions import PathAccessError, WorkspaceError
from nodestep.workspace.base import GrepResult, Workspace
from nodestep.workspace.paths import check_glob_pattern, check_path


class LocalWorkspace(Workspace):
    """Workspace backed by a directory on disk.

    Paths, globs and symlinks cannot reach outside the root, and paths follow
    the rules of ``check_path`` listed in
    [Workspaces](../../../concepts/workspace.md#workspaces).

    Parameters
    ----------
    root_dir : str or Path
        Root directory.
    create : bool, optional
        Create ``root_dir`` and its parents if missing; otherwise a missing root
        raises.
    max_file_read_bytes : int, optional
        Largest file ``read_file`` accepts.

    Raises
    ------
    WorkspaceError
        If ``root_dir`` is missing and ``create`` is false, or is not a
        directory.
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        create: bool = False,
        max_file_read_bytes: int = 256_000,
    ) -> None:
        root = Path(root_dir)
        if create and not root.exists():
            root.mkdir(parents=True)
        if not root.exists():
            raise WorkspaceError(
                f"Workspace root '{root}' does not exist; pass create=True to create it"
            )
        if not root.is_dir():
            raise WorkspaceError(f"Workspace root '{root}' is not a directory")
        self._root_dir = root.resolve()
        self.max_file_read_bytes = max_file_read_bytes

    @property
    def root_dir(self) -> Path:
        """Absolute root directory."""
        return self._root_dir

    def resolve_path(self, path: str | None) -> tuple[Path, str]:
        """Resolve a path inside the root, following symlinks.

        Parameters
        ----------
        path : str or None
            Path relative to the root; ``None``, ``""`` or ``"."`` is the root.

        Returns
        -------
        tuple[Path, str]
            Absolute resolved path and the relative path with ``/`` separators.

        Raises
        ------
        PathAccessError
            If the path, or a symlink target in it, leaves the root, or the path
            fails ``check_path``.
        """
        root = self._root_dir
        if path not in (None, ".", ""):
            check_path(path)
        candidate = root if path in (None, ".", "") else (root / Path(path))
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise PathAccessError(
                f"Path '{path}' is outside the workspace root"
            ) from error
        return resolved, str(resolved.relative_to(root)).replace(os.sep, "/")

    def _inside(self, candidate: Path) -> bool:
        try:
            candidate.resolve().relative_to(self._root_dir)
        except ValueError:
            return False
        return True

    def _read_text(self, resolved: Path, path: str) -> str:
        size = resolved.stat().st_size
        if size > self.max_file_read_bytes:
            raise WorkspaceError(
                f"'{path}' is {size} bytes; the read limit is {self.max_file_read_bytes}"
            )
        try:
            with resolved.open(encoding="utf-8", newline="") as handle:
                return handle.read()
        except UnicodeDecodeError as error:
            raise WorkspaceError(f"'{path}' is not a UTF-8 text file") from error

    def _write_text(self, resolved: Path, content: str) -> None:
        with resolved.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)

    def read_file(self, path: str) -> str:
        """Read a UTF-8 text file, keeping its line endings.

        Parameters
        ----------
        path : str
            Path relative to the workspace root.

        Returns
        -------
        str

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        WorkspaceError
            If the file is larger than ``max_file_read_bytes`` or not UTF-8
            text.
        PathAccessError
            If the path leaves the root or fails ``check_path``.
        """
        resolved, _ = self.resolve_path(path)
        return self._read_text(resolved, path)

    def write_file(self, path: str, content: str) -> None:
        """Write a UTF-8 text file as given, creating parent directories.

        Parameters
        ----------
        path : str
            Path relative to the workspace root.
        content : str

        Raises
        ------
        IsADirectoryError
            If the path is a directory.
        PathAccessError
            If the path leaves the root or fails ``check_path``.
        """
        resolved, _ = self.resolve_path(path)
        if resolved.is_dir():
            raise IsADirectoryError(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._write_text(resolved, content)

    def list_dir(self, path: str = ".") -> list[str]:
        """List a directory; subdirectories end with ``/``.

        Parameters
        ----------
        path : str
            Path relative to the workspace root.

        Returns
        -------
        list[str]
            Sorted entry names.

        Raises
        ------
        FileNotFoundError
            If the path does not exist.
        NotADirectoryError
            If the path is a file.
        PathAccessError
            If the path leaves the root or fails ``check_path``.
        """
        resolved, _ = self.resolve_path(path)
        if not resolved.exists():
            raise FileNotFoundError(path)
        if not resolved.is_dir():
            raise NotADirectoryError(path)
        return sorted(
            entry.name + ("/" if entry.is_dir() else "") for entry in resolved.iterdir()
        )

    def exists(self, path: str) -> bool:
        """Whether a file or directory exists.

        Parameters
        ----------
        path : str
            Path relative to the workspace root.

        Returns
        -------
        bool

        Raises
        ------
        PathAccessError
            If the path leaves the root or fails ``check_path``.
        """
        resolved, _ = self.resolve_path(path)
        return resolved.exists()

    def delete(self, path: str) -> bool:
        """Delete a file, directory or symlink (never its target).

        Parameters
        ----------
        path : str
            Path relative to the workspace root.

        Returns
        -------
        bool
            Whether something was deleted.

        Raises
        ------
        WorkspaceError
            If ``path`` is the workspace root.
        PathAccessError
            If the path leaves the root or fails ``check_path``.
        """
        if path not in (".", ""):
            check_path(path)
        lexical = Path(os.path.normpath(self._root_dir / path))
        if lexical == self._root_dir:
            raise WorkspaceError("Refusing to delete the workspace root")
        try:
            lexical.relative_to(self._root_dir)
        except ValueError as error:
            raise PathAccessError(
                f"Path '{path}' is outside the workspace root"
            ) from error
        if not self._inside(lexical.parent):
            raise PathAccessError(f"Path '{path}' is outside the workspace root")
        if lexical.is_symlink():
            lexical.unlink()
            return True
        resolved, _ = self.resolve_path(path)
        if resolved.is_file():
            resolved.unlink()
            return True
        if resolved.is_dir():
            shutil.rmtree(resolved)
            return True
        return False

    def edit(self, path: str, old: str, new: str, *, replace_all: bool = False) -> int:
        """Replace text in a file, keeping its line endings.

        In a CRLF file, ``old`` and ``new`` written with LF match and are
        written with CRLF.

        Parameters
        ----------
        path : str
            Path relative to the workspace root.
        old : str
            Text to find.
        new : str
            Its replacement.
        replace_all : bool, optional
            Replace every occurrence instead of the first.

        Returns
        -------
        int
            Number of replacements.

        Raises
        ------
        ValueError
            If ``old`` is empty.
        FileNotFoundError
            If the file does not exist.
        WorkspaceError
            If the file is larger than ``max_file_read_bytes`` or not UTF-8
            text.
        PathAccessError
            If the path leaves the root or fails ``check_path``.
        """
        if not old:
            raise ValueError("edit needs non-empty old text")
        resolved, _ = self.resolve_path(path)
        content = self._read_text(resolved, path)
        if old not in content and "\r\n" in content and "\r\n" not in old:
            old = old.replace("\n", "\r\n")
            new = new.replace("\n", "\r\n")
        if replace_all:
            count = content.count(old)
            if count:
                self._write_text(resolved, content.replace(old, new))
            return count
        if old not in content:
            return 0
        self._write_text(resolved, content.replace(old, new, 1))
        return 1

    def glob(self, pattern: str, path: str = ".") -> list[str]:
        """List files matching a glob pattern, without symlinks leaving the root.

        Parameters
        ----------
        pattern : str
            Pattern such as ``"src/**/*.py"``, relative to ``path``.
        path : str, optional
            Directory to search from, relative to the root.

        Returns
        -------
        list[str]
            Sorted matching file paths relative to the root.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        PathAccessError
            If ``pattern`` names a drive, is absolute or contains ``..``, or
            ``path`` leaves the root or fails ``check_path``.
        """
        check_glob_pattern(pattern)
        resolved, _ = self.resolve_path(path)
        if not resolved.exists():
            raise FileNotFoundError(path)
        root = self._root_dir
        return sorted(
            str(file.relative_to(root)).replace(os.sep, "/")
            for file in resolved.glob(pattern)
            if file.is_file() and self._inside(file)
        )

    def grep(
        self,
        pattern: str,
        path: str = ".",
        *,
        glob_pattern: str | None = None,
    ) -> GrepResult:
        """Find lines matching a regular expression inside the root.

        Parameters
        ----------
        pattern : str
            Regular expression, searched in each line.
        path : str, optional
            File or directory to search, relative to the root.
        glob_pattern : str, optional
            Only search the files of a directory ``path`` that match it; a file
            ``path`` is always searched.

        Returns
        -------
        GrepResult
            Matching lines by relative path, and in ``skipped`` the files that
            are larger than ``max_file_read_bytes``, not UTF-8 text, not
            readable, or symlinks that point outside the root.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        re.error
            If ``pattern`` is not a valid regular expression.
        PathAccessError
            If ``glob_pattern`` names a drive, is absolute or contains ``..``,
            or ``path`` leaves the root or fails ``check_path``.
        """
        if glob_pattern:
            check_glob_pattern(glob_pattern)
        resolved, _ = self.resolve_path(path)
        if not resolved.exists():
            raise FileNotFoundError(path)
        regex = re.compile(pattern)
        files: list[Path]
        if resolved.is_file():
            files = [resolved]
        elif glob_pattern:
            files = [file for file in resolved.glob(glob_pattern) if file.is_file()]
        else:
            files = [file for file in resolved.rglob("*") if file.is_file()]
        result = GrepResult()
        for file_path in sorted(files):
            relative = str(file_path.relative_to(self._root_dir)).replace(os.sep, "/")
            if not self._inside(file_path):
                result.skipped.append(relative)
                continue
            try:
                text = self._read_text(file_path, relative)
            except (WorkspaceError, PermissionError):
                result.skipped.append(relative)
                continue
            matching = [line for line in text.splitlines() if regex.search(line)]
            if matching:
                result.matches[relative] = matching
        return result


__all__ = ["LocalWorkspace"]
