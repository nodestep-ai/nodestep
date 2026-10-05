"""`InMemoryWorkspace`, which keeps an agent's files in memory.

See [Workspaces](../../../concepts/workspace.md#workspaces).
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path

from nodestep.exceptions import PathAccessError, WorkspaceError
from nodestep.workspace.base import GrepResult, Workspace
from nodestep.workspace.paths import check_glob_pattern, check_path

_VIRTUAL_ROOT = Path("/virtual")


def _segment_to_regex(segment: str) -> str:
    regex, index = "", 0
    while index < len(segment):
        character = segment[index]
        if character == "*":
            regex += "[^/]*"
        elif character == "?":
            regex += "[^/]"
        elif character == "[" and (end := segment.find("]", index + 1)) != -1:
            body = segment[index + 1 : end].replace("\\", "\\\\")
            if body.startswith("!"):
                body = "^" + body[1:]
            regex += f"[{body}]"
            index = end
        else:
            regex += re.escape(character)
        index += 1
    return regex


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    parts = pattern.split("/")
    if parts[-1] == "**":
        return re.compile("(?!)")
    regex = "".join(
        "(?:[^/]+/)*" if part == "**" else _segment_to_regex(part) + "/"
        for part in parts[:-1]
    )
    return re.compile(f"^{regex}{_segment_to_regex(parts[-1])}$")


class InMemoryWorkspace(Workspace):
    """Workspace kept in memory, with the same rules as ``LocalWorkspace``."""

    def __init__(self) -> None:
        self._files: dict[str, str] = {}

    @property
    def root_dir(self) -> Path:
        """Not available; the workspace has no directory on disk.

        Raises
        ------
        WorkspaceError
        """
        raise WorkspaceError("InMemoryWorkspace has no filesystem root directory")

    def resolve_path(self, path: str | None) -> tuple[Path, str]:
        """Normalize a path, refusing anything outside the root.

        Parameters
        ----------
        path : str or None
            Path relative to the root; ``None``, ``""`` or ``"."`` is the root.

        Returns
        -------
        tuple[Path, str]
            The path under the placeholder root ``/virtual`` and the
            normalized relative path.

        Raises
        ------
        PathAccessError
            If the path leaves the root or fails ``check_path``.
        """
        if path in (None, ".", ""):
            return _VIRTUAL_ROOT, "."
        check_path(path)
        raw = path.replace("\\", "/")
        normalized = posixpath.normpath(raw)
        if raw.startswith("/") or normalized == ".." or normalized.startswith("../"):
            raise PathAccessError(f"Path '{path}' is outside the workspace root")
        if normalized == ".":
            return _VIRTUAL_ROOT, "."
        return _VIRTUAL_ROOT / normalized, normalized

    def _is_dir(self, key: str) -> bool:
        if key == ".":
            return True
        prefix = key.rstrip("/") + "/"
        return any(existing.startswith(prefix) for existing in self._files)

    def read_file(self, path: str) -> str:
        """Read a file.

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
        IsADirectoryError
            If the path is a directory.
        PathAccessError
            If the path leaves the root or fails ``check_path``.
        """
        _, key = self.resolve_path(path)
        if key not in self._files:
            if self._is_dir(key):
                raise IsADirectoryError(path)
            raise FileNotFoundError(path)
        return self._files[key]

    def write_file(self, path: str, content: str) -> None:
        """Write a file; parent directories exist as soon as a file is in them.

        Parameters
        ----------
        path : str
            Path relative to the workspace root.
        content : str

        Raises
        ------
        IsADirectoryError
            If the path is a directory.
        FileExistsError
            If a parent of the path is a file.
        PathAccessError
            If the path leaves the root or fails ``check_path``.
        """
        _, key = self.resolve_path(path)
        if self._is_dir(key):
            raise IsADirectoryError(path)
        parents = key.split("/")[:-1]
        for depth in range(1, len(parents) + 1):
            if "/".join(parents[:depth]) in self._files:
                raise FileExistsError(path)
        self._files[key] = content

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
        _, relative = self.resolve_path(path)
        if relative in self._files:
            raise NotADirectoryError(path)
        if not self._is_dir(relative):
            raise FileNotFoundError(path)
        prefix = "" if relative == "." else relative.rstrip("/") + "/"
        entries: set[str] = set()
        for key in self._files:
            if prefix and not key.startswith(prefix):
                continue
            rest = key[len(prefix) :]
            if not rest:
                continue
            first, _, _ = rest.partition("/")
            if "/" in rest:
                entries.add(first + "/")
            else:
                entries.add(first)
        return sorted(entries)

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
        _, key = self.resolve_path(path)
        return key in self._files or self._is_dir(key)

    def delete(self, path: str) -> bool:
        """Delete a file or directory; the root cannot be deleted.

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
        _, key = self.resolve_path(path)
        if key == ".":
            raise WorkspaceError("Refusing to delete the workspace root")
        if key in self._files:
            del self._files[key]
            return True
        directory_prefix = key.rstrip("/") + "/"
        to_remove = [
            file_key
            for file_key in self._files
            if file_key.startswith(directory_prefix)
        ]
        if not to_remove:
            return False
        for file_key in to_remove:
            del self._files[file_key]
        return True

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
        PathAccessError
            If the path leaves the root or fails ``check_path``.
        """
        if not old:
            raise ValueError("edit needs non-empty old text")
        _, key = self.resolve_path(path)
        if key not in self._files:
            raise FileNotFoundError(path)
        content = self._files[key]
        if old not in content and "\r\n" in content and "\r\n" not in old:
            old = old.replace("\n", "\r\n")
            new = new.replace("\n", "\r\n")
        if replace_all:
            count = content.count(old)
            if count:
                self._files[key] = content.replace(old, new)
            return count
        if old not in content:
            return 0
        self._files[key] = content.replace(old, new, 1)
        return 1

    def glob(self, pattern: str, path: str = ".") -> list[str]:
        """List files matching a glob pattern.

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
        _, relative = self.resolve_path(path)
        if relative not in self._files and not self._is_dir(relative):
            raise FileNotFoundError(path)
        prefix = "" if relative == "." else relative.rstrip("/") + "/"
        regex = _glob_to_regex(pattern)
        return sorted(
            file_key
            for file_key in self._files
            if file_key.startswith(prefix) and regex.match(file_key[len(prefix) :])
        )

    def grep(
        self,
        pattern: str,
        path: str = ".",
        *,
        glob_pattern: str | None = None,
    ) -> GrepResult:
        """Find lines matching a regular expression.

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
            Matching lines by relative path; ``skipped`` is always empty.

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
        _, relative = self.resolve_path(path)
        if relative not in self._files and not self._is_dir(relative):
            raise FileNotFoundError(path)
        regex = re.compile(pattern)
        if relative in self._files:
            candidates = {relative: self._files[relative]}
        else:
            prefix = "" if relative == "." else relative.rstrip("/") + "/"
            keys = (
                self.glob(glob_pattern, path)
                if glob_pattern
                else [
                    file_key for file_key in self._files if file_key.startswith(prefix)
                ]
            )
            candidates = {key: self._files[key] for key in keys}
        result = GrepResult()
        for key, content in sorted(candidates.items()):
            matching = [line for line in content.splitlines() if regex.search(line)]
            if matching:
                result.matches[key] = matching
        return result


__all__ = ["InMemoryWorkspace"]
