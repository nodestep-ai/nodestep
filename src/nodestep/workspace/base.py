from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import Field

from nodestep.models.base import NodestepModel


class GrepResult(NodestepModel):
    """Matching lines by relative path, and the files that could not be read.

    Attributes
    ----------
    matches : dict[str, list[str]]
        Matching lines, keyed by the file's path relative to the root.
    skipped : list[str]
        Relative paths of files that are too large, not UTF-8 text, not
        readable, or symlinks that point outside the root.
    """

    matches: dict[str, list[str]] = Field(default_factory=dict)
    skipped: list[str] = Field(default_factory=list)


class Workspace(ABC):
    """Files an agent can read and change, confined to a root."""

    @property
    @abstractmethod
    def root_dir(self) -> Path:
        """Absolute root directory of the workspace."""
        ...

    @abstractmethod
    def resolve_path(self, path: str | None) -> tuple[Path, str]:
        """Resolve a path inside the workspace.

        Parameters
        ----------
        path : str or None
            Path relative to the root; ``None`` or ``"."`` is the root.

        Returns
        -------
        tuple[Path, str]
            Absolute path and normalized relative path.

        Raises
        ------
        PathAccessError
            If the path leaves the workspace root or breaks a rule of
            ``nodestep.workspace.paths.check_path``.
        """
        ...

    @abstractmethod
    def read_file(self, path: str) -> str:
        """Read a text file.

        Parameters
        ----------
        path : str
            Path relative to the workspace root.
        """
        ...

    @abstractmethod
    def write_file(self, path: str, content: str) -> None:
        """Write a text file, creating parent directories.

        Parameters
        ----------
        path : str
            Path relative to the workspace root.
        content : str
        """
        ...

    @abstractmethod
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
        """
        ...

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Whether a file or directory exists.

        Parameters
        ----------
        path : str
            Path relative to the workspace root.
        """
        ...

    @abstractmethod
    def delete(self, path: str) -> bool:
        """Delete a file or directory.

        Parameters
        ----------
        path : str
            Path relative to the workspace root.

        Returns
        -------
        bool
            Whether something was deleted.
        """
        ...

    @abstractmethod
    def edit(self, path: str, old: str, new: str, *, replace_all: bool = False) -> int:
        """Replace text in a file.

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
        """
        ...

    @abstractmethod
    def glob(self, pattern: str, path: str = ".") -> list[str]:
        """List files matching a glob pattern.

        Parameters
        ----------
        pattern : str
            Pattern such as ``"src/**/*.py"``; it may not leave the root.
        path : str, optional
            Directory to search from.

        Returns
        -------
        list[str]
            Matching paths relative to the root.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        """
        ...

    @abstractmethod
    def grep(
        self, pattern: str, path: str = ".", *, glob_pattern: str | None = None
    ) -> GrepResult:
        """Find lines matching a regular expression.

        Parameters
        ----------
        pattern : str
            Regular expression.
        path : str, optional
            File or directory to search.
        glob_pattern : str, optional
            Only search the files of a directory ``path`` that match this
            pattern; a file ``path`` is searched whatever the pattern.

        Returns
        -------
        GrepResult
            Matching lines keyed by relative path, and the relative paths of
            files skipped because they are too large, not UTF-8 text, not
            readable, or symlinks that point outside the root.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        """
        ...


__all__ = ["GrepResult", "Workspace"]
