from __future__ import annotations

import re
from pathlib import PureWindowsPath

from nodestep.exceptions import PathAccessError

_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"{device}{index}" for device in ("COM", "LPT") for index in range(1, 10)}
    | {
        f"{device}{digit}"
        for device in ("COM", "LPT")
        for digit in "\u00b9\u00b2\u00b3"
    }
)
_INVALID_CHARACTERS = re.compile(r'[<>:"|?*\x00-\x1f]')
_SEPARATORS = re.compile(r"[\\/]")


def _segments(path: str) -> list[str]:
    return [segment for segment in _SEPARATORS.split(path) if segment]


def _check_anchor(path: str, what: str) -> None:
    windows = PureWindowsPath(path)
    if windows.drive:
        raise PathAccessError(f"{what} '{path}' names a drive or network share")
    if windows.root:
        raise PathAccessError(f"{what} '{path}' is absolute; use a relative path")


def check_path(path: str) -> None:
    """Refuse a workspace path that is not portable and relative.

    The rules follow Windows, on every OS.

    Parameters
    ----------
    path : str
        Path relative to the workspace root, with ``/`` or ``\\`` separators.

    Raises
    ------
    PathAccessError
        If the path is absolute, names a drive or UNC share, contains ``:``
        (an NTFS alternate data stream), one of ``<>"|?*`` or a control
        character, uses a reserved device name such as ``CON``, ``nul.txt``,
        ``COM1`` or ``CONIN$``, or has a segment ending in a dot or a space.
    """
    _check_anchor(path, "Path")
    for segment in _segments(path):
        if segment in (".", ".."):
            continue
        invalid = _INVALID_CHARACTERS.search(segment)
        if invalid is not None:
            raise PathAccessError(
                f"Path '{path}' contains {invalid.group()!r} in {segment!r}"
            )
        if segment.split(".", 1)[0].rstrip(" ").upper() in _RESERVED_NAMES:
            raise PathAccessError(
                f"Path '{path}' uses the reserved device name '{segment}'"
            )
        if segment.endswith((".", " ")):
            raise PathAccessError(
                f"Path '{path}' has the segment '{segment}' ending in a dot or space"
            )


def check_glob_pattern(pattern: str) -> None:
    """Refuse a glob pattern that could leave the workspace root.

    Parameters
    ----------
    pattern : str
        Glob pattern relative to the workspace root.

    Raises
    ------
    PathAccessError
        If the pattern is absolute, names a drive or UNC share, or has a
        ``..`` segment.
    """
    _check_anchor(pattern, "Glob pattern")
    if ".." in _segments(pattern):
        raise PathAccessError(
            f"Glob pattern '{pattern}' may not leave the workspace root"
        )


__all__ = ["check_glob_pattern", "check_path"]
