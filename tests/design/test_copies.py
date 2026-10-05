from importlib.resources import files
from pathlib import Path

import pytest

STYLESHEET = Path(__file__).resolve().parents[3] / "nodestep-stylesheet"
STATIC = files("nodestep_sandbox").joinpath("static")


def test_the_sandbox_copies_match_the_stylesheet_repository() -> None:
    if not STYLESHEET.is_dir():
        pytest.skip("nodestep-stylesheet is not checked out next to nodestep")
    for name in ("nodestep.css", "nodestep-theme.js", "nodestep-data.js"):
        assert STATIC.joinpath(name).read_bytes() == (STYLESHEET / name).read_bytes(), (
            name
        )
