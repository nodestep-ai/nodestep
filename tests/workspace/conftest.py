import os
from pathlib import Path

import pytest


@pytest.fixture
def symlinks(tmp_path: Path) -> None:
    probe = tmp_path / "symlink-probe"
    try:
        os.symlink(tmp_path, probe, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"this OS cannot create symlinks here: {error}")
    probe.unlink()
