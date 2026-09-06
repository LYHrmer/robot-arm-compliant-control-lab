"""The pytest console entry point does not automatically prepend the checkout."""

import os
import subprocess
import sys
from pathlib import Path


def test_repository_tools_are_importable_with_console_entrypoint_search_path(monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.setenv("MUJOCO_GL", "egl")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, sys; from pathlib import Path; "
                "assert os.environ['MUJOCO_GL'] == 'disable'; "
                "sys.path = [p for p in sys.path if Path(p).resolve() != Path.cwd()]; "
                "import pytest; raise SystemExit(pytest.console_main())"
            ),
            "--collect-only",
            "-q",
            "tests/test_surface_candidate_evaluation.py",
            "tests/test_surface_mlp_actor.py",
            "tests/test_surface_preparation_evidence.py",
            "tests/test_surface_repartition.py",
        ],
        cwd=root,
        env={
            **os.environ,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "MUJOCO_GL": "disable",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
