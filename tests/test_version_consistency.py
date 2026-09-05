from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _required_match(pattern: str, text: str, source: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise AssertionError(f"version is missing from {source}")
    return match.group(1)


def test_python_and_cpp_project_versions_match() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    python_version = _required_match(r'^version = "([0-9]+\.[0-9]+\.[0-9]+)"$', pyproject, "pyproject")
    cpp_version = _required_match(r"project\([^\n]* VERSION ([0-9]+\.[0-9]+\.[0-9]+)", cmake, "CMake")

    assert python_version == cpp_version == "0.5.1"
