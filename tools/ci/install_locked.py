"""Install the validated Python environment into an explicit virtual environment."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def check_target() -> None:
    if os.environ.get("PYTHONPATH"):
        raise RuntimeError("unset PYTHONPATH before installing; external packages can shadow the lock")
    expected = (ROOT / "environment/python-version").read_text().strip()
    if platform.python_implementation() != "CPython" or platform.python_version() != expected:
        raise RuntimeError(f"locked environment requires CPython {expected}")
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("locked environment requires Linux x86_64")
    libc, version = platform.libc_ver()
    if libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 35):
        raise RuntimeError("locked environment requires glibc >= 2.35")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("create and activate a new virtual environment before installing")
    configuration = Path(sys.prefix) / "pyvenv.cfg"
    if not configuration.is_file() or "include-system-site-packages = false" not in (
        configuration.read_text().lower()
    ):
        raise RuntimeError("create a virtual environment without --system-site-packages")


def install_commands(profile: str) -> list[list[str]]:
    if profile not in {"core", "learning-cpu"}:
        raise ValueError(f"unknown profile: {profile}")
    pip = [sys.executable, "-m", "pip", "--disable-pip-version-check"]
    return [
        [*pip, "install", "--require-hashes", "--only-binary=:all:",
         "--retries", "2", "--timeout", "30", "-r", str(ROOT / f"environment/{profile}.lock")],
        [*pip, "install", "--no-deps", "--no-build-isolation", "-e", str(ROOT)],
        [*pip, "check"],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("core", "learning-cpu"), default="core")
    parser.add_argument("--check-only", action="store_true", help="validate target; do not install")
    args = parser.parse_args()
    check_target()
    for command in install_commands(args.profile):
        print("+", " ".join(command), flush=True)
        if not args.check_only:
            subprocess.run(command, check=True, timeout=900)


if __name__ == "__main__":
    main()
