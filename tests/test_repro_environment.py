from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from tools.ci import check_learning_junit, install_locked

ROOT = Path(__file__).resolve().parents[1]


def _pins(profile: str) -> dict[str, str]:
    text = (ROOT / f"environment/{profile}.lock").read_text()
    pins = dict(re.findall(r"^([\w.-]+)(?:\[[^]]+\])?==([^\s\\]+)", text, re.MULTILINE))
    for entry in text.replace("\\\n", "").splitlines():
        if entry == "--only-binary :all:":
            continue
        if entry and not entry.lstrip().startswith("#"):
            assert "--hash=sha256:" in entry
            assert "==" in entry or entry.startswith("torch @ https://download-r2.pytorch.org/")
    return pins


def test_cpu_lock_preserves_every_core_pin_and_has_no_cuda_packages():
    core = _pins("core")
    learning = _pins("learning-cpu")
    assert {name: learning.get(name) for name in core} == core
    assert {"numpy", "mujoco", "gymnasium", "matplotlib", "setuptools", "wheel", "pip"} <= core.keys()
    text = (ROOT / "environment/learning-cpu.lock").read_text()
    python_parts = (ROOT / "environment/python-version").read_text().strip().split(".")
    abi = "cp" + "".join(python_parts[:2])
    assert f"%2Bcpu-{abi}-{abi}-manylinux_2_28_x86_64.whl" in text
    assert not any(name.startswith(("nvidia-", "cuda", "triton")) for name in learning)


def test_lock_satisfies_direct_inputs_and_preserves_the_official_torch_url():
    core = _pins("core")
    for line in (ROOT / "environment/core.in").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        assert core[requirement.name] in requirement.specifier
    torch_input = next(
        line for line in (ROOT / "environment/learning-cpu.in").read_text().splitlines()
        if line.startswith("torch @ ")
    )
    assert torch_input in (ROOT / "environment/learning-cpu.lock").read_text()


def test_installer_enforces_hashes_and_disables_unpinned_build_dependencies():
    commands = install_locked.install_commands("core")
    assert "--require-hashes" in commands[0] and "--only-binary=:all:" in commands[0]
    assert "--no-deps" in commands[1] and "--no-build-isolation" in commands[1]
    assert commands[-1][-1] == "check"
    with pytest.raises(ValueError):
        install_locked.install_commands("gpu")


@pytest.mark.parametrize("problem", ["python", "platform", "glibc", "global"])
def test_installer_rejects_wrong_target(monkeypatch, problem):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(install_locked.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(install_locked.platform, "python_version", lambda: (
        "0.0.0" if problem == "python" else (ROOT / "environment/python-version").read_text().strip()
    ))
    monkeypatch.setattr(install_locked.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(install_locked.platform, "libc_ver", lambda: (
        "glibc", "2.17" if problem == "glibc" else "2.35"
    ))
    monkeypatch.setattr(install_locked.sys, "platform", "win32" if problem == "platform" else "linux")
    monkeypatch.setattr(install_locked.sys, "base_prefix", "/system")
    monkeypatch.setattr(install_locked.sys, "prefix", "/system" if problem == "global" else "/venv")
    with pytest.raises(RuntimeError):
        install_locked.check_target()


def test_installer_rejects_external_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/opt/ros/site-packages")
    with pytest.raises(RuntimeError, match="unset PYTHONPATH"):
        install_locked.check_target()


def _report(tmp_path, problem=None):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    required = (ROOT / "environment/learning-tests.txt").read_text().splitlines()
    for index, module in enumerate(required):
        if problem == "missing" and index == 0:
            continue
        case = ET.SubElement(suite, "testcase", {
            "classname": module.removesuffix(".py").replace("/", "."), "name": "test_example",
        })
        if problem in {"skipped", "failure", "error"} and index == 0:
            ET.SubElement(case, problem)
    path = tmp_path / "junit.xml"
    ET.ElementTree(root).write(path)
    return path


def test_learning_gate_accepts_all_required_modules(tmp_path):
    expected = len((ROOT / "environment/learning-tests.txt").read_text().splitlines())
    assert check_learning_junit.check_report(_report(tmp_path)) == expected


@pytest.mark.parametrize("problem", ["missing", "skipped", "failure", "error"])
def test_learning_gate_rejects_partial_execution(tmp_path, problem):
    with pytest.raises(ValueError, match="incomplete learning tests"):
        check_learning_junit.check_report(_report(tmp_path, problem))


def test_every_optional_torch_module_is_in_the_required_learning_suite():
    required = set((ROOT / "environment/learning-tests.txt").read_text().splitlines())
    needle = 'importorskip(' + '"torch")'
    optional = {
        str(path.relative_to(ROOT)) for path in (ROOT / "tests").glob("test_*.py")
        if needle in path.read_text()
    }
    assert optional <= required
    assert all((ROOT / path).is_file() for path in required)


def test_ci_separates_locked_compatibility_and_required_learning_jobs():
    workflow = (ROOT / ".github/workflows/tests.yml").read_text()
    for name in ("locked-core", "floating-compatibility", "learning-cpu"):
        assert f"  {name}:" in workflow
    assert workflow.count("python-version-file: environment/python-version") == 2
    assert workflow.count("timeout-minutes:") == 3
    assert "contents: read" in workflow
    assert "python tools/ci/check_learning_junit.py learning-junit.xml" in workflow
    assert "continue-on-error" not in workflow
