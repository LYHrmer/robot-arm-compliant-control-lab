"""Exercise offline installer isolation with real, tiny local wheels."""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def _wheel(directory, name, files):
    wheel = directory / f"{name}-1.0-py3-none-any.whl"
    metadata = f"{name}-1.0.dist-info"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, content in files.items():
            archive.writestr(path, content)
        archive.writestr(f"{metadata}/METADATA",
                         f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n")
        archive.writestr(f"{metadata}/WHEEL",
                         "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(f"{metadata}/RECORD", "")
    return wheel


def _release(directory):
    directory.mkdir()
    wheels = directory / "wheels"
    wheels.mkdir()
    dependency = _wheel(wheels, "installer_audit_dependency", {
        "installer_audit_dependency.py": "READY = True\n",
    })
    project = _wheel(wheels, "installer_audit_project", {
        "tools/__init__.py": "",
        "tools/learning_deployment/__init__.py": "",
        "tools/learning_deployment/__main__.py": (
            "import sys\n"
            "from pathlib import Path\n"
            "import installer_audit_dependency as dependency\n"
            "assert dependency.READY\n"
            "assert Path(dependency.__file__).is_relative_to(Path(sys.prefix))\n"
            "print('fixture verification: PASS')\n"
        ),
    })
    source = Path(__file__).resolve().parents[1] / "tools/learning_deployment/install_release.py"
    shutil.copyfile(source, directory / "install.py")
    digest = hashlib.sha256(dependency.read_bytes()).hexdigest()
    (directory / "runtime.lock").write_text(
        f"--only-binary :all:\ninstaller-audit-dependency==1.0 --hash=sha256:{digest}\n",
    )
    manifest = {
        "schema": "surface_offline_release_v1", "source_commit": "a" * 40,
        "python_version": platform.python_version(),
        "project_wheel": project.relative_to(directory).as_posix(),
        "bundle_manifest_sha256": "b" * 64,
        "artifact_sha256": {
            path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()
        },
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (directory / "COMPLETE").write_text(digest + "\n")
    return digest


@pytest.mark.skipif(platform.system() != "Linux" or platform.machine() != "x86_64",
                    reason="offline installer targets Linux x86_64")
@pytest.mark.parametrize("override", ["PIP_TARGET", "PIP_PREFIX", "user_config", "PIP_CONFIG_FILE"])
def test_installer_ignores_pip_environment_and_configuration(tmp_path, override):
    release = tmp_path / "release"
    digest = _release(release)
    venv = tmp_path / "venv"
    outside = tmp_path / "outside-venv"
    home = tmp_path / "home"
    config_home = home / ".config"
    config_home.mkdir(parents=True)
    environment = {key: value for key, value in os.environ.items() if not key.startswith("PIP_")}
    environment.update(HOME=str(home), XDG_CONFIG_HOME=str(config_home))
    if override in {"PIP_TARGET", "PIP_PREFIX"}:
        environment[override] = str(outside)
    else:
        config = config_home / "pip/pip.conf"
        config.parent.mkdir()
        config.write_text(f"[global]\ntarget = {outside}\n")
        if override == "PIP_CONFIG_FILE":
            explicit = home / "explicit-pip.conf"
            config.rename(explicit)
            environment[override] = str(explicit)
    result = subprocess.run([
        sys.executable, "-I", str(release / "install.py"), "--venv", str(venv),
        "--expected-release-sha256", digest,
    ], env=environment, capture_output=True, text=True, timeout=90, check=False)
    assert not outside.exists(), result.stdout + result.stderr
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fixture verification: PASS" in result.stdout
    assert json.loads(result.stdout.splitlines()[-1])["status"] == "PASS"
