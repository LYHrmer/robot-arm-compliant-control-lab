"""Build a local offline release from the exact commit bound to a bundle."""

import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from tools.learning_deployment.bundle import verify_bundle
from tools.learning_deployment.common import output_path, seal, sha256


def package_release(bundle, expected_sha256, wheelhouse, output):
    manifest = verify_bundle(bundle, expected_sha256)
    root = Path(__file__).resolve().parents[2]
    output = output_path(output)
    if output.exists():
        raise FileExistsError("release output must be a new directory")
    commit = manifest["source_commit"]
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("bundle source commit is invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".learning-release-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        archive = temporary / "source.tar"
        subprocess.run(["git", "-C", str(root), "archive", "--format=tar", "--output", str(archive),
                        commit, "src", "tools", "environment/deployment-runtime.lock",
                        "pyproject.toml", "README.md", "LICENSE"], check=True)
        source = temporary / "source"
        source.mkdir()
        with tarfile.open(archive) as stream:
            for member in stream.getmembers():
                if not (member.isfile() or member.isdir()) or Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                    raise ValueError("source archive contains an unsafe entry")
            stream.extractall(source)  # Generated locally from the pinned Git tree; entries checked above.
        staging = temporary / "release"
        staging.mkdir()
        wheels = staging / "wheels"
        wheels.mkdir()
        lock = source / "environment/deployment-runtime.lock"
        wanted = set(re.findall(r"sha256:([0-9a-f]{64})", lock.read_text()))
        found = set()
        for wheel in sorted(Path(wheelhouse).glob("*.whl")):
            digest = sha256(wheel)
            if digest in wanted:
                shutil.copyfile(wheel, wheels / wheel.name)
                found.add(digest)
        if found != wanted or not wanted:
            raise ValueError("wheelhouse does not contain every hash-pinned runtime dependency")
        environment = dict(os.environ)
        # PYTHONPATH in development may be relative; preserve only build tooling
        # resolution when running the build from the clean archive directory.
        environment["PYTHONPATH"] = os.pathsep.join(str(Path(p).absolute()) for p in sys.path if p)
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-index", "--no-deps",
                        "--no-build-isolation", "--wheel-dir", str(wheels), str(source)],
                       env=environment, cwd=source, check=True)
        project_wheels = list(wheels.glob("compliant_control_lab-*.whl"))
        if len(project_wheels) != 1:
            raise ValueError("build must produce exactly one project wheel")
        shutil.copyfile(lock, staging / "runtime.lock")
        shutil.copyfile(source / "tools/learning_deployment/install_release.py", staging / "install.py")
        shutil.copytree(bundle, staging / "bundle")
        digest = seal(staging, {
            "schema": "surface_offline_release_v1", "source_commit": commit,
            "python_version": manifest["runtime_identity"]["runner"]["python_version"],
            "project_wheel": f"wheels/{project_wheels[0].name}",
            "bundle_manifest_sha256": expected_sha256,
            "target": "Linux_x86_64_MuJoCo_simulation_only",
        })
        if output.exists():
            raise FileExistsError("release output appeared during packaging")
        os.rename(staging, output)
    return {"output": str(output), "release_manifest_sha256": digest,
            "bundle_manifest_sha256": expected_sha256, "source_commit": commit}
