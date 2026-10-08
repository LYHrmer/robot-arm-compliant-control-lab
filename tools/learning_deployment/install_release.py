"""Standalone offline installer copied into a versioned local simulation release."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path, PurePosixPath


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_release(root, expected):
    manifest_path = root / "manifest.json"
    if digest(manifest_path) != expected:
        raise ValueError("release manifest differs from the supplied SHA256")
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "surface_offline_release_v1":
        raise ValueError("unsupported release schema")
    for name, value in manifest["artifact_sha256"].items():
        relative = PurePosixPath(name)
        path = root / name
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
            raise ValueError("unsafe release path")
        if any(p.is_symlink() for p in (path, *path.parents)) or digest(path) != value:
            raise ValueError(f"release artifact changed: {name}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != {*manifest["artifact_sha256"], "manifest.json", "COMPLETE"}:
        raise ValueError("release inventory differs")
    if (root / "COMPLETE").read_text().strip() != expected:
        raise ValueError("incomplete release")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--accept-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    try:
        manifest = validate_release(root, args.expected_release_sha256)
        if (platform.system() != "Linux" or platform.machine() != "x86_64"
                or platform.python_version() != manifest["python_version"]):
            raise ValueError("release requires Linux x86_64 and Python " + manifest["python_version"])
        target = args.venv.absolute()
        if any(p.is_symlink() for p in (target, *target.parents)):
            raise ValueError("venv path must not traverse a symlink")
        if target == root or target in root.parents or root in target.parents:
            raise ValueError("venv must be separate from the immutable release")
        marker = target / ".learning-release-sha256"
        if target.exists() and (not marker.is_file() or marker.read_text().strip() != args.expected_release_sha256):
            raise ValueError("refusing an existing environment belonging to another release")
        if args.dry_run:
            print(json.dumps({"status": "DRY_RUN", "writes": False, "network": False,
                              "source_commit": manifest["source_commit"], "venv": str(target)}))
            return 0
        environment = {key: value for key, value in os.environ.items()
                       if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}}
        environment.update(OPENBLAS_CORETYPE="Haswell", OPENBLAS_NUM_THREADS="1",
                           OMP_NUM_THREADS="1", MUJOCO_GL="disable", PYTHONNOUSERSITE="1")
        if not target.exists():
            target.mkdir(parents=True)
            marker.write_text(args.expected_release_sha256 + "\n")
        # Re-running venv is safe for this marked, release-specific environment.
        subprocess.run([sys.executable, "-I", "-m", "venv", str(target)], env=environment, check=True)
        python = target / "bin/python"
        subprocess.run([str(python), "-I", "-m", "pip", "install", "--no-index", "--require-hashes",
                        "--find-links", str(root / "wheels"), "-r", str(root / "runtime.lock")],
                       env=environment, check=True)
        subprocess.run([str(python), "-I", "-m", "pip", "install", "--no-index", "--no-deps",
                        str(root / manifest["project_wheel"])], env=environment, check=True)
        subprocess.run([str(python), "-I", "-m", "pip", "check"], env=environment, check=True)
        command = [str(python), "-I", "-m", "tools.learning_deployment"]
        pinned = ["--bundle", str(root / "bundle"), "--expected-manifest-sha256",
                  manifest["bundle_manifest_sha256"]]
        subprocess.run([*command, "verify", *pinned, "--profile"], env=environment, check=True)
        if args.accept_output:
            completed = subprocess.run([*command, "accept", *pinned, "--output", str(args.accept_output.absolute())],
                                       env=environment, check=False)
            if completed.returncode:
                return completed.returncode
        print(json.dumps({"status": "PASS", "venv": str(target), "source_commit": manifest["source_commit"],
                          "rollback": "use the previous release and its separate virtual environment"}))
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
