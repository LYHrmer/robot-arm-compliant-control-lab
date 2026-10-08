"""Immutable manifests, explicit source identity and local output boundaries."""

import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

from compliant_control_lab.surface_experiment import _output_path
from tools.surface_mlp_actor import current_runner_identity


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def output_path(path, *, allow_existing=False):
    path = Path(path)
    if allow_existing and path.is_dir():
        # Validate a never-written child so the frozen-tree and symlink checks
        # still apply, while allowing a workspace with completed stages.
        return _output_path(path / ".deployment-path-check").parent
    return _output_path(path)


def runtime_identity():
    directory = Path(__file__).resolve().parent
    tools = directory.parent
    paths = [*directory.glob("*.py"), *(tools / name for name in (
        "train_surface_bc.py", "train_surface_ppo.py", "surface_learning_pilot.py",
    ))]
    return {
        "runner": current_runner_identity(),
        "deployment_sources": {
            path.relative_to(tools).as_posix(): sha256(path) for path in sorted(paths)
        },
    }


def committed_source():
    """Preparation is run from a committed checkout, never an untraceable tree."""
    root = Path(__file__).resolve().parents[2]
    command = ["git", "-C", str(root)]
    try:
        head = subprocess.check_output([*command, "rev-parse", "HEAD"], text=True).strip()
        changes = subprocess.check_output([
            *command, "status", "--porcelain", "--untracked-files=all", "--",
            "src", "tools", "environment", "pyproject.toml",
        ], text=True)
    except subprocess.CalledProcessError as error:
        raise ValueError("prepare requires a committed source checkout") from error
    if changes.strip():
        raise ValueError("commit source/config changes before prepare; --dry-run remains available")
    return head


def seal(directory, metadata):
    directory = Path(directory)
    hashes = {p.relative_to(directory).as_posix(): sha256(p)
              for p in sorted(directory.rglob("*")) if p.is_file()}
    write_new(directory / "manifest.json", {**metadata, "artifact_sha256": hashes})
    digest = sha256(directory / "manifest.json")
    (directory / "COMPLETE").write_text(digest + "\n", encoding="utf-8")
    return digest


def verify_sealed(directory, *, expected_sha256=None):
    """Check inventory and every byte before interpreting any artifact."""
    root = Path(directory)
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("archive path must not traverse a symlink")
    entries = list(root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("archive contains a symlink")
    digest = sha256(root / "manifest.json")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("manifest differs from the externally pinned SHA256")
    if (root / "COMPLETE").read_text().strip() != digest:
        raise ValueError("COMPLETE does not bind manifest")
    manifest = read_json(root / "manifest.json")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("manifest has no artifact inventory")
    for name, expected in hashes.items():
        path = PurePosixPath(name)
        if (not isinstance(expected, str) or len(expected) != 64
                or set(expected) - set("0123456789abcdef")
                or path.is_absolute() or ".." in path.parts or path.as_posix() != name):
            raise ValueError("unsafe artifact path or digest")
        if sha256(root / name) != expected:
            raise ValueError(f"artifact changed: {name}")
    actual = {p.relative_to(root).as_posix() for p in entries if p.is_file()}
    if actual != {*hashes, "manifest.json", "COMPLETE"}:
        raise ValueError("unexpected or missing archive files")
    return manifest
