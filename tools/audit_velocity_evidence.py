"""Read-only audit of the evidence chain behind the four published velocity-cost studies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCOPE = "velocity_cost_evidence_chain"
MAX_ERROR_CHARS = 2000


@dataclass(frozen=True)
class Study:
    name: str
    module: str
    directory: str
    manifest_sha256: str
    decision_field: str | None = None
    expected_decision: str | None = None


STUDIES = (
    Study(
        "budget_transfer",
        "tools.budget_transfer",
        "results/franka_budget_transfer",
        "bce10bd63e2879ab9a7e89ed3375d3e971eefce8fb681e2ad57feb37d49e55c6",
        "screening_status",
        "FAIL",
    ),
    Study(
        "velocity_cost",
        "tools.velocity_cost_study",
        "results/franka_velocity_cost",
        "1179f95aff09ca82174abccdf7cd65b643050ba682f36d4f59509c6f249ab488",
    ),
    Study(
        "onset_observer",
        "tools.onset_observer_study",
        "results/franka_onset_observer",
        "e01db28864f04057d5272c4f0e469e2261c5166e5afe877afe2f00ef2d66a2ce",
    ),
    Study(
        "velocity_time",
        "tools.velocity_time_study",
        "results/franka_velocity_time",
        "803376ba2042d5ca8550fa6d2608676096cedd9678cabe8317d09220e4252f7d",
        "decision",
        "do_not_expand",
    ),
)


def _concise(text: str) -> str:
    """Explain a component failure on one line, without a traceback."""
    message = " ".join(text.split())
    if len(message) > MAX_ERROR_CHARS:
        return message[: MAX_ERROR_CHARS - 3] + "..."
    return message


def _reject_symlinks(path: Path) -> None:
    for current in (path, *path.parents):
        if current.is_symlink():
            raise ValueError(f"symlinked evidence path is not auditable: {current}")


def _reject_non_finite(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _reject_non_finite(item)
    elif isinstance(value, list):
        for item in value:
            _reject_non_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("component report contains a non-finite number")


def _reject_json_constant(name: str) -> float:
    raise ValueError(f"component report contains the non-finite constant {name}")


def _verify_manifest_pin(study: Study, archive: Path) -> None:
    """Pin the archive identity before any child process is allowed to run."""
    _reject_symlinks(archive)
    manifest = archive / "manifest.json"
    if manifest.is_symlink():
        raise ValueError(f"symlinked manifest is not auditable: {manifest}")
    if not archive.is_dir():
        raise FileNotFoundError(f"missing evidence archive: {study.directory}")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest != study.manifest_sha256:
        raise ValueError(
            f"manifest sha256 mismatch: expected {study.manifest_sha256}, found {digest}"
        )


def _run_component_audit(study: Study, archive: Path, root: Path, timeout_s: float) -> str:
    try:
        completed = subprocess.run(
            [sys.executable, "-B", "-m", study.module, "--audit", str(archive)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError(f"{study.module} --audit timed out after {timeout_s:g}s") from error
    if completed.returncode != 0:
        # Python failures may contain import warnings and a full traceback.
        lines = [line.strip() for line in completed.stderr.splitlines() if line.strip()]
        raise ValueError(
            f"{study.module} --audit exited with code {completed.returncode}: "
            f"{lines[-1] if lines else '(no stderr)'}"
        )
    return completed.stdout


def _parse_component_report(study: Study, stdout: str) -> dict[str, Any]:
    try:
        report = json.loads(stdout, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as error:
        raise ValueError(f"{study.module} --audit printed no single JSON object: {error}") from error
    if not isinstance(report, dict):
        raise TypeError(f"{study.module} --audit report is not a JSON object")
    _reject_non_finite(report)
    if report.get("audit_status") != "PASS":
        raise ValueError(f"component audit_status is {report.get('audit_status')!r}, not PASS")
    if report.get("default_changed") is not False:
        raise ValueError(
            f"component default_changed is {report.get('default_changed')!r}, expected false"
        )
    if study.decision_field is not None:
        decision = report.get(study.decision_field)
        if decision != study.expected_decision:
            raise ValueError(
                f"component {study.decision_field} is {decision!r}, "
                f"expected frozen decision {study.expected_decision!r}"
            )
    return report


def _audit_component(study: Study, root: Path, timeout_s: float) -> dict[str, Any]:
    archive = root / study.directory
    entry: dict[str, Any] = {
        "name": study.name,
        "archive": study.directory,
        "manifest_sha256": study.manifest_sha256,
    }
    try:
        _verify_manifest_pin(study, archive)
        stdout = _run_component_audit(study, archive, root, timeout_s)
        report = _parse_component_report(study, stdout)
    except (OSError, TypeError, ValueError, subprocess.SubprocessError) as error:
        entry["audit_status"] = "FAIL"
        entry["experiment_decision"] = None
        entry["archived_summary"] = None
        entry["error"] = _concise(str(error) or type(error).__name__)
        return entry
    entry["audit_status"] = "PASS"
    entry["experiment_decision"] = report[study.decision_field] if study.decision_field else None
    entry["archived_summary"] = report
    return entry


def audit_evidence(root: Path, timeout_s: float = 300.0) -> dict[str, Any]:
    """Audit the four specified studies through their own auditors; runs no simulation."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"timeout must be a positive finite number of seconds: {timeout_s!r}")
    root = Path(root).absolute()
    audits = [_audit_component(study, root, timeout_s) for study in STUDIES]
    passed = all(entry["audit_status"] == "PASS" for entry in audits)
    report: dict[str, Any] = {
        "scope": SCOPE,
        "audit_status": "PASS" if passed else "FAIL",
        "simulations_executed": 0,
    }
    if passed:
        report["default_changed"] = False
    report["audits"] = audits
    return report


def _positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number of seconds: {value}") from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError(f"timeout must be positive and finite: {value}")
    return seconds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=_positive_seconds, default=300.0)
    args = parser.parse_args(argv)
    report = audit_evidence(ROOT, timeout_s=args.timeout_seconds)
    for entry in report["audits"]:
        print(f"{entry['name']}: {entry['audit_status']}", file=sys.stderr)
    print(json.dumps(report, indent=2))
    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
