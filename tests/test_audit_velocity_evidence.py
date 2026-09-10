"""Tests for the read-only velocity-evidence audit; no child process ever runs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools import audit_velocity_evidence as mod

EXPECTED_STUDIES = (
    (
        "budget_transfer",
        "tools.budget_transfer",
        "results/franka_budget_transfer",
        "bce10bd63e2879ab9a7e89ed3375d3e971eefce8fb681e2ad57feb37d49e55c6",
        "screening_status",
        "FAIL",
    ),
    (
        "velocity_cost",
        "tools.velocity_cost_study",
        "results/franka_velocity_cost",
        "1179f95aff09ca82174abccdf7cd65b643050ba682f36d4f59509c6f249ab488",
        None,
        None,
    ),
    (
        "onset_observer",
        "tools.onset_observer_study",
        "results/franka_onset_observer",
        "e01db28864f04057d5272c4f0e469e2261c5166e5afe877afe2f00ef2d66a2ce",
        None,
        None,
    ),
    (
        "velocity_time",
        "tools.velocity_time_study",
        "results/franka_velocity_time",
        "803376ba2042d5ca8550fa6d2608676096cedd9678cabe8317d09220e4252f7d",
        "decision",
        "do_not_expand",
    ),
)

_MISSING = object()


def _native_report(study) -> dict:
    """Minimal successful component report: passing, unchanged defaults, archived counts."""
    report = {
        "audit_status": "PASS",
        "default_changed": False,
    }
    if study.decision_field == "screening_status":
        report["screening_status"] = "FAIL"
        report["new_executions"] = 60
    elif study.decision_field == "decision":
        report["decision"] = "do_not_expand"
        report["new_simulations"] = 4
    else:
        report["new_simulations"] = 8 if study.name == "onset_observer" else 0
    return report


def _budget_stdout(**changes) -> str:
    report = _native_report(mod.STUDIES[0])
    for key, value in changes.items():
        if value is _MISSING:
            report.pop(key)
        else:
            report[key] = value
    return json.dumps(report)


class FakeRunner:
    def __init__(self, outcomes: dict):
        self.outcomes = outcomes
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        outcome = self.outcomes[argv[3]]
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, subprocess.CompletedProcess):
            return outcome
        return subprocess.CompletedProcess(argv, 0, outcome, "")

    @property
    def modules(self) -> list[str]:
        return [argv[3] for argv, _ in self.calls]


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    """Fake four archives on disk and re-pin STUDIES to their real manifest hashes."""
    root = tmp_path / "repo"
    studies = []
    for study in mod.STUDIES:
        archive = root / study.directory
        archive.mkdir(parents=True)
        payload = json.dumps({"archive": study.name}, indent=2).encode()
        (archive / "manifest.json").write_bytes(payload)
        studies.append(dataclasses.replace(study, manifest_sha256=hashlib.sha256(payload).hexdigest()))
    studies = tuple(studies)
    runner = FakeRunner({s.module: json.dumps(_native_report(s)) for s in studies})
    monkeypatch.setattr(mod, "STUDIES", studies)
    monkeypatch.setattr(mod.subprocess, "run", runner)
    return root, studies, runner


def test_studies_are_exactly_the_four_published_pins():
    assert tuple(
        (s.name, s.module, s.directory, s.manifest_sha256, s.decision_field, s.expected_decision)
        for s in mod.STUDIES
    ) == EXPECTED_STUDIES
    assert all(len(s.manifest_sha256) == 64 for s in mod.STUDIES)
    with pytest.raises(dataclasses.FrozenInstanceError):
        mod.STUDIES[0].manifest_sha256 = "0" * 64


def test_success_aggregates_four_children_once(evidence):
    root, studies, runner = evidence
    report = mod.audit_evidence(root)
    assert report["scope"] == mod.SCOPE
    assert report["audit_status"] == "PASS"
    assert report["simulations_executed"] == 0
    assert report["default_changed"] is False
    assert runner.modules == [s.module for s in studies]
    entries = {e["name"]: e for e in report["audits"]}
    assert [e["name"] for e in report["audits"]] == [s.name for s in studies]
    assert all(e["audit_status"] == "PASS" and "error" not in e for e in report["audits"])
    assert entries["budget_transfer"]["experiment_decision"] == "FAIL"
    assert entries["velocity_time"]["experiment_decision"] == "do_not_expand"
    assert entries["velocity_cost"]["experiment_decision"] is None
    assert entries["onset_observer"]["experiment_decision"] is None
    assert entries["budget_transfer"]["archived_summary"]["new_executions"] == 60
    assert entries["onset_observer"]["archived_summary"]["new_simulations"] == 8
    assert entries["velocity_cost"]["archived_summary"] == _native_report(studies[1])
    assert entries["velocity_time"]["manifest_sha256"] == studies[3].manifest_sha256


def test_child_invocation_is_exact_and_read_only(evidence):
    root, studies, runner = evidence
    mod.audit_evidence(root, timeout_s=12.5)
    assert len(runner.calls) == 4
    for study, (argv, kwargs) in zip(studies, runner.calls):
        archive = root / study.directory
        assert argv == [sys.executable, "-B", "-m", study.module, "--audit", str(archive)]
        assert Path(argv[-1]).is_absolute()
        assert "--output" not in argv
        assert kwargs == {"cwd": root, "capture_output": True, "text": True,
                          "timeout": 12.5, "check": False}


@pytest.mark.parametrize(
    "breakage,fragment",
    [("hash", "sha256"), ("missing", "missing evidence archive")],
)
def test_pin_failure_blocks_only_its_own_child(evidence, tmp_path, breakage, fragment):
    root, studies, runner = evidence
    study = studies[1]
    archive = root / study.directory
    if breakage == "hash":
        (archive / "manifest.json").write_bytes(b'{"tampered": true}')
    else:
        archive.rename(tmp_path / "moved_away")
    report = mod.audit_evidence(root)
    entry = report["audits"][1]
    assert report["audit_status"] == "FAIL"
    assert "default_changed" not in report
    assert entry["audit_status"] == "FAIL"
    assert entry["experiment_decision"] is None and entry["archived_summary"] is None
    assert fragment in entry["error"]
    assert runner.modules == [s.module for s in studies if s.module != study.module]


@pytest.mark.parametrize("kind", ["archive", "manifest"])
def test_symlinked_evidence_is_rejected(evidence, tmp_path, kind):
    root, studies, runner = evidence
    archive = root / studies[0].directory
    manifest = archive / "manifest.json"
    if kind == "archive":
        real = tmp_path / "real_archive"
        archive.rename(real)
        archive.symlink_to(real, target_is_directory=True)
    else:
        real = tmp_path / "real_manifest.json"
        real.write_bytes(manifest.read_bytes())
        manifest.unlink()
        manifest.symlink_to(real)
    report = mod.audit_evidence(root)
    assert report["audit_status"] == "FAIL"
    assert "symlink" in report["audits"][0]["error"]
    assert runner.modules == [s.module for s in studies[1:]]


@pytest.mark.parametrize(
    "outcome,fragment",
    [
        (subprocess.TimeoutExpired(["python"], 7.0), "timed out"),
        (subprocess.CompletedProcess(["python"], 3, "", "child exploded"), "child exploded"),
        (subprocess.CompletedProcess(["python"], 3, "", "   "), "no stderr"),
    ],
)
def test_child_failure_is_recorded_and_others_continue(evidence, outcome, fragment):
    root, studies, runner = evidence
    runner.outcomes[studies[0].module] = outcome
    report = mod.audit_evidence(root, timeout_s=7.0)
    entry = report["audits"][0]
    assert report["audit_status"] == "FAIL" and "default_changed" not in report
    assert entry["audit_status"] == "FAIL" and fragment in entry["error"]
    assert entry["archived_summary"] is None
    assert runner.modules == [s.module for s in studies]
    assert all(e["audit_status"] == "PASS" for e in report["audits"][1:])


@pytest.mark.parametrize(
    "stdout,fragment",
    [
        ("this is not json", "JSON"),
        ('{"audit_status": "PASS"} {"audit_status": "PASS"}', "JSON"),
        ("", "JSON"),
        ('[{"audit_status": "PASS"}]', "not a JSON object"),
        ('"PASS"', "not a JSON object"),
        (_budget_stdout(drift=float("nan")), "non-finite"),
        (_budget_stdout(drift=float("inf")), "non-finite"),
        ('{"audit_status": "PASS", "default_changed": false, "drift": 1e999}', "non-finite"),
        (_budget_stdout(nested={"inner": [float("nan")]}), "non-finite"),
        (_budget_stdout(audit_status=_MISSING), "audit_status"),
        (_budget_stdout(audit_status="FAIL"), "audit_status"),
        (_budget_stdout(audit_status="pass"), "audit_status"),
        (_budget_stdout(default_changed=_MISSING), "default_changed"),
        (_budget_stdout(default_changed=True), "default_changed"),
        (_budget_stdout(default_changed=0), "default_changed"),
        (_budget_stdout(default_changed="false"), "default_changed"),
        (_budget_stdout(screening_status=_MISSING), "screening_status"),
        (_budget_stdout(screening_status="MAYBE"), "screening_status"),
        (_budget_stdout(screening_status="PASS"), "frozen decision"),
        (_budget_stdout(screening_status=None), "screening_status"),
    ],
)
def test_unusable_component_report_fails_closed(evidence, stdout, fragment):
    root, studies, runner = evidence
    runner.outcomes[studies[0].module] = stdout
    report = mod.audit_evidence(root)
    entry = report["audits"][0]
    assert report["audit_status"] == "FAIL" and "default_changed" not in report
    assert entry["audit_status"] == "FAIL" and entry["archived_summary"] is None
    assert fragment in entry["error"]
    assert all(e["audit_status"] == "PASS" for e in report["audits"][1:])


@pytest.mark.parametrize("value", ["expand_now", "expand_original_grid", _MISSING])
def test_time_study_decision_is_also_screened(evidence, value):
    root, studies, runner = evidence
    bad = _native_report(studies[3])
    if value is _MISSING:
        bad.pop("decision")
    else:
        bad["decision"] = value
    runner.outcomes[studies[3].module] = json.dumps(bad)
    report = mod.audit_evidence(root)
    assert report["audit_status"] == "FAIL"
    assert "decision" in report["audits"][3]["error"]


@pytest.mark.parametrize("timeout", [0, -1.0, float("nan"), float("inf")])
def test_audit_evidence_rejects_invalid_timeout(evidence, timeout):
    root, _, runner = evidence
    with pytest.raises(ValueError, match="timeout"):
        mod.audit_evidence(root, timeout_s=timeout)
    assert runner.calls == []


def test_main_prints_valid_json_and_returns_zero(evidence, monkeypatch, capsys):
    root, _, _ = evidence
    monkeypatch.setattr(mod, "ROOT", root)
    assert mod.main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["audit_status"] == "PASS"
    assert payload["simulations_executed"] == 0
    assert payload["default_changed"] is False
    assert len(payload["audits"]) == 4


def test_main_returns_one_on_failure(evidence, monkeypatch, capsys):
    root, studies, _ = evidence
    (root / studies[2].directory / "manifest.json").write_bytes(b"{}")
    monkeypatch.setattr(mod, "ROOT", root)
    assert mod.main([]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["audit_status"] == "FAIL"
    assert "default_changed" not in payload


@pytest.mark.parametrize("value", ["0", "-5", "nan", "inf", "soon"])
def test_main_rejects_invalid_timeout_argument(evidence, monkeypatch, value):
    root, _, runner = evidence
    monkeypatch.setattr(mod, "ROOT", root)
    with pytest.raises(SystemExit) as excinfo:
        mod.main(["--timeout-seconds", value])
    assert excinfo.value.code == 2
    assert runner.calls == []


def test_keyboard_interrupt_propagates(evidence):
    root, studies, runner = evidence
    runner.outcomes[studies[0].module] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        mod.audit_evidence(root)
    assert runner.modules == [studies[0].module]


def test_relative_root_is_resolved_before_child_invocation(evidence, monkeypatch):
    root, _, runner = evidence
    monkeypatch.chdir(root.parent)
    assert mod.audit_evidence(Path(root.name))["audit_status"] == "PASS"
    assert all(kwargs["cwd"] == root for _, kwargs in runner.calls)
    assert all(Path(argv[-1]).is_absolute() for argv, _ in runner.calls)


def test_symlink_above_root_blocks_every_child(evidence, tmp_path):
    root, _, runner = evidence
    alias = tmp_path / "alias"
    alias.symlink_to(root.parent, target_is_directory=True)
    report = mod.audit_evidence(alias / root.name)
    assert report["audit_status"] == "FAIL"
    assert all("symlink" in entry["error"] for entry in report["audits"])
    assert runner.calls == []


def test_child_failure_omits_traceback_and_bounds_error(evidence):
    root, studies, runner = evidence
    stderr = "warning\nTraceback (most recent call last):\n  File ...\nValueError: " + "x" * 2500
    runner.outcomes[studies[0].module] = subprocess.CompletedProcess([], 1, "", stderr)
    report = mod.audit_evidence(root)
    error = report["audits"][0]["error"]
    assert "ValueError" in error and "Traceback" not in error
    assert len(error) <= mod.MAX_ERROR_CHARS


def test_registered_pins_match_published_manifests():
    for study in mod.STUDIES:
        manifest = mod.ROOT / study.directory / "manifest.json"
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() == study.manifest_sha256


def test_documented_entrypoints_include_the_new_read_only_command():
    for relative in ("README.md", "docs/README.md", "docs/recruiter_walkthrough.md",
                     "docs/velocity_evidence_audit.md"):
        text = (mod.ROOT / relative).read_text(encoding="utf-8")
        assert "python -m tools.audit_velocity_evidence" in text, relative
    assert "franka-published-results-audit" in (mod.ROOT / "README.md").read_text(encoding="utf-8")
