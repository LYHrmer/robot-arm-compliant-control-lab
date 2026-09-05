from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from compliant_control_lab.franka_control import FrankaHybridController
from compliant_control_lab.franka_simulation import (
    DEFAULT_FRANKA_SCENARIOS,
    FrankaSimulationConfig,
    run_franka_trial,
)
from compliant_control_lab.published_results_audit import PublishedResultAudit
from compliant_control_lab.smoke import SmokeReport, _validate_trial, run_smoke


def _archive() -> PublishedResultAudit:
    return PublishedResultAudit(
        protocol_sha256="abc",
        artifact_count=4,
        row_count=384,
        method_count=8,
        case_count=48,
        pass_counts={"torque_residual_run_00": 24},
        primary_rule_passed=False,
    )


def test_report_keeps_smoke_status_separate_from_frozen_decision() -> None:
    report = SmokeReport(
        archive=_archive(),
        simulation_steps=1000,
        force_rmse_n=2.4,
        peak_force_n=31.2,
        saturation_pct=0.0,
    )

    rendered = report.render()

    assert "archive: PASS" in rendered
    assert "frozen_decision=FAIL" in rendered
    assert rendered.endswith("smoke: PASS")


def test_validate_trial_rejects_nonfinite_signal() -> None:
    trial = run_franka_trial(
        FrankaHybridController(),
        DEFAULT_FRANKA_SCENARIOS[0],
        FrankaSimulationConfig(duration=0.004),
    )
    invalid = replace(trial, raw_normal_force=np.full_like(trial.raw_normal_force, np.nan))

    with pytest.raises(ValueError, match="raw_normal_force"):
        _validate_trial(invalid)


def test_run_smoke_rejects_invalid_duration() -> None:
    with pytest.raises(ValueError, match="duration"):
        run_smoke(Path("unused"), Path("unused"), duration=0.0)
