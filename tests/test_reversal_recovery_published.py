"""Bind recovery claims to all four published pairs, including their costs."""

import json
from pathlib import Path

import numpy as np

from tools.reversal_recovery import study

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/franka_reversal_recovery"


def test_published_recovery_archive_replays_without_default_promotion():
    result = study.audit_archive(ARCHIVE)
    assert result["archive_integrity"] == "PASS"
    assert result["comparison_status"] == "PASS"
    assert result["new_simulations"] == 8
    assert result["default_changed"] is False
    manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    assert manifest["source_commit"] == "64176beffd1a21b0d86b14a92254cc424f7a2674"
    assert manifest["new_holdout"] is False


def test_guide_retains_each_falling_seed_and_high_load_costs():
    report = json.loads((ARCHIVE / "comparison.json").read_text())
    guide = (ROOT / "docs/reversal_recovery.md").read_text()
    assert len(report["runs"]) == 8 and len(report["comparisons"]) == 4
    assert sum(row["audit"]["validated_cycles"] for row in report["runs"]) == 48_000
    assert report["eligible_for_default_change"] is False
    for row in report["runs"]:
        if row["scenario"] == "falling":
            for window in row["windows"]:
                if window["name"] in {"early", "post"}:
                    assert f"{window['tangent_rmse_mm']:.3f}" in guide
    for pair in report["comparisons"]:
        assert pair["status"] == "PASS"
        if pair["scenario"] == "constant_high":
            ramp = next(row for row in pair["window_deltas"] if row["name"] == "ramp")
            for metric in ("tangent_rmse_mm", "tangent_velocity_rmse_mm_s"):
                assert ramp[metric] > 0
                assert f"{ramp[metric]:.3f}" in guide
    assert "不是与默认固定 6 N 的新比较" in guide
    assert "没有重跑仿真" in guide


def test_each_pair_has_the_same_complete_prefix_before_braking():
    for scenario in ("falling", "constant_high"):
        for seed in (11, 29):
            prefix = ARCHIVE / "traces" / f"{scenario}_seed{seed}"
            with (
                np.load(f"{prefix}__adaptive6_8.npz", allow_pickle=False) as baseline,
                np.load(f"{prefix}__hold_cap_tracking.npz", allow_pickle=False) as candidate,
            ):
                mask = baseline["time"] < 4.5
                assert int(mask.sum()) == 2250
                fields = [name for name in baseline.files
                          if baseline[name].ndim and baseline[name].shape[0] == 6000]
                assert len(fields) == 69
                for name in fields:
                    np.testing.assert_array_equal(baseline[name][mask], candidate[name][mask])
