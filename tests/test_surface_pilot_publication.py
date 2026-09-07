import json
import shutil
from pathlib import Path

import pytest

from tools import publish_surface_learning_pilot as publication

ARCHIVE = Path(__file__).resolve().parents[1] / "results" / "franka_surface_learning_pilot"


def test_published_pilot_has_all_seeds_and_recomputable_selection():
    result = publication.audit_public_pilot(ARCHIVE)
    assert result["hashes_match"] and result["selection_recomputed"]
    assert result["all_physics_traces_distributed"] is False
    assert result["new_holdout"] is False
    assert result["distributed_file_count"] > 30
    for stage in ("bc", "ppo"):
        summary = publication._read(ARCHIVE / f"{stage}_summary.json")
        assert [item["seed"] for item in summary["development_test"]] == [11, 29, 47]
        assert all(len(item["report"]["runs"]) == 4 for item in summary["development_test"])


def test_exported_bc_reproduced_offline_mse_but_does_not_imply_closed_loop_success():
    check = publication._read(ARCHIVE / "bc_export_check.json")
    assert check["passed"] and len(check["runs"]) == 3
    assert max(row["absolute_difference"] for row in check["runs"]) < check["absolute_tolerance"]
    report = publication._read(ARCHIVE / "bc_summary.json")
    assert all(item["report"]["all_episodes_succeeded"] for item in report["validation"])
    assert not report["all_selected_development_gates_met"]


def test_public_hash_check_rejects_missing_or_modified_file(tmp_path):
    original = publication._read(ARCHIVE / "manifest.json")
    publication._write_new(tmp_path / "manifest.json", original)
    (tmp_path / "COMPLETE").write_text(publication._sha256(tmp_path / "manifest.json"))
    with pytest.raises((FileNotFoundError, ValueError)):
        publication.audit_public_pilot(tmp_path)


def test_missing_metrics_cannot_count_as_tracking_success(tmp_path):
    row = {
        "method": "failed",
        "episode_success": False,
        "contact_ratio_pct": None,
        "tangent_rmse_mm": None,
        "force_rmse_n": None,
    }
    summary = publication._write_comparison(tmp_path, [row])["failed"]
    assert summary["safe_completions"] == summary["tracking_passes"] == 0
    assert summary["mean_tangent_rmse_mm"] is None
    assert summary["minimum_contact_ratio_pct"] is None


def test_safe_but_poor_tracking_is_not_a_tracking_pass(tmp_path):
    row = {
        "method": "safe_only",
        "episode_success": True,
        "contact_ratio_pct": 100.0,
        "tangent_rmse_mm": 15.0,
        "force_rmse_n": 0.2,
    }
    summary = publication._write_comparison(tmp_path, [row])["safe_only"]
    assert summary["safe_completions"] == 1
    assert summary["tracking_passes"] == 0


def _reseal(directory):
    manifest = publication._read(directory / "manifest.json")
    manifest["artifact_sha256"] = {
        name: publication._sha256(directory / name) for name in manifest["artifact_sha256"]
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "COMPLETE").write_text(publication._sha256(directory / "manifest.json"))


@pytest.mark.parametrize("split", ["validation", "development_test"])
def test_resealed_missing_cohort_cannot_hide_behind_zip(tmp_path, split):
    directory = tmp_path / "archive"
    shutil.copytree(ARCHIVE, directory)
    summary = publication._read(directory / "ppo_summary.json")
    summary[split].pop()
    if split == "validation":
        selection = publication._read(directory / "ppo_selection.json")
        selection[split] = summary[split]
        (directory / "ppo_selection.json").write_text(json.dumps(selection))
        summary["selection_sha256"] = publication._sha256(directory / "ppo_selection.json")
    (directory / "ppo_summary.json").write_text(json.dumps(summary))
    (directory / "ppo_COMPLETE").write_text(publication._sha256(directory / "ppo_summary.json"))
    _reseal(directory)
    with pytest.raises(ValueError, match="evaluation cohort omits a seed"):
        publication.audit_public_pilot(directory)


def test_publication_rejects_unlisted_extra_file(tmp_path):
    directory = tmp_path / "archive"
    shutil.copytree(ARCHIVE, directory)
    (directory / "unlisted.txt").write_text("not part of distributed evidence")
    with pytest.raises(ValueError, match="unlisted files"):
        publication.audit_public_pilot(directory)


def test_resealed_stale_plan_marker_is_rejected(tmp_path):
    directory = tmp_path / "archive"
    shutil.copytree(ARCHIVE, directory)
    (directory / "PLAN_SHA256").write_text("0" * 64)
    _reseal(directory)
    with pytest.raises(ValueError, match="plan SHA differs"):
        publication.audit_public_pilot(directory)
