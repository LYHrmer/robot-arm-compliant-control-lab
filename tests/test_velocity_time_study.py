"""Synthetic publisher tests; these never run robot physics or modify archives."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from tools import velocity_time_study as study


def _row(scale, budget, time):
    row = {key: 0.0 for key in study.transfer.screening.PUBLIC_LIMITS}
    row.update(
        rotation_gain_scale=scale,
        max_force_n=budget,
        velocity_error_time_s=time,
        tangent_velocity_error_rms_m_s=0.004 + 0.0001 * (budget == 8.0),
        contact_ratio_pct=100.0,
        peak_force_n=12.0,
        saturation_pct=0.0,
        projection_pct=0.0,
        minimum_reserved_torque_headroom_nm=4.0,
    )
    if time == 0.10:
        row["tangent_velocity_error_rms_m_s"] -= 0.00005 * (budget == 8.0)
    return row


def _rows():
    return [
        _row(scale, budget, time)
        for scale in (1.0, 2.0)
        for budget in (6.0, 8.0)
        for time in (0.05, 0.10)
    ]


def _tables():
    return {
        "comparison": _rows(),
        "catchup": [{"row": index} for index in range(16)],
        "observer_windows": [{"row": index} for index in range(32)],
        "events": [{"row": index} for index in range(4)],
    }


def _fake_collect(directory, *, execute):
    if execute:
        for _, _, _, time, name in study.specifications():
            if time == 0.10:
                path = Path(directory) / name
                np.savez_compressed(path, velocity_error_time_s=np.array(time))
    return deepcopy(_tables())


@pytest.fixture
def archive(tmp_path, monkeypatch):
    sources = {"tools/source.py": "a" * 64}
    monkeypatch.setattr(study, "input_archive", dict)
    monkeypatch.setattr(study, "source_identity", lambda: sources.copy())
    monkeypatch.setattr(study, "collect", _fake_collect)
    monkeypatch.setattr(
        study.runner,
        "run_trial",
        lambda *_args: pytest.fail("publisher fixture must not run physics"),
    )
    return study.generate(tmp_path / "velocity-time")


def _reseal(root):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"] = {
        name: study._sha256(root / name) for name in manifest["artifact_sha256"]
    }
    study._write_json(manifest_path, manifest)
    (root / "COMPLETE").write_text(study._sha256(manifest_path) + "\n", encoding="utf-8")


def test_specifications_freeze_four_new_and_four_reference_rows():
    specs = list(study.specifications())

    assert len(specs) == 8
    assert {(s, b, t) for _, s, b, t, _ in specs} == {
        (s, b, t) for s in (1.0, 2.0) for b in (6.0, 8.0) for t in (0.05, 0.10)
    }
    assert all(case["case_index"] == 23 for case, *_ in specs)
    assert len({name for *_, name in specs}) == 8


def test_protocol_freezes_parent_counts_and_diagnostic_catchup():
    protocol = study.protocol_document()

    assert protocol["reference"] == study.REFERENCE
    assert protocol["new_simulations"] == protocol["reused_traces"] == 4
    assert protocol["evaluated_rows"] == 8
    assert protocol["only_changed_parameter"] == "velocity_error_time"
    assert protocol["velocity_error_times_s"] == [0.05, 0.10]
    assert protocol["decision"]["per_arm_nonworse"] == [
        "tangent_velocity_error_rms_m_s"
    ]
    assert protocol["decision"]["position_regression_limit_mm"] == 0.10
    assert protocol["decision"]["catchup_statistics_are_diagnostic"] is True
    assert protocol["default_changed"] is protocol["new_holdout"] is False


def test_constructor_documents_change_only_velocity_error_time():
    for constructor in study.protocol_document()["constructors"]:
        original = study.public.controller_document(
            constructor["rotation_gain_scale"], constructor["max_force_n"]
        )
        expected = deepcopy(original)
        expected["tangential"]["velocity_error_time"] = constructor[
            "velocity_error_time_s"
        ]
        assert constructor["constructor"] == expected


def test_screen_accepts_reduced_budget_cost_without_worsening_any_arm():
    result = study.screen(_rows())

    assert len(result["arm_pairs"]) == 4
    assert len(result["budget_pairs"]) == 2
    assert all(pair["status"] == "PASS" for pair in result["arm_pairs"])
    assert all(pair["velocity_cost_reduced"] for pair in result["budget_pairs"])
    assert result["eligible_for_full_grid"] is True
    assert result["decision"] == "expand_original_grid"


@pytest.mark.parametrize("budget", (6.0, 8.0))
def test_screen_rejects_per_arm_velocity_worsening(budget):
    rows = _rows()
    candidate = next(
        row for row in rows
        if row["rotation_gain_scale"] == 1.0
        and row["max_force_n"] == budget
        and row["velocity_error_time_s"] == 0.10
    )
    reference = next(
        row for row in rows
        if row["rotation_gain_scale"] == 1.0
        and row["max_force_n"] == budget
        and row["velocity_error_time_s"] == 0.05
    )
    candidate["tangent_velocity_error_rms_m_s"] = (
        reference["tangent_velocity_error_rms_m_s"] + 2e-12
    )

    result = study.screen(rows)

    pair = next(
        pair for pair in result["arm_pairs"]
        if pair["rotation_gain_scale"] == 1.0 and pair["max_force_n"] == budget
    )
    assert pair["status"] == "FAIL"
    assert "tangent_velocity_error_rms_m_s" in pair["failed_checks"]
    assert result["eligible_for_full_grid"] is False


def test_screen_requires_strict_budget_cost_reduction_for_both_gains():
    rows = _rows()
    candidate = next(
        row for row in rows
        if row["rotation_gain_scale"] == 2.0
        and row["max_force_n"] == 8.0
        and row["velocity_error_time_s"] == 0.10
    )
    candidate["tangent_velocity_error_rms_m_s"] += 0.00005

    result = study.screen(rows)

    pair = next(p for p in result["budget_pairs"] if p["rotation_gain_scale"] == 2.0)
    assert pair["velocity_cost_reduced"] is False
    assert result["decision"] == "do_not_expand"


@pytest.mark.parametrize("failure", ("relative", "absolute"))
def test_screen_retains_original_candidate_compatibility_gates(failure):
    rows = _rows()
    candidate = next(
        row for row in rows
        if row["rotation_gain_scale"] == 1.0
        and row["max_force_n"] == 8.0
        and row["velocity_error_time_s"] == 0.10
    )
    if failure == "relative":
        candidate["force_rmse_n"] = 0.3
    else:
        candidate["contact_ratio_pct"] = np.nextafter(99.0, 0.0)

    result = study.screen(rows)

    pair = next(p for p in result["budget_pairs"] if p["rotation_gain_scale"] == 1.0)
    assert pair["compatibility"]["0.1"]["status"] == "FAIL"
    assert result["eligible_for_full_grid"] is False


@pytest.mark.parametrize("tamper", ("missing", "duplicate", "nonfinite"))
def test_screen_rejects_invalid_grid_or_metric(tamper):
    rows = _rows()
    if tamper == "missing":
        rows.pop()
    elif tamper == "duplicate":
        rows.append(deepcopy(rows[0]))
    else:
        rows[0]["force_rmse_n"] = np.nan

    with pytest.raises(ValueError, match="incomplete|duplicate|nonfinite"):
        study.screen(rows)


def test_collect_reuses_four_parents_and_reads_four_new_traces_without_physics(
    tmp_path, monkeypatch
):
    vectors = np.zeros((2, 3))
    base_trace = {
        "time": np.array([0.0, 0.002]),
        "target_position": vectors.copy(),
        "target_linear_velocity": vectors.copy(),
        "target_normal_force": np.zeros(2),
    }

    def load_trace(path):
        trace = deepcopy(base_trace)
        if "tv100ms" in str(path):
            trace["velocity_error_time_s"] = np.array(0.10)
        return trace

    def catchup(_trace, _normal, start, end):
        return {
            "start_s": start,
            "end_s": end,
            "samples": 1,
            "tangent_rmse_mm": 0.0,
            "tangent_velocity_error_rms_m_s": 0.0,
            "positive_along_lag_area_mm_s": 0.0,
        }

    monkeypatch.setattr(study, "input_archive", lambda: tmp_path / "parent")
    monkeypatch.setattr(study.transfer, "load_trace", load_trace)
    monkeypatch.setattr(study.public, "compact", lambda _trace: {})
    monkeypatch.setattr(study.transfer.validation, "validate_public", lambda *_args: None)
    monkeypatch.setattr(
        study.onset.validation,
        "validate_observation",
        lambda *_args: {"validated_cycles": 2},
    )
    monkeypatch.setattr(study.metrics, "catchup_metrics", catchup)
    monkeypatch.setattr(
        study.public,
        "metrics",
        lambda *_args: {"tangent_rmse_mm": 0.0, "tangent_velocity_error_rms_m_s": 0.0},
    )
    monkeypatch.setattr(
        study.onset,
        "window_rows",
        lambda _trace: [{"window": name} for name, *_ in study.onset.WINDOWS],
    )
    monkeypatch.setattr(
        study.onset,
        "pair_events",
        lambda *_args: {
            "first_6n_cap_s": 1.0,
            "first_6n_cap_blocks_positive_update_s": 1.1,
        },
    )
    monkeypatch.setattr(
        study.runner,
        "run_trial",
        lambda *_args: pytest.fail("execute=False must not run physics"),
    )

    tables = study.collect(tmp_path, execute=False)

    assert {name: len(rows) for name, rows in tables.items()} == {
        "comparison": 8,
        "catchup": 16,
        "observer_windows": 32,
        "events": 4,
    }
    assert [row["row_origin"] for row in tables["comparison"]].count("executed") == 4
    assert all("first_reference_cap_s" in row for row in tables["events"])


def test_generate_rejects_occupied_and_symlinked_output_before_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        study,
        "input_archive",
        lambda: pytest.fail("unsafe output must fail before parent reads"),
    )
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(FileExistsError, match="output already exists"):
        study.generate(occupied)

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        study.generate(link / "archive")


def test_generate_source_mutation_leaves_no_publication(tmp_path, monkeypatch):
    output = tmp_path / "velocity-time"
    sources = iter(({"source.py": "a" * 64}, {"source.py": "b" * 64}))
    monkeypatch.setattr(study, "input_archive", dict)
    monkeypatch.setattr(study, "source_identity", lambda: next(sources))
    monkeypatch.setattr(study, "collect", _fake_collect)

    with pytest.raises(ValueError, match="sources changed during execution"):
        study.generate(output)

    assert not output.exists()
    assert not list(tmp_path.glob(".velocity-time-*"))


def test_audit_binds_complete_to_manifest(archive):
    (archive / "COMPLETE").write_text("0" * 64 + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete manifest"):
        study.audit_archive(archive)


def test_audit_rejects_resealed_inventory_tamper(archive):
    path = archive / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"].pop("events.csv")
    study._write_json(path, manifest)
    (archive / "COMPLETE").write_text(study._sha256(path) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected/missing archive files|inventory"):
        study.audit_archive(archive)


@pytest.mark.parametrize("artifact", ("protocol.json", "source_hashes.json", "comparison.csv", "screening.json"))
def test_audit_rejects_resealed_frozen_or_recomputed_tamper(archive, artifact):
    path = archive / artifact
    if artifact.endswith(".json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        document["tampered"] = True
        study._write_json(path, document)
    else:
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("0.004", "0.003", 1), encoding="utf-8")
    _reseal(archive)

    with pytest.raises(ValueError, match="frozen input differs|reconstructed metric mismatch|summary"):
        study.audit_archive(archive)
