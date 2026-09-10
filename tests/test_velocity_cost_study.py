"""Synthetic publisher/audit contract tests; no real archive is validated here."""

import csv
import json
from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from tools import velocity_cost_study as study


def _analysis_result(mse_delta=0.0):
    along_6 = 0.010
    cross_6 = 0.004
    along_8 = along_6 + 0.6 * mse_delta
    cross_8 = cross_6 + 0.4 * mse_delta

    def row(start, end, samples, weighted):
        return {
            "start_s": start,
            "end_s": end,
            "samples": samples,
            "velocity_rms_6_m_s": 0.1,
            "velocity_rms_8_m_s": 0.1,
            "velocity_rms_delta_m_s": 0.0,
            "along_mse_6_m2_s2": along_6,
            "along_mse_8_m2_s2": along_8,
            "cross_mse_6_m2_s2": cross_6,
            "cross_mse_8_m2_s2": cross_8,
            "along_mean_6_m_s": 0.01,
            "along_mean_8_m_s": 0.02,
            "cross_mean_6_m_s": -0.01,
            "cross_mean_8_m_s": -0.02,
            "along_variance_6_m2_s2": 0.001,
            "along_variance_8_m2_s2": 0.002,
            "cross_variance_6_m2_s2": 0.003,
            "cross_variance_8_m2_s2": 0.004,
            "position_rms_6_mm": 0.2,
            "position_rms_8_mm": 0.2,
            "position_rms_delta_mm": 0.0,
            "position_along_mean_6_mm": 0.02,
            "position_along_mean_8_mm": 0.03,
            "request_along_mean_6_n": 1.0,
            "request_along_mean_8_n": 1.2,
            "request_along_std_6_n": 0.1,
            "request_along_std_8_n": 0.2,
            "request_delta_vector_rms_n": 0.3,
            "mse_delta_m2_s2": mse_delta,
            "weighted_mse_delta_m2_s2": weighted,
        }

    return {
        "overall": row(1.5, 4.5, 1_500, mse_delta),
        "windows": [
            row(start, start + 0.5, 250, mse_delta / 6.0)
            for start in np.arange(1.5, 4.5, 0.5)
        ],
        "timing": {
            "first_request_difference_s": 1.0,
            "first_velocity_difference_s": 1.2,
            "first_position_difference_s": 1.4,
        },
    }


def _synthetic_tables():
    tables = {name: [] for name in study.TABLES}
    for index, (case, scale) in enumerate(product(range(24), (1.0, 2.0))):
        mse_delta = (index - 24) * 1e-5
        result = _analysis_result(mse_delta)
        context = {
            "case_index": case,
            "rotation_gain_scale": scale,
            "wall_yaw_deg": (-15.0, 0.0, 15.0)[case % 3],
            "wall_time_constant_s": 0.02,
            "tool_mass_kg": 1.0,
            "simulation_seed": 11,
            "original_compatibility": "FAIL" if index < 25 else "PASS",
            "reference_trace": f"parent/reference-{case}-{scale}.npz",
            "candidate_trace": f"parent/candidate-{case}-{scale}.npz",
        }
        tables["pairs"].append({**context, **result["overall"]})
        tables["windows"].extend(
            {**context, **window} for window in result["windows"]
        )
        tables["timing"].append({**context, **result["timing"]})
    return tables


def _write_parent_fixture(tmp_path, monkeypatch, mutate_rows=None):
    transfer_root = tmp_path / "transfer"
    public_root = tmp_path / "public24"
    transfer_root.mkdir()
    public_root.mkdir()

    cases = []
    for case in range(24):
        cases.append(
            {
                "case_index": case,
                "scenario": {
                    "wall_yaw_deg": (-15.0, 0.0, 15.0)[case % 3],
                    "wall_time_constant": 0.02,
                    "tool_mass_kg": 1.0,
                },
                "config": {
                    "evaluation_start": 1.5,
                    "duration": 4.5,
                    "timestep": 0.002,
                    "seed": 11,
                },
            }
        )
    decisions = [
        {
            "case_index": case,
            "rotation_gain_scale": scale,
            "status": "FAIL" if index < 25 else "PASS",
            "tangent_velocity_error_rms_m_s": 0.0,
        }
        for index, (case, scale) in enumerate(product(range(24), (1.0, 2.0)))
    ]
    study._write_json(
        transfer_root / "protocol.json", {"public24": {"cases": cases}}
    )
    study._write_json(
        transfer_root / "screening.json", {"public24": {"pairs": decisions}}
    )

    rows = []
    traces = {}
    for case, scale, budget in product(range(24), (1.0, 2.0), (6.0, 8.0)):
        angle = np.deg2rad((-15.0, 0.0, 15.0)[case % 3])
        tangent = np.array([-np.sin(angle), np.cos(angle), 0.0])
        time = np.arange(2_250) * 0.002
        target_velocity = np.tile(0.05 * tangent, (len(time), 1))
        target_position = np.zeros((len(time), 3))
        relative = (
            f"traces/s{int(scale)}__online__case_{case:02d}__compact.npz"
            if budget == 6.0
            else f"traces/public24__case{case:02d}__s{int(scale)}__f8n.npz"
        )
        root = public_root if budget == 6.0 else transfer_root
        traces[root / relative] = {
            "case_index": np.array(case),
            "rotation_gain_scale": np.array(scale),
            "max_force_n": np.array(budget),
            "method": np.array("online"),
            "time": time,
            "position": target_position + 0.0002 * tangent,
            "target_position": target_position,
            "linear_velocity": target_velocity + 0.1 * tangent,
            "target_linear_velocity": target_velocity,
            "requested_tangential_force_world": np.tile(tangent, (len(time), 1)),
        }
        rows.append(
            {
                "case_index": case,
                "rotation_gain_scale": scale,
                "max_force_n": budget,
                "trace_path": (
                    f"{study.PARENTS['public24']['directory']}/{relative}"
                    if budget == 6.0
                    else relative
                ),
                "row_origin": "pinned_reference" if budget == 6.0 else "executed",
                "tangent_velocity_error_rms_m_s": 0.1,
                "tangent_rmse_mm": 0.2,
            }
        )
    if mutate_rows is not None:
        mutate_rows(rows)
    study._write_csv(transfer_root / "public24.csv", rows)

    monkeypatch.setattr(
        study,
        "input_archives",
        lambda: {"transfer": transfer_root, "public24": public_root},
    )
    monkeypatch.setattr(study.transfer, "load_trace", lambda path: deepcopy(traces[path]))
    return transfer_root, public_root


@pytest.fixture
def generated_archive(tmp_path, monkeypatch):
    tables = _synthetic_tables()
    sources = {"tools/velocity_cost_study.py": "a" * 64}
    monkeypatch.setattr(study, "collect", lambda: deepcopy(tables))
    monkeypatch.setattr(study, "source_identity", lambda: sources.copy())
    return study.generate(tmp_path / "velocity-cost")


def _reseal(root):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"] = {
        name: study._sha256(root / name) for name in manifest["artifact_sha256"]
    }
    study._write_json(manifest_path, manifest)
    (root / "COMPLETE").write_text(
        study._sha256(manifest_path) + "\n", encoding="utf-8"
    )


def test_generate_rejects_occupied_and_symlinked_outputs_before_inputs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        study,
        "source_identity",
        lambda: pytest.fail("unsafe output must be rejected before source inspection"),
    )
    monkeypatch.setattr(
        study, "collect", lambda: pytest.fail("unsafe output must not read parents")
    )

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(FileExistsError, match="output already exists"):
        study.generate(occupied)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match=r"output (?:path )?must not contain symlinks"):
        study.generate(linked_parent / "new-archive")


def test_input_archives_uses_both_declared_manifest_pins(tmp_path, monkeypatch):
    calls = []

    def verify(path, digest):
        calls.append((path, digest))

    monkeypatch.setattr(study.transfer, "verify_archive", verify)
    roots = study.input_archives()

    assert set(roots) == set(study.PARENTS)
    assert calls == [
        (roots[name], study.PARENTS[name]["manifest_sha256"])
        for name in study.PARENTS
    ]


def test_collect_requires_exact_96_parent_identities_paths_and_origins(
    tmp_path, monkeypatch
):
    _write_parent_fixture(tmp_path, monkeypatch)

    tables = study.collect()

    assert {name: len(rows) for name, rows in tables.items()} == {
        "pairs": 48,
        "windows": 288,
        "timing": 48,
    }
    assert {
        (row["case_index"], row["rotation_gain_scale"])
        for row in tables["pairs"]
    } == set(product(range(24), (1.0, 2.0)))
    first = tables["pairs"][0]
    assert first["reference_trace"].endswith(
        "/traces/s1__online__case_00__compact.npz"
    )
    assert first["candidate_trace"].endswith(
        "/traces/public24__case00__s1__f8n.npz"
    )
    assert sum(row["original_compatibility"] == "FAIL" for row in tables["pairs"]) == 25


@pytest.mark.parametrize("tamper", ("missing", "duplicate", "invalid"))
def test_collect_rejects_incomplete_duplicate_or_invalid_identity_grid(
    tmp_path, monkeypatch, tamper
):
    def mutate(rows):
        if tamper == "missing":
            rows.pop()
        elif tamper == "duplicate":
            rows[-1] = rows[0].copy()
        else:
            rows[0]["case_index"] = 24

    _write_parent_fixture(tmp_path, monkeypatch, mutate)

    with pytest.raises(ValueError, match="exactly the 96 frozen identities"):
        study.collect()


@pytest.mark.parametrize("field,value", (("trace_path", "wrong.npz"), ("row_origin", "executed")))
def test_collect_rejects_changed_reference_path_or_origin(
    tmp_path, monkeypatch, field, value
):
    def mutate(rows):
        reference = next(row for row in rows if row["max_force_n"] == 6.0)
        reference[field] = value

    _write_parent_fixture(tmp_path, monkeypatch, mutate)

    with pytest.raises(ValueError, match="trace path/origin"):
        study.collect()


def test_summary_preserves_25_failure_pairs_and_window_contributions():
    tables = _synthetic_tables()

    summary = study.summarize(tables)

    assert summary["pairs"] == 48
    assert summary["windows"] == 288
    assert summary["original_compatibility_passed"] == 23
    assert summary["groups"]["original_fail"]["pairs"] == 25
    for group in summary["groups"].values():
        windows = group["window_contributions"]
        assert len(windows) == 6
        assert sum(w["mean_weighted_mse_delta_m2_s2"] for w in windows) == pytest.approx(
            group["mean_mse_delta_m2_s2"], abs=1e-15
        )


def test_check_summary_allows_roundoff_but_rejects_material_mse_change():
    expected = study.summarize(_synthetic_tables())
    actual = deepcopy(expected)
    actual["groups"]["all"]["mean_mse_delta_m2_s2"] += 1e-18
    study.check_summary(actual, expected)

    actual["groups"]["all"]["mean_mse_delta_m2_s2"] += 1e-9
    with pytest.raises(ValueError, match="summary numeric value differs"):
        study.check_summary(actual, expected)


@pytest.mark.parametrize(
    "path,value,message",
    (
        (("pairs",), 47, "summary value differs"),
        (("default_changed",), True, "summary value differs"),
        (("pairs",), 48.0, "summary type differs"),
        (
            ("groups", "all", "mean_mse_delta_m2_s2"),
            np.nan,
            "summary numeric value differs",
        ),
    ),
)
def test_check_summary_rejects_count_boolean_type_and_nan(path, value, message):
    expected = study.summarize(_synthetic_tables())
    actual = deepcopy(expected)
    target = actual
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        study.check_summary(actual, expected)


def test_audit_binds_complete_to_manifest(generated_archive):
    (generated_archive / "COMPLETE").write_text("0" * 64 + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete manifest"):
        study.audit_archive(generated_archive)


def test_audit_rejects_resealed_parent_pin_tamper(generated_archive):
    manifest_path = generated_archive / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parents"]["transfer"]["manifest_sha256"] = "0" * 64
    study._write_json(manifest_path, manifest)
    (generated_archive / "COMPLETE").write_text(
        study._sha256(manifest_path) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="parents/execution count"):
        study.audit_archive(generated_archive)


@pytest.mark.parametrize("artifact", ("protocol.json", "source_hashes.json", "summary.json", "pairs.csv"))
def test_audit_rejects_resealed_semantic_tampering(generated_archive, artifact):
    path = generated_archive / artifact
    if artifact.endswith(".json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        if artifact == "protocol.json":
            document["identity"] = "tampered"
        elif artifact == "source_hashes.json":
            document["tools/fabricated.py"] = "0" * 64
        else:
            document["pairs"] = 47
        study._write_json(path, document)
    else:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["velocity_rms_6_m_s"] = "999"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    _reseal(generated_archive)

    with pytest.raises(ValueError):
        study.audit_archive(generated_archive)
