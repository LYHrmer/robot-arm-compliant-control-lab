import csv
import json
from copy import deepcopy
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import budget_transfer as transfer
from tools import budget_transfer_screen as screening
from tools import combined_residual_ablation as old
from tools import compensation_budget_screen as budget_screen
from tools import rotation_gain_regression as rotation

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_ARCHIVE = ROOT / "results/franka_budget_transfer"


def _archive(root: Path) -> tuple[Path, str]:
    root.mkdir()
    artifact = root / "artifact.txt"
    artifact.write_text("original\n", encoding="utf-8")
    manifest = {"artifact_sha256": {artifact.name: transfer._sha256(artifact)}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    digest = transfer._sha256(root / "manifest.json")
    (root / "COMPLETE").write_text(digest + "\n", encoding="utf-8")
    return root, digest


@pytest.mark.parametrize("relative", ("../escape", "/tmp/absolute", "nested/../file"))
def test_safe_path_rejects_escaping_or_noncanonical_names(tmp_path, relative):
    with pytest.raises(ValueError, match="unsafe archive artifact"):
        transfer.safe_path(tmp_path, relative)


def test_safe_path_rejects_symlinked_archive_components(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        transfer.safe_path(linked, "artifact.txt")


def test_generate_rejects_occupied_or_symlinked_output_before_loading_inputs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        transfer,
        "references",
        lambda: pytest.fail("unsafe output must be rejected before reading inputs"),
    )
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(FileExistsError, match="output already exists"):
        transfer.generate(occupied)

    real = tmp_path / "real-output-parent"
    real.mkdir()
    linked = tmp_path / "linked-output-parent"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match=r"output (?:path )?must not contain symlinks"):
        transfer.generate(linked / "new-archive")


def test_verify_archive_binds_manifest_complete_pin_and_artifact_hashes(tmp_path):
    root, digest = _archive(tmp_path / "valid")
    assert transfer.verify_archive(root, digest)["artifact_sha256"]

    with pytest.raises(ValueError, match="pinned reference manifest differs"):
        transfer.verify_archive(root, "0" * 64)

    (root / "COMPLETE").write_text("wrong\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete manifest"):
        transfer.verify_archive(root)
    (root / "COMPLETE").write_text(digest + "\n", encoding="utf-8")

    (root / "artifact.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash differs"):
        transfer.verify_archive(root)


def test_verify_archive_rejects_unsafe_and_uninventoried_artifacts(tmp_path):
    unsafe, _ = _archive(tmp_path / "unsafe")
    manifest_path = unsafe / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"] = {"../outside": "0" * 64}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (unsafe / "COMPLETE").write_text(transfer._sha256(manifest_path), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe archive artifact"):
        transfer.verify_archive(unsafe)

    extra, digest = _archive(tmp_path / "extra")
    (extra / "unlisted.txt").write_text("not inventoried\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected/missing archive files"):
        transfer.verify_archive(extra, digest)


def _public_rows():
    rows = []
    for case, scale, budget in product(range(24), (1.0, 2.0), (6.0, 8.0)):
        candidate = budget == 8.0
        rows.append(
            {
                **{metric: 0.0 for metric in rotation.ALL_METRICS},
                "case_index": case,
                "rotation_gain_scale": scale,
                "max_force_n": budget,
                "wall_yaw_deg": (-15, 0, 15)[case % 3],
                "wall_time_constant_s": 0.02,
                "tool_mass_kg": 1.0,
                "tangent_rmse_mm": screening.PUBLIC_LIMITS["tangent_rmse_mm"]
                if candidate
                else 0.0,
                "tangent_velocity_error_rms_m_s": screening.PUBLIC_LIMITS[
                    "tangent_velocity_error_rms_m_s"
                ]
                if candidate
                else 0.0,
                "force_rmse_n": screening.PUBLIC_LIMITS["force_rmse_n"]
                if candidate
                else 0.0,
                "orientation_rmse_deg": screening.PUBLIC_LIMITS["orientation_rmse_deg"]
                if candidate
                else 0.0,
                "peak_force_n": 35.0 if candidate else 34.0,
                "measurement_rmse_n": 0.1,
                "max_penetration_mm": 0.2,
                "minimum_reserved_torque_headroom_nm": 1.0 if candidate else 2.0,
                "max_compensation_force_n": budget,
                "contact_ratio_pct": 99.0 if candidate else 100.0,
                "saturation_pct": 0.0,
                "projection_pct": 0.0,
            }
        )
    return rows


def test_public_screen_requires_the_exact_96_row_grid_and_accepts_boundaries():
    result = screening.public_screen(_public_rows())

    assert len(result["pairs"]) == 48
    assert len(result["physical_groups"]) == 24
    assert result["passed_pairs"] == 48
    assert result["status"] == "PASS"


@pytest.mark.parametrize("tamper", ("duplicate", "missing", "invalid"))
def test_public_screen_rejects_duplicate_missing_and_invalid_identities(tamper):
    rows = _public_rows()
    if tamper == "duplicate":
        rows.append(deepcopy(rows[0]))
        message = "duplicate transfer identity"
    elif tamper == "missing":
        rows.pop()
        message = "incomplete or unexpected transfer identities"
    else:
        rows[0]["case_index"] = 24
        message = "incomplete or unexpected transfer identities"

    with pytest.raises(ValueError, match=message):
        screening.public_screen(rows)


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_public_screen_rejects_nonfinite_metrics(value):
    rows = _public_rows()
    rows[1]["tangent_rmse_mm"] = value
    with pytest.raises(ValueError, match="nonfinite metric: tangent_rmse_mm"):
        screening.public_screen(rows)


def test_public_screen_distinguishes_inclusive_limit_from_excess():
    rows = _public_rows()
    assert screening.public_screen(rows)["pairs"][0]["status"] == "PASS"

    rows[1]["tangent_rmse_mm"] = np.nextafter(
        screening.PUBLIC_LIMITS["tangent_rmse_mm"], np.inf
    )
    pair = screening.public_screen(rows)["pairs"][0]
    assert pair["status"] == "FAIL"
    assert "tangent_rmse_mm" in pair["failed_checks"]

    rows = _public_rows()
    rows[1]["contact_ratio_pct"] = np.nextafter(99.0, 0.0)
    pair = screening.public_screen(rows)["pairs"][0]
    assert pair["status"] == "FAIL"
    assert "contact" in pair["failed_checks"]


def _dynamic_tables():
    rows, phases, diagnostics = [], [], []
    for yaw, variant, scale, budget in product(
        (-15, 0, 15), ("combined", "no_bias"), (1.0, 2.0), (6.0, 8.0)
    ):
        context = {
            "surface_yaw_deg": yaw,
            "variant": variant,
            "rotation_gain_scale": scale,
            "max_force_n": budget,
        }
        rows.append(
            {
                **context,
                "contact_ratio_pct": 99.0,
                "peak_force_n": 33.0 + (scale - 1.0) + (budget - 6.0) / 2.0,
                "saturation_pct": 0.0,
                "projection_pct": 0.0,
                "minimum_reserved_torque_headroom_nm": 1.0,
                "max_uncapped_tangential_amplitude_n": np.nextafter(6.0, np.inf)
                if budget == 8.0
                else 5.0,
                "max_matched_request_difference_n": 1e-8 if budget == 8.0 else 0.0,
            }
        )
        for phase in budget_screen.PHASES:
            phases.append(
                {
                    **context,
                    "phase": phase,
                    "force_rmse_n": (scale - 1.0) * 0.2 + (budget == 8.0) * 0.2,
                    "orientation_rmse_deg": (scale - 1.0) * 0.1
                    + (budget == 8.0) * 0.1,
                }
            )
        for window in budget_screen.WINDOWS:
            if (yaw, variant) == (-15, "combined"):
                tangent = 2.5 if budget == 6.0 else (3.0 if window == "late_post" else 2.0)
            elif (yaw, variant) == (-15, "no_bias"):
                tangent = 4.0 if budget == 6.0 else 3.0
            elif (yaw, variant) == (0, "combined"):
                tangent = 1.0 if budget == 6.0 else (scale - 1.0) * 0.1
            else:
                tangent = 2.0 if budget == 6.0 else 1.0
            diagnostics.append(
                {
                    **{metric: 0.0 for metric in old.PAIRED_DIAGNOSTICS},
                    **context,
                    "window": window,
                    "tangent_rmse_mm": tangent,
                    "normal_force_error_rms_n": 0.5,
                    "orientation_error_rms_deg": 0.1,
                }
            )
    return rows, phases, diagnostics


def test_dynamic_screen_requires_exact_grid_and_accepts_declared_boundaries():
    rows, phases, diagnostics = _dynamic_tables()
    result = screening.dynamic_screen(rows, phases, diagnostics)

    assert len(rows) == 24
    assert len(result["gain_pairs"]) == 12
    assert len(result["descriptive_interactions"]) == 6
    assert all(len(group["pairs"]) == 6 for group in result["budget_by_gain"].values())
    assert result["passed_gain_pairs"] == 12
    assert result["status"] == "PASS"


@pytest.mark.parametrize("table_index", (0, 1, 2))
def test_dynamic_screen_rejects_a_missing_identity_in_every_table(table_index):
    tables = list(_dynamic_tables())
    tables[table_index].pop()
    with pytest.raises(ValueError, match="incomplete or unexpected transfer identities"):
        screening.dynamic_screen(*tables)


def test_dynamic_screen_rejects_duplicate_and_invalid_identities():
    rows, phases, diagnostics = _dynamic_tables()
    rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError, match="duplicate transfer identity"):
        screening.dynamic_screen(rows, phases, diagnostics)

    rows, phases, diagnostics = _dynamic_tables()
    diagnostics[0]["window"] = "unknown"
    with pytest.raises(ValueError, match="incomplete or unexpected transfer identities"):
        screening.dynamic_screen(rows, phases, diagnostics)


def test_dynamic_screen_rejects_nonfinite_metrics():
    rows, phases, diagnostics = _dynamic_tables()
    diagnostics[0]["normal_force_error_rms_n"] = np.nan
    with pytest.raises(ValueError, match="nonfinite metric: normal_force_error_rms_n"):
        screening.dynamic_screen(rows, phases, diagnostics)


def test_dynamic_screen_enforces_inclusive_and_exclusive_boundaries():
    rows, phases, diagnostics = _dynamic_tables()
    result = screening.dynamic_screen(rows, phases, diagnostics)
    gain_pair = next(
        pair
        for pair in result["gain_pairs"]
        if pair["surface_yaw_deg"] == 0
        and pair["variant"] == "combined"
        and pair["max_force_n"] == 8.0
    )
    assert gain_pair["post_tangent_rmse_mm"] == screening.GAIN_LIMITS[
        "post_tangent_rmse_mm"
    ]
    assert gain_pair["status"] == "PASS"

    post = next(
        row
        for row in diagnostics
        if row["surface_yaw_deg"] == -15
        and row["variant"] == "combined"
        and row["rotation_gain_scale"] == 1.0
        and row["max_force_n"] == 8.0
        and row["window"] == "post"
    )
    post["tangent_rmse_mm"] = np.nextafter(2.0, np.inf)
    budget_decision = screening.dynamic_screen(rows, phases, diagnostics)["budget_by_gain"][
        "1"
    ]["pairs"][0]
    assert budget_decision["status"] == "FAIL"
    assert "post_reduction" in budget_decision["failed_checks"]

    rows, phases, diagnostics = _dynamic_tables()
    candidate = next(
        row
        for row in rows
        if row["surface_yaw_deg"] == -15
        and row["variant"] == "combined"
        and row["rotation_gain_scale"] == 1.0
        and row["max_force_n"] == 8.0
    )
    candidate["max_uncapped_tangential_amplitude_n"] = 6.0
    decision = screening.dynamic_screen(rows, phases, diagnostics)["budget_by_gain"]["1"][
        "pairs"
    ][0]
    assert decision["status"] == "FAIL"
    assert "budget_exercised" in decision["failed_checks"]


def test_collect_and_decision_complete_full_grid_without_running_physics(tmp_path, monkeypatch):
    refs = transfer.references()
    dynamic_cases = transfer.dynamic.cases()

    def public_trial(case, scale, budget):
        trace = transfer.load_trace(
            refs["public24"] / transfer.public.reference_trace(case, scale)
        )
        trace = {name: np.asarray(value).copy() for name, value in trace.items()}
        trace["max_force_n"] = np.array(budget)
        return SimpleNamespace(trace=trace)

    def dynamic_trial(case, scale, budget):
        variant = next(name for name, expected in dynamic_cases if case == expected)
        trace = transfer.load_trace(
            refs["dynamic"] / transfer.dynamic.reference_trace(variant, case, budget)
        )
        trace = {name: np.asarray(value).copy() for name, value in trace.items()}
        trace["rotation_gain_scale"] = np.array(scale)
        return SimpleNamespace(trace=trace)

    monkeypatch.setattr(transfer.public, "run_trial", public_trial)
    monkeypatch.setattr(transfer.dynamic, "run_trial", dynamic_trial)
    (tmp_path / "traces").mkdir()

    tables, trace_files = transfer.collect(tmp_path, refs, execute=True)
    result = transfer.decision(tables)

    assert {name: len(rows) for name, rows in tables.items()} == {
        "public24": 96,
        "dynamic": 24,
        "phases": 72,
        "diagnostics": 72,
    }
    assert len(trace_files) == 60
    assert len(list((tmp_path / "traces").glob("*.npz"))) == 60
    assert result["status"] == "PASS"
    assert result["default_changed"] is False
    assert result["new_holdout"] is False

    transfer._write_json(tmp_path / "protocol.json", transfer.protocol_document())
    transfer._write_json(tmp_path / "source_hashes.json", transfer.source_identity())
    for name, rows in tables.items():
        transfer._write_csv(tmp_path / f"{name}.csv", rows)
    transfer._write_json(tmp_path / "screening.json", result)
    artifacts = {
        path.relative_to(tmp_path).as_posix(): transfer._sha256(path)
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "identity": transfer.IDENTITY,
        "evaluated_rows": 120,
        "new_executions": 60,
        "pinned_reference_rows": 60,
        "default_changed": False,
        "new_holdout": False,
        "input_sha256": {
            name: artifacts[name] for name in ("protocol.json", "source_hashes.json")
        },
        "artifact_sha256": artifacts,
    }
    transfer._write_json(tmp_path / "manifest.json", manifest)
    (tmp_path / "COMPLETE").write_text(
        transfer._sha256(tmp_path / "manifest.json") + "\n", encoding="utf-8"
    )

    monkeypatch.setattr(
        transfer.public,
        "run_trial",
        lambda *_args: pytest.fail("audit must not execute the public simulator"),
    )
    monkeypatch.setattr(
        transfer.dynamic,
        "run_trial",
        lambda *_args: pytest.fail("audit must not execute the dynamic simulator"),
    )
    monkeypatch.setattr(transfer, "pinned_metric_preflight", lambda _refs: None)
    assert transfer.audit_archive(tmp_path)["audit_status"] == "PASS"

    def reseal():
        current = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        current["artifact_sha256"] = {
            name: transfer._sha256(tmp_path / name) for name in current["artifact_sha256"]
        }
        current["input_sha256"] = {
            name: current["artifact_sha256"][name]
            for name in ("protocol.json", "source_hashes.json")
        }
        (tmp_path / "manifest.json").write_text(json.dumps(current), encoding="utf-8")
        (tmp_path / "COMPLETE").write_text(
            transfer._sha256(tmp_path / "manifest.json") + "\n", encoding="utf-8"
        )

    protocol_path = tmp_path / "protocol.json"
    original_protocol = protocol_path.read_bytes()
    protocol = json.loads(original_protocol)
    protocol["default_changed"] = True
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    reseal()
    with pytest.raises(ValueError, match="frozen protocol differs"):
        transfer.audit_archive(tmp_path)
    protocol_path.write_bytes(original_protocol)
    reseal()

    sources_path = tmp_path / "source_hashes.json"
    original_sources = sources_path.read_bytes()
    sources = json.loads(original_sources)
    sources["tools/fabricated.py"] = "0" * 64
    sources_path.write_text(json.dumps(sources), encoding="utf-8")
    reseal()
    with pytest.raises(ValueError, match="source identity differs"):
        transfer.audit_archive(tmp_path)
    sources_path.write_bytes(original_sources)
    reseal()

    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluated_rows"] = 119
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "COMPLETE").write_text(
        transfer._sha256(manifest_path) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="wrong manifest scope/count: evaluated_rows"):
        transfer.audit_archive(tmp_path)
    manifest["evaluated_rows"] = 120
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "COMPLETE").write_text(
        transfer._sha256(manifest_path) + "\n", encoding="utf-8"
    )

    table_path = tmp_path / "public24.csv"
    original_table = table_path.read_bytes()
    with table_path.open(newline="", encoding="utf-8") as stream:
        published = list(csv.DictReader(stream))
    published[0]["row_origin"] = "executed"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(published[0]))
        writer.writeheader()
        writer.writerows(published)
    reseal()
    with pytest.raises(ValueError, match="public24"):
        transfer.audit_archive(tmp_path)
    table_path.write_bytes(original_table)
    reseal()

    trace_path = tmp_path / "traces/public24__case00__s1__f8n.npz"
    original_trace = trace_path.read_bytes()
    trace = transfer.load_trace(trace_path)
    trace["case_index"] = np.array(1)
    np.savez_compressed(trace_path, **trace)
    reseal()
    with pytest.raises(ValueError, match="case_index differs"):
        transfer.audit_archive(tmp_path)
    trace_path.write_bytes(original_trace)
    reseal()


def test_published_archive_reaudits_without_running_physics(monkeypatch):
    if not PUBLISHED_ARCHIVE.exists():
        pytest.skip("budget-transfer archive has not been generated")
    monkeypatch.setattr(
        transfer.public,
        "run_trial",
        lambda *_args: pytest.fail("audit must not execute the public simulator"),
    )
    monkeypatch.setattr(
        transfer.dynamic,
        "run_trial",
        lambda *_args: pytest.fail("audit must not execute the dynamic simulator"),
    )

    result = transfer.audit_archive(PUBLISHED_ARCHIVE)

    assert result["audit_status"] == "PASS"
    assert result["evaluated_rows"] == 120
    assert result["new_executions"] == 60
    assert result["pinned_reference_rows"] == 60
    assert result["pre_onset_exact_pairs"] == 12
    assert result["default_changed"] is False
