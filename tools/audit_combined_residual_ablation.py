"""Recompute the small input-ablation archive, including original/prefix identities."""

import argparse
import json
from pathlib import Path

import numpy as np

from compliant_control_lab.online_compensation_experiment import (
    _case_document,
    _compact_trace,
    _sha256,
)
from compliant_control_lab.surface_simulation import yaw_frame
from tools import audit_online_compensation_errors as legacy
from tools import combined_residual_ablation as study
from tools.combined_residual_diagnostics import analyze_trace
from tools.combined_residual_validation import check_unchanged_prefix, close, validate_trace
from tools.cross_surface_regression import read_csv


def key(row, suffix=None):
    value = (int(float(row["surface_yaw_deg"])), row["variant"])
    return value + ((row[suffix],) if suffix else ())


def indexed(rows, suffix=None):
    result = {key(row, suffix): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate table identity")
    return result


def check_row(published, expected, label):
    for name, value in expected.items():
        legacy._cell(published, name, value, label)


def physical_checks(row):
    failures = [name for failed, name in (
        (row["contact_ratio_pct"] < 99, "contact_ratio"),
        (row["peak_force_n"] > 35, "raw_peak_force"),
        (row["saturation_pct"] != 0, "saturation"),
    ) if failed]
    return {"failed_absolute_checks": ";".join(failures),
            "absolute_checks_pass": "no" if failures else "yes",
            "comparator_gates_evaluated": "no", "comparator_gate_status": "not_evaluated"}


def independent_diagnostics(trace, case, row, start, end):
    """Use direct projections and direct windows, not the analyzer's cumulative sum."""
    mask = (trace["time"] >= start) & (trace["time"] < end)
    normal = yaw_frame(case.scenario.wall_yaw_deg).rotation[:, 0]
    delta = trace["position"][mask] - trace["target_position"][mask]
    tangent = delta - (delta @ normal)[:, None] * normal
    square = np.sum(tangent**2, axis=1)
    close(row["tangent_rmse_mm"], 1000 * np.sqrt(np.mean(square)), "independent tangent RMSE")
    velocity = trace["target_linear_velocity"][mask]
    velocity = velocity - (velocity @ normal)[:, None] * normal
    speed = np.linalg.norm(velocity, axis=1)
    moving = speed > 1e-9
    if np.any(moving):
        direction = velocity[moving] / speed[moving, None]
        along = np.sum(tangent[moving] * direction, axis=1)
        cross = np.sum(tangent[moving] * np.cross(normal, direction), axis=1)
        close(row["along_track_rmse_mm"], 1000 * np.sqrt(np.mean(along**2)), "along RMSE")
        close(row["cross_track_rmse_mm"], 1000 * np.sqrt(np.mean(cross**2)), "cross RMSE")
        close(row["moving_tangent_rmse_mm"]**2,
              row["along_track_rmse_mm"]**2 + row["cross_track_rmse_mm"]**2, "split energy")
    span = round(.25 / case.config.timestep)
    rolling = np.array([1000 * np.sqrt(np.mean(square[i:i + span]))
                        for i in range(len(square) - span + 1)])
    close(row["rolling_window_pass_pct"], 100 * np.mean(rolling <= 3), "direct rolling windows")
    close(row["sample_count"], int(mask.sum()), "window sample count")


def audit_archive(directory):
    root = Path(directory)
    if root.is_symlink():
        raise ValueError("archive must not be a symlink")
    manifest_path = root / "manifest.json"
    if (root / "COMPLETE").read_text().strip() != _sha256(manifest_path):
        raise ValueError("incomplete manifest")
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["artifact_sha256"].items():
        relative = Path(name)
        path = root / relative
        if relative.is_absolute() or ".." in relative.parts or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("unsafe archive artifact")
        if _sha256(path) != digest:
            raise ValueError(f"artifact hash differs: {name}")
    if manifest["identity"] != study.IDENTITY or manifest["runs"] != 12:
        raise ValueError("wrong ablation identity/count")
    for name in ("new_holdout", "default_changed", "comparator_gates_evaluated"):
        if manifest[name] is not False:
            raise ValueError(f"wrong experiment scope: {name}")
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol != json.loads(json.dumps(study.protocol_document())):
        raise ValueError("ablation protocol differs")
    if json.loads((root / "source_hashes.json").read_text()) != study.source_identity():
        raise ValueError("source identity differs")
    for name, digest in manifest["input_sha256"].items():
        if digest != manifest["artifact_sha256"].get(name):
            raise ValueError("input hash not bound to artifact")
    reference = study.verify_reference()
    rows = indexed(read_csv(root / "comparison.csv"))
    phases = indexed(read_csv(root / "phase_metrics.csv"), "phase")
    diagnostics = indexed(read_csv(root / "diagnostics.csv"), "window")
    pairs = indexed(read_csv(root / "paired_diagnostics.csv"), "window")
    if tuple(map(len, (rows, phases, diagnostics, pairs))) != (12, 36, 36, 27):
        raise ValueError("incomplete ablation tables")
    expected_files = {"protocol.json", "source_hashes.json", "comparison.csv", "phase_metrics.csv",
                      "diagnostics.csv", "paired_diagnostics.csv"}
    raw, rebuilt = {}, {}
    old_protocol = {"recovery": {"window_s": .25, "absolute_force_error_n": 1., "tangent_rmse_mm": 3.},
                    "controller_constructor_configurations": {"online": study.controller_document()}}
    for variant, case in study.ablation_cases():
        context = study.identity(variant, case)
        identity = key(context)
        name = f"traces/{study.stem(variant, case)}__diagnostic.npz"
        expected_files.add(name)
        with np.load(root / name, allow_pickle=False) as archive:
            trace = {k: archive[k] for k in archive.files}
        if set(trace) != set(study.TRACE_FIELDS):
            raise ValueError("diagnostic trace fields differ")
        validate_trace(trace, case)
        raw[identity] = trace
        compact = _compact_trace(trace, case)
        whole, windows = legacy._summarize(compact, _case_document(case), "online", old_protocol)
        overall_expected = {**context, **whole, **physical_checks(whole),
                            "controller_yaw_error_deg": case.controller_yaw_error_deg}
        low, high, torque = (trace[name] for name in
                             ("lower_torque_limit", "upper_torque_limit", "commanded_torque"))
        overall_expected.update(
            projection_pct=float(100 * np.mean(trace["torque_projection_scale"] < 1 - 1e-12)),
            minimum_torque_headroom_nm=float(min(np.min(torque - low), np.min(high - torque))),
            minimum_reserved_torque_headroom_nm=float(min(
                np.min(torque - low - (high - low) / 20),
                np.min(high - (high - low) / 20 - torque))),
        )
        for name in study.DIAGNOSTIC_FIELDS[1:]:
            overall_expected[f"{name}_pct"] = float(100 * np.mean(trace[name][trace["time"] >= 1.5]))
        check_row(rows[identity], overall_expected, "overall")
        error = float(rows[identity]["slew_reconstruction_max_error_n"])
        if not np.isfinite(error) or not 0 <= error <= 1e-9:
            raise ValueError("uncertain observer reconstruction")
        legacy._cell(rows[identity], "slew_reconstruction_mismatch_cycles", 0, "observer reconstruction")
        for row, window in zip(windows, case.phases, strict=True):
            expected = {**context, **row, **physical_checks(row)}
            mask = (trace["time"] >= window.start_s) & (trace["time"] < window.end_s)
            expected.update({f"{name}_pct": float(100 * np.mean(trace[name][mask]))
                             for name in study.DIAGNOSTIC_FIELDS[1:]})
            check_row(phases[(*identity, row["phase"])], expected, "phase")
        for label, start, end in study.DIAGNOSTIC_WINDOWS:
            row = analyze_trace(trace, surface_yaw_deg=case.scenario.wall_yaw_deg,
                                controller_yaw_error_deg=case.controller_yaw_error_deg,
                                start_s=start, end_s=end)
            independent_diagnostics(trace, case, row, start, end)
            check_row(diagnostics[(*identity, label)], {**context, "window": label, **row}, "diagnostic")
            rebuilt[(*identity, label)] = row
        if variant == "combined":
            old_stem = f"yaw_{int(case.scenario.wall_yaw_deg)}__modest_combined__seed_11__online__s1"
            with np.load(reference / "traces" / f"{old_stem}__compact.npz", allow_pickle=False) as old:
                for field, values in compact.items():
                    if not np.array_equal(values, old[field]):
                        raise ValueError(f"original combined compact changed: {field}")
            with np.load(reference / "traces" / f"{old_stem}__inputs.npz", allow_pickle=False) as old:
                for field in set(trace) & set(old.files):
                    if not np.array_equal(trace[field], old[field]):
                        raise ValueError(f"original combined raw changed: {field}")
    if set(manifest["artifact_sha256"]) != expected_files:
        raise ValueError("unexpected/missing artifact inventory")
    for yaw in study.YAWS:
        for variant, onset in (("no_bias", 6.), ("constant_initial_friction", 4.)):
            check_unchanged_prefix(raw[yaw, "combined"], raw[yaw, variant], onset)
        for variant in study.VARIANTS[1:]:
            for label, _, _ in study.DIAGNOSTIC_WINDOWS:
                identity = (yaw, variant, label)
                expected = {"surface_yaw_deg": yaw, "variant": variant, "window": label,
                            "reference_variant": "combined", "delta_direction": "variant_minus_combined"}
                for metric in study.PAIRED_DIAGNOSTICS:
                    expected[f"delta_{metric}"] = rebuilt[identity][metric] - rebuilt[yaw, "combined", label][metric]
                check_row(pairs[identity], expected, "diagnostic pair")
    return {"status": "PASS", "runs": 12, "phase_rows": 36, "diagnostic_rows": 36,
            "paired_rows": 27, "original_combined_exact_runs": 3,
            "pre_onset_exact_pairs": 6, "comparator_gates_evaluated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    print(json.dumps(audit_archive(parser.parse_args().audit), indent=2))


if __name__ == "__main__":
    main()
