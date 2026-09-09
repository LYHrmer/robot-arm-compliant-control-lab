"""Independently recompute the fixed 6/8 N archive, including cost-screen failures."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from compliant_control_lab.online_compensation_experiment import (
    _case_document,
    _compact_trace,
    _sha256,
)
from tools import audit_online_compensation_errors as legacy
from tools import combined_residual_ablation as original
from tools import compensation_budget_study as study
from tools.audit_combined_residual_ablation import independent_diagnostics, physical_checks
from tools.combined_residual_diagnostics import analyze_trace
from tools.combined_residual_validation import check_unchanged_prefix
from tools.compensation_budget_screen import screen
from tools.compensation_budget_validation import validate_trace
from tools.cross_surface_regression import read_csv


def key(row, suffix=None):
    identity = (float(row["surface_yaw_deg"]), row["variant"], float(row["max_force_n"]))
    return identity + ((row[suffix],) if suffix else ())


def indexed(rows, suffix=None):
    result = {key(row, suffix): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate table identity")
    return result


def check_row(published, expected, label):
    for name, value in expected.items():
        legacy._cell(published, name, value, label)


def audit_archive(directory):
    root = Path(directory)
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("archive must not contain symlinks")
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
    if (manifest["schema_version"] != 1 or manifest["identity"] != study.IDENTITY or manifest["runs"] != 12
            or manifest["budgets_n"] != [6., 8.]):
        raise ValueError("wrong budget identity/count")
    for name in ("new_holdout", "default_changed", "comparator_gates_evaluated"):
        if manifest[name] is not False:
            raise ValueError(f"wrong experiment scope: {name}")
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol != json.loads(json.dumps(study.protocol_document())):
        raise ValueError("budget protocol differs")
    if json.loads((root / "source_hashes.json").read_text()) != study.source_identity():
        raise ValueError("source identity differs")
    expected_inputs = {name: manifest["artifact_sha256"][name]
                       for name in ("protocol.json", "source_hashes.json")}
    if manifest["input_sha256"] != expected_inputs:
        raise ValueError("input hashes not bound to artifacts")
    reference = study.verify_reference()
    rows = indexed(read_csv(root / "comparison.csv"))
    phases = indexed(read_csv(root / "phase_metrics.csv"), "phase")
    diagnostics = indexed(read_csv(root / "diagnostics.csv"), "window")
    pairs = read_csv(root / "paired_diagnostics.csv")
    if (len(rows), len(phases), len(diagnostics), len(pairs)) != (12, 36, 36, 18):
        raise ValueError("incomplete budget tables")
    expected_files = {"protocol.json", "source_hashes.json", "comparison.csv", "phase_metrics.csv",
                      "diagnostics.csv", "paired_diagnostics.csv", "screening.json"}
    raw, rebuilt_rows, rebuilt_phases, rebuilt_diagnostics = {}, [], [], []
    for variant, case in study.study_cases():
        for budget in study.BUDGETS:
            context = study.identity(variant, case, budget)
            identity = key(context)
            name = f"traces/{study.stem(variant, case, budget)}__diagnostic.npz"
            expected_files.add(name)
            with np.load(root / name, allow_pickle=False) as archive:
                trace = {k: archive[k] for k in archive.files}
            if set(trace) != {*original.TRACE_FIELDS, "max_force_n"}:
                raise ValueError("diagnostic trace fields differ")
            validate_trace(trace, case, budget)
            raw[identity] = trace
            compact = _compact_trace(trace, case)
            old_protocol = {
                "recovery": {"window_s": .25, "absolute_force_error_n": 1., "tangent_rmse_mm": 3.},
                "controller_constructor_configurations": {"online": study.controller_document(budget)},
            }
            whole, windows = legacy._summarize(compact, _case_document(case), "online", old_protocol)
            whole.pop("compensation_limit_observed")
            if whole["tangent_recovery_status"] == "not_recovered_with_limit_active":
                whole["tangent_recovery_status"] = "not_recovered"
            low, high, torque = (trace[k] for k in
                                 ("lower_torque_limit", "upper_torque_limit", "commanded_torque"))
            overall = {**context, **whole, **physical_checks(whole),
                       "controller_yaw_error_deg": case.controller_yaw_error_deg,
                       "projection_pct": float(100 * np.mean(trace["torque_projection_scale"] < 1 - 1e-12)),
                       "minimum_torque_headroom_nm": float(min(np.min(torque - low), np.min(high - torque))),
                       "minimum_reserved_torque_headroom_nm": float(min(
                           np.min(torque - low - (high - low) / 20),
                           np.min(high - (high - low) / 20 - torque))),
                       "max_uncapped_tangential_amplitude_n": float(np.max(
                           trace["controller_coefficient_before_compute"] * trace["diagnostic_corrected_force_n"])),
                       "max_matched_request_difference_n": float(np.max(np.linalg.norm(
                           trace["requested_tangential_force_world"]
                           - raw[identity[:2] + (6.,)]["requested_tangential_force_world"], axis=1))),
                       }
            for field in original.DIAGNOSTIC_FIELDS[1:]:
                overall[f"{field}_pct"] = float(100 * np.mean(trace[field][trace["time"] >= 1.5]))
            check_row(rows[identity], overall, "overall")
            error = float(rows[identity]["slew_reconstruction_max_error_n"])
            if not np.isfinite(error) or not 0 <= error <= 1e-9:
                raise ValueError("uncertain observer reconstruction")
            legacy._cell(rows[identity], "slew_reconstruction_mismatch_cycles", 0, "observer")
            rebuilt_rows.append(overall)
            for window, phase in zip(windows, case.phases, strict=True):
                expected = {**context, **window, **physical_checks(window)}
                mask = (trace["time"] >= phase.start_s) & (trace["time"] < phase.end_s)
                for field in original.DIAGNOSTIC_FIELDS[1:]:
                    expected[f"{field}_pct"] = float(100 * np.mean(trace[field][mask]))
                check_row(phases[(*identity, phase.name)], expected, "phase")
                rebuilt_phases.append(expected)
            for label, start, end in original.DIAGNOSTIC_WINDOWS:
                diagnostic = analyze_trace(trace, surface_yaw_deg=case.scenario.wall_yaw_deg,
                                           controller_yaw_error_deg=case.controller_yaw_error_deg,
                                           start_s=start, end_s=end)
                independent_diagnostics(trace, case, diagnostic, start, end)
                expected = {**context, "window": label, **diagnostic}
                check_row(diagnostics[(*identity, label)], expected, "diagnostic")
                rebuilt_diagnostics.append(expected)
            if budget == 6.:
                old_path = reference / "traces" / f"{original.stem(variant, case)}__diagnostic.npz"
                with np.load(old_path, allow_pickle=False) as archived:
                    for field in original.TRACE_FIELDS:
                        if not np.array_equal(trace[field], archived[field]):
                            raise ValueError(f"6 N reference differs: {field}")
    if set(manifest["artifact_sha256"]) != expected_files:
        raise ValueError("unexpected/missing artifact inventory")
    actual_files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual_files != expected_files | {"manifest.json", "COMPLETE"}:
        raise ValueError("unexpected archive files")
    for yaw, budget in product(study.YAWS, study.BUDGETS):
        check_unchanged_prefix(raw[yaw, "combined", budget], raw[yaw, "no_bias", budget], 6.)
    rebuilt_lookup = indexed(rebuilt_diagnostics, "window")
    expected_pairs = []
    for yaw, variant, window in product(study.YAWS, study.VARIANTS,
                                        (w[0] for w in original.DIAGNOSTIC_WINDOWS)):
        candidate = rebuilt_lookup[yaw, variant, 8., window]
        baseline = rebuilt_lookup[yaw, variant, 6., window]
        expected_pairs.append({
            "surface_yaw_deg": yaw, "variant": variant, "window": window,
            "max_force_n": 8., "reference_max_force_n": 6.,
            "delta_direction": "budget_8n_minus_6n",
            **{f"delta_{metric}": candidate[metric] - baseline[metric]
               for metric in original.PAIRED_DIAGNOSTICS},
        })
    pair_key = lambda r: (float(r["surface_yaw_deg"]), r["variant"], r["window"])
    lookup = {pair_key(row): row for row in pairs}
    if len(lookup) != 18 or set(lookup) != {pair_key(r) for r in expected_pairs}:
        raise ValueError("wrong paired identities")
    for expected in expected_pairs:
        check_row(lookup[pair_key(expected)], expected, "pair")
    screening = screen(rebuilt_rows, rebuilt_phases, rebuilt_diagnostics)
    if json.loads((root / "screening.json").read_text()) != screening:
        raise ValueError("screening result differs")
    return {"status": "PASS", "runs": 12, "exact_reference_runs": 6,
            "pre_onset_exact_pairs": 6, "diagnostic_rows": 36, "paired_rows": 18,
            "screening_status": screening["status"], "screening_passed_pairs": screening["passed_pairs"],
            "default_changed": False, "comparator_gates_evaluated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    print(json.dumps(audit_archive(parser.parse_args().audit), indent=2))


if __name__ == "__main__":
    main()
