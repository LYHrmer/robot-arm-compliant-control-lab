"""Read-only, paired velocity-cost decomposition of the frozen public24 traces.

No simulation, controller update, new screening gate or causal attribution. The
derived archive points to pinned parent traces rather than duplicating them.
"""

import argparse
import csv
import json
import os
import platform
import tempfile
from pathlib import Path

import numpy as np

from compliant_control_lab.online_compensation_experiment import _sha256, _write_csv, _write_json
from tools import budget_transfer as transfer
from tools import velocity_cost_analysis as analysis
from tools.audit_online_compensation_errors import _cell

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = "public24-velocity-cost-v1"
PARENTS = {
    "transfer": {
        "directory": "results/franka_budget_transfer",
        "manifest_sha256": "bce10bd63e2879ab9a7e89ed3375d3e971eefce8fb681e2ad57feb37d49e55c6",
    },
    "public24": dict(transfer.public.REFERENCE),
}
TABLES = ("pairs", "windows", "timing")
ARTIFACTS = {*(f"{name}.csv" for name in TABLES), "protocol.json", "source_hashes.json", "summary.json"}


def protocol_document():
    return {
        "identity": IDENTITY, "parents": PARENTS, "pairs": 48, "input_traces": 96,
        "new_simulations": 0, "default_changed": False, "new_holdout": False,
        "evaluation_interval_s": [1.5, 4.5], "interval_convention": "start <= t < end",
        "window_s": 0.5, "windows_per_pair": 6, "sample_period_s": 0.002,
        "error_sign": "actual_minus_target", "frame": "true normal; unit target tangent; normal cross along",
        "zero_target_speed": "reject evaluated tangent speed <= 1e-12 m/s",
        "mse_contribution": "window samples / total samples * (MSE8 - MSE6)",
        "variance": "population variance in each moving along/cross coordinate, ddof=0",
        "first_difference": "full-trace vector norm > 1e-12 in recorded field units",
        "summary_float_audit_tolerance": {"rtol": 1e-10, "atol": 1e-15, "counts_and_labels": "exact"},
        "timing_convention": "state x[k] precedes command u[k] within the same recorded row",
        "limitations": [
            "Public-development observations, not a new blind holdout or independent physical samples.",
            "Requested compensation is not measured contact friction or identified excess force.",
            "No compact-trace cap, update-ready, slew, filtered-force or applied Cartesian wrench inference.",
            "Window means/variance do not identify oscillation, noise or causal mediation.",
            "First differences are descriptive chronology, not measured system delay.",
            "All original compatibility thresholds and FAIL decisions remain unchanged.",
        ],
    }


def source_identity():
    # Preserve the parent audit dependency closure; add only these two new tools.
    result = transfer.source_identity()
    for filename in (__file__, analysis.__file__):
        path = Path(filename).resolve()
        result[path.relative_to(ROOT).as_posix()] = _sha256(path)
    return dict(sorted(result.items()))


def input_archives():
    roots = {}
    for name, record in PARENTS.items():
        roots[name] = transfer.safe_path(ROOT, record["directory"])
        transfer.verify_archive(roots[name], record["manifest_sha256"])
    return roots


def check_decomposition(reference, candidate, normal, result):
    """Check partitions and vector-MSE closure without the analyzer's helpers."""
    time = np.asarray(reference["time"])
    if not np.allclose(time, np.arange(2250) * .002, rtol=0, atol=1e-12):
        raise ValueError("public24 trace time grid differs")
    overall, windows = result["overall"], result["windows"]
    if overall["samples"] != 1500 or len(windows) != 6:
        raise ValueError("wrong evaluated sample/window count")
    for i, row in enumerate(windows):
        if (row["start_s"], row["end_s"], row["samples"]) != (1.5 + .5*i, 2. + .5*i, 250):
            raise ValueError("windows do not partition the frozen evaluation interval")
    for row in [overall, *windows]:
        mask = (time >= row["start_s"]) & (time < row["end_s"])
        mse = []
        for arm, trace in ((6, reference), (8, candidate)):
            error = trace["linear_velocity"][mask] - trace["target_linear_velocity"][mask]
            error -= np.outer(error @ normal, normal)
            mse.append(float(np.mean(np.einsum("ij,ij->i", error, error))))
            components = row[f"along_mse_{arm}_m2_s2"] + row[f"cross_mse_{arm}_m2_s2"]
            if not np.isclose(components, mse[-1], rtol=1e-12, atol=1e-15):
                raise ValueError("along/cross vector MSE does not close")
            for direction in ("along", "cross"):
                rebuilt = row[f"{direction}_mean_{arm}_m_s"]**2 + row[f"{direction}_variance_{arm}_m2_s2"]
                if not np.isclose(rebuilt, row[f"{direction}_mse_{arm}_m2_s2"], rtol=1e-12, atol=1e-15):
                    raise ValueError("mean/variance decomposition does not close")
        if not np.isclose(mse[1] - mse[0], row["mse_delta_m2_s2"], rtol=1e-12, atol=1e-15):
            raise ValueError("paired vector MSE delta does not close")
        if not np.isclose(row["weighted_mse_delta_m2_s2"], row["samples"] / 1500 * (mse[1] - mse[0]), rtol=1e-12, atol=1e-15):
            raise ValueError("weighted window contribution differs")
    if not np.isclose(sum(w["weighted_mse_delta_m2_s2"] for w in windows), overall["mse_delta_m2_s2"], rtol=1e-12, atol=1e-15):
        raise ValueError("window contributions do not close")
    timing = result["timing"]
    first = timing["first_request_difference_s"]
    if first is None:
        if any(timing[k] is not None for k in ("first_velocity_difference_s", "first_position_difference_s")):
            raise ValueError("states differ without a recorded request difference")
        return
    prefix = time < first
    for name in set(reference) & set(candidate):
        left, right = np.asarray(reference[name]), np.asarray(candidate[name])
        if left.ndim and left.shape[0] == len(time) and not np.array_equal(left[prefix], right[prefix]):
            raise ValueError(f"recorded prefix differs before request divergence: {name}")
    for name in ("first_velocity_difference_s", "first_position_difference_s"):
        if timing[name] is not None and timing[name] <= first:
            raise ValueError("pre-control state differs before the changed request can act")


def collect():
    roots = input_archives()
    protocol = json.loads((roots["transfer"] / "protocol.json").read_text())
    cases = {c["case_index"]: c for c in protocol["public24"]["cases"]}
    screening = json.loads((roots["transfer"] / "screening.json").read_text())["public24"]
    decisions = {(p["case_index"], p["rotation_gain_scale"]): p for p in screening["pairs"]}
    with (roots["transfer"] / "public24.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {(int(r["case_index"]), float(r["rotation_gain_scale"]), float(r["max_force_n"])): r for r in rows}
    expected = {(case, scale, budget) for case in range(24) for scale in (1., 2.) for budget in (6., 8.)}
    if len(rows) != 96 or set(indexed) != expected:
        raise ValueError("parent public24 must contain exactly the 96 frozen identities")
    tables = {name: [] for name in TABLES}
    for case in range(24):
        document = cases[case]
        scenario, config = document["scenario"], document["config"]
        if (config["evaluation_start"], config["duration"], config["timestep"]) != (1.5, 4.5, .002):
            raise ValueError("parent evaluation grid changed")
        angle = np.deg2rad(scenario["wall_yaw_deg"])
        normal = np.array([np.cos(angle), np.sin(angle), 0.])
        for scale in (1., 2.):
            traces, paths = [], []
            for budget in (6., 8.):
                row = indexed[case, scale, budget]
                relative = (f"traces/s{int(scale)}__online__case_{case:02d}__compact.npz" if budget == 6.
                            else f"traces/public24__case{case:02d}__s{int(scale)}__f8n.npz")
                expected_path = f"{PARENTS['public24']['directory']}/{relative}" if budget == 6. else relative
                origin = "pinned_reference" if budget == 6. else "executed"
                if row["trace_path"] != expected_path or row["row_origin"] != origin:
                    raise ValueError("parent trace path/origin differs from frozen identity")
                trace = transfer.load_trace(transfer.safe_path(roots["public24" if budget == 6. else "transfer"], relative))
                if (int(trace["case_index"]) != case or float(trace["rotation_gain_scale"]) != scale
                        or str(trace["method"]) != "online"):
                    raise ValueError("parent trace metadata mismatch")
                if budget == 8. and float(trace["max_force_n"]) != budget:
                    raise ValueError("parent force budget metadata mismatch")
                traces.append(trace)
                paths.append(f"{PARENTS['public24' if budget == 6. else 'transfer']['directory']}/{relative}")
            result = analysis.analyze_pair(*traces, normal)
            check_decomposition(*traces, normal, result)
            for budget in (6., 8.):
                row = indexed[case, scale, budget]
                _cell(row, "tangent_velocity_error_rms_m_s", result["overall"][f"velocity_rms_{int(budget)}_m_s"], "parent velocity")
                _cell(row, "tangent_rmse_mm", result["overall"][f"position_rms_{int(budget)}_mm"], "parent position")
            delta = result["overall"]["velocity_rms_delta_m_s"]
            decision = decisions[case, scale]
            if not np.isclose(delta, decision["tangent_velocity_error_rms_m_s"], rtol=0, atol=1e-12):
                raise ValueError("parent paired velocity delta differs")
            context = {
                "case_index": case, "rotation_gain_scale": scale,
                "wall_yaw_deg": scenario["wall_yaw_deg"], "wall_time_constant_s": scenario["wall_time_constant"],
                "tool_mass_kg": scenario["tool_mass_kg"], "simulation_seed": config["seed"],
                "original_compatibility": decision["status"],
                "reference_trace": paths[0], "candidate_trace": paths[1],
            }
            tables["pairs"].append({**context, **result["overall"]})
            tables["windows"].extend({**context, **w} for w in result["windows"])
            tables["timing"].append({**context, **result["timing"]})
    return tables


def summarize(tables):
    pairs, windows = tables["pairs"], tables["windows"]
    groups = {}
    for label, selected in (("all", pairs), ("original_fail", [p for p in pairs if p["original_compatibility"] == "FAIL"])):
        identities = {(p["case_index"], p["rotation_gain_scale"]) for p in selected}
        delta = np.array([p["mse_delta_m2_s2"] for p in selected])
        along = np.array([p["along_mse_8_m2_s2"] - p["along_mse_6_m2_s2"] for p in selected])
        window_groups = []
        for start in np.arange(1.5, 4.5, .5):
            chosen = [w for w in windows if w["start_s"] == start and (w["case_index"], w["rotation_gain_scale"]) in identities]
            contributions = [w["weighted_mse_delta_m2_s2"] for w in chosen]
            window_groups.append({"start_s": float(start), "end_s": float(start + .5),
                                  "positive_pairs": sum(v > 0 for v in contributions),
                                  "negative_pairs": sum(v < 0 for v in contributions),
                                  "mean_weighted_mse_delta_m2_s2": float(np.mean(contributions))})
        groups[label] = {
            "pairs": len(selected), "positive_total_mse_pairs": int(np.sum(delta > 0)),
            "positive_along_mse_pairs": int(np.sum(along > 0)),
            "positive_cross_mse_pairs": sum(p["cross_mse_8_m2_s2"] > p["cross_mse_6_m2_s2"] for p in selected),
            "mean_mse_delta_m2_s2": float(np.mean(delta)),
            "mean_along_mse_delta_m2_s2": float(np.mean(along)),
            "mean_cross_mse_delta_m2_s2": float(np.mean(delta - along)),
            "window_contributions": window_groups,
        }
    return {"identity": IDENTITY, "analysis_kind": "descriptive_paired_decomposition",
            "new_simulations": 0, "pairs": len(pairs), "windows": len(windows),
            "original_compatibility_passed": sum(p["original_compatibility"] == "PASS" for p in pairs),
            "default_changed": False, "groups": groups}


def check_summary(actual, expected):
    """Keep counts/labels exact; allow only roundoff in recomputed float statistics."""
    if type(actual) is not type(expected):
        raise ValueError("summary type differs")
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError("summary keys differ")
        for key in expected:
            check_summary(actual[key], expected[key])
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError("summary length differs")
        for left, right in zip(actual, expected, strict=True):
            check_summary(left, right)
    elif isinstance(expected, float):
        if not np.isfinite(actual) or not np.isclose(actual, expected, rtol=1e-10, atol=1e-15):
            raise ValueError("summary numeric value differs")
    elif actual != expected:
        raise ValueError("summary value differs")


def audit_archive(directory):
    root = Path(directory).absolute()
    manifest = transfer.verify_archive(root)
    if manifest.get("identity") != IDENTITY or set(manifest["artifact_sha256"]) != ARTIFACTS:
        raise ValueError("derived archive identity/inventory differs")
    if manifest.get("parents") != PARENTS or manifest.get("new_simulations") != 0:
        raise ValueError("derived archive parents/execution count differs")
    for filename, expected in (("protocol.json", protocol_document()), ("source_hashes.json", source_identity())):
        if json.loads((root / filename).read_text()) != expected:
            raise ValueError(f"frozen input differs: {filename}")
    tables = collect()
    for name in TABLES:
        transfer.check_table(root / f"{name}.csv", tables[name])
    check_summary(json.loads((root / "summary.json").read_text()), summarize(tables))
    return {"audit_status": "PASS", "pairs": len(tables["pairs"]), "windows": len(tables["windows"]),
            "new_simulations": 0, "original_compatibility_passed": summarize(tables)["original_compatibility_passed"],
            "default_changed": False}


def generate(directory):
    output = transfer.rotation._output_path(directory)
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("output must not contain symlinks")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    sources = source_identity()
    tables = collect()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".velocity-cost-", dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        for name, value in (("protocol.json", protocol_document()), ("source_hashes.json", sources), ("summary.json", summarize(tables))):
            _write_json(staging / name, value)
        for name in TABLES:
            _write_csv(staging / f"{name}.csv", tables[name])
        if source_identity() != sources:
            raise ValueError("sources changed during derivation")
        manifest = {"identity": IDENTITY, "parents": PARENTS, "new_simulations": 0,
                    "versions": {"python": platform.python_version(), "numpy": np.__version__},
                    "artifact_sha256": {name: _sha256(staging / name) for name in sorted(ARTIFACTS)}}
        _write_json(staging / "manifest.json", manifest)
        (staging / "COMPLETE").write_text(_sha256(staging / "manifest.json") + "\n")
        audit_archive(staging)
        if output.exists():
            raise FileExistsError("output appeared during derivation")
        os.rename(staging, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--audit", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_archive(args.audit)) if args.audit else generate(args.output))


if __name__ == "__main__":
    main()
