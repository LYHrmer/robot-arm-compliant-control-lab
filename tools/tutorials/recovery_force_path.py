"""Read-only, post-hoc force-path diagnosis of six public high-load reversal pairs."""

import argparse
import json

import numpy as np

from tools.budget_transfer import verify_archive
from tools.reversal_recovery import study as original
from tools.reversal_recovery_transfer import study
from tools.tutorials.reversal_recovery import window_metrics

ARCHIVE = study.ROOT / "results/franka_reversal_recovery_transfer"
MANIFEST_SHA256 = "eef733d9b228267f2987a0e4c743020c73a27541010dbeedd8f571a94681c44a"
WINDOWS = ((7.35, 7.65), (7.5, 8.0))


def force_path(trace, parameters, *, coefficient=None, budget=None):
    """One-step substitutions on recorded states, never a recursive rollout.

    Every cycle keeps the candidate's recorded previous output, normal force,
    blend and target velocity. An override changes only the named scalar input.
    The output is an auxiliary request BEFORE nominal torque projection.
    """
    count = len(trace["time"])
    mu = trace["controller_coefficient_before_compute"] if coefficient is None else coefficient
    cap = trace["load_budget_applied_n"] if budget is None else budget
    mu, cap = np.asarray(mu), np.asarray(cap)
    if any(x.shape != (count,) or not np.all(np.isfinite(x)) or np.any(x < 0)
           for x in (mu, cap)):
        raise ValueError("coefficient and budget must be finite nonnegative cycle arrays")
    velocity = trace["target_linear_velocity"] @ trace["controller_frame_rotation"]
    velocity[:, 0] = 0
    speed = np.linalg.norm(velocity, axis=1)
    direction = velocity / np.sqrt(speed[:, None]**2 + parameters["velocity_scale"]**2)
    raw = mu * trace["diagnostic_corrected_force_n"]
    amplitude = np.minimum(raw, cap)
    desired = trace["contact_blend"][:, None] * amplitude[:, None] * direction
    previous = np.vstack((np.zeros(3), trace["load_compensation_force_local"][:-1]))
    delta = desired - previous
    distance = np.linalg.norm(delta, axis=1)
    step = parameters["force_slew_rate"] * float(trace["dt"])
    output = previous + delta * np.minimum(1, step / np.maximum(distance, 1e-12))[:, None]
    active = trace["diagnostic_compensation_active"]
    output[~active] = 0
    return {"raw_amplitude_n": raw, "capped_amplitude_n": amplitude,
            "desired_n": desired, "output_n": output,
            "cap_active": active & (raw >= cap), "slew_active": active & (distance > step)}


def analyze_pair(baseline, candidate, parameters, yaw):
    """Separate observed differences from one-variable, fixed-state sensitivities."""
    for field in ("time", "dt", "controller_frame_rotation", "target_linear_velocity"):
        if not np.array_equal(baseline[field], candidate[field]):
            raise ValueError(f"unmatched paired input: {field}")
    paths = [force_path(trace, parameters) for trace in (baseline, candidate)]
    for trace, path in zip((baseline, candidate), paths, strict=True):
        if not np.allclose(path["output_n"], trace["load_compensation_force_local"],
                           rtol=0, atol=1e-10):
            raise ValueError("one-step force reconstruction differs")
        for key, field in (("cap_active", "diagnostic_amplitude_capped"),
                           ("slew_active", "diagnostic_slew_limited")):
            if not np.array_equal(path[key], trace[field]):
                raise ValueError(f"one-step flag reconstruction differs: {key}")
    mu_only = force_path(candidate, parameters,
                         coefficient=baseline["controller_coefficient_before_compute"])
    budget_only = force_path(candidate, parameters, budget=baseline["load_budget_applied_n"])
    time = candidate["time"]
    ready, first_increase = [], []
    for trace in (baseline, candidate):
        indices = np.flatnonzero((time >= 7) & trace["controller_update_ready_after_compute"])
        ready.append(float(time[indices[0]]) if len(indices) else None)
        indices = np.flatnonzero((time >= 7) & (trace["controller_coefficient_after_compute"]
                                               > trace["controller_coefficient_before_compute"]))
        first_increase.append(float(time[indices[0]]) if len(indices) else None)
    hold = (time >= 6) & (time < 7)
    velocity = candidate["target_linear_velocity"] @ candidate["controller_frame_rotation"]
    low_speed = np.linalg.norm(velocity[:, 1:], axis=1) < parameters["min_update_speed"]
    released = np.flatnonzero(
        (time >= 5) & (time < 7) & low_speed & candidate["diagnostic_compensation_active"]
        & (candidate["controller_coefficient_after_compute"]
           < candidate["controller_coefficient_before_compute"])
    )
    rows = []
    for start, end in WINDOWS:
        mask = (time >= start) & (time < end)
        metrics = [window_metrics(trace, yaw, start, end) for trace in (baseline, candidate)]
        norms = [np.linalg.norm(path["output_n"][mask], axis=1)
                 for path in (*paths, mu_only, budget_only)]
        desired = [np.linalg.norm(path["desired_n"][mask], axis=1)
                   for path in (*paths, mu_only)]
        rows.append({
            "start_s": start, "end_s": end, "samples": int(np.count_nonzero(mask)),
            "position_delta_mm": metrics[1]["position_rmse_mm"] - metrics[0]["position_rmse_mm"],
            "velocity_delta_mm_s": 1000 * (
                metrics[1]["velocity_rms_m_s"] - metrics[0]["velocity_rms_m_s"]),
            "baseline_cap_cycles": int(np.count_nonzero(paths[0]["cap_active"][mask])),
            "candidate_cap_cycles": int(np.count_nonzero(paths[1]["cap_active"][mask])),
            "baseline_slew_cycles": int(np.count_nonzero(paths[0]["slew_active"][mask])),
            "candidate_slew_cycles": int(np.count_nonzero(paths[1]["slew_active"][mask])),
            "mu_only_slew_cycles": int(np.count_nonzero(mu_only["slew_active"][mask])),
            "observed_desired_gap_mean_n": float(np.mean(desired[1] - desired[0])),
            "mu_only_desired_change_mean_n": float(np.mean(desired[2] - desired[1])),
            "mu_only_desired_remaining_gap_mean_n": float(np.mean(desired[2] - desired[0])),
            "observed_request_gap_mean_n": float(np.mean(norms[1] - norms[0])),
            "mu_only_request_change_mean_n": float(np.mean(norms[2] - norms[1])),
            "budget_only_request_change_mean_n": float(np.mean(norms[3] - norms[1])),
            "mu_only_remaining_gap_mean_n": float(np.mean(norms[2] - norms[0])),
        })
    # This is the original 7--8 s gate, not a gate fitted to the new subwindows.
    ramp = [window_metrics(trace, yaw, 7, 8) for trace in (baseline, candidate)]
    return {"baseline_first_ready_s": ready[0], "candidate_first_ready_s": ready[1],
            "baseline_first_coefficient_increase_s": first_increase[0],
            "candidate_first_coefficient_increase_s": first_increase[1],
            "candidate_first_low_speed_release_s": float(time[released[0]]) if len(released) else None,
            "candidate_last_low_speed_release_s": float(time[released[-1]]) if len(released) else None,
            "candidate_low_speed_release_cycles": len(released),
            "hold_output_equal": bool(np.array_equal(
                baseline["load_compensation_force_local"][hold],
                candidate["load_compensation_force_local"][hold])),
            "hold_budget_equal": bool(np.array_equal(
                baseline["load_budget_applied_n"][hold], candidate["load_budget_applied_n"][hold])),
            "hold_max_position_difference_m": float(np.max(np.abs(
                baseline["position"][hold] - candidate["position"][hold]))),
            "ramp_position_delta_mm": ramp[1]["position_rmse_mm"] - ramp[0]["position_rmse_mm"],
            "ramp_velocity_delta_mm_s": 1000 * (
                ramp[1]["velocity_rms_m_s"] - ramp[0]["velocity_rms_m_s"]),
            "windows": rows}


def diagnose():
    """Pin both archives; validate only the 12 selected traces without physics."""
    verify_archive(ARCHIVE, MANIFEST_SHA256)
    comparison = json.loads((ARCHIVE / "comparison.json").read_text())
    protocol = study.protocol_document()
    reference = study.ROOT / protocol["reference"]["directory"]
    manifest = verify_archive(reference, protocol["reference"]["manifest_sha256"])
    original.verify_comparison(json.loads((ARCHIVE / "protocol.json").read_text()), protocol)
    if json.loads((ARCHIVE / "source_hashes.json").read_text()) != study.source_identity():
        raise ValueError("source identity differs")
    original.verify_sources(json.loads((reference / "source_hashes.json").read_text()),
                            manifest["source_commit"])
    selected = [s for s in study.specifications(protocol)
                if s["scenario"] == "constant_high" and s["error_profile"] == "clean"]
    pairs = {}
    for spec in selected:
        _, parameters = original.make_controller(spec, protocol)
        trace = original.previous._load_trace(study.trace_location(ARCHIVE, spec, protocol))
        study.validate_trace(spec, trace, parameters, protocol)
        key = spec["surface_yaw_deg"], spec["seed"]
        pairs.setdefault(key, {})[spec["method"]] = (trace, parameters)
    rows = []
    for (yaw, seed), pair in pairs.items():
        baseline, parameters = pair["adaptive6_8"]
        candidate, candidate_parameters = pair["hold_cap_tracking"]
        if parameters != candidate_parameters:
            raise ValueError("paired parameters differ")
        rows.append({"surface_yaw_deg": yaw, "seed": seed,
                     **analyze_pair(baseline, candidate, parameters, yaw)})
    return {"analysis": "post-hoc fixed-state one-step sensitivity; not closed-loop evidence",
            "new_simulations": 0, "validated_traces": len(selected), "default_changed": False,
            "archived_comparison_status": comparison["status"],
            "archived_passing_pairs": sum(p["status"] == "PASS" for p in comparison["comparisons"]),
            "archived_total_pairs": len(comparison["comparisons"]),
            "manifest_sha256": MANIFEST_SHA256,
            "reference_manifest_sha256": protocol["reference"]["manifest_sha256"],
            "windows_are_post_hoc": True, "pairs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print complete diagnostic values")
    args = parser.parse_args()
    report = diagnose()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("pinned archive hashes + auxiliary replay: PASS; 12 old traces; no new physics")
    print(f"original comparison: {report['archived_passing_pairs']}/{report['archived_total_pairs']} "
          f"pairs passed; overall {report['archived_comparison_status']} (unchanged)")
    print("post-hoc windows; request magnitudes; substitutions hold previous output fixed")
    print("yaw seed ready_baseline/candidate  window     gap_N    mu_only_change_N budget_change_N")
    for pair in report["pairs"]:
        for row in pair["windows"]:
            print(f"{pair['surface_yaw_deg']:3} {pair['seed']:4} "
                  f"{pair['baseline_first_ready_s']:.3f}/{pair['candidate_first_ready_s']:.3f}"
                  f"                {row['start_s']:g}-{row['end_s']:g} "
                  f"{row['observed_request_gap_mean_n']:+.6f} "
                  f"{row['mu_only_request_change_mean_n']:+.6f} "
                  f"{row['budget_only_request_change_mean_n']:+.6f}")
    print("No substitute trajectory, tracking improvement or promotion follows from these probes.")


if __name__ == "__main__":
    main()
