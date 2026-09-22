"""Inspect archived stop/reverse recovery; print answers without running physics."""

import argparse
import json
from pathlib import Path

import numpy as np

from compliant_control_lab.surface_control import SurfaceFrame
from tools.budget_transfer import verify_archive
from tools.measured_budget_validation import validate_trace

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "results/franka_measured_budget_robustness"
MANIFEST_SHA256 = "8ae404cdbaf9dc7fcaa91ca77b5b3a2b58a2ef52d75b3be7f8b4666c29ee94bd"
SCENARIO = "yaw0_falling_friction_stop_reverse_fresh"
METHODS = ("adaptive6_8", "fixed6", "fixed8")
WINDOWS = ((7.0, 8.0), (8.0, 10.0), (10.0, 12.0), (8.0, 12.0))


def load_experiment(directory=ARCHIVE):
    """Pin the manifest, verify every artifact, then load the three selected traces."""
    directory = Path(directory)
    verify_archive(directory, MANIFEST_SHA256)
    report = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
    result = {}
    for method in METHODS:
        rows = [
            row for row in report["runs"]
            if row["scenario_id"] == SCENARIO and row["method"] == method
        ]
        if len(rows) != 1:
            raise ValueError("expected exactly one archived run per method")
        row = rows[0]
        expected_path = f"traces/{SCENARIO}__{method}.npz"
        if row["trace_path"] != expected_path:
            raise ValueError("unexpected trace origin")
        with np.load(directory / expected_path, allow_pickle=False) as saved:
            trace = {name: saved[name] for name in saved.files}
        result[method] = (trace, row)
    return result


def window_metrics(trace, surface_yaw_deg, start_s, end_s):
    """Project actual-minus-target errors onto the physical surface tangent plane."""
    time = trace["time"]
    indices = (time >= start_s) & (time < end_s)
    expected_count = round((end_s - start_s) / float(trace["dt"]))
    if expected_count <= 0 or np.count_nonzero(indices) != expected_count:
        raise ValueError("window must contain every scheduled sample")
    angle = np.deg2rad(surface_yaw_deg)
    normal = np.array([np.cos(angle), np.sin(angle), 0.0])
    output = {"start_s": start_s, "end_s": end_s, "samples": expected_count}
    for field, key, scale in (
        ("position", "position_rmse_mm", 1000.0),
        ("linear_velocity", "velocity_rms_m_s", 1.0),
    ):
        error = trace[field][indices] - trace[f"target_{field}"][indices]
        tangent = error - np.outer(error @ normal, normal)
        if not np.all(np.isfinite(tangent)):
            raise ValueError("nonfinite physical tracking error")
        output[key] = scale * float(np.sqrt(np.mean(np.sum(tangent * tangent, axis=1))))
    return output


def analyze(trace, archived_row):
    """Replay auxiliary states and compare recomputed metrics with saved diagnostics."""
    replay = validate_trace(
        trace, SurfaceFrame(trace["controller_frame_rotation"]),
        minimum_force=float(trace["minimum_force_n"]), max_force=float(trace["max_force_n"]),
    )
    windows = [window_metrics(trace, archived_row["surface_yaw_deg"], *w) for w in WINDOWS]
    # The existing report has a 7--8 phase plus 8--12 and 10--12 diagnostic windows.
    references = [
        next(row for row in archived_row["phases"] if row["phase"] == "low_reverse_ramp"),
        next(row for row in archived_row["diagnostics"] if row["window"] == "late_post"),
        next(row for row in archived_row["diagnostics"] if row["window"] == "post"),
    ]
    for measured, reference in zip((windows[0], windows[2], windows[3]), references):
        for key, archived_key in (
            ("position_rmse_mm", "tangent_rmse_mm"),
            ("velocity_rms_m_s", "tangent_velocity_error_rms_m_s"),
        ):
            if not np.isclose(measured[key], reference[archived_key], rtol=0, atol=1e-10):
                raise ValueError(f"recomputed metric differs from archive: {archived_key}")

    time = trace["time"]
    angle = np.deg2rad(archived_row["surface_yaw_deg"])
    normal = np.array([np.cos(angle), np.sin(angle), 0.0])
    velocity = trace["target_linear_velocity"]
    tangent_velocity = velocity - np.outer(velocity @ normal, normal)
    moving = np.linalg.norm(tangent_velocity, axis=1) > 1e-12
    first_motion = int(np.flatnonzero((time >= 7.0) & moving)[0])
    first_ready = int(np.flatnonzero(
        (time >= 7.0) & trace["controller_update_ready_after_compute"]
    )[0])
    snapshots = []
    for stamp in (5.5, 7.0, float(time[first_ready]), 8.0):
        index = round(stamp / float(trace["dt"]))
        snapshots.append({
            "time_s": float(time[index]),
            "budget_n": float(trace["load_budget_applied_n"][index]),
            "coefficient": float(trace["controller_coefficient_before_compute"][index]),
            "request_n": float(np.linalg.norm(trace["load_compensation_force_local"][index])),
            "ready": bool(trace["controller_update_ready_after_compute"][index]),
            "slew_limited": bool(trace["diagnostic_slew_limited"][index]),
            "amplitude_capped": bool(trace["diagnostic_amplitude_capped"][index]),
        })
    return {
        "windows": windows, "snapshots": snapshots, "replay": replay,
        "first_motion_s": float(time[first_motion]), "first_ready_s": float(time[first_ready]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", action="store_true", help="also print hold/restart states")
    arguments = parser.parse_args()
    results = {method: analyze(*data) for method, data in load_experiment().items()}
    cycles = sum(value["replay"]["validated_cycles"] for value in results.values())
    print("archive SHA256: PASS; 3 existing traces, no new physics")
    print(f"auxiliary-state replay: PASS ({cycles} cycles; not full control replay)")
    print("physical tangent errors; half-open windows [start, end)")
    print("method       window_s   position_mm  velocity_m_s")
    for method, result in results.items():
        for row in result["windows"]:
            window = f"{row['start_s']:g}-{row['end_s']:g}"
            print(f"{method:12} {window:8} {row['position_rmse_mm']:11.6f}  "
                  f"{row['velocity_rms_m_s']:.9f}")
    if arguments.events:
        print("events: current coefficient/budget, slew-limited request, update gates")
        for method, result in results.items():
            print(f"{method}: first reverse target={result['first_motion_s']:.3f} s; "
                  f"first ready={result['first_ready_s']:.3f} s")
            for row in result["snapshots"]:
                print(f"  t={row['time_s']:.3f} budget={row['budget_n']:.6f} N "
                      f"mu={row['coefficient']:.9f} request={row['request_n']:.6f} N "
                      f"ready={int(row['ready'])} slew={int(row['slew_limited'])} "
                      f"cap={int(row['amplitude_capped'])}")


if __name__ == "__main__":
    main()
