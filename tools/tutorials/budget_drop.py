"""Read and independently replay a published packet fault; never run new physics."""

import argparse
from pathlib import Path

import numpy as np

from compliant_control_lab.surface_control import SurfaceFrame
from tools.measured_budget_validation import validate_trace

ROOT = Path(__file__).resolve().parents[2]
EVENTS = {"missing": 0, "stale": 2}


def load_trace(event="missing"):
    if event not in EVENTS:
        raise ValueError("event must be missing or stale")
    path = ROOT / (
        "results/franka_measured_budget_robustness/traces/"
        f"yaw0_combined_{event}__adaptive6_8.npz"
    )
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def analyze(trace, event="missing"):
    """Recompute every state transition before extracting the worked example."""
    if event not in EVENTS:
        raise ValueError("event must be missing or stale")
    frame = SurfaceFrame(trace["controller_frame_rotation"])
    validation = validate_trace(trace, frame)
    rejected = np.flatnonzero(trace["load_packet_status"] == EVENTS[event])
    if not len(rejected) or rejected[0] == 0 or rejected[-1] + 1 >= len(trace["time"]):
        raise ValueError("example requires a fault after startup and a recorded recovery")
    index = int(rejected[0])
    force = trace["load_compensation_force_local"]
    magnitude = np.linalg.norm(force, axis=1)
    below = rejected[magnitude[rejected] <= 6.0]
    if not len(below):
        raise ValueError("example never returns below 6 N during the fault")

    # One worked step, with the same surface coordinates used by the controller.
    velocity = trace["target_linear_velocity"][index] @ frame.rotation
    velocity[0] = 0.0
    direction = velocity / np.sqrt(velocity @ velocity + 0.005**2)
    amplitude = min(
        trace["controller_coefficient_before_compute"][index]
        * trace["diagnostic_corrected_force_n"][index],
        trace["load_budget_applied_n"][index],
    )
    desired = trace["contact_blend"][index] * amplitude * direction
    delta = desired - force[index - 1]
    step_limit = 20.0 * float(trace["dt"])
    reconstructed = force[index - 1] + delta * min(
        1.0, step_limit / max(float(np.linalg.norm(delta)), 1e-12)
    )
    np.testing.assert_allclose(reconstructed, force[index], rtol=0, atol=1e-12)
    return {
        "event": event,
        "validation": validation,
        "index": index,
        "time_s": float(trace["time"][index]),
        "rejected_packets": len(rejected),
        "recovery_s": float(trace["time"][rejected[-1] + 1]),
        "previous_budget_n": float(trace["load_budget_applied_n"][index - 1]),
        "budget_n": float(trace["load_budget_applied_n"][index]),
        "previous_load_n": float(trace["load_estimate_n"][index - 1]),
        "load_n": float(trace["load_estimate_n"][index]),
        "mu_before": float(trace["controller_coefficient_before_compute"][index]),
        "mu_after": float(trace["controller_coefficient_after_compute"][index]),
        "previous_force_n": force[index - 1].copy(),
        "desired_force_n": desired,
        "force_n": force[index].copy(),
        "step_n": float(np.linalg.norm(force[index] - force[index - 1])),
        "magnitude_before_n": float(magnitude[index - 1]),
        "magnitude_n": float(magnitude[index]),
        "first_below_index": int(below[0]),
        "first_below_time_s": float(trace["time"][below[0]]),
    }


def check_tamper(trace, index):
    """Corrupt an in-memory copy; success means the independent audit rejects it."""
    altered = dict(trace)
    field = "controller_coefficient_after_compute"
    altered[field] = trace[field].copy()
    altered[field][index] += 1e-4
    try:
        validate_trace(altered, SurfaceFrame(trace["controller_frame_rotation"]))
    except ValueError as error:
        if "coefficient transition" not in str(error):
            raise
        return
    raise AssertionError("the altered coefficient was not rejected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", choices=EVENTS, default="missing")
    parser.add_argument("--check-tamper", action="store_true")
    arguments = parser.parse_args()
    trace = load_trace(arguments.event)
    result = analyze(trace, arguments.event)
    print(f"full-state replay: PASS ({result['validation']['validated_cycles']} cycles)")
    print(f"event: {result['event']}; rejected packets: {result['rejected_packets']}")
    print(f"first rejection: k={result['index']}, t={result['time_s']:.3f} s")
    print(f"applied budget: {result['previous_budget_n']:.6f} -> {result['budget_n']:.6f} N")
    print(f"load estimate: {result['previous_load_n']:.6f} -> {result['load_n']:.6f} N")
    print(f"coefficient: {result['mu_before']:.9f} -> {result['mu_after']:.9f}")
    for key in ("previous_force_n", "desired_force_n", "force_n"):
        print(f"{key}: {np.array2string(result[key], precision=6, floatmode='fixed')}")
    print(f"vector change: {result['step_n']:.6f} N (limit 0.040000 N/step)")
    print(f"request norm: {result['magnitude_before_n']:.6f} -> {result['magnitude_n']:.6f} N")
    print(f"first norm <= 6 N: k={result['first_below_index']}, t={result['first_below_time_s']:.3f} s")
    print(f"packet recovery: t={result['recovery_s']:.3f} s")
    if arguments.check_tamper:
        check_tamper(trace, result["index"])
        print("in-memory coefficient +0.0001: REJECTED (expected)")


if __name__ == "__main__":
    main()
