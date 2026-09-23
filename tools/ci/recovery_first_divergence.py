"""Read-only arithmetic diagnostic for one published recovery trace."""

import json
import math
import sys

import numpy as np

from tools.reversal_recovery import study


def _hex(value):
    return float(value).hex()


def _first(saved, rebuilt, names, time):
    changed = np.flatnonzero(saved != rebuilt)
    if not len(changed):
        return None
    index, column = np.unravel_index(int(changed[0]), saved.shape)
    return {
        "index": int(index), "column": names[column], "time_s": float(time[index]),
        "saved_hex": _hex(saved[index, column]),
        "rebuilt_hex": _hex(rebuilt[index, column]),
        "absolute_difference": float(abs(saved[index, column] - rebuilt[index, column])),
    }


def first_divergence():
    """Observe one frozen replay without replacing any implementation function."""
    protocol = study.protocol_document()
    spec = study.specifications(protocol)[0]  # Falling, seed 11, adaptive6_8.
    _, parameters = study.make_controller(spec, protocol)
    relative = "results/franka_reversal_recovery/" + spec["trace_path"]
    trace = study.previous._load_trace(study.ROOT / relative)
    captured = {}
    replay_code = study.replay_compensation.__code__

    def observe(frame, event, _argument):
        if frame.f_code is not replay_code:
            return None
        if event == "call":
            frame.f_trace_lines = False
            return observe
        if event == "return" and _argument is not None:
            for name in ("alpha", "scheduler", "outputs", "before", "after",
                         "force", "velocity", "speed"):
                captured[name] = frame.f_locals[name]
        return None

    previous_trace = sys.gettrace()
    try:
        sys.settrace(observe)
        audit = study.replay_compensation(
            trace, parameters, spec["method"], protocol["timestep_s"],
        )
    finally:
        sys.settrace(previous_trace)

    names = ("load_budget_applied_n", "load_budget_next_n",
             "load_estimate_n", "load_projected_n")
    saved_scheduler = np.column_stack([trace[name] for name in names])
    scheduler = _first(saved_scheduler, captured["scheduler"], names, trace["time"])
    if scheduler is not None:
        index = scheduler["index"]
        speed = captured["speed"][index]
        unit = captured["velocity"][index] / speed if speed > 0 else None
        scheduler.update({
            "previous_load_hex": {
                "saved": _hex(saved_scheduler[index - 1, 2]) if index else _hex(0),
                "rebuilt": _hex(captured["scheduler"][index - 1, 2]) if index else _hex(0),
            },
            "projected_hex": {
                "saved": _hex(saved_scheduler[index, 3]),
                "rebuilt": _hex(captured["scheduler"][index, 3]),
            },
            "force_hex": [_hex(value) for value in captured["force"][index]],
            "unit_velocity_hex": [_hex(value) for value in unit] if unit is not None else None,
            "speed_hex": _hex(speed),
        })
    coefficients = ("controller_coefficient_before_compute", "controller_coefficient_after_compute")
    alpha_argument = -protocol["timestep_s"] / parameters["load_time_constant"]
    return {
        "trace_path": relative,
        "identity": {name: spec[name] for name in ("scenario", "seed", "method")},
        "alpha_argument_hex": _hex(alpha_argument),
        "alpha_hex": _hex(captured["alpha"]),
        "math_alpha_hex": _hex(-math.expm1(alpha_argument)),
        "first_scheduler_difference": scheduler,
        "first_compensation_difference": _first(
            trace["load_compensation_force_local"], captured["outputs"],
            ("x", "y", "z"), trace["time"],
        ),
        "first_coefficient_difference": _first(
            np.column_stack([trace[name] for name in coefficients]),
            np.column_stack([captured["before"], captured["after"]]), coefficients, trace["time"],
        ),
        "audit": audit,
    }


if __name__ == "__main__":
    print(json.dumps(first_divergence(), indent=2, allow_nan=False))
