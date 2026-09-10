"""Bounded-compensation force-budget arm of the frozen public-24 rotation-gain grid.

This adapter reuses the frozen rotation-gain tooling unchanged: the same public
24-case grid and schedules, the same online compensation arm at rotation gain
scales 1 and 2, the same control loop and the same read-only audit arithmetic.
The only input that changes is the tangential compensation force budget, so no
default is changed; 6 N remains the shipped default recorded in the archive and
8 N is executed as a counterfactual only.
"""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import asdict, replace

import numpy as np

from compliant_control_lab.surface_control import SurfaceFrame
from compliant_control_lab.surface_experiment import _constructor_config
from compliant_control_lab.surface_simulation import (
    SurfaceSimulator,
    SurfaceTrialResult,
    yaw_frame,
)
from tools import audit_rotation_gain_regression as audit
from tools import rotation_gain_regression as rotation

IDENTITY = "budget-rotation-gain-public24-v1"
METHOD = "online"
SCALES = rotation.SCALES
BUDGETS_N = (6.0, 8.0)
DEFAULT_MAX_FORCE_N = 6.0
COMPACT_FIELDS = rotation.COMPACT_FIELDS
TRACE_METADATA = (*audit.TRACE_METADATA, "max_force_n")
REFERENCE = {
    "directory": "results/franka_rotation_gain_public24",
    "manifest_sha256": "bbd94ed631c5b5858308eaba762d41d4412c1f69400298e62571b2f27c3f3fee",
}


def _choice(value, allowed: tuple[float, ...], label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{label} must be a real number: {value!r}")
    number = float(value)
    if not math.isfinite(number) or number not in allowed:
        raise ValueError(f"{label} must be one of {allowed}: {value!r}")
    return number


def cases() -> list[dict]:
    """Return the original ordered 24 case dicts with every frozen schedule intact."""
    _, configurations, _ = rotation.load_reference()
    return rotation.make_protocol(list(range(24)), configurations)[1]


def _with_budget(control, budget: float):
    """Replace only max_force on the fresh online compensation of this controller."""
    compensation = control._base.tangential
    if compensation is None or compensation.mode != METHOD:
        raise ValueError("the force-budget arm requires the online compensation")
    if float(compensation.max_force) != DEFAULT_MAX_FORCE_N:
        raise ValueError(
            f"the default compensation force budget is no longer {DEFAULT_MAX_FORCE_N} N"
        )
    control._base.tangential = replace(compensation, max_force=budget)
    return control


def controller(scale, budget):
    """Fresh online controller at one rotation gain with only the force budget replaced."""
    gain = _choice(scale, SCALES, "rotation gain scale")
    limit = _choice(budget, BUDGETS_N, "force budget")
    return _with_budget(rotation.controller(yaw_frame(0.0), METHOD, gain), limit)


def controller_document(scale, budget) -> dict:
    """Return the actual constructor configuration that executes this pair."""
    return _constructor_config(controller(scale, budget)._base)


def run_trial(case: dict, scale, budget) -> SurfaceTrialResult:
    """Run one public-24 case on the online arm at one gain and one executed budget."""
    gain = _choice(scale, SCALES, "rotation gain scale")
    limit = _choice(budget, BUDGETS_N, "force budget")
    angle = np.deg2rad(case["task"].yaw_deg)
    frame = SurfaceFrame.from_normal(np.array([np.cos(angle), np.sin(angle), 0.0]))
    control = _with_budget(rotation.controller(frame, METHOD, gain), limit)
    sim = SurfaceSimulator(frame, case["scenario"], case["config"], case["task"],
                           rotation.METHODS[METHOD])
    for step in range(round(case["config"].duration / case["config"].timestep)):
        sample = sim.sample()
        if step == 0:
            control.reset(sample.state)
        sim.step(control.compute(sample.state, sample.target, sim.config.timestep), control)
    result = sim.result()
    result.trace.update(rotation_gain_scale=np.array(gain), case_index=np.array(case["case_index"]),
                        method=np.array(METHOD), max_force_n=np.array(limit))
    return result


def compact(trace: dict) -> dict:
    """Keep the frozen compact fields plus the scalar gain, case, method and budget."""
    names = (*COMPACT_FIELDS, *TRACE_METADATA)
    missing = [name for name in names if name not in trace]
    if missing:
        raise ValueError(f"trace is missing compact fields: {missing}")
    return {name: np.asarray(trace[name]) for name in names}


def case_document(case: dict) -> dict:
    """Return the plain JSON case document in the archived scenario/config/task shape."""
    return json.loads(json.dumps({
        "case_index": case["case_index"],
        **{key: asdict(case[key]) for key in ("scenario", "config", "task")},
    }))


def metrics(trace: dict, case: dict) -> dict:
    """Reconstruct every archived metric independently through the read-only audit."""
    return audit._metrics(trace, case_document(case))


def reference_trace(case: dict, scale) -> str:
    """Relative POSIX name of the archived default-budget compact trace of this case."""
    gain = _choice(scale, SCALES, "rotation gain scale")
    return f"traces/s{int(gain)}__{METHOD}__case_{case['case_index']:02d}__compact.npz"
