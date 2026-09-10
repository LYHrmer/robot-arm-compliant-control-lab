"""One-parameter velocity-error time arm of the observed public-24 online seam.

This runner reuses the observed public-24 online controller of tools.onset_observer
unchanged: the same frozen 24-case grid and schedules, the same rotation gain scales,
the same force budgets, the same control loop and the same recorded observation. The
only input that changes is the tangential compensation velocity-error time, so no
default is changed; 0.05 s remains the shipped default recorded in the archive and
0.10 s is executed as a counterfactual only. Cases come from the existing public
choice functions (tools.budget_public24.cases); only case 23 is executed here.
"""

from __future__ import annotations

import json
from dataclasses import fields, replace

import numpy as np

from compliant_control_lab.surface_control import SurfaceFrame
from compliant_control_lab.surface_experiment import _constructor_config
from compliant_control_lab.surface_simulation import (
    SurfaceSimulator,
    SurfaceTrialResult,
    yaw_frame,
)
from tools import budget_public24 as public24
from tools import onset_observer as observer
from tools import rotation_gain_regression as rotation

IDENTITY = "velocity-error-time-rotation-gain-public24-v1"
TIMES_S = (0.05, 0.10)
DEFAULT_VELOCITY_ERROR_TIME_S = 0.05
CASE_INDEX = 23
OBSERVATION_FIELDS = observer.OBSERVATION_FIELDS


def controller(frame: SurfaceFrame, scale, budget, velocity_error_time):
    """Fresh observed public-24 controller with only velocity_error_time replaced."""
    value = public24._choice(velocity_error_time, TIMES_S, "velocity error time")
    control = observer.controller(frame, scale, budget)
    compensation = control._base.tangential
    if float(compensation.velocity_error_time) != DEFAULT_VELOCITY_ERROR_TIME_S:
        raise ValueError(
            "the default compensation velocity error time is no longer "
            f"{DEFAULT_VELOCITY_ERROR_TIME_S} s"
        )
    control._base.tangential = replace(compensation, velocity_error_time=value)
    return control


def parameters(budget, velocity_error_time) -> dict:
    """Return the executed compensation constructor fields of one budget/time pair."""
    compensation = controller(yaw_frame(0.0), 1.0, budget, velocity_error_time)._base.tangential
    values = {f.name: getattr(compensation, f.name) for f in fields(compensation) if f.init}
    return json.loads(json.dumps(values, default=float))


def controller_document(scale, budget, velocity_error_time) -> dict:
    """Return the actual constructor configuration that executes this triple."""
    return _constructor_config(
        controller(yaw_frame(0.0), scale, budget, velocity_error_time)._base
    )


def run_trial(case: dict, scale, budget, velocity_error_time) -> SurfaceTrialResult:
    """Run the accepted public-24 case at one gain, budget and velocity-error time."""
    gain = public24._choice(scale, public24.SCALES, "rotation gain scale")
    limit = public24._choice(budget, public24.BUDGETS_N, "force budget")
    time_s = public24._choice(velocity_error_time, TIMES_S, "velocity error time")
    if case["case_index"] != CASE_INDEX:
        raise ValueError(f"only public-24 case {CASE_INDEX} is executed: {case['case_index']!r}")
    angle = np.deg2rad(case["task"].yaw_deg)
    frame = SurfaceFrame.from_normal(np.array([np.cos(angle), np.sin(angle), 0.0]))
    control = controller(frame, gain, limit, time_s)
    compensation = control._base.tangential
    sim = SurfaceSimulator(frame, case["scenario"], case["config"], case["task"],
                           rotation.METHODS[public24.METHOD])
    rows = []
    for step in range(round(case["config"].duration / case["config"].timestep)):
        sample = sim.sample()
        if step == 0:
            control.reset(sample.state)
        wrench = control.compute(sample.state, sample.target, sim.config.timestep)
        rows.append(compensation.observation)
        sim.step(wrench, control)
    result = sim.result()
    result.trace.update(rotation_gain_scale=np.array(gain), case_index=np.array(case["case_index"]),
                        method=np.array(public24.METHOD), max_force_n=np.array(limit),
                        velocity_error_time_s=np.array(time_s))
    result.trace.update(
        {name: np.asarray([row[name] for row in rows]) for name in OBSERVATION_FIELDS}
    )
    result.trace.setdefault("controller_frame_rotation", frame.rotation.copy())
    return result
