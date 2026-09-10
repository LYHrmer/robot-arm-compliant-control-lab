"""Rotation-gain by force-budget interaction adapter for the frozen budget cases.

The six preselected seed-11 combined/no_bias cases run at rotation gain 1 and 2
under the 6 N and 8 N compensation budgets. Gain 1 is reused from the pinned
force-budget archive, so this module only builds the actual-gain controller, runs
one trial with honest gain metadata and rebuilds every metric independently from
the trace arrays. Nothing here publishes an archive and the matched
gain-2-minus-gain-1 comparison stays with the caller.
"""

from __future__ import annotations

import math
import numbers

import numpy as np

from compliant_control_lab.online_compensation_experiment import (
    ARMS,
    RECOVERY_ABSOLUTE_FORCE_ERROR_N,
    RECOVERY_TANGENT_RMSE_MM,
    RECOVERY_WINDOW_S,
    ScheduledSurfaceSimulator,
    _case_document,
    _compact_trace,
    _constructor_config,
)
from compliant_control_lab.surface_control import SurfaceAdaptiveController
from compliant_control_lab.surface_simulation import yaw_frame
from tools import audit_online_compensation_errors as audit
from tools import combined_residual_ablation as old
from tools import combined_residual_diagnostics, cross_surface_regression
from tools import compensation_budget_study as study

GAIN_SCALES = (1., 2.)
REFERENCE = {
    "directory": "results/franka_compensation_budget",
    "manifest_sha256": "6f183c8112ca5d1f6512357650b26f8e499d53ed8489b666c9274c9e12047dd2",
}
RECOVERY = {
    "window_s": RECOVERY_WINDOW_S,
    "absolute_force_error_n": RECOVERY_ABSOLUTE_FORCE_ERROR_N,
    "tangent_rmse_mm": RECOVERY_TANGENT_RMSE_MM,
}


def _scale(value) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"rotation gain scale must be a real number: {value!r}")
    number = float(value)
    if not math.isfinite(number) or number not in GAIN_SCALES:
        raise ValueError(f"rotation gain scale must be one of {GAIN_SCALES}: {value!r}")
    return number


def cases() -> list[tuple[str, object]]:
    """Return the six frozen (variant, ProtocolCase) pairs of the force-budget screen."""
    return study.study_cases()


def controller(case, scale, budget):
    """Build the online arm of this case at the actual gain scale and force budget."""
    gain, force = _scale(scale), study._budget(budget)
    return study._with_budget(
        SurfaceAdaptiveController(
            yaw_frame(case.task.yaw_deg + case.controller_yaw_error_deg),
            tangential_mode=ARMS[study.ARM],
            rotation_gain_scale=gain,
        ),
        force,
    )


def controller_document(scale, budget) -> dict:
    """Constructor configuration of the executed base; the surface frame is not a field."""
    return _constructor_config(controller(cases()[0][1], scale, budget)._base)


def run_trial(case, scale, budget):
    """Run one case at the actual rotation gain scale and force budget on the online arm."""
    gain, force = _scale(scale), study._budget(budget)
    frame = yaw_frame(case.task.yaw_deg + case.controller_yaw_error_deg)
    simulator = ScheduledSurfaceSimulator(case, frame, study.ARM)
    result = old.run_observed_loop(case, controller(case, gain, force), simulator)
    # The legacy observed loop stamps the gain-1 default, so record what actually ran.
    result.trace["rotation_gain_scale"] = np.array(gain)
    result.trace["max_force_n"] = np.array(force)
    return result


def compact(trace: dict) -> dict:
    return study.diagnostic_trace(trace)


def case_document(variant: str, case) -> dict:
    return {"variant": variant, **_case_document(case)}


def reference_trace(variant: str, case, budget) -> str:
    """Return the archive-relative gain-1 diagnostic trace of this case and budget."""
    return f"traces/{study.stem(variant, case, budget)}__diagnostic.npz"


def _summary_protocol(scale, budget) -> dict:
    """Return the ablation protocol plus the two fields the independent summary reads.

    combined_residual_ablation.protocol_document carries neither the recovery
    thresholds nor controller_constructor_configurations, so both are supplied here.
    The online entry is the constructor of the run being summarized, so the bounds
    that gate the summary follow the actual budget instead of the 6 N default.
    """
    return {
        **old.protocol_document(),
        "recovery": RECOVERY,
        "controller_constructor_configurations": {
            "online": {"safe_adaptive_base": controller_document(scale, budget)},
        },
    }


def metrics(trace: dict, case, budget) -> tuple[dict, list[dict], list[dict]]:
    """Rebuild overall, phase and window metrics from the trace arrays alone.

    The executed gain scale comes from the trace, not from an argument, so the
    summary always describes the run that produced these arrays.
    """
    force = study._budget(budget)
    scale = _scale(float(np.asarray(trace["rotation_gain_scale"])))
    summary, windows = audit._summarize(
        _compact_trace(trace, case), _case_document(case), study.ARM,
        _summary_protocol(scale, force),
    )
    time = np.asarray(trace["time"])
    overall = study.screen_metrics(summary)
    overall = {
        **overall,
        **cross_surface_regression.extra_metrics(trace, slice(None)),
        **old.diagnostic_rates(trace, time >= case.config.evaluation_start),
        # Archived diagnostic traces do not carry the observer scalars of their run.
        **{name: float(trace[name]) for name in old.AUDIT_FIELDS if name in trace},
        **old.absolute_checks(overall),
        "max_uncapped_tangential_amplitude_n": float(np.max(
            trace["controller_coefficient_before_compute"]
            * trace["diagnostic_corrected_force_n"])),
    }
    phases = []
    for window, phase in zip(windows, case.phases, strict=True):
        row = study.screen_metrics(window)
        mask = (time >= phase.start_s) & (time < phase.end_s)
        phases.append({**row, **old.diagnostic_rates(trace, mask), **old.absolute_checks(row)})
    diagnostics = [
        {"window": name, **combined_residual_diagnostics.analyze_trace(
            trace, surface_yaw_deg=case.scenario.wall_yaw_deg,
            controller_yaw_error_deg=case.controller_yaw_error_deg,
            start_s=start, end_s=end)}
        for name, start, end in old.DIAGNOSTIC_WINDOWS
    ]
    return overall, phases, diagnostics
