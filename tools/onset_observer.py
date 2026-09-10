"""Exact-input observation of the bounded online compensation force/advance seam.

The observer subclasses the shipped compensation and copies the arguments and the
resulting internal state around ONE call of the parent force/advance. It is not a
second evaluator and not a reconstruction of a pre-governor target: every recorded
number is either an argument the executing controller passed, a value the parent
stored, or a formula evaluated on those copies. Nothing computed here is written
back, so the executed trajectory is the plain public-24 online arm trajectory.

Vectors are recorded in the local surface frame, exactly as the controller seam
delivers them, and the velocity/error vectors are the tangential projections the
compensation itself consumes. The proposed drive and increment fields describe
what an accepted update would apply; they are not evidence that integration ran.
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np

from compliant_control_lab.surface_control import SurfaceFrame
from compliant_control_lab.surface_simulation import SurfaceSimulator, SurfaceTrialResult
from compliant_control_lab.tangential_compensation import TangentialCompensation, _tangent
from tools import budget_public24 as public24
from tools import rotation_gain_regression as rotation

OBSERVATION_FIELDS = (
    "obs_normal_local",
    "obs_position_error_local_m",
    "obs_velocity_error_local_m_s",
    "obs_target_velocity_local_m_s",
    "obs_measured_velocity_local_m_s",
    "obs_direction_local",
    "obs_previous_direction_local",
    "obs_previous_force_local_n",
    "obs_requested_force_local_n",
    "obs_corrected_force_n",
    "obs_contact_blend",
    "obs_target_normal_force_n",
    "obs_in_contact",
    "obs_coefficient_before_force",
    "obs_coefficient_after_force",
    "obs_coefficient_after_advance",
    "obs_motion_elapsed_before_s",
    "obs_motion_elapsed_after_s",
    "obs_active",
    "obs_update_ready",
    "obs_amplitude_capped",
    "obs_slew_limited",
    "obs_position_drive_m",
    "obs_velocity_drive_m",
    "obs_drive_m",
    "obs_candidate_increment",
    "obs_limited_increment",
    "obs_advance_called",
    "obs_allow_integration",
    "obs_dt_s",
)
AGREEMENT_TOLERANCE = 1e-12


def _copy(vector) -> np.ndarray:
    return np.array(vector, dtype=float)


def _observed_tangent(normal, vector) -> np.ndarray:
    """Project for the record only; an unobservable argument never raises into control."""
    try:
        return _tangent(normal, vector)
    except (TypeError, ValueError):
        return np.full(3, np.nan)


class ObservedCompensation(TangentialCompensation):
    """Compensation that records the governed arguments of its own seam.

    The constructor, its validation and every parameter default stay inherited.
    force() starts a fresh record before the parent call, so the reset() that the
    parent performs on an inactive cycle clears compensation state but not the
    record being built; advance() only appends to the record of that same cycle.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self._observation: dict = {}

    @property
    def observation(self) -> dict:
        """Copy of the most recent force (plus advance) cycle; arrays are never shared."""
        return {
            name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in self._observation.items()
        }

    def force(self, state, target, normal, normal_force, contact_blend, in_contact, *, dt=None):
        record = {
            "obs_normal_local": _copy(normal),
            "obs_previous_direction_local": self._previous_direction.copy(),
            "obs_previous_force_local_n": self._last_force.copy(),
            "obs_coefficient_before_force": float(self._equivalent_mu),
            "obs_motion_elapsed_before_s": float(self._motion_elapsed),
            "obs_advance_called": False,
            "obs_allow_integration": False,
        }
        self._observation = record
        requested = super().force(
            state, target, normal, normal_force, contact_blend, in_contact, dt=dt
        )
        local_normal = record["obs_normal_local"]
        error = _observed_tangent(local_normal, np.subtract(target.position, state.position))
        velocity_error = _observed_tangent(
            local_normal, np.subtract(target.linear_velocity, state.linear_velocity)
        )
        direction = self._direction.copy()
        active = bool(self._active)
        blend, force_n = float(contact_blend), float(normal_force)
        coefficient = float(self._equivalent_mu)
        dt_s = float(dt) if dt is not None else float("nan")
        usable = active and np.isfinite(dt_s)
        amplitude = min(coefficient * force_n, self.max_force)
        slew_delta = blend * amplitude * direction - _observed_tangent(
            local_normal, record["obs_previous_force_local_n"]
        )
        drive = float(direction @ (error + self.velocity_error_time * velocity_error))
        candidate = (
            dt_s * self.adaptation_gain * force_n / (force_n**2 + self.force_regularizer**2) * drive
            if usable
            else 0.0
        )
        limit = self.coefficient_rate_limit * dt_s
        record.update(
            obs_position_error_local_m=error,
            obs_velocity_error_local_m_s=velocity_error,
            obs_target_velocity_local_m_s=_observed_tangent(local_normal, target.linear_velocity),
            obs_measured_velocity_local_m_s=_observed_tangent(local_normal, state.linear_velocity),
            obs_direction_local=direction,
            obs_requested_force_local_n=_copy(requested),
            obs_corrected_force_n=force_n,
            obs_contact_blend=blend,
            obs_target_normal_force_n=float(target.normal_force),
            obs_in_contact=bool(in_contact),
            obs_coefficient_after_force=coefficient,
            obs_coefficient_after_advance=coefficient,
            obs_motion_elapsed_after_s=float(self._motion_elapsed),
            obs_active=active,
            obs_update_ready=bool(self._update_ready),
            obs_amplitude_capped=bool(active and coefficient * force_n >= self.max_force),
            obs_slew_limited=bool(
                active and float(np.linalg.norm(slew_delta)) > self.force_slew_rate * dt_s
            ),
            obs_position_drive_m=float(direction @ error),
            obs_velocity_drive_m=float(self.velocity_error_time * (direction @ velocity_error)),
            obs_drive_m=drive,
            obs_candidate_increment=candidate,
            obs_limited_increment=float(np.clip(candidate, -limit, limit)) if usable else 0.0,
            obs_dt_s=dt_s,
        )
        return requested

    def advance(self, state, target, normal, dt, allow_integration):
        record = self._observation
        if not record:
            raise ValueError("advance requires an observed force cycle")
        if np.isfinite(dt) and dt > 0:
            self._verify(state, target, normal, dt, record)
        super().advance(state, target, normal, dt, allow_integration)
        record["obs_advance_called"] = True
        record["obs_allow_integration"] = bool(allow_integration)
        record["obs_coefficient_after_advance"] = float(self._equivalent_mu)

    def _verify(self, state, target, normal, dt, record) -> None:
        """Confirm the advance seam still sees the governed inputs of the recorded force."""
        observed = {
            "obs_normal_local": _copy(normal),
            "obs_position_error_local_m": _observed_tangent(
                normal, np.subtract(target.position, state.position)
            ),
            "obs_velocity_error_local_m_s": _observed_tangent(
                normal, np.subtract(target.linear_velocity, state.linear_velocity)
            ),
        }
        for name, value in observed.items():
            expected = record[name]
            if not np.all(np.isfinite(expected)):
                continue
            if value.shape != expected.shape or np.any(
                np.abs(value - expected) > AGREEMENT_TOLERANCE
            ):
                raise ValueError(f"advance {name} disagrees with the observed force cycle")
        expected_dt = record["obs_dt_s"]
        if np.isfinite(expected_dt) and abs(float(dt) - expected_dt) > AGREEMENT_TOLERANCE:
            raise ValueError("advance dt disagrees with the observed force cycle")


def controller(frame: SurfaceFrame, scale, budget):
    """Fresh public-24 online controller with only the compensation instance observed."""
    gain = public24._choice(scale, public24.SCALES, "rotation gain scale")
    limit = public24._choice(budget, public24.BUDGETS_N, "force budget")
    control = public24._with_budget(rotation.controller(frame, public24.METHOD, gain), limit)
    compensation = control._base.tangential
    control._base.tangential = ObservedCompensation(
        **{f.name: getattr(compensation, f.name) for f in fields(compensation) if f.init}
    )
    return control


def run_trial(case: dict, scale, budget) -> SurfaceTrialResult:
    """Run one public-24 case on the observed online arm at one gain and one budget."""
    gain = public24._choice(scale, public24.SCALES, "rotation gain scale")
    limit = public24._choice(budget, public24.BUDGETS_N, "force budget")
    angle = np.deg2rad(case["task"].yaw_deg)
    frame = SurfaceFrame.from_normal(np.array([np.cos(angle), np.sin(angle), 0.0]))
    control = controller(frame, gain, limit)
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
                        method=np.array(public24.METHOD), max_force_n=np.array(limit))
    result.trace.update(
        {name: np.asarray([row[name] for row in rows]) for name in OBSERVATION_FIELDS}
    )
    result.trace.setdefault("controller_frame_rotation", frame.rotation.copy())
    return result
