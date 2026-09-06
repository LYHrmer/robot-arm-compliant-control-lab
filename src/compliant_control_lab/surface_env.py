"""Gymnasium Adapter: 50 Hz residual decisions over the same 500 Hz surface simulator."""

from copy import deepcopy
from dataclasses import replace
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from compliant_control_lab.surface_control import SurfaceFrame
from compliant_control_lab.surface_policy import (
    OBSERVATION_NAMES as POLICY_OBSERVATION_NAMES,
)
from compliant_control_lab.surface_policy import (
    OBSERVATION_SCALES as POLICY_OBSERVATION_SCALES,
)
from compliant_control_lab.surface_policy import (
    OBSERVATION_UNITS as POLICY_OBSERVATION_UNITS,
)
from compliant_control_lab.surface_policy import (
    SurfaceResidualConfig,
    SurfaceResidualController,
)
from compliant_control_lab.surface_simulation import (
    SurfaceScenario,
    SurfaceSimulationConfig,
    SurfaceSimulator,
    SurfaceTask,
    SurfaceTrialResult,
    yaw_frame,
)

PHYSICS_DT = 0.002
POLICY_SUBSTEPS = 10
MAX_REWARD_PER_SECOND = 1.0
REWARD_WEIGHTS = {
    "force": 0.2,
    "tangent": 0.2,
    "contact": 0.2,
    "orientation": 0.1,
    "action": 0.1,
    "intervention": 0.2,
}
OBSERVATION_NAMES = (
    *POLICY_OBSERVATION_NAMES,
    *(f"joint_position_{i}" for i in range(7)),
    *(f"joint_velocity_{i}" for i in range(7)),
    "kinematic_age",
    "force_age",
)
ENCODER_SCALES = np.array([*(np.pi,) * 7, *(2.0,) * 7, 0.1, 0.1])
OBSERVATION_SCHEMA = "surface_env_v1"
OBSERVATION_SCALES = (*POLICY_OBSERVATION_SCALES, *ENCODER_SCALES)
OBSERVATION_UNITS = (*POLICY_OBSERVATION_UNITS, *("rad",) * 7, *("rad/s",) * 7, "s", "s")
_FAIL_CLOSED_REASONS = {
    "invalid_action",
    "torque_missing_context",
    "torque_nominal_outside",
    "torque_nonfinite",
    "torque_verification_failed",
}


class SurfaceLearningEnv(gym.Env):
    """Actor inputs are measured controller inputs, current encoders and sample ages.

    The first 33 components follow surface_v1; seven q/pi, seven qd/2 and two
    ages/.1 complete the 49-vector. All are float64 clipped to [-3, 3]. No ideal
    contact or scenario parameters enter this observation. Model-context use in
    the nominal controller retains the simulator's known-dynamics assumption.

    Each reward is a time integral: 1/s alive minus six nonnegative bounded costs
    whose weights sum to one. Force, tangent and orientation scales are 12 N,
    .02 m and .2 rad; their squared normalized errors are clipped to [0,1].
    A failure deducts the ENTIRE horizon's maximum reward plus one, so every
    failed episode has undiscounted return <= -1, while a normal episode has
    return >= 0. Gates are independent of reward and checked every physics step.

    prepare advances nominal once even for the final next observation, without
    applying that command. Invalid terminal observations use the last finite
    observation with an explicit validity flag. Evaluator truth stays in info
    and result(), never in actor observations; callers must not feed info to actors.
    """

    metadata: ClassVar[dict] = {"render_modes": [], "render_fps": 50}

    def __init__(
        self,
        controller_frame: SurfaceFrame,
        *,
        scenario=None,
        config=None,
        task=None,
        nominal_kind="friction",
        residual_config=None,
    ):
        self.frame = controller_frame
        self.scenario = scenario or SurfaceScenario()
        self.config = config or SurfaceSimulationConfig(contact_model="smooth")
        self.task = task or SurfaceTask()
        if self.config.timestep != PHYSICS_DT:
            raise ValueError("SurfaceLearningEnv requires a 500 Hz physics timestep")
        if nominal_kind not in {"friction", "adaptive"}:
            raise ValueError("nominal_kind must be friction or adaptive")
        self.nominal_kind = nominal_kind
        self.residual_config = residual_config or SurfaceResidualConfig()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float64)
        self.observation_space = spaces.Box(-3.0, 3.0, shape=(49,), dtype=np.float64)
        self.render_mode = None
        self._simulator = None
        self._active = False

    def _prepare_observation(self, sample):
        policy = self._controller.prepare(sample.state, sample.target, PHYSICS_DT)
        ages = [
            sample.time - sample.measured_kinematic_sample_time,
            sample.time - sample.measured_wrench_sample_time,
        ]
        values = np.r_[policy, np.r_[sample.q, sample.joint_velocity, ages] / ENCODER_SCALES]
        if values.shape != (49,) or not np.all(np.isfinite(values)) or min(ages) < -1e-12:
            raise ValueError("invalid measured observation or sample age")
        self._observation = np.clip(values, -3.0, 3.0).astype(np.float64)
        self._observation_time = float(sample.time)
        return self._observation.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError(
                "configure scenario/task explicitly at construction, not reset options"
            )
        self._simulation_seed = int(self.np_random.integers(0, 2**31 - 1))
        self._episode_config = replace(self.config, seed=self._simulation_seed)
        self._simulator = SurfaceSimulator(
            self.frame,
            self.scenario,
            self._episode_config,
            self.task,
            controller_kind="surface_residual",
        )
        self._controller = SurfaceResidualController(
            self.frame, self.nominal_kind, self.residual_config
        )
        sample = self._simulator.sample()
        self._controller.reset(sample.state)
        self._actions, self._executed, self._policy_step = [], 0, 0
        self._lost_contact = 0.0
        observation = self._prepare_observation(sample)
        self._active = True
        return observation, {
            "simulation_seed": self._simulation_seed,
            "observation_time_s": 0.0,
            "action_stage": "normalized_local_residual",
            "terminal_observation_valid": True,
        }

    def _stage(self, command):
        valid = command.raw_action.shape == (3,) and np.all(np.isfinite(command.raw_action))
        return {
            "policy_action": command.raw_action.copy() if valid else np.zeros(3),
            "policy_action_valid": bool(valid),
            "policy_step_index": self._policy_step,
            "requested_residual_force_local": command.requested_force.copy(),
            "filtered_residual_force_local": command.filtered_force.copy(),
            "applied_residual_force_local": command.applied_force.copy(),
            "residual_projection_scale": float(command.projection_scale),
            "reasons": command.reasons,
            "endpoint_time": np.nan,
            "endpoint_position": np.full(3, np.nan),
            "endpoint_linear_velocity": np.full(3, np.nan),
            "endpoint_contact_gap_m": np.nan,
            "endpoint_valid": False,
        }

    def _safety_reasons(self, row):
        if not all(np.all(np.isfinite(value)) for value in row.values()):
            return ["nonfinite"]
        force = float(row["true_normal_force"])
        self._lost_contact = (
            self._lost_contact + PHYSICS_DT
            if row["time"] >= self.config.evaluation_start and force <= 0.5
            else 0.0
        )
        return [
            reason
            for failed, reason in (
                (force > 35.0, "raw_force_limit"),
                (-float(row["true_contact_gap_m"]) > 0.002, "penetration_limit"),
                (np.linalg.norm(row["linear_velocity"]) > 0.3, "speed_limit"),
                (
                    np.any(np.abs(row["commanded_torque"] - row["applied_torque"]) > 1e-9),
                    "actuator_clip",
                ),
                (self._lost_contact + 1e-12 >= 0.10, "contact_lost_timeout"),
            )
            if failed
        ]

    def _costs(self, row, stage):
        if not all(np.all(np.isfinite(value)) for value in row.values()):
            return {name: 1.0 for name in REWARD_WEIGHTS}
        normal = yaw_frame(self.scenario.wall_yaw_deg).rotation[:, 0]
        error = row["position"] - row["target_position"]
        tangent = error - normal * (normal @ error)
        return {
            "force": float(
                np.clip(((row["true_normal_force"] - row["target_normal_force"]) / 12) ** 2, 0, 1)
            ),
            "tangent": float(np.clip(np.sum(tangent**2) / 0.02**2, 0, 1)),
            "contact": float(
                row["time"] >= self.config.evaluation_start and row["true_normal_force"] <= 0.5
            ),
            "orientation": float(np.clip((row["orientation_error_rad"] / 0.2) ** 2, 0, 1)),
            "action": float(np.mean(np.clip(stage["policy_action"], -1, 1) ** 2))
            if stage["policy_action_valid"]
            else 1.0,
            "intervention": float(
                any(
                    reason not in {"contact_lost", "contact_not_ready"}
                    for reason in stage["reasons"]
                )
            ),
        }

    def step(self, action):
        if not self._active:
            raise RuntimeError("reset is required before step or after termination/close")
        issue, shape = None, None
        try:
            held_action = np.asarray(action, dtype=np.float64).copy()
            shape = held_action.shape
            if shape != (3,):
                issue = "shape"
            elif not np.all(np.isfinite(held_action)):
                issue = "nonfinite"
            elif np.any(np.abs(held_action) > 1):
                issue = "out_of_space"
        except (ValueError, TypeError):
            issue = "conversion"
        if issue:
            held_action = None  # The controller's invalid-input path applies zero residual.
        reward, reasons, stages, block_rows = 0.0, [], [], []
        weighted_costs = dict.fromkeys(REWARD_WEIGHTS, 0.0)
        observation_valid, exception_detail = True, None
        start = self._executed
        for _ in range(POLICY_SUBSTEPS):
            try:
                wrench = self._controller.apply(held_action)
                stage = self._stage(self._controller.last_command)
            except (ValueError, RuntimeError, FloatingPointError) as error:
                reasons.append("controller_failure")
                exception_detail, observation_valid = str(error), False
                break
            reasons.extend(reason for reason in stage["reasons"] if reason in _FAIL_CLOSED_REASONS)
            if issue == "out_of_space":
                reasons.append("out_of_space")
            stages.append({**deepcopy(stage), "execution_status": "unconfirmed"})
            try:
                row = self._simulator.step(wrench, controller=self._controller.nominal)
            except (ValueError, RuntimeError, FloatingPointError) as error:
                reasons.append("simulator_failure")
                exception_detail, observation_valid = str(error), False
                # A nonfinite executed row is retained by the simulator, not sanitized.
                try:
                    trace = self._simulator.result().trace
                except RuntimeError:
                    break
                count = len(trace["time"])
                if count != self._executed + 1:
                    break
                row = {
                    key: value[-1]
                    for key, value in trace.items()
                    if np.ndim(value) > 0
                    and len(value) == count
                    and key != "controller_frame_rotation"
                }
            self._executed += 1
            self._actions.append(stage)
            stages[-1]["execution_status"] = "recorded"
            block_rows.append(row)
            reasons.extend(self._safety_reasons(row))
            costs = self._costs(row, stage)
            for name, weight in REWARD_WEIGHTS.items():
                weighted_costs[name] += PHYSICS_DT * weight * costs[name]
            reward += PHYSICS_DT * max(
                0.0,
                MAX_REWARD_PER_SECOND - sum(REWARD_WEIGHTS[name] * costs[name] for name in costs),
            )
            if observation_valid:
                try:
                    sample = self._simulator.sample()
                    endpoint = self._simulator.evaluator_kinematics()
                    if not all(np.all(np.isfinite(value)) for value in endpoint.values()):
                        raise ValueError("nonfinite integrated endpoint")
                    stage.update(endpoint)
                    stages[-1].update(deepcopy(endpoint))
                    if np.linalg.norm(endpoint["endpoint_linear_velocity"]) > 0.3:
                        reasons.append("speed_limit")
                    if -endpoint["endpoint_contact_gap_m"] > 0.002:
                        reasons.append("penetration_limit")
                    self._prepare_observation(sample)
                except (ValueError, RuntimeError, FloatingPointError) as error:
                    reasons.append("observation_failure")
                    exception_detail, observation_valid = str(error), False
            if reasons or self._executed >= round(self.config.duration / PHYSICS_DT):
                break
        terminated = bool(reasons)
        truncated = not terminated and self._executed >= round(self.config.duration / PHYSICS_DT)
        penalty = self.config.duration * MAX_REWARD_PER_SECOND + 1.0 if terminated else 0.0
        self._active = not (terminated or truncated)
        self._policy_step += 1
        info = {
            "simulation_seed": self._simulation_seed,
            "action_stage": "normalized_local_residual",
            "action_stages": stages,
            "attempted_substeps": len(stages),
            "physics_substeps": self._executed - start,
            "step_duration_s": (self._executed - start) * PHYSICS_DT,
            "elapsed_s": self._executed * PHYSICS_DT,
            "observation_time_s": self._observation_time,
            "terminal_observation_valid": observation_valid,
            "termination_reasons": tuple(dict.fromkeys(reasons)),
            "reward_cost_integrals": weighted_costs,
            "termination_penalty": penalty,
            "exception_detail": exception_detail,
            "input_action_issue": issue,
            "input_action_shape": shape,
            "invalid_input_repr": repr(action)[:256] if issue else None,
        }
        if block_rows:
            block_trace = {
                key: np.asarray([row[key] for row in block_rows]) for key in block_rows[0]
            }
            info["evaluator_step_metrics"] = SurfaceTrialResult(
                block_trace, self.scenario, self._episode_config, self.task
            ).metrics()
        if (terminated or truncated) and self._executed:
            info["evaluator_episode_metrics"] = self.result().metrics()
        return self._observation.copy(), float(reward - penalty), terminated, truncated, info

    def result(self):
        """Owned executed trace; evaluator truth may include nonfinite failure evidence."""
        if self._simulator is None:
            raise RuntimeError("reset and at least one executed step are required")
        result = self._simulator.result()
        if len(result.trace["time"]) != len(self._actions):
            raise RuntimeError("executed action trace is incomplete")
        result.trace.update(
            {
                key: np.asarray([stage[key] for stage in self._actions])
                for key in self._actions[0]
                if key != "reasons"
            }
        )
        return result

    def close(self):
        self._active = False
