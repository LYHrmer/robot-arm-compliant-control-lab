"""Opt-in surface task with causal tool-end F/T sensing; legacy trials are unchanged."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass

import mujoco
import numpy as np

from compliant_control_lab.franka_adaptive import FrankaSafeAdaptiveController
from compliant_control_lab.franka_control import (
    FrankaActuationContext,
    FrankaState,
    FrankaTarget,
    capture_franka_controller_telemetry,
    damped_nullspace_projector,
    orientation_error,
)
from compliant_control_lab.franka_simulation import _normal_contact_force, _site_state
from compliant_control_lab.surface_control import SurfaceAdaptiveController, SurfaceFrame
from compliant_control_lab.surface_sensing import ToolWrenchSensor, franka_surface_model_path


def yaw_frame(yaw_deg: float) -> SurfaceFrame:
    if not np.isfinite(yaw_deg):
        raise ValueError("yaw must be finite")
    angle = np.deg2rad(yaw_deg)
    return SurfaceFrame.from_normal(np.array([np.cos(angle), np.sin(angle), 0.0]))


def _finite_positive(name: str, value: float, *, allow_zero: bool = False) -> None:
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")


@dataclass(frozen=True)
class SurfaceTask:
    """Commanded plane pose, independent of physics and controller calibration.

    This development protocol supplies an accurate task plane from a nominal fixture
    description. It does not test discovering an unknown surface. All compared arms
    receive exactly the same world-coordinate target, including the oracle-frame arm.
    """

    yaw_deg: float = 0.0
    nominal_plane_x_m: float = 0.400

    def __post_init__(self) -> None:
        if not np.all(np.isfinite([self.yaw_deg, self.nominal_plane_x_m])):
            raise ValueError("task plane pose must be finite")

    def target_at(
        self,
        time: float,
        initial_position: np.ndarray,
        initial_rotation: np.ndarray,
        target_force: float,
    ) -> FrankaTarget:
        frame = yaw_frame(self.yaw_deg)
        normal, tangent1, tangent2 = frame.rotation.T
        phase = float(np.clip((time - 0.10) / 0.90, 0.0, 1.0))
        approach = phase**2 * (3.0 - 2.0 * phase)
        approach_rate = 6.0 * phase * (1.0 - phase) / 0.90
        anchor = np.array([self.nominal_plane_x_m, 0.0, 0.0])
        # Sphere radius .025 m; nominal center reference extends .010 m into the plane.
        displacement = normal * (normal @ (anchor - initial_position) - 0.025 + 0.010)
        position = initial_position + approach * displacement
        velocity = approach_rate * displacement
        if time > 1.20:
            elapsed, ramp, omega = time - 1.20, 0.50, 2.0 * np.pi * 0.20
            u = min(elapsed / ramp, 1.0)
            angle = omega * (ramp * (u**3 - 0.5 * u**4) if elapsed < ramp else elapsed - ramp / 2)
            rate = omega * u**2 * (3.0 - 2.0 * u)
            position += 0.055 * np.sin(angle) * tangent1
            position += 0.040 * (np.cos(angle) - 1.0) * tangent2
            velocity += rate * (0.055 * np.cos(angle) * tangent1 - 0.040 * np.sin(angle) * tangent2)
        return FrankaTarget(
            position, initial_rotation.copy(), velocity, np.zeros(3), target_force * approach
        )


@dataclass(frozen=True)
class SurfaceScenario:
    name: str = "nominal"
    wall_yaw_deg: float = 0.0
    wall_time_constant: float = 0.012
    wall_sliding_friction: float = 0.45
    tool_sliding_friction: float = 0.45
    tool_mass_kg: float = 0.10
    nominal_tool_mass_kg: float = 0.10
    position_noise_std_m: float = 0.0002
    force_noise_std_n: float = 0.15
    torque_noise_std_nm: float = 0.002
    force_bias_sensor_n: tuple[float, float, float] = (0.2, 0.0, 0.0)
    torque_bias_sensor_nm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    delay_steps: int = 0
    bias_compensation_scale: float = 1.0

    def __post_init__(self) -> None:
        yaw_frame(self.wall_yaw_deg)
        for name in ("wall_time_constant", "tool_mass_kg"):
            _finite_positive(name, getattr(self, name))
        for name in (
            "wall_sliding_friction",
            "tool_sliding_friction",
            "nominal_tool_mass_kg",
            "position_noise_std_m",
            "force_noise_std_n",
            "torque_noise_std_nm",
            "bias_compensation_scale",
        ):
            _finite_positive(name, getattr(self, name), allow_zero=True)
        for name in ("force_bias_sensor_n", "torque_bias_sensor_nm"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite 3-vector")
            object.__setattr__(self, name, tuple(float(x) for x in value))
        if (
            isinstance(self.delay_steps, bool)
            or not isinstance(self.delay_steps, (int, np.integer))
            or self.delay_steps < 0
        ):
            raise ValueError("delay_steps must be a nonnegative integer")


@dataclass(frozen=True)
class SurfaceSimulationConfig:
    duration: float = 4.5
    timestep: float = 0.002
    target_force: float = 12.0
    force_filter_time_constant: float = 0.02
    evaluation_start: float = 1.5
    seed: int = 11
    contact_model: str = "legacy"

    def __post_init__(self) -> None:
        if self.contact_model not in {"legacy", "smooth"}:
            raise ValueError("contact_model must be legacy or smooth")
        for name in ("duration", "timestep", "target_force"):
            _finite_positive(name, getattr(self, name))
        for name in ("force_filter_time_constant", "evaluation_start"):
            _finite_positive(name, getattr(self, name), allow_zero=True)
        steps = self.duration / self.timestep
        if round(steps) < 1 or not np.isclose(steps, round(steps), rtol=0, atol=1e-9):
            raise ValueError("duration must be a positive integer multiple of timestep")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, (int, np.integer))
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")


@dataclass
class SurfaceTrialResult:
    trace: dict[str, np.ndarray]
    scenario: SurfaceScenario
    config: SurfaceSimulationConfig
    task: SurfaceTask

    def metrics(self) -> dict:
        log = self.trace
        force, time = log["true_normal_force"], log["time"]
        mask = time >= self.config.evaluation_start
        contact = np.flatnonzero(force > 0)
        output = {
            "has_raw_contact": bool(contact.size),
            "first_raw_contact_time_s": float(time[contact[0]]) if contact.size else None,
            "peak_force_n": float(np.max(force)),
            "seconds_over_35_n": float(np.count_nonzero(force > 35) * self.config.timestep),
            "saturation_pct": float(
                100
                * np.mean(
                    np.any(np.abs(log["commanded_torque"] - log["applied_torque"]) > 1e-9, axis=1)
                )
            ),
            "evaluation_observed": bool(np.any(mask)),
        }
        for name in (
            "force_rmse_n",
            "tangent_rmse_mm",
            "contact_ratio_pct",
            "orientation_rmse_deg",
            "measurement_rmse_n",
        ):
            output[name] = None
        if not np.any(mask):
            return output
        # Truth enters metrics only. The controller never receives these projected errors.
        normal = yaw_frame(self.scenario.wall_yaw_deg).rotation[:, 0]
        error = log["position"][mask] - log["target_position"][mask]
        tangent_error = error - np.outer(error @ normal, normal)
        output.update(
            force_rmse_n=float(
                np.sqrt(np.mean((force[mask] - log["target_normal_force"][mask]) ** 2))
            ),
            tangent_rmse_mm=float(1000 * np.sqrt(np.mean(np.sum(tangent_error**2, axis=1)))),
            contact_ratio_pct=float(100 * np.mean(force[mask] > 0.5)),
            orientation_rmse_deg=float(
                np.rad2deg(np.sqrt(np.mean(log["orientation_error_rad"][mask] ** 2)))
            ),
            measurement_rmse_n=float(
                np.sqrt(np.mean((log["measured_normal_force"][mask] - force[mask]) ** 2))
            ),
        )
        return output


@dataclass(frozen=True)
class SurfaceSample:
    """Action-before-solve input snapshot, never a post-action force measurement."""

    state: FrankaState
    target: FrankaTarget
    time: float
    measured_kinematic_sample_time: float
    measured_wrench_sample_time: float
    q: np.ndarray
    joint_velocity: np.ndarray


class SurfaceSimulator:
    """One causal 500 Hz simulation path for both batch trials and interactive callers.

    sample() prepares the next control input at most once and returns an owned copy.
    step() applies one world wrench and consumes that input. It does not run a controller.
    Construct a new instance to reset model, sensor, RNG, filter and delay state together.
    A final sample at the horizon is available for next-observation semantics, not another action.
    """

    def __init__(
        self,
        controller_frame: SurfaceFrame,
        scenario: SurfaceScenario | None = None,
        config: SurfaceSimulationConfig | None = None,
        task: SurfaceTask | None = None,
        controller_kind: str = "surface_adaptive",
    ) -> None:
        self.frame = controller_frame
        self.scenario = scenario = scenario or SurfaceScenario()
        self.config = config = config or SurfaceSimulationConfig()
        self.task = task or SurfaceTask()
        self.controller_kind = controller_kind
        self._model = model = mujoco.MjModel.from_xml_path(str(franka_surface_model_path()))
        model.opt.timestep = config.timestep
        if model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
            raise ValueError("surface trials require a single-step integrator")
        self._wall_id, self._tool_id, self._site_id = (
            model.geom("contact_wall").id,
            model.geom("tool_tip").id,
            model.site("ee_site").id,
        )
        wall_id, tool_id = self._wall_id, self._tool_id
        self._wall_normal = wall_normal = yaw_frame(scenario.wall_yaw_deg).rotation[:, 0]
        yaw = np.deg2rad(scenario.wall_yaw_deg)
        model.geom_pos[wall_id, :2] = (
            np.array([0.400, 0.0]) + model.geom_size[wall_id, 0] * wall_normal[:2]
        )
        model.geom_quat[wall_id] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        model.geom_solref[wall_id, 0] = scenario.wall_time_constant
        model.geom_friction[wall_id, 0] = scenario.wall_sliding_friction
        model.geom_friction[tool_id, 0] = scenario.tool_sliding_friction
        if config.contact_model == "smooth":
            # Both equal-priority geoms retain the existing explicit compliance choice.
            model.geom_solimp[[tool_id, wall_id], 0] = 0.0
        body_id = model.body("contact_tool").id
        model.body_inertia[body_id] *= scenario.tool_mass_kg / model.body_mass[body_id]
        model.body_mass[body_id] = scenario.tool_mass_kg
        self._data = data = mujoco.MjData(model)
        mujoco.mj_setConst(model, data)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        self._initial_position, self._initial_rotation, _, _, _ = _site_state(
            model, data, self._site_id
        )
        self._nominal_q = data.qpos[:7].copy()
        self._limits = model.actuator_ctrlrange[:7].copy()
        sensor_seed, position_seed = np.random.SeedSequence(config.seed).spawn(2)
        self._position_rng = np.random.default_rng(position_seed)
        self._sensor = ToolWrenchSensor(
            model,
            nominal_mass_kg=scenario.nominal_tool_mass_kg,
            force_bias_sensor_n=scenario.force_bias_sensor_n,
            torque_bias_sensor_nm=scenario.torque_bias_sensor_nm,
            force_noise_std_n=scenario.force_noise_std_n,
            torque_noise_std_nm=scenario.torque_noise_std_nm,
            rng=np.random.default_rng(sensor_seed),
        )
        self._previous_wrench = self._sensor.read_world(data)
        self._filtered_wrench = np.zeros(6)
        self._alpha = config.timestep / (config.force_filter_time_constant + config.timestep)
        self._history: deque = deque(maxlen=scenario.delay_steps + 1)
        self._rows: list[dict] = []
        self._step = 0
        self._pending: SurfaceSample | None = None
        self._input_row: dict | None = None

    def _prepare(self) -> None:
        if self._pending is not None:
            return
        model, data, config = self._model, self._data, self.config
        time = self._step * config.timestep
        mujoco.mj_step1(model, data)
        if abs(data.time - time) > 1e-9:
            raise RuntimeError("MuJoCo reset or time drift invalidated surface timestamps")
        position, rotation, linear, angular, jacobian = _site_state(model, data, self._site_id)
        self._filtered_wrench += self._alpha * (self._previous_wrench - self._filtered_wrench)
        feedback_time = max(0, self._step - 1) * config.timestep
        self._history.append(
            (
                position + self._position_rng.normal(0, self.scenario.position_noise_std_m, 3),
                rotation,
                linear,
                angular,
                self._filtered_wrench.copy(),
                time,
                feedback_time,
            )
        )
        (
            measured_position,
            measured_rotation,
            measured_linear,
            measured_angular,
            measured_wrench,
            measured_time,
            measured_wrench_time,
        ) = self._history[0]
        posture = 10.0 * (self._nominal_q - data.qpos[:7]) - 2.5 * data.qvel[:7]
        actuation = FrankaActuationContext(
            jacobian,
            self.scenario.bias_compensation_scale * data.qfrc_bias[:7]
            + damped_nullspace_projector(jacobian) @ posture,
            self._limits[:, 0],
            self._limits[:, 1],
        )
        state = FrankaState(
            measured_position,
            measured_rotation,
            measured_linear,
            measured_angular,
            float(self.frame.rotation[:, 0] @ measured_wrench[:3]),
            actuation,
        )
        target = self.task.target_at(
            time, self._initial_position, self._initial_rotation, config.target_force
        )
        self._pending = SurfaceSample(
            state,
            target,
            time,
            measured_time,
            measured_wrench_time,
            data.qpos[:7].copy(),
            data.qvel[:7].copy(),
        )
        self._input_row = {
            "time": time,
            "kinematic_sample_time": time,
            "raw_wrench_sample_time": time,
            "feedback_wrench_sample_time": feedback_time,
            "measured_kinematic_sample_time": measured_time,
            "measured_wrench_sample_time": measured_wrench_time,
            "q": data.qpos[:7].copy(),
            "joint_velocity": data.qvel[:7].copy(),
            "position": position,
            "linear_velocity": linear,
            "measured_position": state.position,
            "measured_rotation": state.rotation,
            "measured_linear_velocity": state.linear_velocity,
            "measured_angular_velocity": state.angular_velocity,
            "measured_normal_force": state.normal_force,
            "measured_wrench_world": measured_wrench,
            "filtered_wrench_world": self._filtered_wrench.copy(),
            "feedback_raw_wrench_world": self._previous_wrench.copy(),
            "cartesian_jacobian": actuation.cartesian_jacobian,
            "joint_torque_offset": actuation.joint_torque_offset,
            "lower_torque_limit": actuation.lower_torque_limit,
            "upper_torque_limit": actuation.upper_torque_limit,
            "target_position": target.position,
            "target_rotation": target.rotation,
            "target_linear_velocity": target.linear_velocity,
            "target_angular_velocity": target.angular_velocity,
            "target_normal_force": target.normal_force,
            "orientation_error_rad": np.linalg.norm(orientation_error(rotation, target.rotation)),
        }

    def sample(self) -> SurfaceSample:
        self._prepare()
        return deepcopy(self._pending)

    def evaluator_kinematics(self) -> dict:
        """Fresh x[k] geometry/twist, including x[N]; never an actor observation.

        Uses the same cached step1 refresh as sample(). It does not integrate or
        solve a new contact force. Existing trace rows keep their original x[k]
        convention; interactive callers can separately record x[k+1].
        """
        self._prepare()
        row = self._input_row
        return {
            "endpoint_time": float(row["time"]),
            "endpoint_position": row["position"].copy(),
            "endpoint_linear_velocity": row["linear_velocity"].copy(),
            "endpoint_contact_gap_m": float(
                self._wall_normal @ (np.array([0.400, 0.0, 0.0]) - row["position"]) - 0.025
            ),
            "endpoint_valid": True,
        }

    def step(self, wrench: np.ndarray, controller: object | None = None) -> dict:
        if self._step >= round(self.config.duration / self.config.timestep):
            raise RuntimeError("surface horizon reached; create a new simulator to reset")
        wrench = np.asarray(wrench, dtype=float)
        if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
            raise ValueError("wrench must be a finite 6-vector")
        self._prepare()
        model, data, config = self._model, self._data, self.config
        time = self._pending.time
        row = self._input_row.copy()
        actuation = self._pending.state.actuation
        commanded_torque = actuation.joint_torque(wrench)
        applied_torque = np.clip(commanded_torque, self._limits[:, 0], self._limits[:, 1])
        telemetry = capture_franka_controller_telemetry(controller)
        row.update(
            {
                "commanded_wrench": wrench.copy(),
                "commanded_torque": commanded_torque,
                "applied_torque": applied_torque,
                "contact_blend": telemetry.contact_blend
                if telemetry.contact_blend is not None
                else 0.0,
                "governed_normal_lead_m": (
                    telemetry.governed_normal_lead_m
                    if telemetry.governed_normal_lead_m is not None
                    else 0.0
                ),
                "torque_projection_scale": (
                    telemetry.torque_projection_scale
                    if telemetry.torque_projection_scale is not None
                    else 1.0
                ),
                "requested_tangential_force_world": (
                    getattr(controller, "requested_tangential_force_world", np.zeros(3))
                ),
            }
        )
        data.ctrl[:7], data.ctrl[7] = applied_torque, 0.0
        mujoco.mj_step2(model, data)
        if abs(data.time - (time + config.timestep)) > 1e-9:
            raise RuntimeError("MuJoCo reset or time drift invalidated surface timestamps")
        # site_xmat and sensordata still describe x[k], not the integrated x[k+1].
        self._previous_wrench = self._sensor.read_world(data)
        row["raw_wrench_world"] = self._previous_wrench.copy()
        # Ideal contact sum is evaluator-only, never supplied to controller/sensor.
        row["true_normal_force"] = _normal_contact_force(model, data, self._tool_id, self._wall_id)
        row["true_contact_gap_m"] = float(
            self._wall_normal @ (np.array([0.400, 0.0, 0.0]) - row["position"]) - 0.025
        )
        tangent_load_world = np.zeros(3)
        contact_wrench = np.zeros(6)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if {contact.geom1, contact.geom2} == {self._tool_id, self._wall_id}:
                mujoco.mj_contactForce(model, data, contact_index, contact_wrench)
                tangent_load_world += contact.frame.reshape(3, 3)[1:].T @ contact_wrench[1:3]
        row["true_tangent_force_n"] = float(np.linalg.norm(tangent_load_world))
        self._rows.append(row)
        self._step += 1
        self._pending = self._input_row = None
        if not all(np.all(np.isfinite(value)) for value in row.values()):
            raise RuntimeError("nonfinite surface trace")
        return deepcopy(row)

    def result(self) -> SurfaceTrialResult:
        if not self._rows:
            raise RuntimeError("no executed surface samples")
        trace = {name: np.asarray([row[name] for row in self._rows]) for name in self._rows[0]}
        trace.update(
            schema_version=np.array(1),
            dt=np.array(self.config.timestep),
            controller_kind=np.array(self.controller_kind),
            contact_model=np.array(self.config.contact_model),
            controller_frame_rotation=self.frame.rotation.copy(),
            force_filter_alpha=np.array(self._alpha),
        )
        return SurfaceTrialResult(trace, self.scenario, self.config, self.task)


def run_surface_trial(
    controller_frame: SurfaceFrame,
    scenario: SurfaceScenario | None = None,
    config: SurfaceSimulationConfig | None = None,
    task: SurfaceTask | None = None,
    controller_kind: str = "surface_adaptive",
) -> SurfaceTrialResult:
    """Batch adapter over the same causal surface simulator used by interactive callers."""
    if controller_kind == "surface_adaptive":
        controller = SurfaceAdaptiveController(controller_frame)
    elif controller_kind in {"surface_integral", "surface_friction"}:
        controller = SurfaceAdaptiveController(
            controller_frame, tangential_mode=controller_kind.removeprefix("surface_")
        )
    elif controller_kind == "world_safe_adaptive":
        if not np.array_equal(controller_frame.rotation, np.eye(3)):
            raise ValueError("world_safe_adaptive requires the identity frame")
        controller = FrankaSafeAdaptiveController()
    else:
        raise ValueError("unknown controller_kind")
    simulator = SurfaceSimulator(controller_frame, scenario, config, task, controller_kind)
    for step in range(round(simulator.config.duration / simulator.config.timestep)):
        sample = simulator.sample()
        if step == 0:
            controller.reset(sample.state)
        wrench = controller.compute(sample.state, sample.target, simulator.config.timestep)
        simulator.step(wrench, controller)
    return simulator.result()
