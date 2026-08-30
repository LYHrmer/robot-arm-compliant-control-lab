"""MuJoCo benchmark harness for 7-DOF Franka contact control."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

import mujoco
import numpy as np

from compliant_control_lab.franka_control import (
    FrankaActuationContext,
    FrankaController,
    FrankaState,
    FrankaTarget,
    damped_nullspace_projector,
    orientation_error,
)

WALL_SURFACE_X = 0.400
TOOL_RADIUS = 0.025
CONTACT_CENTER_X = WALL_SURFACE_X - TOOL_RADIUS


@dataclass(frozen=True)
class FrankaScenario:
    name: str
    wall_time_constant: float = 0.012
    wall_sliding_friction: float = 0.45
    wall_yaw_deg: float = 0.0
    position_noise_std: float = 0.0
    force_noise_std: float = 0.05
    force_bias_n: float = 0.0
    delay_steps: int = 0
    bias_compensation_scale: float = 1.0


@dataclass(frozen=True)
class FrankaSimulationConfig:
    duration: float = 4.5
    timestep: float = 0.002
    target_force: float = 12.0
    force_filter_time_constant: float = 0.02
    seed: int = 11


@dataclass
class FrankaTrialResult:
    controller: str
    scenario: str
    time: np.ndarray
    q: np.ndarray
    position: np.ndarray
    desired_position: np.ndarray
    normal_force: np.ndarray
    raw_normal_force: np.ndarray
    desired_force: np.ndarray
    orientation_error_rad: np.ndarray
    torque: np.ndarray
    controller_time_us: np.ndarray
    saturated: np.ndarray

    def metrics(self, evaluation_start: float = 1.5) -> dict[str, float | str]:
        mask = self.time >= evaluation_start
        if not np.any(mask):
            mask = np.ones_like(self.time, dtype=bool)
        force_error = self.normal_force[mask] - self.desired_force[mask]
        tangent_error = self.position[mask, 1:] - self.desired_position[mask, 1:]
        torque_norm = np.linalg.norm(self.torque[mask], axis=1)
        return {
            "controller": self.controller,
            "scenario": self.scenario,
            "force_rmse_n": float(np.sqrt(np.mean(force_error**2))),
            "peak_force_n": float(np.max(self.raw_normal_force)),
            "tangent_rmse_mm": float(1_000.0 * np.sqrt(np.mean(tangent_error**2))),
            "orientation_rmse_deg": float(
                np.rad2deg(np.sqrt(np.mean(self.orientation_error_rad[mask] ** 2)))
            ),
            "contact_ratio_pct": float(100.0 * np.mean(self.normal_force[mask] > 0.5)),
            "torque_rms_nm": float(np.sqrt(np.mean(torque_norm**2))),
            "saturation_pct": float(100.0 * np.mean(self.saturated)),
            "controller_p95_us": float(np.percentile(self.controller_time_us[mask], 95)),
        }


DEFAULT_FRANKA_SCENARIOS = (
    FrankaScenario(name="nominal"),
    FrankaScenario(name="stiff_wall", wall_time_constant=0.005),
    FrankaScenario(
        name="noisy_delay",
        position_noise_std=0.0005,
        force_noise_std=0.6,
        delay_steps=10,
    ),
)


def franka_model_path() -> Path:
    return Path(__file__).with_name("assets") / "franka_scene.xml"


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _target_at(
    time: float,
    initial_position: np.ndarray,
    initial_rotation: np.ndarray,
    config: FrankaSimulationConfig,
) -> FrankaTarget:
    approach = _smoothstep((time - 0.10) / 0.90)
    contact_position = initial_position.copy()
    contact_position[0] = CONTACT_CENTER_X + 0.010
    position = initial_position + approach * (contact_position - initial_position)
    velocity = np.zeros(3)

    if time > 1.20:
        phase = 2.0 * np.pi * 0.20 * (time - 1.20)
        position[1] = initial_position[1] + 0.055 * np.sin(phase)
        position[2] = initial_position[2] - 0.040 + 0.040 * np.cos(phase)
        velocity[1] = 0.055 * 2.0 * np.pi * 0.20 * np.cos(phase)
        velocity[2] = -0.040 * 2.0 * np.pi * 0.20 * np.sin(phase)

    return FrankaTarget(
        position=position,
        rotation=initial_rotation,
        linear_velocity=velocity,
        angular_velocity=np.zeros(3),
        normal_force=config.target_force * approach,
    )


def _normal_contact_force(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tool_geom_id: int,
    wall_geom_id: int,
) -> float:
    total = 0.0
    contact_wrench = np.zeros(6)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom_pair = {contact.geom1, contact.geom2}
        if geom_pair != {tool_geom_id, wall_geom_id}:
            continue
        mujoco.mj_contactForce(model, data, contact_index, contact_wrench)
        total += max(0.0, float(contact_wrench[0]))
    return total


def _site_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    jacobian_position = np.zeros((3, model.nv))
    jacobian_rotation = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacobian_position, jacobian_rotation, site_id)
    jacobian = np.vstack((jacobian_position[:, :7], jacobian_rotation[:, :7]))
    twist = jacobian @ data.qvel[:7]
    return (
        data.site_xpos[site_id].copy(),
        data.site_xmat[site_id].reshape(3, 3).copy(),
        twist[:3],
        twist[3:],
        jacobian,
    )


def run_franka_trial(
    controller: FrankaController,
    scenario: FrankaScenario = DEFAULT_FRANKA_SCENARIOS[0],
    config: FrankaSimulationConfig | None = None,
) -> FrankaTrialResult:
    """Run one deterministic 7-DOF contact-control trial."""
    config = config or FrankaSimulationConfig()
    model = mujoco.MjModel.from_xml_path(str(franka_model_path()))
    model.opt.timestep = config.timestep
    wall_id = model.geom("contact_wall").id
    tool_id = model.geom("tool_tip").id
    site_id = model.site("ee_site").id
    model.geom_solref[wall_id, 0] = scenario.wall_time_constant
    model.geom_friction[wall_id, 0] = scenario.wall_sliding_friction
    wall_yaw = np.deg2rad(scenario.wall_yaw_deg)
    wall_normal = np.array([np.cos(wall_yaw), np.sin(wall_yaw)])
    wall_half_thickness = model.geom_size[wall_id, 0]
    model.geom_pos[wall_id, :2] = np.array([WALL_SURFACE_X, 0.0])
    model.geom_pos[wall_id, :2] += wall_half_thickness * wall_normal
    model.geom_quat[wall_id] = np.array(
        [np.cos(0.5 * wall_yaw), 0.0, 0.0, np.sin(0.5 * wall_yaw)]
    )

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    data.ctrl[7] = 0.0
    mujoco.mj_forward(model, data)
    initial_position, initial_rotation, _, _, _ = _site_state(model, data, site_id)
    nominal_q = data.qpos[:7].copy()

    step_count = round(config.duration / config.timestep)
    rng = np.random.default_rng(config.seed)
    history: deque[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = deque(
        maxlen=scenario.delay_steps + 1
    )

    time_log = np.empty(step_count)
    q_log = np.empty((step_count, 7))
    position_log = np.empty((step_count, 3))
    desired_position_log = np.empty((step_count, 3))
    force_log = np.empty(step_count)
    raw_force_log = np.empty(step_count)
    desired_force_log = np.empty(step_count)
    orientation_error_log = np.empty(step_count)
    torque_log = np.empty((step_count, 7))
    controller_time_log = np.empty(step_count)
    saturated_log = np.empty(step_count, dtype=bool)

    initial_state = FrankaState(
        position=initial_position,
        rotation=initial_rotation,
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        normal_force=0.0,
    )
    controller.reset(initial_state)
    actuator_limits = model.actuator_ctrlrange[:7].copy()
    filtered_force = 0.0
    filter_alpha = config.timestep / (config.force_filter_time_constant + config.timestep)

    for step in range(step_count):
        time = step * config.timestep
        position, rotation, linear_velocity, angular_velocity, jacobian = _site_state(
            model, data, site_id
        )
        raw_force = _normal_contact_force(model, data, tool_id, wall_id)
        filtered_force += filter_alpha * (raw_force - filtered_force)
        history.append(
            (
                position.copy(),
                rotation.copy(),
                linear_velocity.copy(),
                angular_velocity.copy(),
                filtered_force,
            )
        )
        measured_position, measured_rotation, measured_linear, measured_angular, measured_force = (
            history[0]
        )
        measured_position = measured_position + rng.normal(
            0.0, scenario.position_noise_std, size=3
        )
        measured_force = (
            measured_force
            + scenario.force_bias_n
            + rng.normal(0.0, scenario.force_noise_std)
        )
        nullspace = damped_nullspace_projector(jacobian)
        posture_torque = 10.0 * (nominal_q - data.qpos[:7]) - 2.5 * data.qvel[:7]
        actuation = FrankaActuationContext(
            cartesian_jacobian=jacobian,
            joint_torque_offset=(
                scenario.bias_compensation_scale * data.qfrc_bias[:7]
                + nullspace @ posture_torque
            ),
            lower_torque_limit=actuator_limits[:, 0],
            upper_torque_limit=actuator_limits[:, 1],
        )
        state = FrankaState(
            position=measured_position,
            rotation=measured_rotation,
            linear_velocity=measured_linear,
            angular_velocity=measured_angular,
            normal_force=measured_force,
            actuation=actuation,
        )
        target = _target_at(time, initial_position, initial_rotation, config)

        start_ns = perf_counter_ns()
        wrench = controller.compute(state, target, config.timestep)
        controller_time_log[step] = (perf_counter_ns() - start_ns) / 1_000.0

        torque_unclipped = actuation.joint_torque(wrench)
        torque = np.clip(
            torque_unclipped,
            actuator_limits[:, 0],
            actuator_limits[:, 1],
        )
        data.ctrl[:7] = torque
        data.ctrl[7] = 0.0

        time_log[step] = time
        q_log[step] = data.qpos[:7]
        position_log[step] = position
        desired_position_log[step] = target.position
        force_log[step] = filtered_force
        raw_force_log[step] = raw_force
        desired_force_log[step] = target.normal_force
        orientation_error_log[step] = np.linalg.norm(
            orientation_error(rotation, target.rotation)
        )
        torque_log[step] = torque
        saturated_log[step] = bool(np.any(np.abs(torque_unclipped - torque) > 1e-9))
        mujoco.mj_step(model, data)

    return FrankaTrialResult(
        controller=controller.name,
        scenario=scenario.name,
        time=time_log,
        q=q_log,
        position=position_log,
        desired_position=desired_position_log,
        normal_force=force_log,
        raw_normal_force=raw_force_log,
        desired_force=desired_force_log,
        orientation_error_rad=orientation_error_log,
        torque=torque_log,
        controller_time_us=controller_time_log,
        saturated=saturated_log,
    )
