"""MuJoCo benchmark harness for 7-DOF Franka contact control."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter_ns
from typing import Literal

import mujoco
import numpy as np

from compliant_control_lab.franka_control import (
    FrankaActuationContext,
    FrankaController,
    FrankaControllerTelemetrySnapshot,
    FrankaState,
    FrankaTarget,
    capture_franka_controller_telemetry,
    damped_nullspace_projector,
    orientation_error,
)

WALL_SURFACE_X = 0.400
TOOL_RADIUS = 0.025
CONTACT_CENTER_X = WALL_SURFACE_X - TOOL_RADIUS
WIPING_START_TIME_S = 1.20


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
    control_timing: Literal["legacy", "split_step"] = "legacy"
    approach_reference: Literal["legacy", "consistent"] = "legacy"

    def __post_init__(self) -> None:
        if self.control_timing not in {"legacy", "split_step"}:
            raise ValueError("control_timing must be legacy or split_step")
        if self.approach_reference not in {"legacy", "consistent"}:
            raise ValueError("approach_reference must be legacy or consistent")


@dataclass
class FrankaTrialResult:
    """One rollout with an explicit control/measurement timing convention.

    In legacy mode, cached contact/site/Jacobian values retain the previous forward pass,
    while joint velocity has already been integrated by the preceding step. A row
    is a control-cycle record, not a fully synchronized physical-state snapshot.
    Its kinematic timestamp dates pose/J only; twist mixes that J with newer qvel.

    In split_step mode, position, joint state, Jacobian and velocity belong to
    ``time[k]``. The new wrench is applied in mj_step2; its solved raw contact force
    also belongs to time[k], although it is read after integration. It is cached
    as a scalar for the NEXT control cycle, before mj_step1 rebuilds contacts.
    ``normal_force`` is the causal filtered feedback before added scenario delay,
    bias or noise; ``measured_normal_force`` is the actual controller input.
    Force timestamps identify the latest raw input to the filter, not a claim
    that the filtered signal has zero lag. Startup repeats the reset observation.

    Torque headroom is the smallest signed distance from the induced
    joint torque to either actuator limit; negative values indicate an exceedance
    (the saturation flag separately uses a 1e-9 Nm numerical tolerance).
    """

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
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    target_linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    commanded_wrench: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    minimum_torque_headroom_nm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    controller_snapshots: tuple[FrankaControllerTelemetrySnapshot, ...] = ()
    control_timing: str = "legacy"
    approach_reference: str = "legacy"
    joint_velocity: np.ndarray = field(default_factory=lambda: np.zeros((0, 7)))
    kinematic_sample_time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    raw_force_sample_time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    feedback_force_sample_time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    measured_kinematic_sample_time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    measured_force_sample_time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    measured_normal_force: np.ndarray = field(default_factory=lambda: np.zeros(0))

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
    if config.approach_reference == "consistent" and 0.10 < time < 1.0:
        phase = (time - 0.10) / 0.90
        approach_rate = 6.0 * phase * (1.0 - phase) / 0.90
        velocity = approach_rate * (contact_position - initial_position)

    if time > WIPING_START_TIME_S:
        phase = 2.0 * np.pi * 0.20 * (time - WIPING_START_TIME_S)
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
    split_step = config.control_timing == "split_step"
    if split_step and model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
        raise ValueError("split_step requires a single-step integrator, not RK4")
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
    model.geom_quat[wall_id] = np.array([np.cos(0.5 * wall_yaw), 0.0, 0.0, np.sin(0.5 * wall_yaw)])

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    data.ctrl[7] = 0.0
    mujoco.mj_forward(model, data)
    initial_position, initial_rotation, _, _, _ = _site_state(model, data, site_id)
    nominal_q = data.qpos[:7].copy()

    step_count = round(config.duration / config.timestep)
    rng = np.random.default_rng(config.seed)
    history: deque[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]] = (
        deque(maxlen=scenario.delay_steps + 1)
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
    linear_velocity_log = np.empty((step_count, 3))
    target_linear_velocity_log = np.empty((step_count, 3))
    commanded_wrench_log = np.empty((step_count, 6))
    minimum_torque_headroom_log = np.empty(step_count)
    controller_snapshots: list[FrankaControllerTelemetrySnapshot] = []
    joint_velocity_log = np.empty((step_count, 7))
    kinematic_time_log = np.empty(step_count)
    raw_force_time_log = np.empty(step_count)
    feedback_force_time_log = np.empty(step_count)
    measured_kinematic_time_log = np.empty(step_count)
    measured_force_time_log = np.empty(step_count)
    measured_force_log = np.empty(step_count)

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
    # Cache the solved scalar before mj_step1 can replace contact/constraint indices.
    previous_solved_force = (
        _normal_contact_force(model, data, tool_id, wall_id) if split_step else 0.0
    )

    for step in range(step_count):
        time = step * config.timestep
        previous_time = max(0, step - 1) * config.timestep
        if split_step:
            mujoco.mj_step1(model, data)
            if abs(data.time - time) > 1e-9:
                raise RuntimeError("MuJoCo reset or time drift invalidated split-step timestamps")
        position, rotation, linear_velocity, angular_velocity, jacobian = _site_state(
            model, data, site_id
        )
        raw_force = (
            previous_solved_force
            if split_step
            else _normal_contact_force(model, data, tool_id, wall_id)
        )
        kinematic_time = time if split_step else previous_time
        filtered_force += filter_alpha * (raw_force - filtered_force)
        history.append(
            (
                position.copy(),
                rotation.copy(),
                linear_velocity.copy(),
                angular_velocity.copy(),
                filtered_force,
                kinematic_time,
                previous_time,
            )
        )
        (
            measured_position,
            measured_rotation,
            measured_linear,
            measured_angular,
            measured_force,
            measured_kinematic_time,
            measured_force_time,
        ) = history[0]
        measured_position = measured_position + rng.normal(0.0, scenario.position_noise_std, size=3)
        measured_force = (
            measured_force + scenario.force_bias_n + rng.normal(0.0, scenario.force_noise_std)
        )
        nullspace = damped_nullspace_projector(jacobian)
        posture_torque = 10.0 * (nominal_q - data.qpos[:7]) - 2.5 * data.qvel[:7]
        actuation = FrankaActuationContext(
            cartesian_jacobian=jacobian,
            joint_torque_offset=(
                scenario.bias_compensation_scale * data.qfrc_bias[:7] + nullspace @ posture_torque
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
        controller_snapshots.append(capture_franka_controller_telemetry(controller))

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
        joint_velocity_log[step] = data.qvel[:7]
        position_log[step] = position
        desired_position_log[step] = target.position
        force_log[step] = filtered_force
        raw_force_log[step] = raw_force
        kinematic_time_log[step] = kinematic_time
        raw_force_time_log[step] = time if split_step else previous_time
        feedback_force_time_log[step] = previous_time
        measured_kinematic_time_log[step] = measured_kinematic_time
        measured_force_time_log[step] = measured_force_time
        measured_force_log[step] = measured_force
        desired_force_log[step] = target.normal_force
        orientation_error_log[step] = np.linalg.norm(orientation_error(rotation, target.rotation))
        torque_log[step] = torque
        saturated_log[step] = bool(np.any(np.abs(torque_unclipped - torque) > 1e-9))
        linear_velocity_log[step] = linear_velocity
        target_linear_velocity_log[step] = target.linear_velocity
        commanded_wrench_log[step] = wrench
        minimum_torque_headroom_log[step] = float(
            np.min(
                np.minimum(
                    torque_unclipped - actuator_limits[:, 0],
                    actuator_limits[:, 1] - torque_unclipped,
                )
            )
        )
        if split_step:
            mujoco.mj_step2(model, data)
            if abs(data.time - (time + config.timestep)) > 1e-9:
                raise RuntimeError("MuJoCo reset or time drift invalidated split-step timestamps")
            # Constraint force still describes x[k], u[k], not integrated x[k+1].
            previous_solved_force = _normal_contact_force(model, data, tool_id, wall_id)
            raw_force_log[step] = previous_solved_force
        else:
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
        linear_velocity=linear_velocity_log,
        target_linear_velocity=target_linear_velocity_log,
        commanded_wrench=commanded_wrench_log,
        minimum_torque_headroom_nm=minimum_torque_headroom_log,
        controller_snapshots=tuple(controller_snapshots),
        control_timing=config.control_timing,
        approach_reference=config.approach_reference,
        joint_velocity=joint_velocity_log,
        kinematic_sample_time=kinematic_time_log,
        raw_force_sample_time=raw_force_time_log,
        feedback_force_sample_time=feedback_force_time_log,
        measured_kinematic_sample_time=measured_kinematic_time_log,
        measured_force_sample_time=measured_force_time_log,
        measured_normal_force=measured_force_log,
    )
