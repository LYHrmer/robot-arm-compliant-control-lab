"""MuJoCo simulation harness and reproducible benchmark scenarios."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

import mujoco
import numpy as np

from compliant_control_lab.controllers import CartesianController, ControlTarget, Observation
from compliant_control_lab.kinematics import forward_kinematics, inverse_kinematics, jacobian

WALL_SURFACE_X = 0.695
FINGERTIP_RADIUS = 0.045
CONTACT_CENTER_X = WALL_SURFACE_X - FINGERTIP_RADIUS


@dataclass(frozen=True)
class Scenario:
    name: str
    wall_time_constant: float = 0.008
    position_noise_std: float = 0.0
    force_noise_std: float = 0.05
    delay_steps: int = 0


@dataclass(frozen=True)
class SimulationConfig:
    duration: float = 4.0
    timestep: float = 0.002
    target_force: float = 12.0
    force_filter_time_constant: float = 0.02
    seed: int = 7


@dataclass
class TrialResult:
    controller: str
    scenario: str
    time: np.ndarray
    q: np.ndarray
    position: np.ndarray
    desired_position: np.ndarray
    normal_force: np.ndarray
    raw_normal_force: np.ndarray
    desired_force: np.ndarray
    torque: np.ndarray
    controller_time_us: np.ndarray
    saturated: np.ndarray

    def metrics(self, evaluation_start: float = 1.2) -> dict[str, float | str]:
        mask = self.time >= evaluation_start
        if not np.any(mask):
            mask = np.ones_like(self.time, dtype=bool)
        force_error = self.normal_force[mask] - self.desired_force[mask]
        y_error = self.position[mask, 1] - self.desired_position[mask, 1]
        torque_norm = np.linalg.norm(self.torque[mask], axis=1)
        return {
            "controller": self.controller,
            "scenario": self.scenario,
            "force_rmse_n": float(np.sqrt(np.mean(force_error**2))),
            "peak_force_n": float(np.max(self.raw_normal_force)),
            "y_rmse_mm": float(1_000.0 * np.sqrt(np.mean(y_error**2))),
            "contact_ratio_pct": float(100.0 * np.mean(self.normal_force[mask] > 0.5)),
            "torque_rms_nm": float(np.sqrt(np.mean(torque_norm**2))),
            "saturation_pct": float(100.0 * np.mean(self.saturated)),
            "controller_p95_us": float(np.percentile(self.controller_time_us[mask], 95)),
        }


DEFAULT_SCENARIOS = (
    Scenario(name="nominal", wall_time_constant=0.016),
    Scenario(name="stiff_wall", wall_time_constant=0.006),
    Scenario(
        name="noisy_delay",
        position_noise_std=0.001,
        force_noise_std=0.6,
        delay_steps=10,
    ),
)


def _model_path() -> Path:
    return Path(__file__).with_name("assets") / "planar_arm.xml"


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def target_at(time: float, config: SimulationConfig) -> ControlTarget:
    """Generate a smooth contact approach followed by tangential wall tracking."""
    approach = _smoothstep((time - 0.15) / 0.85)
    start_position = np.array([0.56, 0.05])
    contact_position = np.array([CONTACT_CENTER_X + 0.025, 0.05])
    position = start_position + approach * (contact_position - start_position)

    if time > 1.0:
        phase = 2.0 * np.pi * 0.20 * (time - 1.0)
        position[1] = 0.05 + 0.06 * np.sin(phase)
        y_velocity = 0.06 * 2.0 * np.pi * 0.20 * np.cos(phase)
    else:
        y_velocity = 0.0

    return ControlTarget(
        position=position,
        velocity=np.array([0.0, y_velocity]),
        normal_force=config.target_force * approach,
    )


def _normal_contact_force(
    model: mujoco.MjModel, data: mujoco.MjData, fingertip_geom_id: int
) -> float:
    total = 0.0
    contact_wrench = np.zeros(6)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if contact.geom1 != fingertip_geom_id and contact.geom2 != fingertip_geom_id:
            continue
        mujoco.mj_contactForce(model, data, contact_index, contact_wrench)
        total += max(0.0, float(contact_wrench[0]))
    return total


def run_trial(
    controller: CartesianController,
    scenario: Scenario = DEFAULT_SCENARIOS[0],
    config: SimulationConfig | None = None,
) -> TrialResult:
    """Run one deterministic controller/scenario trial."""
    config = config or SimulationConfig()
    model = mujoco.MjModel.from_xml_path(str(_model_path()))
    model.opt.timestep = config.timestep
    wall_id = model.geom("wall").id
    fingertip_id = model.geom("fingertip").id
    model.geom_solref[wall_id, 0] = scenario.wall_time_constant

    data = mujoco.MjData(model)
    data.qpos[:] = inverse_kinematics(np.array([0.56, 0.05]))
    mujoco.mj_forward(model, data)

    step_count = round(config.duration / config.timestep)
    rng = np.random.default_rng(config.seed)
    history: deque[tuple[np.ndarray, np.ndarray, float]] = deque(
        maxlen=scenario.delay_steps + 1
    )

    time_log = np.empty(step_count)
    q_log = np.empty((step_count, 2))
    position_log = np.empty((step_count, 2))
    desired_position_log = np.empty((step_count, 2))
    force_log = np.empty(step_count)
    raw_force_log = np.empty(step_count)
    desired_force_log = np.empty(step_count)
    torque_log = np.empty((step_count, 2))
    controller_time_log = np.empty(step_count)
    saturated_log = np.empty(step_count, dtype=bool)

    initial_position = forward_kinematics(data.qpos)
    initial_observation = Observation(initial_position, np.zeros(2), 0.0)
    controller.reset(initial_observation)
    actuator_limits = model.actuator_ctrlrange.copy()
    filtered_force = 0.0
    force_filter_alpha = config.timestep / (config.force_filter_time_constant + config.timestep)

    for step in range(step_count):
        time = step * config.timestep
        q = data.qpos.copy()
        q_velocity = data.qvel.copy()
        position = forward_kinematics(q)
        task_velocity = jacobian(q) @ q_velocity
        raw_normal_force = _normal_contact_force(model, data, fingertip_id)
        filtered_force += force_filter_alpha * (raw_normal_force - filtered_force)
        history.append((position.copy(), task_velocity.copy(), filtered_force))
        measured_position, measured_velocity, measured_force = history[0]

        measured_position = measured_position + rng.normal(
            0.0, scenario.position_noise_std, size=2
        )
        measured_force = max(0.0, measured_force + rng.normal(0.0, scenario.force_noise_std))
        observation = Observation(measured_position, measured_velocity, measured_force)
        target = target_at(time, config)

        start_ns = perf_counter_ns()
        wrench = controller.compute(observation, target, config.timestep)
        controller_time_log[step] = (perf_counter_ns() - start_ns) / 1_000.0

        torque_unclipped = jacobian(q).T @ wrench - 0.12 * q_velocity
        torque = np.clip(
            torque_unclipped,
            actuator_limits[:, 0],
            actuator_limits[:, 1],
        )
        data.ctrl[:] = torque

        time_log[step] = time
        q_log[step] = q
        position_log[step] = position
        desired_position_log[step] = target.position
        force_log[step] = filtered_force
        raw_force_log[step] = raw_normal_force
        desired_force_log[step] = target.normal_force
        torque_log[step] = torque
        saturated_log[step] = bool(np.any(np.abs(torque_unclipped - torque) > 1e-9))
        mujoco.mj_step(model, data)

    return TrialResult(
        controller=controller.name,
        scenario=scenario.name,
        time=time_log,
        q=q_log,
        position=position_log,
        desired_position=desired_position_log,
        normal_force=force_log,
        raw_normal_force=raw_force_log,
        desired_force=desired_force_log,
        torque=torque_log,
        controller_time_us=controller_time_log,
        saturated=saturated_log,
    )
