"""Simulator-independent 6D Cartesian controllers for the Franka Panda."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


def _array(values: tuple[float, ...]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def orientation_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """Return a world-frame small-angle orientation error from current to desired."""
    current = np.asarray(current, dtype=float).reshape(3, 3)
    desired = np.asarray(desired, dtype=float).reshape(3, 3)
    return 0.5 * sum(
        (np.cross(current[:, axis], desired[:, axis]) for axis in range(3)),
        start=np.zeros(3),
    )


def damped_nullspace_projector(jacobian: np.ndarray, damping: float = 0.03) -> np.ndarray:
    """Return a torque-space projector onto the Jacobian null space."""
    jacobian = np.asarray(jacobian, dtype=float)
    gram = jacobian @ jacobian.T + damping**2 * np.eye(jacobian.shape[0])
    jacobian_transpose_pinv = np.linalg.solve(gram, jacobian)
    return np.eye(jacobian.shape[1]) - jacobian.T @ jacobian_transpose_pinv


@dataclass(frozen=True)
class FrankaActuationContext:
    """Numeric robot-model data needed to check a Cartesian wrench before actuation."""

    cartesian_jacobian: np.ndarray
    joint_torque_offset: np.ndarray
    lower_torque_limit: np.ndarray
    upper_torque_limit: np.ndarray

    def __post_init__(self) -> None:
        jacobian = np.asarray(self.cartesian_jacobian, dtype=float)
        offset = np.asarray(self.joint_torque_offset, dtype=float)
        lower = np.asarray(self.lower_torque_limit, dtype=float)
        upper = np.asarray(self.upper_torque_limit, dtype=float)
        if jacobian.ndim != 2 or jacobian.shape[0] != 6:
            raise ValueError("cartesian_jacobian must have shape (6, dof)")
        dof = jacobian.shape[1]
        if offset.shape != (dof,) or lower.shape != (dof,) or upper.shape != (dof,):
            raise ValueError("actuation vectors must match the Jacobian dof")
        if not all(np.all(np.isfinite(values)) for values in (jacobian, offset, lower, upper)):
            raise ValueError("actuation context must be finite")
        if np.any(lower >= upper):
            raise ValueError("lower torque limits must be below upper limits")
        object.__setattr__(self, "cartesian_jacobian", jacobian.copy())
        object.__setattr__(self, "joint_torque_offset", offset.copy())
        object.__setattr__(self, "lower_torque_limit", lower.copy())
        object.__setattr__(self, "upper_torque_limit", upper.copy())

    def joint_torque(self, wrench: np.ndarray) -> np.ndarray:
        wrench = np.asarray(wrench, dtype=float)
        if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
            raise ValueError("wrench must be a finite 6-vector")
        return self.cartesian_jacobian.T @ wrench + self.joint_torque_offset


@dataclass(frozen=True)
class FrankaState:
    position: np.ndarray
    rotation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    normal_force: float
    actuation: FrankaActuationContext | None = None


@dataclass(frozen=True)
class FrankaTarget:
    position: np.ndarray
    rotation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    normal_force: float


class FrankaController(Protocol):
    name: str

    def reset(self, state: FrankaState) -> None: ...

    def compute(self, state: FrankaState, target: FrankaTarget, dt: float) -> np.ndarray: ...


def _orientation_wrench(
    state: FrankaState,
    target: FrankaTarget,
    stiffness: np.ndarray,
    damping: np.ndarray,
) -> np.ndarray:
    rotation_error = orientation_error(state.rotation, target.rotation)
    angular_velocity_error = target.angular_velocity - state.angular_velocity
    return stiffness * rotation_error + damping * angular_velocity_error


@dataclass
class FrankaImpedanceController:
    translational_stiffness: np.ndarray = field(
        default_factory=lambda: _array((300.0, 450.0, 450.0))
    )
    translational_damping: np.ndarray = field(
        default_factory=lambda: _array((38.0, 38.0, 38.0))
    )
    rotational_stiffness: np.ndarray = field(
        default_factory=lambda: _array((20.0, 20.0, 20.0))
    )
    rotational_damping: np.ndarray = field(
        default_factory=lambda: _array((5.0, 5.0, 5.0))
    )
    name: str = "impedance"

    def reset(self, state: FrankaState) -> None:
        del state

    def compute(self, state: FrankaState, target: FrankaTarget, dt: float) -> np.ndarray:
        del dt
        force = self.translational_stiffness * (target.position - state.position)
        force += self.translational_damping * (
            target.linear_velocity - state.linear_velocity
        )
        torque = _orientation_wrench(
            state,
            target,
            self.rotational_stiffness,
            self.rotational_damping,
        )
        return np.concatenate((force, torque))


@dataclass
class FrankaAdmittanceController:
    normal: np.ndarray = field(default_factory=lambda: _array((1.0, 0.0, 0.0)))
    virtual_mass: float = 3.0
    virtual_damping: float = 70.0
    virtual_stiffness: float = 8.0
    inner_stiffness: np.ndarray = field(
        default_factory=lambda: _array((300.0, 450.0, 450.0))
    )
    inner_damping: np.ndarray = field(
        default_factory=lambda: _array((40.0, 38.0, 38.0))
    )
    rotational_stiffness: np.ndarray = field(
        default_factory=lambda: _array((20.0, 20.0, 20.0))
    )
    rotational_damping: np.ndarray = field(
        default_factory=lambda: _array((5.0, 5.0, 5.0))
    )
    max_normal_offset: float = 0.06
    name: str = "admittance"
    _normal_reference: float | None = field(default=None, init=False, repr=False)
    _normal_velocity: float = field(default=0.0, init=False, repr=False)

    def reset(self, state: FrankaState) -> None:
        self.normal = self.normal / np.linalg.norm(self.normal)
        self._normal_reference = float(self.normal @ state.position)
        self._normal_velocity = 0.0

    def compute(self, state: FrankaState, target: FrankaTarget, dt: float) -> np.ndarray:
        if self._normal_reference is None:
            self.reset(state)
        assert self._normal_reference is not None

        target_normal_position = float(self.normal @ target.position)
        force_error = target.normal_force - state.normal_force
        displacement = self._normal_reference - target_normal_position
        acceleration = (
            force_error
            - self.virtual_damping * self._normal_velocity
            - self.virtual_stiffness * displacement
        ) / self.virtual_mass
        self._normal_velocity += acceleration * dt
        self._normal_reference += self._normal_velocity * dt
        self._normal_reference = float(
            np.clip(
                self._normal_reference,
                target_normal_position - self.max_normal_offset,
                target_normal_position + self.max_normal_offset,
            )
        )

        tangent_projector = np.eye(3) - np.outer(self.normal, self.normal)
        reference_position = tangent_projector @ target.position
        reference_position += self.normal * self._normal_reference
        reference_velocity = tangent_projector @ target.linear_velocity
        reference_velocity += self.normal * self._normal_velocity
        force = self.inner_stiffness * (reference_position - state.position)
        force += self.inner_damping * (reference_velocity - state.linear_velocity)
        torque = _orientation_wrench(
            state,
            target,
            self.rotational_stiffness,
            self.rotational_damping,
        )
        return np.concatenate((force, torque))


@dataclass
class FrankaHybridController:
    """Impact-aware normal-force control with tangential pose impedance."""

    normal: np.ndarray = field(default_factory=lambda: _array((1.0, 0.0, 0.0)))
    force_kp: float = 0.35
    force_ki: float = 1.5
    normal_damping: float = 10.0
    tangential_stiffness: np.ndarray = field(
        default_factory=lambda: _array((0.0, 450.0, 450.0))
    )
    tangential_damping: np.ndarray = field(
        default_factory=lambda: _array((0.0, 38.0, 38.0))
    )
    rotational_stiffness: np.ndarray = field(
        default_factory=lambda: _array((20.0, 20.0, 20.0))
    )
    rotational_damping: np.ndarray = field(
        default_factory=lambda: _array((5.0, 5.0, 5.0))
    )
    integral_limit: float = 2.0
    max_normal_command: float = 25.0
    approach_stiffness: float = 800.0
    approach_damping: float = 120.0
    max_approach_command: float = 12.0
    contact_threshold: float = 3.0
    contact_confirm_time: float = 0.02
    contact_release_threshold: float = 1.0
    contact_release_time: float = 0.05
    force_transition_time: float = 0.15
    name: str = "hybrid"
    _force_integral: float = field(default=0.0, init=False, repr=False)
    _contact_confirm_elapsed: float = field(default=0.0, init=False, repr=False)
    _contact_release_elapsed: float = field(default=0.0, init=False, repr=False)
    _force_blend: float = field(default=0.0, init=False, repr=False)
    _in_contact: bool = field(default=False, init=False, repr=False)

    @property
    def in_contact(self) -> bool:
        return self._in_contact

    @property
    def force_blend(self) -> float:
        return self._force_blend

    def reset(self, state: FrankaState) -> None:
        self.normal = self.normal / np.linalg.norm(self.normal)
        self._force_integral = 0.0
        self._contact_confirm_elapsed = 0.0
        self._contact_release_elapsed = 0.0
        self._in_contact = state.normal_force >= self.contact_threshold
        self._force_blend = 1.0 if self._in_contact else 0.0

    def compute(self, state: FrankaState, target: FrankaTarget, dt: float) -> np.ndarray:
        safe_dt = max(0.0, dt)
        if self._in_contact:
            if state.normal_force < self.contact_release_threshold:
                self._contact_release_elapsed += safe_dt
                if self._contact_release_elapsed >= self.contact_release_time:
                    self._in_contact = False
                    self._contact_confirm_elapsed = 0.0
            else:
                self._contact_release_elapsed = 0.0
        elif state.normal_force >= self.contact_threshold:
            self._contact_confirm_elapsed += safe_dt
            if self._contact_confirm_elapsed >= self.contact_confirm_time:
                self._in_contact = True
                self._contact_release_elapsed = 0.0
        else:
            self._contact_confirm_elapsed = 0.0

        blend_step = safe_dt / self.force_transition_time
        blend_target = 1.0 if self._in_contact else 0.0
        self._force_blend += float(
            np.clip(blend_target - self._force_blend, -blend_step, blend_step)
        )

        force_error = target.normal_force - state.normal_force
        if self._in_contact:
            self._force_integral = float(
                np.clip(
                    self._force_integral + force_error * safe_dt,
                    -self.integral_limit,
                    self.integral_limit,
                )
            )

        normal_velocity = float(self.normal @ state.linear_velocity)
        force_command = (
            target.normal_force
            + self.force_kp * force_error
            + self.force_ki * self._force_integral
            - self.normal_damping * normal_velocity
        )
        force_command = float(np.clip(force_command, 0.0, self.max_normal_command))
        target_normal_velocity = float(self.normal @ target.linear_velocity)
        normal_position_error = float(self.normal @ (target.position - state.position))
        approach_command = self.approach_stiffness * normal_position_error
        approach_command += self.approach_damping * (
            target_normal_velocity - normal_velocity
        )
        approach_command = float(
            np.clip(approach_command, 0.0, self.max_approach_command)
        )
        normal_command = (
            (1.0 - self._force_blend) * approach_command
            + self._force_blend * force_command
        )

        tangent_projector = np.eye(3) - np.outer(self.normal, self.normal)
        position_error = tangent_projector @ (target.position - state.position)
        velocity_error = tangent_projector @ (
            target.linear_velocity - state.linear_velocity
        )
        force = self.normal * normal_command
        force += self.tangential_stiffness * position_error
        force += self.tangential_damping * velocity_error
        torque = _orientation_wrench(
            state,
            target,
            self.rotational_stiffness,
            self.rotational_damping,
        )
        return np.concatenate((force, torque))


def default_franka_controllers() -> dict[str, FrankaController]:
    return {
        "impedance": FrankaImpedanceController(),
        "admittance": FrankaAdmittanceController(),
        "hybrid": FrankaHybridController(),
    }
