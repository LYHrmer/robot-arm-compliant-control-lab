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
class FrankaState:
    position: np.ndarray
    rotation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    normal_force: float


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
    name: str = "hybrid"
    _force_integral: float = field(default=0.0, init=False, repr=False)

    def reset(self, state: FrankaState) -> None:
        del state
        self.normal = self.normal / np.linalg.norm(self.normal)
        self._force_integral = 0.0

    def compute(self, state: FrankaState, target: FrankaTarget, dt: float) -> np.ndarray:
        force_error = target.normal_force - state.normal_force
        if state.normal_force > 0.5:
            self._force_integral = float(
                np.clip(
                    self._force_integral + force_error * dt,
                    -self.integral_limit,
                    self.integral_limit,
                )
            )

        normal_velocity = float(self.normal @ state.linear_velocity)
        normal_command = (
            target.normal_force
            + self.force_kp * force_error
            + self.force_ki * self._force_integral
            - self.normal_damping * normal_velocity
        )
        normal_command = float(np.clip(normal_command, 0.0, self.max_normal_command))

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
