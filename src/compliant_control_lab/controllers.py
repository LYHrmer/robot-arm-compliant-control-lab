"""Cartesian controllers independent of the simulator implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


def _vec2(x: tuple[float, float]) -> np.ndarray:
    return np.asarray(x, dtype=float)


@dataclass(frozen=True)
class Observation:
    """Measured Cartesian state and wall-normal contact force magnitude."""

    position: np.ndarray
    velocity: np.ndarray
    normal_force: float


@dataclass(frozen=True)
class ControlTarget:
    """Cartesian motion and contact-force reference."""

    position: np.ndarray
    velocity: np.ndarray
    normal_force: float


class CartesianController(Protocol):
    name: str

    def reset(self, observation: Observation) -> None: ...

    def compute(
        self, observation: Observation, target: ControlTarget, dt: float
    ) -> np.ndarray: ...


@dataclass
class PositionController:
    """High-stiffness Cartesian PD baseline."""

    stiffness: np.ndarray = field(default_factory=lambda: _vec2((650.0, 500.0)))
    damping: np.ndarray = field(default_factory=lambda: _vec2((45.0, 38.0)))
    name: str = "position"

    def reset(self, observation: Observation) -> None:
        del observation

    def compute(self, observation: Observation, target: ControlTarget, dt: float) -> np.ndarray:
        del dt
        position_error = target.position - observation.position
        velocity_error = target.velocity - observation.velocity
        return self.stiffness * position_error + self.damping * velocity_error


@dataclass
class ImpedanceController:
    """Low-stiffness Cartesian impedance controller."""

    stiffness: np.ndarray = field(default_factory=lambda: _vec2((140.0, 260.0)))
    damping: np.ndarray = field(default_factory=lambda: _vec2((30.0, 32.0)))
    name: str = "impedance"

    def reset(self, observation: Observation) -> None:
        del observation

    def compute(self, observation: Observation, target: ControlTarget, dt: float) -> np.ndarray:
        del dt
        position_error = target.position - observation.position
        velocity_error = target.velocity - observation.velocity
        return self.stiffness * position_error + self.damping * velocity_error


@dataclass
class AdmittanceController:
    """Force-driven reference generator followed by Cartesian position control."""

    virtual_mass: float = 3.0
    virtual_damping: float = 70.0
    virtual_stiffness: float = 10.0
    inner_stiffness: np.ndarray = field(default_factory=lambda: _vec2((240.0, 780.0)))
    inner_damping: np.ndarray = field(default_factory=lambda: _vec2((42.0, 60.0)))
    max_normal_offset: float = 0.07
    name: str = "admittance"
    _normal_reference: float | None = field(default=None, init=False, repr=False)
    _normal_velocity: float = field(default=0.0, init=False, repr=False)

    def reset(self, observation: Observation) -> None:
        self._normal_reference = float(observation.position[0])
        self._normal_velocity = 0.0

    def compute(self, observation: Observation, target: ControlTarget, dt: float) -> np.ndarray:
        if self._normal_reference is None:
            self.reset(observation)

        assert self._normal_reference is not None
        force_error = target.normal_force - observation.normal_force
        displacement = self._normal_reference - target.position[0]
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
                target.position[0] - self.max_normal_offset,
                target.position[0] + self.max_normal_offset,
            )
        )

        reference = np.array([self._normal_reference, target.position[1]])
        reference_velocity = np.array([self._normal_velocity, target.velocity[1]])
        return self.inner_stiffness * (reference - observation.position) + self.inner_damping * (
            reference_velocity - observation.velocity
        )


@dataclass
class HybridForcePositionController:
    """Normal-force PI control on x and position PD control on tangential y."""

    force_kp: float = 0.35
    force_ki: float = 1.5
    normal_damping: float = 12.0
    tangential_stiffness: float = 900.0
    tangential_damping: float = 68.0
    integral_limit: float = 2.0
    max_normal_command: float = 24.0
    name: str = "hybrid"
    _force_integral: float = field(default=0.0, init=False, repr=False)

    def reset(self, observation: Observation) -> None:
        del observation
        self._force_integral = 0.0

    def compute(self, observation: Observation, target: ControlTarget, dt: float) -> np.ndarray:
        force_error = target.normal_force - observation.normal_force
        # Integrate only near contact. This avoids winding up while the arm approaches
        # the wall or briefly loses contact during tangential motion.
        if observation.normal_force > 0.5:
            self._force_integral = float(
                np.clip(
                    self._force_integral + force_error * dt,
                    -self.integral_limit,
                    self.integral_limit,
                )
            )
        normal_command = (
            target.normal_force
            + self.force_kp * force_error
            + self.force_ki * self._force_integral
            - self.normal_damping * observation.velocity[0]
        )
        normal_command = float(np.clip(normal_command, 0.0, self.max_normal_command))
        tangential_command = (
            self.tangential_stiffness * (target.position[1] - observation.position[1])
            + self.tangential_damping * (target.velocity[1] - observation.velocity[1])
        )
        return np.array([normal_command, tangential_command])


def default_controllers() -> dict[str, CartesianController]:
    """Return fresh controller instances used by the benchmark."""
    return {
        "position": PositionController(),
        "impedance": ImpedanceController(),
        "admittance": AdmittanceController(),
        "hybrid": HybridForcePositionController(),
    }
