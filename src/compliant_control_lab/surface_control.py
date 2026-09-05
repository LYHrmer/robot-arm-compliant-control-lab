"""Fixed surface-coordinate control behind the existing numeric controller seam."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from compliant_control_lab.franka_adaptive import (
    FrankaAdaptiveHybridController,
    FrankaSafeAdaptiveController,
)
from compliant_control_lab.franka_control import FrankaHybridController, FrankaState, FrankaTarget


def _vector(values: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValueError(f"expected a finite {size}-vector")
    return values


@dataclass(frozen=True)
class SurfaceFrame:
    """A fixed orientation with world columns [normal, tangent1, tangent2].

    World and surface coordinates share an origin. Wrench transformations only
    rotate force and moment at the same TCP; they do not shift the moment origin.
    This frame is specified for a trial, not estimated from contact online.
    """

    rotation: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=float)
        if (
            rotation.shape != (3, 3)
            or not np.all(np.isfinite(rotation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0.0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0.0)
        ):
            raise ValueError("rotation must be a finite proper orthonormal 3x3 matrix")
        rotation = rotation.copy()
        rotation.setflags(write=False)
        object.__setattr__(self, "rotation", rotation)

    @classmethod
    def from_normal(
        cls,
        normal: np.ndarray,
        tangent_hint: np.ndarray | None = None,
    ) -> SurfaceFrame:
        """Normalize a normal and project a nonparallel tangent hint onto its plane.

        Without a hint, tangent2 follows world +z projected onto the surface
        (world +y supplies the up direction for nearly vertical normals). Rotate
        an explicit hint together with the normal when checking frame covariance.
        """
        normal = _vector(normal, 3)
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("normal must be nonzero with finite norm")
        direction = normal / norm
        if tangent_hint is None:
            up = np.array([0.0, 0.0, 1.0])
            if abs(direction[2]) > 0.99:
                up = np.array([0.0, 1.0, 0.0])
            hint = np.cross(up, direction)
        else:
            hint = _vector(tangent_hint, 3)
        tangent = hint - direction * float(direction @ hint)
        tangent_norm = float(np.linalg.norm(tangent))
        if not np.isfinite(tangent_norm) or tangent_norm <= 1e-12:
            raise ValueError("tangent_hint must have a nonzero component tangent to the normal")
        tangent /= tangent_norm
        return cls(np.column_stack((direction, tangent, np.cross(direction, tangent))))

    def vector_to_local(self, vector: np.ndarray) -> np.ndarray:
        return self.rotation.T @ _vector(vector, 3)

    def vector_to_world(self, vector: np.ndarray) -> np.ndarray:
        return self.rotation @ _vector(vector, 3)

    def point_to_local(self, point: np.ndarray) -> np.ndarray:
        return self.vector_to_local(point)

    def point_to_world(self, point: np.ndarray) -> np.ndarray:
        return self.vector_to_world(point)

    def wrench_to_local(self, wrench: np.ndarray) -> np.ndarray:
        return (_vector(wrench, 6).reshape(2, 3) @ self.rotation).reshape(6)

    def wrench_to_world(self, wrench: np.ndarray) -> np.ndarray:
        return (_vector(wrench, 6).reshape(2, 3) @ self.rotation.T).reshape(6)


class SurfaceAdaptiveController:
    """Execute torque-safe adaptive control in one fixed local surface frame.

    The public interface remains reset(state), compute(state, target, dt) and
    read-only telemetry. Inputs and output use world coordinates at the same TCP;
    normal_force remains a positive compression scalar along the surface normal.
    This wrapper owns its mutable controller state and accepts no MuJoCo objects.
    A supplied base must be configured for local +x normal control and exclusively
    owned by this controller. The frame and its axes remain fixed between resets.
    """

    name = "surface_adaptive_hybrid"

    def __init__(
        self,
        frame: SurfaceFrame,
        base: FrankaSafeAdaptiveController | None = None,
    ) -> None:
        self._frame = frame
        if base is None:
            nominal = FrankaHybridController(
                normal=np.array([1.0, 0.0, 0.0]), force_transition_time=0.50
            )
            adaptive = FrankaAdaptiveHybridController(base=nominal)
            base = FrankaSafeAdaptiveController(base=adaptive)
        self._base = base
        self._initialized = False

    @property
    def frame(self) -> SurfaceFrame:
        return self._frame

    @property
    def corrected_force_n(self) -> float:
        return self._base.corrected_force_n

    @property
    def filtered_force_rate_n_s(self) -> float:
        return self._base.filtered_force_rate_n_s

    @property
    def contact_blend(self) -> float:
        return self._base.contact_blend

    @property
    def last_governed_normal_lead_m(self) -> float:
        return self._base.last_governed_normal_lead_m

    @property
    def last_torque_projection_scale(self) -> float:
        return self._base.last_torque_projection_scale

    @property
    def torque_projection_pct(self) -> float:
        return self._base.torque_projection_pct

    @property
    def mean_torque_projection_scale(self) -> float:
        return self._base.mean_torque_projection_scale

    @property
    def torque_projection_fallback_count(self) -> int:
        return self._base.torque_projection_fallback_count

    def _local_state(self, state: FrankaState) -> FrankaState:
        actuation = state.actuation
        if actuation is not None:
            rotation = self._frame.rotation.T
            jacobian = actuation.cartesian_jacobian
            actuation = replace(
                actuation,
                cartesian_jacobian=np.concatenate(
                    (rotation @ jacobian[:3], rotation @ jacobian[3:]), axis=0
                ),
            )
        return replace(
            state,
            position=self._frame.point_to_local(state.position),
            rotation=self._frame.rotation.T @ state.rotation,
            linear_velocity=self._frame.vector_to_local(state.linear_velocity),
            angular_velocity=self._frame.vector_to_local(state.angular_velocity),
            actuation=actuation,
        )

    def _local_target(self, target: FrankaTarget) -> FrankaTarget:
        return replace(
            target,
            position=self._frame.point_to_local(target.position),
            rotation=self._frame.rotation.T @ target.rotation,
            linear_velocity=self._frame.vector_to_local(target.linear_velocity),
            angular_velocity=self._frame.vector_to_local(target.angular_velocity),
        )

    def reset(self, state: FrankaState) -> None:
        self._base.reset(self._local_state(state))
        self._initialized = True

    def compute(self, state: FrankaState, target: FrankaTarget, dt: float) -> np.ndarray:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        local_state = self._local_state(state)
        local_target = self._local_target(target)
        if not self._initialized:
            self._base.reset(local_state)
            self._initialized = True
        wrench = self._base.compute(local_state, local_target, dt)
        return self._frame.wrench_to_world(wrench)
