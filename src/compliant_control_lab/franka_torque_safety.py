"""Joint-torque safety projection for Cartesian Franka control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compliant_control_lab.franka_control import FrankaActuationContext


@dataclass(frozen=True)
class TorqueProjection:
    """Result of scaling an additive wrench along its original ray."""

    additive_wrench: np.ndarray
    scale: float
    status: str

    def __post_init__(self) -> None:
        additive_wrench = np.asarray(self.additive_wrench, dtype=float)
        if additive_wrench.shape != (6,) or not np.all(np.isfinite(additive_wrench)):
            raise ValueError("additive_wrench must be a finite 6-vector")
        if not np.isfinite(self.scale) or not 0.0 <= self.scale <= 1.0:
            raise ValueError("scale must be finite and in [0, 1]")
        object.__setattr__(self, "additive_wrench", additive_wrench.copy())

    @property
    def residual_force(self) -> np.ndarray:
        return self.additive_wrench[:3].copy()


def _validate_reserve_fraction(reserve_fraction: float) -> float:
    reserve_fraction = float(reserve_fraction)
    if not np.isfinite(reserve_fraction) or not 0.0 <= reserve_fraction < 1.0:
        raise ValueError("reserve_fraction must be finite and in [0, 1)")
    return reserve_fraction


def _wrench(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (6,):
        raise ValueError(f"{name} must have shape (6,)")
    return values


def _safe_limits(
    context: FrankaActuationContext,
    reserve_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    span = context.upper_torque_limit - context.lower_torque_limit
    reserve = 0.5 * reserve_fraction * span
    return context.lower_torque_limit + reserve, context.upper_torque_limit - reserve


def _inside_limits(
    torque: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> bool:
    limit_scale = max(1.0, float(np.max(np.abs(np.concatenate((lower, upper))))))
    tolerance = 32.0 * np.finfo(float).eps * limit_scale
    return bool(np.all(torque >= lower - tolerance) and np.all(torque <= upper + tolerance))


def _fallback(status: str) -> TorqueProjection:
    return TorqueProjection(additive_wrench=np.zeros(6), scale=0.0, status=status)


def project_wrench_to_torque_limits(
    context: FrankaActuationContext | None,
    nominal_wrench: np.ndarray,
    additive_wrench: np.ndarray,
    reserve_fraction: float = 0.10,
) -> TorqueProjection:
    """Scale a 6D additive wrench so all resulting joint torques remain admissible.

    The nominal wrench is never modified. If its joint torque is already outside the reserved
    envelope, the additive wrench is disabled instead of being used to repair nominal control.
    """

    reserve_fraction = _validate_reserve_fraction(reserve_fraction)
    nominal_wrench = _wrench(nominal_wrench, "nominal_wrench")
    additive_wrench = _wrench(additive_wrench, "additive_wrench")
    if context is None:
        return _fallback("missing_context")
    if not np.all(np.isfinite(nominal_wrench)) or not np.all(np.isfinite(additive_wrench)):
        return _fallback("nonfinite")

    lower, upper = _safe_limits(context, reserve_fraction)
    with np.errstate(over="ignore", invalid="ignore"):
        nominal_torque = context.joint_torque(nominal_wrench)
    if not np.all(np.isfinite(nominal_torque)):
        return _fallback("nonfinite")
    if not _inside_limits(nominal_torque, lower, upper):
        return _fallback("nominal_outside")

    with np.errstate(over="ignore", invalid="ignore"):
        additive_torque = context.cartesian_jacobian.T @ additive_wrench
        requested_torque = nominal_torque + additive_torque
    if not np.all(np.isfinite(additive_torque)) or not np.all(np.isfinite(requested_torque)):
        return _fallback("nonfinite")
    if _inside_limits(requested_torque, lower, upper):
        return TorqueProjection(additive_wrench.copy(), 1.0, "unchanged")

    positive = additive_torque > 0.0
    negative = additive_torque < 0.0
    ratios = np.full(additive_torque.shape, np.inf)
    ratios[positive] = (upper[positive] - nominal_torque[positive]) / additive_torque[
        positive
    ]
    ratios[negative] = (lower[negative] - nominal_torque[negative]) / additive_torque[
        negative
    ]
    scale = float(np.clip(np.min(ratios), 0.0, 1.0))
    if scale < 1.0 and scale > 0.0:
        scale = float(np.nextafter(scale, 0.0))
    projected = scale * additive_wrench

    with np.errstate(over="ignore", invalid="ignore"):
        projected_torque = context.joint_torque(nominal_wrench + projected)
    if not np.all(np.isfinite(projected_torque)) or not _inside_limits(
        projected_torque, lower, upper
    ):
        return _fallback("verification_failed")
    return TorqueProjection(projected, scale, "scaled")


def project_residual_force(
    context: FrankaActuationContext | None,
    nominal_wrench: np.ndarray,
    residual_force: np.ndarray,
    reserve_fraction: float = 0.10,
) -> TorqueProjection:
    """Project a translational residual through the complete wrench-to-torque map."""

    residual_force = np.asarray(residual_force, dtype=float)
    if residual_force.shape != (3,):
        raise ValueError("residual_force must have shape (3,)")
    additive_wrench = np.zeros(6)
    additive_wrench[:3] = residual_force
    return project_wrench_to_torque_limits(
        context,
        nominal_wrench,
        additive_wrench,
        reserve_fraction,
    )


def residual_torque_headroom(
    context: FrankaActuationContext | None,
    nominal_wrench: np.ndarray,
    action_bounds: np.ndarray,
    reserve_fraction: float = 0.10,
) -> np.ndarray:
    """Return normalized torque headroom for +x, -x, +y, -y, +z and -z residuals."""

    _validate_reserve_fraction(reserve_fraction)
    nominal_wrench = _wrench(nominal_wrench, "nominal_wrench")
    action_bounds = np.asarray(action_bounds, dtype=float)
    if (
        action_bounds.shape != (3,)
        or not np.all(np.isfinite(action_bounds))
        or np.any(action_bounds <= 0.0)
    ):
        raise ValueError("action_bounds must be a finite positive 3-vector")

    headroom = np.zeros(6)
    for axis in range(3):
        for direction_index, direction in enumerate((1.0, -1.0)):
            residual = np.zeros(3)
            residual[axis] = direction * action_bounds[axis]
            projection = project_residual_force(
                context,
                nominal_wrench,
                residual,
                reserve_fraction,
            )
            if projection.status in {"unchanged", "scaled"}:
                headroom[2 * axis + direction_index] = projection.scale
    return headroom
