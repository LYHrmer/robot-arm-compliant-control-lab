"""One impedance calculation with a synthetic Jacobian; no robot actuation."""

from __future__ import annotations

import argparse

import numpy as np

from compliant_control_lab.franka_control import (
    FrankaActuationContext,
    FrankaImpedanceController,
    FrankaState,
    FrankaTarget,
)

# Rows are [vx, vy, vz, wx, wy, wz]. This is not a Franka model sample.
JACOBIAN = np.array(
    [
        [0.0, 0.2, 0.0, -0.1, 0.0, 0.0, 0.1],
        [0.1, 0.0, 0.3, 0.0, -0.2, 0.0, 0.0],
        [0.0, -0.1, 0.0, 0.2, 0.0, 0.1, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    ]
)


def compute_example(x_error_mm: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Return the production controller's wrench and zero-offset task torque."""
    if x_error_mm not in (10.0, 20.0):
        raise ValueError("x_error_mm must be 10 or 20 for this exercise")
    state = FrankaState(
        position=np.zeros(3),
        rotation=np.eye(3),
        linear_velocity=np.array([0.02, -0.01, 0.0]),
        angular_velocity=np.array([-0.1, 0.2, -0.05]),
        normal_force=0.0,
    )
    target = FrankaTarget(
        position=np.array([x_error_mm / 1000.0, -0.02, 0.005]),
        rotation=np.eye(3),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        normal_force=0.0,
    )
    controller = FrankaImpedanceController(
        translational_stiffness=np.array([300.0, 200.0, 400.0]),
        translational_damping=np.array([20.0, 30.0, 40.0]),
        rotational_stiffness=np.array([20.0, 20.0, 20.0]),
        rotational_damping=np.array([2.0, 3.0, 4.0]),
    )
    controller.reset(state)
    wrench = controller.compute(state, target, dt=0.002)
    context = FrankaActuationContext(
        cartesian_jacobian=JACOBIAN,
        joint_torque_offset=np.zeros(7),
        # Required placeholders: joint_torque() does not check or apply limits.
        lower_torque_limit=np.full(7, -100.0),
        upper_torque_limit=np.full(7, 100.0),
    )
    return wrench, context.joint_torque(wrench)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-error-mm", type=float, choices=(10.0, 20.0), default=10.0)
    args = parser.parse_args(argv)
    wrench, torque = compute_example(args.x_error_mm)

    def vector(values: np.ndarray) -> str:
        return "[" + ", ".join(f"{value:.3f}" for value in values) + "]"

    print("Synthetic Jacobian; single-step calculation, not a hardware command.")
    print(f"x position error (mm): {args.x_error_mm:.1f}")
    print(f"force [fx, fy, fz] (N): {vector(wrench[:3])}")
    print(f"moment [mx, my, mz] (N m): {vector(wrench[3:])}")
    print(f"task torque [tau1..tau7] (N m): {vector(torque)}")
    print("No bias, nullspace torque, safety projection, or physics integration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
