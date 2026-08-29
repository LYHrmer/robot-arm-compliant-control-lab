import numpy as np
import pytest

from compliant_control_lab.franka_adaptive import FrankaAdaptiveHybridController
from compliant_control_lab.franka_control import FrankaState, FrankaTarget


def _state(force: float = 0.0, normal_position: float = 0.36) -> FrankaState:
    return FrankaState(
        position=np.array([normal_position, 0.0, 0.45]),
        rotation=np.eye(3),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        normal_force=force,
    )


def _target(force: float = 12.0) -> FrankaTarget:
    return FrankaTarget(
        position=np.array([0.38, 0.04, 0.42]),
        rotation=np.eye(3),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        normal_force=force,
    )


def test_adaptive_controller_estimates_precontact_force_bias() -> None:
    controller = FrankaAdaptiveHybridController()
    controller.reset(_state())

    for _ in range(200):
        controller.compute(_state(force=1.25), _target(force=0.0), dt=0.002)

    assert controller.estimated_force_bias_n == pytest.approx(1.25, abs=0.02)
    assert controller.corrected_force_n < 0.02


def test_adaptive_controller_reduces_force_gains_for_stiff_fast_contact() -> None:
    controller = FrankaAdaptiveHybridController()
    controller.reset(_state())

    for index in range(1, 15):
        controller.compute(
            _state(force=2.0 * index, normal_position=0.36 + 0.0001 * index),
            _target(),
            dt=0.002,
        )

    assert controller.estimated_contact_stiffness_n_m > controller.reference_contact_stiffness
    assert controller.force_gain_scale < 1.0


def test_adaptive_controller_keeps_guarded_wrench_finite_and_bounded() -> None:
    controller = FrankaAdaptiveHybridController()
    controller.reset(_state())
    wrench = controller.compute(_state(force=50.0), _target(), dt=0.002)

    assert np.all(np.isfinite(wrench))
    assert wrench[0] >= -controller.max_retreat_command - 1.0e-12
    assert wrench[0] <= controller.base.max_normal_command + 1.0e-12
