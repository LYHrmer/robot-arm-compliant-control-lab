import numpy as np

from compliant_control_lab.controllers import (
    AdmittanceController,
    ControlTarget,
    HybridForcePositionController,
    Observation,
)


def _observation(force=0.0):
    return Observation(
        position=np.array([0.64, 0.02]),
        velocity=np.zeros(2),
        normal_force=force,
    )


def _target(force=12.0):
    return ControlTarget(
        position=np.array([0.66, 0.10]),
        velocity=np.zeros(2),
        normal_force=force,
    )


def test_admittance_moves_reference_toward_wall_when_force_is_low():
    controller = AdmittanceController()
    observation = _observation(force=0.0)
    controller.reset(observation)
    wrench = controller.compute(observation, _target(force=12.0), dt=0.01)
    assert wrench[0] > 0.0


def test_admittance_moves_away_when_force_is_too_high():
    controller = AdmittanceController()
    observation = _observation(force=20.0)
    controller.reset(observation)
    wrench = controller.compute(observation, _target(force=5.0), dt=0.01)
    assert wrench[0] < 0.0


def test_hybrid_controller_separates_force_and_position_axes():
    controller = HybridForcePositionController()
    observation = _observation(force=6.0)
    wrench = controller.compute(observation, _target(force=12.0), dt=0.002)
    assert wrench[0] > 12.0
    assert wrench[1] > 0.0

