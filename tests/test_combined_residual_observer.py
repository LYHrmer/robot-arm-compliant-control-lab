"""Recorded-input checks for the modest-combined residual diagnosis."""

from pathlib import Path

import numpy as np
import pytest

from compliant_control_lab.franka_control import (
    FrankaActuationContext,
    FrankaState,
    FrankaTarget,
)
from compliant_control_lab.surface_control import SurfaceAdaptiveController, SurfaceFrame
from compliant_control_lab.tangential_compensation import TangentialCompensation

TRACE = (
    Path(__file__).parents[1]
    / "results/franka_cross_surface_dynamic/traces"
    / "yaw_-15__modest_combined__seed_11__online__s1__full.npz"
)


class Observer(SurfaceAdaptiveController):
    """Capture online state on each side of the unchanged control seam."""

    def __init__(self, frame):
        super().__init__(frame, tangential_mode="online")
        self.before = []
        self.after = []
        self.ready_before = []
        self.ready_after = []
        self.requests = []
        self.uncapped_amplitudes = []

    def compute(self, state, target, dt):
        self.before.append(self.equivalent_tangential_coefficient)
        self.ready_before.append(self.tangential_update_ready)
        wrench = super().compute(state, target, dt)
        self.after.append(self.equivalent_tangential_coefficient)
        self.ready_after.append(self.tangential_update_ready)
        self.requests.append(self.requested_tangential_force_world)
        self.uncapped_amplitudes.append(self.before[-1] * self.corrected_force_n)
        return wrench


def test_online_amplitude_ceiling_can_be_hidden_by_velocity_smoothing():
    compensation = TangentialCompensation("online")
    state = FrankaState(
        np.zeros(3), np.eye(3), np.array([0.0, 0.02, 0.0]), np.zeros(3), 20.0
    )
    target = FrankaTarget(
        np.zeros(3), np.eye(3), np.array([0.0, 0.02, 0.0]), np.zeros(3), 12.0
    )

    for _ in range(200):
        request = compensation.force(state, target, [1.0, 0.0, 0.0], 20.0, 1.0, True, dt=0.002)

    assert compensation.equivalent_mu * 20.0 > compensation.max_force
    expected = compensation.max_force * 0.02 / np.sqrt(0.02**2 + 0.005**2)
    assert np.linalg.norm(request) == pytest.approx(expected)
    assert np.linalg.norm(request) < compensation.max_force


def test_recorded_inputs_replay_original_combined_controller_and_observe_cap():
    with np.load(TRACE, allow_pickle=False) as archive:
        trace = {name: archive[name] for name in archive.files}
    controller = Observer(SurfaceFrame(trace["controller_frame_rotation"]))
    count = len(trace["time"])
    wrenches = np.empty_like(trace["commanded_wrench"])
    torques = np.empty_like(trace["commanded_torque"])

    for index in range(count):
        actuation = FrankaActuationContext(
            trace["cartesian_jacobian"][index],
            trace["joint_torque_offset"][index],
            trace["lower_torque_limit"][index],
            trace["upper_torque_limit"][index],
        )
        state = FrankaState(
            trace["measured_position"][index],
            trace["measured_rotation"][index],
            trace["measured_linear_velocity"][index],
            trace["measured_angular_velocity"][index],
            trace["measured_normal_force"][index],
            actuation,
        )
        target = FrankaTarget(
            trace["target_position"][index],
            trace["target_rotation"][index],
            trace["target_linear_velocity"][index],
            trace["target_angular_velocity"][index],
            trace["target_normal_force"][index],
        )
        if index == 0:
            controller.reset(state)
        wrenches[index] = controller.compute(state, target, float(trace["dt"]))
        torques[index] = actuation.joint_torque(wrenches[index])

    np.testing.assert_array_equal(wrenches, trace["commanded_wrench"])
    np.testing.assert_array_equal(torques, trace["commanded_torque"])
    np.testing.assert_array_equal(controller.before, trace["controller_coefficient_before_compute"])
    np.testing.assert_array_equal(controller.after, trace["controller_coefficient_after_compute"])
    np.testing.assert_array_equal(controller.ready_before, trace["controller_update_ready_before_compute"])
    np.testing.assert_array_equal(controller.ready_after, trace["controller_update_ready_after_compute"])
    np.testing.assert_array_equal(controller.requests, trace["requested_tangential_force_world"])

    post = trace["time"] >= 8.0
    capped = np.asarray(controller.uncapped_amplitudes) >= 6.0
    assert np.mean(capped[post]) == pytest.approx(0.98)
    assert np.mean(np.asarray(controller.ready_after)[post]) == 1.0
    assert np.max(np.linalg.norm(np.asarray(controller.requests)[post], axis=1)) < 6.0
