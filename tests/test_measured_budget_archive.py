"""Published packet faults must not perturb the fixed-budget control baselines."""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "results/franka_measured_budget_robustness/traces"
CONTROL_FIELDS = (
    "position", "linear_velocity", "measured_position", "measured_linear_velocity",
    "measured_normal_force", "commanded_wrench", "commanded_torque",
    "controller_coefficient_after_compute",
)


@pytest.mark.parametrize("method", ["fixed6", "fixed8"])
def test_auxiliary_faults_leave_fixed_control_outputs_bit_exact(method):
    with np.load(TRACES / f"yaw0_combined_fresh__{method}.npz", allow_pickle=False) as archive:
        reference = {key: archive[key] for key in CONTROL_FIELDS}
    for fault in ("bias_plus", "bias_minus", "scale_0p8", "scale_1p2", "missing", "stale"):
        with np.load(TRACES / f"yaw0_combined_{fault}__{method}.npz", allow_pickle=False) as trace:
            for key in CONTROL_FIELDS:
                np.testing.assert_array_equal(trace[key], reference[key], err_msg=f"{fault}/{key}")


@pytest.mark.parametrize("fault", ["missing", "stale"])
def test_real_packet_rejection_exercises_legal_falling_budget(fault):
    with np.load(TRACES / f"yaw0_combined_{fault}__adaptive6_8.npz", allow_pickle=False) as trace:
        force = trace["load_compensation_force_local"]
        applied = trace["load_budget_applied_n"]
        active = trace["diagnostic_compensation_active"]
        rejected = ~trace["load_measurement_available"]
        assert np.any(np.linalg.norm(force, axis=1)[rejected] > applied[rejected] + 1e-10)
        assert np.max(np.linalg.norm(force, axis=1)) <= 8.0 + 1e-10
        delta = np.linalg.norm(np.diff(np.vstack((np.zeros(3), force)), axis=0), axis=1)
        assert np.max(delta[active]) <= 20.0 * float(trace["dt"]) + 1e-10
