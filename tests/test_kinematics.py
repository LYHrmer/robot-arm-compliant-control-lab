import numpy as np
import pytest

from compliant_control_lab.kinematics import forward_kinematics, inverse_kinematics, jacobian


def test_jacobian_matches_finite_difference():
    q = np.array([-0.35, 1.1])
    epsilon = 1e-7
    numerical = np.column_stack(
        [
            (forward_kinematics(q + epsilon * np.eye(2)[joint]) - forward_kinematics(q))
            / epsilon
            for joint in range(2)
        ]
    )
    np.testing.assert_allclose(jacobian(q), numerical, atol=1e-6)


@pytest.mark.parametrize("position", [np.array([0.56, 0.05]), np.array([0.62, -0.12])])
def test_inverse_kinematics_round_trip(position):
    q = inverse_kinematics(position)
    np.testing.assert_allclose(forward_kinematics(q), position, atol=1e-9)


def test_inverse_kinematics_rejects_unreachable_position():
    with pytest.raises(ValueError):
        inverse_kinematics(np.array([2.0, 0.0]))

