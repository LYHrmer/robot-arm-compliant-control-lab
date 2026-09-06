"""Independent numeric specifications for smooth bounded friction feedforward."""

import numpy as np
import pytest

from compliant_control_lab.tangential_compensation import bounded_friction_force


@pytest.mark.parametrize(
    "velocity,force,mu", [([0, 0, 5], 1, 0.5), ([1, 2, 3], -5, 0.5), ([1, 2, 3], 5, 0)]
)
def test_no_tangent_speed_or_no_compression_gives_zero(velocity, force, mu):
    np.testing.assert_array_equal(
        bounded_friction_force([0, 0, 1], velocity, force, mu, 10, 1), np.zeros(3)
    )


def test_known_numeric_example():
    force = bounded_friction_force([0, 0, 1], [3, 4, 5], 2, 0.5, 10, 1)
    np.testing.assert_allclose(force, np.array([3, 4, 0]) / np.sqrt(26), atol=1e-12)


def test_tangent_only_and_norm_cap():
    force = bounded_friction_force([0, 0, 1], [100, 0, 0], 1000, 1, 5, 1)
    assert force.shape == (3,) and force[1] == force[2] == 0
    assert 0 < np.linalg.norm(force) < 5


def test_reversal_and_near_zero_continuity():
    positive = bounded_friction_force([0, 0, 1], [1e-6, 0, 0], 2, 0.5, 10, 1)
    negative = bounded_friction_force([0, 0, 1], [-1e-6, 0, 0], 2, 0.5, 10, 1)
    np.testing.assert_allclose(positive, -negative, atol=1e-12)
    assert np.linalg.norm(positive) < 1e-3


def test_so3_covariance():
    rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    normal, velocity = np.array([0, 0, 1]), np.array([1, 2, 3])
    force = bounded_friction_force(normal, velocity, 3, 0.4, 8, 1.5)
    rotated = bounded_friction_force(rotation @ normal, rotation @ velocity, 3, 0.4, 8, 1.5)
    np.testing.assert_allclose(rotation @ force, rotated, atol=1e-12)


@pytest.mark.parametrize(
    "index,value",
    [
        (0, [1, 1, 0]),
        (0, [0, 0]),
        (1, [1, 0]),
        (2, np.nan),
        (3, -0.1),
        (4, 0),
        (4, -1),
        (5, 0),
        (5, -1),
        (0, [np.nan, 0, 1]),
        (1, [np.nan, 0, 0]),
        (3, np.inf),
        (4, np.nan),
        (5, np.inf),
    ],
)
def test_invalid_parameters_raise(index, value):
    args = [[0, 0, 1], [1, 0, 0], 1, 0.5, 1, 1]
    args[index] = value
    with pytest.raises(ValueError):
        bounded_friction_force(*args)
