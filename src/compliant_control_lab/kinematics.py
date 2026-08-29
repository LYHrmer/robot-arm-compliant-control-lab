"""Analytic kinematics for the two-link planar arm used in the lab."""

from __future__ import annotations

import numpy as np

LINK_LENGTHS = np.array([0.45, 0.35], dtype=float)


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """Return end-effector ``[x, y]`` for joint angles ``[q1, q2]``."""
    q1, q2 = np.asarray(q, dtype=float)
    l1, l2 = LINK_LENGTHS
    return np.array(
        [
            l1 * np.cos(q1) + l2 * np.cos(q1 + q2),
            l1 * np.sin(q1) + l2 * np.sin(q1 + q2),
        ]
    )


def jacobian(q: np.ndarray) -> np.ndarray:
    """Return the 2x2 geometric Jacobian mapping joint to Cartesian velocity."""
    q1, q2 = np.asarray(q, dtype=float)
    l1, l2 = LINK_LENGTHS
    return np.array(
        [
            [
                -l1 * np.sin(q1) - l2 * np.sin(q1 + q2),
                -l2 * np.sin(q1 + q2),
            ],
            [
                l1 * np.cos(q1) + l2 * np.cos(q1 + q2),
                l2 * np.cos(q1 + q2),
            ],
        ]
    )


def inverse_kinematics(position: np.ndarray, elbow_up: bool = False) -> np.ndarray:
    """Solve the planar arm IK problem for a reachable Cartesian position."""
    x, y = np.asarray(position, dtype=float)
    l1, l2 = LINK_LENGTHS
    cos_q2 = (x * x + y * y - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if not -1.0 <= cos_q2 <= 1.0:
        raise ValueError(f"Position {position} is outside the arm workspace")

    q2 = np.arccos(np.clip(cos_q2, -1.0, 1.0))
    if elbow_up:
        q2 = -q2
    q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    return np.array([q1, q2])

