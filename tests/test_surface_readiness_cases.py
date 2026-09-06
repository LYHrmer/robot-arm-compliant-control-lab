from collections import Counter

import numpy as np
import pytest

from compliant_control_lab.learning_surface_task import LearningSurfaceTask
from compliant_control_lab.surface_readiness_cases import preparation_cases
from compliant_control_lab.surface_splits import case_group_id


def test_fixed_domain_has_grouped_noise_replicas_and_known_plane():
    cases = preparation_cases()
    assert len(cases) == 24 and cases == preparation_cases()
    assert len({case["case_id"] for case in cases}) == 24
    assert set(Counter(case["group_id"] for case in cases).values()) == {2}
    assert {case["split"] for case in cases} == {"train", "validation", "development_test"}
    assert Counter({case["group_id"]: case["split"] for case in cases}.values()) == {
        "train": 8,
        "validation": 2,
        "development_test": 2,
    }
    assert len({case["task"]["phase_rad"] for case in cases}) > 6
    assert {case["task"]["direction"] for case in cases} == {-1, 1}
    for case in cases:
        assert case_group_id(case) == case["group_id"]
        task, scenario, config = case["task"], case["scenario"], case["config"]
        assert task["yaw_deg"] == scenario["wall_yaw_deg"]
        assert scenario["tool_sliding_friction"] == scenario["wall_sliding_friction"]
        assert 0.25 <= scenario["wall_sliding_friction"] <= 0.65
        assert 0 <= scenario["delay_steps"] <= 3
        assert 8 <= config["target_force"] <= 16 and config["duration"] == 12
        frame = np.asarray(case["controller_frame_rotation"])
        controller_yaw = np.rad2deg(np.arctan2(frame[1, 0], frame[0, 0]))
        assert abs(controller_yaw - task["yaw_deg"]) <= 5
        LearningSurfaceTask(**task)


@pytest.mark.parametrize("group_count", [True, 0, 5, 6.0])
def test_invalid_group_count_rejected(group_count):
    with pytest.raises(ValueError):
        preparation_cases(group_count)
