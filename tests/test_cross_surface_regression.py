from dataclasses import asdict

import numpy as np
import pytest

from compliant_control_lab.online_compensation_experiment import protocol_cases
from compliant_control_lab.surface_simulation import yaw_frame
from tools import cross_surface_regression as study


def test_cases_keep_original_physics_except_surface_yaw_and_rotated_bias():
    originals = {c.name: c for c in protocol_cases() if c.config.seed == 11}
    cases = study.make_cases()
    assert len(cases) == 6
    assert {(c.name, c.scenario.wall_yaw_deg) for c in cases} == {
        (name, yaw) for name in study.PROFILES for yaw in (-15, 0, 15)
    }
    for case in cases:
        old = originals[case.name]
        if case.scenario.wall_yaw_deg == 15:
            assert asdict(case) == asdict(old)
        assert case.config == old.config
        assert (case.config.seed, case.config.duration, case.config.timestep,
                case.config.evaluation_start, case.config.contact_model) == (11, 12, .002, 1.5, "smooth")
        assert case.phases == old.phases
        assert case.friction == old.friction
        assert case.recovery_start_s == old.recovery_start_s
        assert case.task.nominal_plane_x_m == old.task.nominal_plane_x_m
        assert case.task.trajectory == old.task.trajectory
        assert case.controller_yaw_error_deg == old.controller_yaw_error_deg
        left, right = asdict(case.scenario), asdict(old.scenario)
        for key in ("name", "wall_yaw_deg"):
            left.pop(key)
            right.pop(key)
        assert left == right
        np.testing.assert_array_equal(case.wrench_bias_world.at(5.998), np.zeros(6))
        normal = yaw_frame(case.task.yaw_deg).rotation[:, 0]
        expected = np.r_[.75 * normal, np.zeros(3)] if case.name == "modest_combined" else np.zeros(6)
        np.testing.assert_array_equal(case.wrench_bias_world.at(6.0), expected)


def test_execution_frames_and_trace_identity_match_actual_controller():
    cases = study.make_cases()
    entries = study.executions(cases)
    assert len(entries) == len({e["stem"] for e in entries}) == 36
    for entry in entries:
        error = 3 if entry["case"] == "modest_combined" else 0
        np.testing.assert_allclose(entry["controller"]["surface_frame_rotation"],
                                   yaw_frame(entry["surface_yaw_deg"] + error).rotation)
        base = entry["controller"]["safe_adaptive_base"]["base"]["base"]
        np.testing.assert_allclose(base["rotational_stiffness"], 20 * entry["scale"])
        np.testing.assert_allclose(base["rotational_damping"], 5 * np.sqrt(entry["scale"]))
    assert sum(e["surface_yaw_deg"] == -15 and e["arm"] == "online" for e in entries) == 4
    assert sum(len(c.phases) for c in cases) * 3 * 2 == 144
    assert study.make_protocol()["default_changed"] is False
    assert study.make_protocol()["new_holdout"] is False


def test_reference_hashes_and_original_case_documents_are_pinned():
    references = study.load_references()
    assert set(references) == {1, 2}
    for _, tables in references.values():
        assert len(tables["comparison"]) == 80
        assert len(tables["phase_metrics"]) == 168


def test_reference_guard_rejects_numeric_or_category_change():
    assert study.compare_reference({"x": 1.0, "y": True}, {"x": "1", "y": "True"}) == 0
    with pytest.raises(ValueError, match="metric differs"):
        study.compare_reference({"x": 1.0001}, {"x": "1"})
    with pytest.raises(ValueError, match="category differs"):
        study.compare_reference({"x": "no"}, {"x": "yes"})


def test_occupied_output_preserves_existing_data(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("user data")
    with pytest.raises(FileExistsError):
        study.generate(tmp_path)
    assert marker.read_text() == "user data"


def test_headroom_uses_unclipped_requests_and_masks():
    trace = {"commanded_torque": np.array([[11.0], [8.0]]),
             "lower_torque_limit": np.full((2, 1), -10.0),
             "upper_torque_limit": np.full((2, 1), 10.0),
             "torque_projection_scale": np.array([0.8, 1.0])}
    whole = study.extra_metrics(trace, slice(None))
    assert whole == {"projection_pct": 50.0, "minimum_torque_headroom_nm": -1.0,
                     "minimum_reserved_torque_headroom_nm": -2.0}
    assert study.extra_metrics(trace, np.array([False, True]))["minimum_reserved_torque_headroom_nm"] == 1
