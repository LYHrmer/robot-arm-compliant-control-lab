import numpy as np
import pytest

from compliant_control_lab.surface_control import SurfaceFrame
from tools import rotation_gain_regression as regression


def _row(scale, method="online", value=0.0, *, seed=11):
    return {
        "scale": scale,
        "method": method,
        "case_index": 0,
        "simulation_seed": seed,
        "wall_yaw_deg": -15.0,
        "wall_time_constant_s": 0.005,
        "tool_mass_kg": 0.10,
        **{name: value for name in regression.ALL_METRICS},
        "frozen_metric_max_abs_error": 0.0 if scale == 1 else None,
        "all_gates_pass": "yes",
        "failed_gates": "",
    }


def test_pinned_reference_and_protocol_preserve_public_24_grid_and_four_methods():
    reference, configurations, identity = regression.load_reference()
    expected = {
        (method, index) for method in regression.METHODS for index in range(24)
    }
    assert len(reference) == len(configurations) == 96
    assert set(reference) == set(configurations) == expected
    assert identity["manifest_sha256"] == regression.REFERENCE_MANIFEST_SHA256

    protocol, cases = regression.make_protocol(list(range(24)), configurations)
    assert protocol["methods"] == regression.METHODS
    assert protocol["selected_case_indices"] == list(range(24))
    assert protocol["is_subset"] is False
    assert [case["case_index"] for case in cases] == list(range(24))
    physical_groups = {}
    for case in cases:
        scenario, config = case["scenario"], case["config"]
        key = (
            scenario.wall_yaw_deg,
            scenario.wall_time_constant,
            scenario.tool_mass_kg,
        )
        physical_groups.setdefault(key, set()).add(config.seed)
    assert len(physical_groups) == 12
    assert all(seeds == {11, 29} for seeds in physical_groups.values())

    for method in regression.METHODS:
        scale_1 = protocol["controller_parameters"]["1"][method]["base"]["base"]
        scale_2 = protocol["controller_parameters"]["2"][method]["base"]["base"]
        np.testing.assert_allclose(scale_2["rotational_stiffness"], [40.0] * 3)
        np.testing.assert_allclose(scale_2["rotational_damping"], np.sqrt(2) * 5.0)
        for name in (
            "normal",
            "force_kp",
            "force_ki",
            "normal_damping",
            "tangential_stiffness",
            "tangential_damping",
        ):
            assert scale_2[name] == scale_1[name]


@pytest.mark.parametrize(
    "indices",
    [[], [0, 0], [-1], [24], [True], [1.5], ["1"]],
)
def test_protocol_rejects_invalid_case_indices(indices):
    _, configurations, _ = regression.load_reference()
    with pytest.raises(ValueError, match="case indices"):
        regression.make_protocol(indices, configurations)


@pytest.mark.parametrize("scale", [0.0, -1.0, 3.0, np.nan, True, "1"])
def test_controller_rejects_scales_outside_fixed_matched_study(scale):
    with pytest.raises(ValueError, match="fixed scale 1 or 2"):
        regression.controller(SurfaceFrame(np.eye(3)), "baseline", scale)


def test_controller_rejects_method_outside_fixed_four():
    with pytest.raises(ValueError, match="four methods"):
        regression.controller(SurfaceFrame(np.eye(3)), "unknown", 1.0)


def test_pairs_are_same_method_case_seed_and_scale_two_minus_scale_one():
    rows = [
        _row(1, "baseline", 10.0),
        _row(1, "online", 20.0),
        _row(2, "online", 23.0),
        _row(2, "baseline", 11.0),
    ]
    pairs = regression.pair_rows(rows)
    assert [(row["method"], row["case_index"], row["simulation_seed"]) for row in pairs] == [
        ("online", 0, 11),
        ("baseline", 0, 11),
    ]
    assert pairs[0]["orientation_rmse_deg"] == 3.0
    assert pairs[1]["orientation_rmse_deg"] == 1.0
    assert all(row["from_scale"] == 1.0 and row["to_scale"] == 2.0 for row in pairs)

    rows[2]["simulation_seed"] = 29
    with pytest.raises(ValueError, match="case/seed identity differs"):
        regression.pair_rows(rows)


def test_projection_and_reserved_headroom_use_inclusive_tolerance_boundaries():
    friction = {"force_rmse_n": 1.0, "orientation_rmse_deg": 2.0}
    row = {
        "has_raw_contact": True,
        "contact_ratio_pct": 99.0,
        "peak_force_n": 35.0,
        "saturation_pct": 0.0,
        "force_rmse_n": 1.2,
        "orientation_rmse_deg": 2.2,
        "projection_pct": 1e-10,
        "minimum_reserved_torque_headroom_nm": -1e-10,
    }
    assert regression.failures(row, friction) == ()
    row["projection_pct"] = np.nextafter(1e-10, np.inf)
    row["minimum_reserved_torque_headroom_nm"] = np.nextafter(-1e-10, -np.inf)
    assert regression.failures(row, friction)[-2:] == (
        "torque_projection",
        "reserved_torque_headroom",
    )


def test_summary_retains_every_failed_case():
    rows = [_row(scale, method, float(scale)) for scale in (1, 2)
            for method in regression.METHODS]
    rows[1].update(all_gates_pass="no", failed_gates="torque_projection")
    rows[-1].update(
        all_gates_pass="no",
        failed_gates="torque_projection;reserved_torque_headroom",
    )
    result = regression.summary(rows, regression.pair_rows(rows), [0])
    assert result["failures"] == [
        {
            "scale": 1,
            "method": "integral",
            "case_index": 0,
            "failed_gates": "torque_projection",
        },
        {
            "scale": 2,
            "method": "online",
            "case_index": 0,
            "failed_gates": "torque_projection;reserved_torque_headroom",
        },
    ]
    assert result["gate_counts"]["1"]["integral"] == 0
    assert result["gate_counts"]["2"]["online"] == 0


def test_generate_refuses_occupied_output_without_running_trial(tmp_path, monkeypatch):
    output = tmp_path / "occupied"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("user data", encoding="utf-8")
    monkeypatch.setattr(
        regression,
        "run_trial",
        lambda *_: pytest.fail("occupied output must be rejected before simulation"),
    )
    with pytest.raises((FileExistsError, ValueError)):
        regression.generate(output, case_indices=[0])
    assert marker.read_text(encoding="utf-8") == "user data"
