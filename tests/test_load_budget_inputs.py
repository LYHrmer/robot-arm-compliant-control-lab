"""Information-boundary tests for causal load-budget inputs."""

from types import SimpleNamespace

import numpy as np
import pytest

from compliant_control_lab.surface_control import SurfaceFrame
from tools.load_budget_inputs import measured_force_input

ALLOWED = {"time", "measured_wrench_sample_time", "measured_wrench_world"}


class PoisonMapping:
    """Fail if extraction inspects anything beyond the three approved row fields."""

    def __init__(self, values):
        self.values = values
        self.accessed = []

    def __getitem__(self, name):
        if name not in ALLOWED:
            raise AssertionError(f"forbidden input accessed: {name}")
        self.accessed.append(name)
        return self.values[name]

    def __iter__(self):
        raise AssertionError("input-row enumeration is forbidden")

    def get(self, *args):
        raise AssertionError("input-row probing is forbidden")


def _fixture(wrench=None, *, row_time=0.2, measurement_time=0.198, normal_force=12.0):
    if wrench is None:
        wrench = np.array([12.0, 2.0, -3.0, 0.1, 0.2, 0.3])
    row = {
        "time": row_time,
        "measured_wrench_sample_time": measurement_time,
        "measured_wrench_world": wrench,
    }
    sample = SimpleNamespace(
        time=row_time,
        measured_wrench_sample_time=measurement_time,
        state=SimpleNamespace(normal_force=normal_force),
    )
    return row, sample


def test_identity_frame_preserves_robot_on_environment_sign_and_tcp_force():
    row, sample = _fixture()

    result = measured_force_input(row, sample, SurfaceFrame(np.eye(3)))

    np.testing.assert_array_equal(result["force_local"], [12.0, 2.0, -3.0])
    np.testing.assert_array_equal(result["wrench_world"], row["measured_wrench_world"])
    assert result["measurement_time_s"] == 0.198
    assert result["measurement_age_s"] == pytest.approx(0.002)


def test_rotated_frame_uses_rotation_only_without_force_torque_mixing():
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    frame = SurfaceFrame(rotation)
    local_wrench = np.array([12.0, 2.0, -3.0, 1e6, -2e6, 3e6])
    world_wrench = frame.wrench_to_world(local_wrench)
    row, sample = _fixture(world_wrench, normal_force=12.0)

    result = measured_force_input(row, sample, frame)

    np.testing.assert_allclose(result["force_local"], local_wrench[:3], atol=1e-12)


def test_poison_mapping_proves_only_three_approved_fields_are_read():
    values, sample = _fixture()
    values.update(
        {
            "true_normal_force": object(),
            "true_tangent_force_n": object(),
            "applied_raw_wrench_bias_world": object(),
            "feedback_raw_wrench_world": object(),
            "_previous_wrench": object(),
        }
    )
    row = PoisonMapping(values)

    measured_force_input(row, sample, SurfaceFrame(np.eye(3)))

    assert row.accessed == ["time", "measured_wrench_sample_time", "measured_wrench_world"]


def test_outputs_are_owned_and_repeated_calls_do_not_mutate_inputs():
    row, sample = _fixture()
    original = row["measured_wrench_world"].copy()

    first = measured_force_input(row, sample, SurfaceFrame(np.eye(3)))
    first["force_local"][:] = 99.0
    first["wrench_world"][:] = 88.0
    second = measured_force_input(row, sample, SurfaceFrame(np.eye(3)))

    np.testing.assert_array_equal(row["measured_wrench_world"], original)
    np.testing.assert_array_equal(second["force_local"], original[:3])
    np.testing.assert_array_equal(second["wrench_world"], original)


def test_declared_time_tolerances_are_accepted():
    row, sample = _fixture()
    row["time"] += 0.9e-12
    row["measured_wrench_sample_time"] -= 0.9e-12

    result = measured_force_input(row, sample, SurfaceFrame(np.eye(3)))

    assert result["measurement_time_s"] == 0.198 - 0.9e-12
    assert result["measurement_age_s"] == pytest.approx(0.002 + 0.9e-12)


@pytest.mark.parametrize("measurement_time", (-1e-15, 0.201))
def test_rejects_negative_or_future_measurement_time(measurement_time):
    row, sample = _fixture(measurement_time=measurement_time)

    with pytest.raises(ValueError, match=r"within \[0, sample time\]"):
        measured_force_input(row, sample, SurfaceFrame(np.eye(3)))


@pytest.mark.parametrize("which", ("row", "measurement"))
def test_rejects_row_and_sample_timestamp_disagreement(which):
    row, sample = _fixture()
    if which == "row":
        sample.time += 2e-12
        message = "row time differs"
    else:
        sample.measured_wrench_sample_time += 2e-12
        message = "wrench time differs"

    with pytest.raises(ValueError, match=message):
        measured_force_input(row, sample, SurfaceFrame(np.eye(3)))


@pytest.mark.parametrize(
    "target",
    ("row_time", "measurement_time", "sample_time", "sample_measurement_time", "normal_force"),
)
def test_rejects_nonfinite_times_and_normal_force(target):
    row, sample = _fixture()
    if target == "row_time":
        row["time"] = np.nan
    elif target == "measurement_time":
        row["measured_wrench_sample_time"] = np.inf
    elif target == "sample_time":
        sample.time = np.nan
    elif target == "sample_measurement_time":
        sample.measured_wrench_sample_time = -np.inf
    else:
        sample.state.normal_force = np.nan

    with pytest.raises(ValueError, match="must be finite"):
        measured_force_input(row, sample, SurfaceFrame(np.eye(3)))


@pytest.mark.parametrize("wrench", (np.zeros(5), np.zeros((2, 3)), np.full(6, np.nan)))
def test_rejects_wrong_shape_or_nonfinite_wrench(wrench):
    row, sample = _fixture(wrench, normal_force=0.0)

    with pytest.raises(ValueError, match="finite 6-vector"):
        measured_force_input(row, sample, SurfaceFrame(np.eye(3)))


@pytest.mark.parametrize("missing", (None, "time", "measured_wrench_world"))
def test_rejects_none_or_missing_input_row(missing):
    row, sample = _fixture()
    if missing is None:
        row = None
    else:
        del row[missing]

    with pytest.raises(ValueError, match="input row"):
        measured_force_input(row, sample, SurfaceFrame(np.eye(3)))


def test_rejects_wrench_that_disagrees_with_sample_normal_force():
    row, sample = _fixture(normal_force=11.9)

    with pytest.raises(ValueError, match="normal component differs"):
        measured_force_input(row, sample, SurfaceFrame(np.eye(3)))


def test_normal_projection_tolerance_is_absolute_one_e_minus_ten():
    row, sample = _fixture(normal_force=12.0 + 0.9e-10)
    measured_force_input(row, sample, SurfaceFrame(np.eye(3)))

    sample.state.normal_force = 12.0 + 1.1e-10
    with pytest.raises(ValueError, match="normal component differs"):
        measured_force_input(row, sample, SurfaceFrame(np.eye(3)))


def test_controller_yaw_error_has_analytic_normal_to_tangent_leakage():
    angle = np.deg2rad(5.0)
    frame = SurfaceFrame.from_normal(np.array([np.cos(angle), np.sin(angle), 0.0]))
    world_wrench = np.array([12.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    row, sample = _fixture(world_wrench, normal_force=12.0 * np.cos(angle))

    force = measured_force_input(row, sample, frame)["force_local"]

    assert force[0] == pytest.approx(12.0 * np.cos(angle))
    assert force[1] == pytest.approx(-12.0 * np.sin(angle))
    assert force[2] == pytest.approx(0.0)


def test_mass_calibration_gravity_residual_remains_tangential():
    residual = (0.13 - 0.10) * 9.81
    row, sample = _fixture(
        np.array([12.0, 0.0, residual, 0.0, 0.0, 0.0]), normal_force=12.0
    )

    force = measured_force_input(row, sample, SurfaceFrame(np.eye(3)))["force_local"]

    np.testing.assert_allclose(force, [12.0, 0.0, residual], atol=1e-15)
