from dataclasses import asdict

import numpy as np
import pytest

from compliant_control_lab.franka_adaptive import FrankaSafeAdaptiveController
from compliant_control_lab.franka_control import FrankaActuationContext, FrankaState, FrankaTarget
from compliant_control_lab.surface_control import SurfaceAdaptiveController, SurfaceFrame
from compliant_control_lab.surface_replay import replay_surface_trace, save_surface_trace


def _history(*, world=False, controller=None):
    """Nonconstant measured inputs exercising reset, contact transition and controller memory."""
    count, dt = 360, 0.002
    time = np.arange(count) * dt
    angle = 0.0 if world else np.deg2rad(23.0)
    frame = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    position = np.column_stack(
        (0.37 + 0.001 * np.sin(2 * time), 0.04 * np.sin(3 * time), 0.5 + 0.005 * np.cos(time))
    )
    velocity = np.column_stack(
        (0.002 * np.cos(2 * time), 0.12 * np.cos(3 * time), -0.005 * np.sin(time))
    )
    jacobian = np.random.default_rng(17).normal(0, 0.12, (count, 6, 7))
    arrays = {
        "schema_version": np.array(1),
        "dt": np.array(dt),
        "controller_kind": np.array("world_safe_adaptive" if world else "surface_adaptive"),
        "controller_frame_rotation": frame,
        "measured_position": position @ frame.T,
        "measured_rotation": np.tile(frame, (count, 1, 1)),
        "measured_linear_velocity": velocity @ frame.T,
        "measured_angular_velocity": np.column_stack(
            (np.zeros(count), np.zeros(count), 0.02 * np.sin(time))
        ),
        "measured_normal_force": np.where(time < 0.04, 0.0, 12 + np.sin(9 * time)),
        "cartesian_jacobian": jacobian,
        "joint_torque_offset": np.tile(np.arange(7) / 2, (count, 1)) + np.sin(time[:, None]),
        "lower_torque_limit": np.tile(-np.arange(6, 13, dtype=float), (count, 1)),
        "upper_torque_limit": np.tile(np.arange(5, 12, dtype=float), (count, 1)),
        "target_position": (position + np.array([0.02, 0.002, -0.001])) @ frame.T,
        "target_rotation": np.tile(frame, (count, 1, 1)),
        "target_linear_velocity": 0.8 * velocity @ frame.T,
        "target_angular_velocity": np.zeros((count, 3)),
        "target_normal_force": np.full(count, 12.0),
        "commanded_wrench": np.empty((count, 6)),
        "commanded_torque": np.empty((count, 7)),
        "applied_torque": np.empty((count, 7)),
        "time": time,
    }
    if controller is None:
        controller = (
            FrankaSafeAdaptiveController()
            if world
            else SurfaceAdaptiveController(SurfaceFrame(frame))
        )
    for index in range(count):
        context = FrankaActuationContext(
            **{
                key: arrays[key][index]
                for key in (
                    "cartesian_jacobian",
                    "joint_torque_offset",
                    "lower_torque_limit",
                    "upper_torque_limit",
                )
            }
        )
        state = FrankaState(
            **{
                key: arrays[f"measured_{key}"][index]
                for key in (
                    "position",
                    "rotation",
                    "linear_velocity",
                    "angular_velocity",
                    "normal_force",
                )
            },
            actuation=context,
        )
        target = FrankaTarget(
            **{
                key: arrays[f"target_{key}"][index]
                for key in (
                    "position",
                    "rotation",
                    "linear_velocity",
                    "angular_velocity",
                    "normal_force",
                )
            }
        )
        if index == 0:
            controller.reset(state)
        wrench = controller.compute(state, target, dt)
        torque = context.joint_torque(wrench)
        arrays["commanded_wrench"][index] = wrench
        arrays["commanded_torque"][index] = torque
        arrays["applied_torque"][index] = np.clip(
            torque, context.lower_torque_limit, context.upper_torque_limit
        )
    return arrays


@pytest.fixture
def trace():
    return _history()


def test_stateful_surface_controller_roundtrip_uses_full_history(tmp_path, trace):
    path = save_surface_trace(tmp_path / "history.npz", trace)
    result = replay_surface_trace(path)

    assert result.sample_count == 360
    assert result.matches
    assert result.max_wrench_error == 0
    assert result.max_torque_error == 0
    assert result.controller_kind == "surface_adaptive"
    assert result.controller_name == "surface_adaptive_hybrid"
    assert result.controller_supplied is False
    assert asdict(result)["matches"] is True
    assert np.ptp(trace["commanded_wrench"], axis=0).max() > 1.0
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["time"], trace["time"])


def test_custom_world_controller_resets_once_from_row_zero(tmp_path):
    parameters = {"max_normal_lead": 0.004}
    trace = _history(world=True, controller=FrankaSafeAdaptiveController(**parameters))
    path = save_surface_trace(tmp_path / "world.npz", trace)

    class RecordingController:
        name = "recording_world_adaptive"

        def __init__(self):
            self.inner = FrankaSafeAdaptiveController(**parameters)
            self.resets = []
            self.states = []

        def reset(self, state):
            self.resets.append(state)
            self.inner.reset(state)

        def compute(self, state, target, dt):
            self.states.append(state)
            return self.inner.compute(state, target, dt)

    controller = RecordingController()
    result = replay_surface_trace(path, controller)

    assert result.matches
    assert result.controller_supplied is True
    assert result.controller_name == "recording_world_adaptive"
    assert len(controller.resets) == 1
    assert len(controller.states) == 360
    reset = controller.resets[0]
    for key in ("position", "rotation", "linear_velocity", "angular_velocity", "normal_force"):
        np.testing.assert_array_equal(getattr(reset, key), trace[f"measured_{key}"][0])
        np.testing.assert_array_equal(getattr(reset, key), getattr(controller.states[0], key))
    for key in (
        "cartesian_jacobian",
        "joint_torque_offset",
        "lower_torque_limit",
        "upper_torque_limit",
    ):
        np.testing.assert_array_equal(getattr(reset.actuation, key), trace[key][0])


def test_default_world_controller_is_selected_from_metadata(tmp_path):
    trace = _history(world=True)
    path = save_surface_trace(tmp_path / "world.npz", trace)
    result = replay_surface_trace(path)

    assert result.matches
    assert result.controller_kind == "world_safe_adaptive"
    assert result.controller_name == "safe_adaptive_hybrid"


@pytest.mark.parametrize("field", ("commanded_wrench", "target_position"))
def test_changed_commands_or_target_are_replay_mismatch_not_invalid_archive(tmp_path, trace, field):
    trace[field][100, 1] += 0.01
    path = save_surface_trace(tmp_path / "changed.npz", trace)
    result = replay_surface_trace(path)

    assert result.matches is False
    assert result.max_wrench_error > 1e-10


def test_unclipped_torque_is_compared_even_when_applied_torque_is_valid(tmp_path, trace):
    trace["commanded_torque"][100, 0] = 100
    trace["applied_torque"][100, 0] = trace["upper_torque_limit"][100, 0]
    path = save_surface_trace(tmp_path / "request.npz", trace)
    result = replay_surface_trace(path)

    assert result.matches is False
    assert result.max_wrench_error == 0
    assert result.max_torque_error > 90


@pytest.mark.parametrize(
    "mutate, message",
    (
        (lambda arrays: arrays.pop("target_rotation"), "missing fields"),
        (lambda arrays: arrays.update(schema_version=np.array(2)), "schema_version"),
        (lambda arrays: arrays.update(dt=np.array(0.0)), "dt"),
        (lambda arrays: arrays.update(dt=np.array(np.inf)), "finite"),
        (lambda arrays: arrays.update(measured_position=np.zeros((1, 3))), "shape"),
        (lambda arrays: arrays["measured_linear_velocity"].__setitem__((0, 0), np.nan), "finite"),
        (
            lambda arrays: arrays.update(controller_frame_rotation=np.diag([1.0, 1.0, -1.0])),
            "orthonormal",
        ),
        (lambda arrays: arrays["target_rotation"].__setitem__((0, 0, 0), 2.0), "rotations"),
        (
            lambda arrays: arrays.update(lower_torque_limit=arrays["upper_torque_limit"].copy()),
            "lower torque",
        ),
        (lambda arrays: arrays["applied_torque"].__setitem__((0, 0), 99), "clipped"),
        (lambda arrays: arrays.update(controller_kind=np.array("unknown")), "controller_kind"),
        (lambda arrays: arrays.update(controller_kind=np.array("world_safe_adaptive")), "identity"),
        (lambda arrays: arrays.update(extra=np.array([object()], dtype=object)), "Object arrays"),
    ),
)
def test_replay_rejects_malformed_loaded_trace_before_controller_reset(
    tmp_path, trace, mutate, message
):
    mutate(trace)
    path = tmp_path / "invalid.npz"
    np.savez_compressed(path, **trace)

    class NeverCalled:
        def reset(self, _state):
            pytest.fail("validation must precede reset")

    with pytest.raises(ValueError, match=message):
        replay_surface_trace(path, NeverCalled())


def test_empty_trace_is_rejected_before_save(tmp_path, trace):
    trace["measured_normal_force"] = np.zeros(0)
    path = tmp_path / "empty.npz"
    with pytest.raises(ValueError, match="nonempty"):
        save_surface_trace(path, trace)
    assert not path.exists()


def test_save_adds_schema_without_mutating_input_and_refuses_existing_path(tmp_path, trace):
    trace.pop("schema_version")
    path = save_surface_trace(tmp_path / "new.npz", trace)
    assert "schema_version" not in trace
    contents = path.read_bytes()
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["schema_version"].item() == 1
    with pytest.raises(FileExistsError):
        save_surface_trace(path, trace)
    assert path.read_bytes() == contents
    assert not list(tmp_path.glob(".surface-trace-*"))


@pytest.mark.parametrize("parent_link", (False, True))
def test_save_refuses_symlink_paths(tmp_path, trace, parent_link):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target if parent_link else target / "uncreated.npz")
    path = link / "trace.npz" if parent_link else link
    with pytest.raises(ValueError, match="symlinks"):
        save_surface_trace(path, trace)
    assert not list(target.iterdir())


def test_publication_failure_cleans_staging_without_partial_output(tmp_path, trace, monkeypatch):
    def fail_link(*_args):
        raise OSError("publication failure")

    monkeypatch.setattr("compliant_control_lab.surface_replay.os.link", fail_link)
    with pytest.raises(OSError, match="publication failure"):
        save_surface_trace(tmp_path / "trace.npz", trace)
    assert not list(tmp_path.iterdir())
