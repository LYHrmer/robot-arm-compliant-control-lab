"""Synthetic checks for the velocity-error time coefficient tradeoff figure."""

from copy import deepcopy

import numpy as np
import pytest

from tools import plot_velocity_time as figure_module

YAW = np.deg2rad(15.0)
NORMAL = np.array([np.cos(YAW), np.sin(YAW), 0.0])
TANGENT = np.array([-np.sin(YAW), np.cos(YAW), 0.0])


def _trace(lag_mm=0.0, velocity_mm_s=0.0, speed=0.05, count=1200):
    """Build a trace whose along-path lag and velocity error are known exactly."""
    time = np.arange(count, dtype=float) * 0.002
    target_velocity = np.tile(speed * TANGENT + 0.01 * NORMAL, (count, 1))
    target_position = time[:, None] * target_velocity
    return {
        "time": time,
        "target_position": target_position,
        "target_linear_velocity": target_velocity,
        "position": target_position - (lag_mm / 1_000.0) * TANGENT + 0.004 * NORMAL,
        "linear_velocity": target_velocity + (velocity_mm_s / 1_000.0) * TANGENT - 0.02 * NORMAL,
    }


def _archive(root, directory, suffix, gain, name, velocity_time_s=None):
    traces = root / directory / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    extra = {} if velocity_time_s is None else {"velocity_error_time_s": velocity_time_s}
    np.savez(
        traces / name,
        **_trace(lag_mm=2.0 if suffix == "s1" else 3.0, velocity_mm_s=-1.5 * gain),
        case_index=23,
        method="online",
        max_force_n=8.0,
        rotation_gain_scale=gain,
        **extra,
    )


def test_paired_series_signs_units_and_window():
    paired = figure_module.paired_series(
        _trace(lag_mm=2.0, velocity_mm_s=-3.0), _trace(lag_mm=0.5, velocity_mm_s=1.5)
    )
    assert set(paired) == {
        "time",
        "lag_005_mm",
        "lag_010_mm",
        "velocity_005_mm_s",
        "velocity_010_mm_s",
    }
    assert paired["time"][0] == pytest.approx(1.5)
    assert paired["time"].min() >= 1.5 and paired["time"].max() < 2.2
    assert len(paired["time"]) == 350
    assert paired["lag_005_mm"] == pytest.approx(2.0)
    assert paired["lag_010_mm"] == pytest.approx(0.5)
    assert paired["velocity_005_mm_s"] == pytest.approx(-3.0)
    assert paired["velocity_010_mm_s"] == pytest.approx(1.5)


def test_paired_series_rejects_mismatched_and_invalid_traces():
    before = _trace(lag_mm=1.0)
    with pytest.raises(ValueError, match="time grid"):
        figure_module.paired_series(before, _trace(lag_mm=1.0, count=1100))
    with pytest.raises(ValueError, match="target position and velocity"):
        figure_module.paired_series(before, _trace(lag_mm=1.0, speed=0.06))
    broken = _trace(lag_mm=1.0)
    broken["position"] = broken["position"][:, :2]
    with pytest.raises(ValueError, match="position"):
        figure_module.paired_series(broken, before)
    broken = _trace(lag_mm=1.0)
    broken["linear_velocity"][800, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        figure_module.paired_series(before, broken)
    with pytest.raises(ValueError, match="tangential speed"):
        figure_module.paired_series(before, _trace(lag_mm=1.0, speed=0.0))
    del before["target_position"]
    with pytest.raises(ValueError, match="target_position"):
        figure_module.paired_series(before, _trace(lag_mm=1.0))


def test_paired_series_does_not_mutate_inputs():
    before, after = _trace(lag_mm=2.0, velocity_mm_s=1.0), _trace(lag_mm=0.5)
    expected_before, expected_after = deepcopy(before), deepcopy(after)
    figure_module.paired_series(before, after)
    for trace, expected in ((before, expected_before), (after, expected_after)):
        assert set(trace) == set(expected)
        assert all(np.array_equal(trace[name], expected[name]) for name in expected)


def test_render_writes_two_by_two_svg_from_verified_archives(tmp_path, monkeypatch):
    new_directory = figure_module.NEW_REFERENCE[0]
    baseline_directory = figure_module.BASELINE_REFERENCE[0]
    for suffix, gain in figure_module.GAINS:
        _archive(tmp_path, baseline_directory, suffix, gain, f"case23__{suffix}__f8n.npz")
        _archive(tmp_path, new_directory, suffix, gain, f"case23__{suffix}__f8n__tv100ms.npz", 0.10)
    verified = []
    monkeypatch.setattr(
        figure_module, "verify_archive", lambda directory, pin: verified.append((directory, pin))
    )

    output = tmp_path / "figures" / "velocity_time.svg"
    assert figure_module.render(tmp_path, output) == output
    assert verified == [
        (tmp_path / new_directory, figure_module.NEW_REFERENCE[1]),
        (tmp_path / baseline_directory, figure_module.BASELINE_REFERENCE[1]),
    ]
    svg = output.read_text(encoding="utf-8")
    assert all(line == line.rstrip() for line in svg.splitlines())
    assert figure_module.NEW_REFERENCE[1] in svg and figure_module.BASELINE_REFERENCE[1] in svg
    assert "Along-path position lag [mm]" in svg and "true evaluator states" in svg

    paired = figure_module.paired_series(_trace(lag_mm=1.0), _trace(lag_mm=0.5))
    assert len(figure_module._draw({gain: paired for _, gain in figure_module.GAINS}).axes) == 4

    with pytest.raises(FileExistsError):
        figure_module.render(tmp_path, output)
    with pytest.raises(ValueError, match="svg"):
        figure_module.render(tmp_path, tmp_path / "figure.png")
    with pytest.raises(ValueError, match="inside a frozen archive"):
        figure_module.render(tmp_path, tmp_path / new_directory / "figure.svg")


@pytest.mark.parametrize("reference", [figure_module.NEW_REFERENCE, figure_module.BASELINE_REFERENCE])
def test_render_rejects_symlink_parent_into_frozen_archive(tmp_path, monkeypatch, reference):
    archive = tmp_path / reference[0]
    archive.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(archive, target_is_directory=True)
    monkeypatch.setattr(figure_module, "verify_archive", lambda *_: None)
    with pytest.raises(ValueError, match="inside a frozen archive"):
        figure_module.render(tmp_path, alias / "figure.svg")
    assert not (archive / "figure.svg").exists()


def test_render_rejects_dangling_output_symlink(tmp_path):
    target = tmp_path / "missing.svg"
    output = tmp_path / "figure.svg"
    output.symlink_to(target)
    with pytest.raises(FileExistsError):
        figure_module.render(tmp_path, output)
    assert output.is_symlink()
    assert not target.exists()
