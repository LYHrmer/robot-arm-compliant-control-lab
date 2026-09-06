"""Keep the diagnostic's sampling intervention distinct from changing the controller."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from compliant_control_lab import surface_simulation as sim
from compliant_control_lab.surface_replay import replay_surface_trace, save_surface_trace

spec = importlib.util.spec_from_file_location(
    "contact_step_refinement",
    Path(__file__).parents[1] / "tools/diagnostics/contact_step_refinement.py",
)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


@pytest.mark.parametrize("profile", ["legacy", "smooth"])
def test_one_substep_is_exact_ordinary_trial(profile):
    config = sim.SurfaceSimulationConfig(duration=0.65, evaluation_start=0, contact_model=profile)
    refined, report, samples = diagnostic.run_refinement(1, config=config)
    ordinary = sim.run_surface_trial(sim.yaw_frame(15), refined.scenario, config, refined.task)
    assert refined.metrics() == ordinary.metrics()
    for name, value in ordinary.trace.items():
        np.testing.assert_array_equal(refined.trace[name], value, err_msg=name)
    assert report["control_grid"] == report["all_physics"]
    assert len(samples) == 325
    assert np.max(samples[:, 1]) > 0.5  # Exercise solved contact, not just free flight.


@pytest.mark.parametrize("substeps", [4, 8])
def test_fixed_control_clock_first_phase_noise_cadence_and_replay(substeps, tmp_path):
    config = sim.SurfaceSimulationConfig(duration=0.65, contact_model="smooth")
    trial, report, samples = diagnostic.run_refinement(substeps, config=config)
    trace = trial.trace
    assert len(trace["time"]) == 325
    assert len(samples) == 325 * substeps
    assert report["sensor_reads_including_reset"] == 326
    assert report["control_period_s"] == float(trace["dt"]) == 0.002
    assert report["physics_period_s"] == 0.002 / substeps
    assert np.max(samples[:, 1]) > 0.5
    assert report["resolved_contact_pair"]["solimp"][0] == 0
    np.testing.assert_allclose(samples[:, 0], np.arange(len(samples)) * 0.002 / substeps, atol=1e-15)
    np.testing.assert_allclose(samples[::substeps, 0], trace["raw_wrench_sample_time"], atol=1e-15)
    np.testing.assert_array_equal(samples[::substeps, 1], trace["true_normal_force"])
    np.testing.assert_array_equal(samples[::substeps, 2], trace["true_tangent_force_n"])
    np.testing.assert_array_equal(trace["feedback_raw_wrench_world"][1:], trace["raw_wrench_world"][:-1])
    np.testing.assert_array_equal(trace["measured_wrench_sample_time"][1:], trace["time"][:-1])
    save_surface_trace(tmp_path / "refined.npz", trace)
    assert replay_surface_trace(tmp_path / "refined.npz").matches


@pytest.mark.parametrize("substeps", [0, -1, True, 1.5, "4"])
def test_reject_invalid_substeps(substeps):
    with pytest.raises(ValueError, match="positive integer"):
        diagnostic.run_refinement(substeps)


def test_reject_control_period_change():
    with pytest.raises(ValueError, match="500 Hz"):
        diagnostic.run_refinement(4, config=sim.SurfaceSimulationConfig(timestep=0.001))


def test_all_physics_metrics_do_not_hide_between_control_peaks():
    samples = np.array([[0, 1, 0, -0.001, 0, 12], [0.0005, 40, 18, -0.001, 0, 12],
                        [0.001, 0, 0, 0.001, 0, 12], [0.0015, 2, 0.9, -0.001, 0, 12]])
    all_grid = diagnostic._grid_metrics(samples, 0, 0.0005)
    control_grid = diagnostic._grid_metrics(samples[::4], 0, 0.002)
    assert all_grid["peak_force_n"] == 40
    assert all_grid["contact_ratio_pct"] == 75
    assert all_grid["geometry_separation_pct"] == 25
    assert all_grid["seconds_over_35_n"] == 0.0005
    assert control_grid["peak_force_n"] == 1
    assert control_grid["contact_ratio_pct"] == 100


def test_output_rejects_existing_files_and_symlink_components(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("must validate output before running simulations")

    monkeypatch.setattr(diagnostic, "run_refinement", forbidden)
    existing = tmp_path / "existing.json"
    existing.write_text("preserve", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "missing")
    for target in (existing, alias / "new.json", dangling):
        with pytest.raises(ValueError, match="new.*symlink"):
            diagnostic.main(["--output", str(target)])
    assert existing.read_text(encoding="utf-8") == "preserve"


def test_json_includes_independent_script_and_source_provenance(tmp_path, monkeypatch):
    calls = []

    def fake_run(count, *, config):
        calls.append((count, config))
        return None, {"substeps": count, "control_grid": {}, "all_physics": {}}, None

    monkeypatch.setattr(diagnostic, "run_refinement", fake_run)
    destination = tmp_path / "new.json"
    diagnostic.main(["--output", str(destination)])
    result = json.loads(destination.read_text(encoding="utf-8"))
    assert [count for count, _ in calls] == [1, 4, 8]
    assert all(config.duration == 4.5 and config.timestep == 0.002 for _, config in calls)
    assert result["script_sha256"] == hashlib.sha256(Path(diagnostic.__file__).read_bytes()).hexdigest()
    assert result["source_and_assets_sha256"] == diagnostic._source_hashes()
    assert result["engine_versions"]["mujoco"] == diagnostic.mujoco.__version__
    assert result["sampling"]["control_and_filter_hz"] == 500
