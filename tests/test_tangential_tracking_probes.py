import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools/diagnostics/tangential_tracking_probes.py"
)
_spec = importlib.util.spec_from_file_location("tangential_tracking_probes", MODULE_PATH)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


@pytest.fixture
def fake_trials(monkeypatch):
    calls = []
    nominal_constructor = probe.FrankaHybridController
    nominal_parameters = []

    def record_nominal(**parameters):
        nominal_parameters.append(parameters)
        return nominal_constructor(**parameters)

    def fake_run(frame, scenario, config, task):
        model = SimpleNamespace(
            geom_friction=np.ones((2, 3)),
            geom=lambda name: SimpleNamespace(id=0 if name == "tool_tip" else 1),
        )
        probe.mujoco.mj_setConst(model, None)
        probe.sim.SurfaceAdaptiveController(frame)
        calls.append((scenario, config, task, model.geom_friction.copy(), nominal_parameters[-1]))
        return SimpleNamespace(
            trace={
                "time": np.array([1.5, 2.0, 2.5]),
                "true_normal_force": np.array([6.0, 12.0, 18.0]),
                "true_contact_gap_m": np.array([-0.001, -0.002, -0.003]),
            },
            metrics=lambda: {"contact_ratio_pct": 100.0, "tangent_rmse_mm": 1.0},
        )

    monkeypatch.setattr(probe.mujoco, "mj_setConst", lambda *_: None)
    monkeypatch.setattr(probe, "FrankaHybridController", record_nominal)
    monkeypatch.setattr(probe.sim, "run_surface_trial", fake_run)
    return calls


def test_invalid_probe_stops_before_simulation(fake_trials):
    with pytest.raises(ValueError, match="unknown"):
        probe.run_probe("unknown")
    assert not fake_trials


@pytest.mark.parametrize("name", probe.PROBES)
def test_fixed_probe_changes_only_declared_settings(name, fake_trials):
    row = probe.run_probe(name)
    scenario, config, task, friction, nominal = fake_trials[0]
    static = name.startswith("load_")
    assert config.duration == 3.0 and config.timestep == 0.002 and config.seed == 11
    assert config.contact_model == "smooth" and config.force_filter_time_constant == 0.02
    assert config.target_force == (float(name.split("_")[1]) if static else 12.0)
    assert config.evaluation_start == (2.0 if static else 1.5)
    assert scenario.wall_yaw_deg == 0.0 and scenario.wall_time_constant == 0.012
    assert scenario.tool_mass_kg == scenario.nominal_tool_mass_kg == 0.10
    assert (
        scenario.position_noise_std_m
        == scenario.force_noise_std_n
        == scenario.torque_noise_std_nm
        == 0.0
    )
    assert scenario.force_bias_sensor_n == scenario.torque_bias_sensor_nm == (0.0, 0.0, 0.0)
    np.testing.assert_array_equal(friction[:, 0], 0.0 if name == "friction_zero" else 1.0)
    np.testing.assert_array_equal(friction[:, 1:], np.ones((2, 2)))
    np.testing.assert_array_equal(
        nominal["tangential_stiffness"], np.array([0, 450, 450]) * (2 if name == "double_k" else 1)
    )
    assert nominal["force_transition_time"] == 0.5
    assert isinstance(task, probe.HoldTask) == static
    assert isinstance(task, probe.SlowTask) == (name == "slow")
    assert row["median_penetration_mm"] == pytest.approx(2.5 if static else 2.0)
    assert row["actual_normal_force_median_n"] == pytest.approx(15.0 if static else 12.0)
    assert row["max_penetration_mm"] == 3.0


def test_task_variants_preserve_approach_and_consistent_slow_velocity():
    position, rotation = np.array([0.36, 0.0, 0.45]), np.eye(3)
    base, slow, hold = probe.sim.SurfaceTask(), probe.SlowTask(), probe.HoldTask()
    for task in (slow, hold):
        np.testing.assert_array_equal(
            task.target_at(0.7, position, rotation, 12).position,
            base.target_at(0.7, position, rotation, 12).position,
        )
    expected = base.target_at(1.6, position, rotation, 12)
    actual = slow.target_at(2.0, position, rotation, 12)
    np.testing.assert_array_equal(actual.position, expected.position)
    np.testing.assert_array_equal(actual.linear_velocity, 0.5 * expected.linear_velocity)
    np.testing.assert_array_equal(
        hold.target_at(2.0, position, rotation, 12).position,
        base.target_at(1.2, position, rotation, 12).position,
    )


@pytest.mark.parametrize("kind", ["existing", "symlink", "symlink_parent"])
def test_bad_output_is_rejected_before_work(tmp_path, monkeypatch, kind):
    output = tmp_path / "result.json"
    retained = tmp_path / "retained"
    retained.mkdir()
    if kind == "existing":
        output.write_text("retain")
    else:
        output.symlink_to(retained)
        if kind == "symlink_parent":
            output /= "new.json"
    monkeypatch.setattr(sys, "argv", ["probe", "--output", str(output)])
    monkeypatch.setattr(probe, "run_probe", lambda *_: pytest.fail("must not simulate"))
    with pytest.raises(ValueError, match="new"):
        probe.main()


@pytest.mark.parametrize("drift", [False, True])
def test_aggregate_publication_records_provenance_and_rejects_drift(tmp_path, monkeypatch, drift):
    output = tmp_path / "report.json"
    calls = []
    monkeypatch.setattr(sys, "argv", ["probe", "--output", str(output)])
    monkeypatch.setattr(probe, "run_probe", lambda name: calls.append(name) or {"probe": name})
    monkeypatch.setattr(probe, "analyze_reference_trace", lambda: {"reference_sha256": "c" * 64})
    monkeypatch.setattr(probe, "_hash", lambda *_: "c" * 64)
    monkeypatch.setattr(
        probe, "_source_hashes", lambda: {"source": "changed" if drift and calls else "original"}
    )
    if drift:
        with pytest.raises(RuntimeError, match="changed"):
            probe.main()
        assert not output.exists()
    else:
        probe.main()
        payload = json.loads(output.read_text())
        assert len(payload["tracking_probes"]) == 4 and len(payload["static_force_depth"]) == 3
        assert payload["script_sha256"] == "c" * 64
        assert payload["engine_versions"]["mujoco"]
        assert "not full raw traces" in payload["scope"]
        assert "not real-material" in payload["force_depth_scope"]
    assert calls == list(probe.PROBES)


def test_existing_trace_directional_analysis_reconstructs_recorded_pd_command():
    row = probe.analyze_reference_trace()
    assert row["samples"] == 1500
    assert row["behind_target_pct"] == 100.0
    assert row["pd_reconstruction_max_error_n"] < 1e-10
    assert len(row["reference_sha256"]) == 64
