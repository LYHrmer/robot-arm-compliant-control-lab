"""Check diagnostic interventions and reject unsafe output before simulation."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools/diagnostics/contact_oscillation_probes.py"
)
_spec = importlib.util.spec_from_file_location("contact_oscillation_probes", MODULE_PATH)
cop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cop)


def make_trial():
    trial = MagicMock()
    trial.trace = {
        "time": np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        "true_normal_force": np.array([0.0, 0.0, 1.0, 2.0, 3.0]),
        "true_tangent_force_n": np.array([0.0, 0.0, 0.5, 0.6, 0.7]),
        "true_contact_gap_m": np.array([1e-3, 1e-3, -1e-5, -2e-5, -3e-5]),
    }
    trial.metrics.return_value = {"peak_force_n": 3.0}
    return trial


def test_invalid_probe_rejects_before_simulation():
    with (
        patch.object(cop.sim, "run_surface_trial") as run,
        pytest.raises(ValueError, match="unknown contact probe"),
    ):
        cop.run_probe("not_a_probe")
    run.assert_not_called()


@pytest.mark.parametrize(
    "name, expected_filter, expected_model, hold",
    [
        ("baseline", 0.02, "legacy", False),
        ("hold_position", 0.02, "legacy", True),
        ("no_filter", 0.0, "legacy", False),
        ("filter_5ms", 0.005, "legacy", False),
        ("smooth", 0.02, "smooth", False),
    ],
)
def test_config_and_task_overrides(name, expected_filter, expected_model, hold):
    with patch.object(cop.sim, "run_surface_trial", return_value=make_trial()) as run:
        result = cop.run_probe(name)
    _, _, config, task = run.call_args.args
    assert config.force_filter_time_constant == pytest.approx(expected_filter)
    assert config.contact_model == expected_model
    assert isinstance(task, cop.HoldTask) == hold
    assert result["metrics"] == {"peak_force_n": 3.0}


@pytest.mark.parametrize(
    "name, key, expected",
    [
        ("zero_p", "force_kp", 0),
        ("zero_i", "force_ki", 0),
        ("normal_damping_60", "normal_damping", 60),
        ("friction_zero", "both_geom_sliding_friction", 0),
        ("friction_01", "both_geom_sliding_friction", 0.1),
        ("impedance_half", "both_geom_initial_impedance", 0.5),
        ("impratio_10", "impratio", 10),
    ],
)
def test_overrides_dict_is_one_factor(name, key, expected):
    with patch.object(cop.sim, "run_surface_trial", return_value=make_trial()):
        result = cop.run_probe(name)
    assert {k: v for k, v in result["overrides"].items() if v is not None} == {key: expected}


@pytest.mark.parametrize(
    "name, expected",
    [
        ("baseline", {}),
        ("zero_p", {"force_kp": 0}),
        ("zero_i", {"force_ki": 0}),
        ("normal_damping_60", {"normal_damping": 60}),
    ],
)
def test_controller_receives_only_the_stated_gain_change(name, expected):
    def fake_run(frame, *_rest):
        cop.sim.SurfaceAdaptiveController(frame)
        return make_trial()

    with (
        patch.object(cop, "FrankaHybridController") as hybrid,
        patch.object(cop, "FrankaAdaptiveHybridController"),
        patch.object(cop, "FrankaSafeAdaptiveController"),
        patch.object(cop, "SurfaceAdaptiveController"),
        patch.object(cop.sim, "run_surface_trial", side_effect=fake_run),
    ):
        cop.run_probe(name)
    hybrid.assert_called_once_with(force_transition_time=0.5, **expected)


@pytest.mark.parametrize("existing", [True, False])
def test_cli_rejects_existing_output_or_duplicate_probes(tmp_path, monkeypatch, existing):
    output = tmp_path / "out.json"
    if existing:
        output.write_text("{}")
    probes = ["baseline"] if existing else ["baseline", "baseline"]
    monkeypatch.setattr(sys, "argv", ["prog", "--output", str(output), "--probes", *probes])
    with (
        patch.object(cop, "run_probe", side_effect=AssertionError("must not run")),
        pytest.raises(ValueError, match="must be new" if existing else "unique"),
    ):
        cop.main()
    assert output.read_text() == "{}" if existing else not output.exists()
