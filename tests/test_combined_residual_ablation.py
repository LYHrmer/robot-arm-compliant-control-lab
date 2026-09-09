import hashlib
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from compliant_control_lab.online_compensation_experiment import GATES
from compliant_control_lab.surface_simulation import yaw_frame
from compliant_control_lab.tangential_compensation import TangentialCompensation
from tools import combined_residual_ablation as ablation
from tools import cross_surface_regression

ROOT = Path(__file__).parents[1]


def _originals():
    return {
        int(case.scenario.wall_yaw_deg): case
        for case in cross_surface_regression.make_cases()
        if case.name == ablation.PROFILE
    }


def _fake_controller(compensation, *, coefficient, corrected_force_n, contact_blend):
    normal = np.array([1.0, 0.0, 0.0])
    return SimpleNamespace(
        _base=SimpleNamespace(
            tangential=compensation,
            base=SimpleNamespace(base=SimpleNamespace(normal=normal)),
        ),
        frame=SimpleNamespace(vector_to_local=lambda vector: np.asarray(vector, dtype=float)),
        equivalent_tangential_coefficient=coefficient,
        corrected_force_n=corrected_force_n,
        contact_blend=contact_blend,
    )


def _observed_cycle(observer, controller, velocity, dt, *, executed_velocity=None):
    compensation = controller._base.tangential
    snapshot = observer.before_compute(SimpleNamespace(linear_velocity=velocity), dt)
    executed = velocity if executed_velocity is None else executed_velocity
    compensation._active = True
    compensation._equivalent_mu = controller.equivalent_tangential_coefficient
    compensation._online_force(
        SimpleNamespace(linear_velocity=executed),
        SimpleNamespace(linear_velocity=executed),
        controller._base.base.base.normal,
        controller.corrected_force_n,
        controller.contact_blend,
        dt,
    )
    return observer.after_compute(snapshot)


def test_twelve_deterministic_identities_preserve_the_original_cases():
    cases = ablation.ablation_cases()
    identities = [ablation.identity(variant, case) for variant, case in cases]
    assert identities == [
        {
            "surface_yaw_deg": yaw,
            "variant": variant,
            "case": ablation.PROFILE,
            "arm": "online",
            "simulation_seed": 11,
            "rotation_gain_scale": 1.0,
        }
        for yaw in (-15, 0, 15)
        for variant in ("combined", "no_yaw_error", "no_bias", "constant_initial_friction")
    ]
    assert identities == [
        ablation.identity(variant, case) for variant, case in ablation.ablation_cases()
    ]
    stems = [ablation.stem(variant, case) for variant, case in cases]
    assert len(set(stems)) == 12
    originals = _originals()
    for variant, case in cases:
        if variant == "combined":
            assert case == originals[int(case.scenario.wall_yaw_deg)]


def test_each_variant_replaces_only_its_declared_field():
    originals = _originals()
    expected = {
        "no_yaw_error": 0.0,
        "no_bias": ((0.0, (0.0,) * 6),),
        "constant_initial_friction": ((0.0, ablation.CONSTANT_FRICTION),),
    }
    for variant, case in ablation.ablation_cases():
        original = originals[int(case.scenario.wall_yaw_deg)]
        replaced = ablation.REPLACED_FIELD[variant]
        untouched = [name for name in (f.name for f in fields(case)) if name != replaced]
        assert [getattr(case, name) for name in untouched] == [
            getattr(original, name) for name in untouched
        ]
        if variant == "no_yaw_error":
            assert case.controller_yaw_error_deg == expected[variant]
            assert original.controller_yaw_error_deg == 3.0
        elif variant == "no_bias":
            assert case.wrench_bias_world.knots == expected[variant]
        elif variant == "constant_initial_friction":
            assert case.friction.knots == expected[variant]
            assert case.friction.at(12.0) == ablation.CONSTANT_FRICTION


def test_retained_world_bias_follows_the_true_surface_normal():
    for variant, case in ablation.ablation_cases():
        bias = np.asarray(case.wrench_bias_world.at(6.0), dtype=float)
        assert np.allclose(bias[3:], 0.0)
        if variant == "no_bias":
            assert np.allclose(bias[:3], 0.0)
            continue
        normal = yaw_frame(case.scenario.wall_yaw_deg).rotation[:, 0]
        assert bias[:3] == pytest.approx(0.75 * normal)
        assert np.allclose(case.wrench_bias_world.at(0.0), 0.0)


def test_timings_phases_seed_and_rotation_gain_are_fixed():
    cases = ablation.ablation_cases()
    protocol = ablation.protocol_document()
    reference = cases[0][1]
    for _, case in cases:
        assert case.config.duration == 12.0
        assert case.config.timestep == 0.002
        assert case.config.evaluation_start == 1.5
        assert case.config.seed == ablation.SEED == 11
        assert case.recovery_start_s == reference.recovery_start_s == 8.0
        assert case.phases == reference.phases
        assert case.task.trajectory == reference.task.trajectory
    assert (protocol["duration_s"], protocol["timestep_s"], protocol["evaluation_start_s"]) == (
        12.0, 0.002, 1.5,
    )
    assert protocol["seed"] == 11
    assert protocol["rotation_gain_scale"] == ablation.ROTATION_GAIN_SCALE == 1.0
    assert protocol["controller"]["rotation_gain_scale"] == 1.0
    assert protocol["arms_executed"] == ["online"]
    assert all(stem.endswith("__seed_11__online__s1") for stem in (
        ablation.stem(variant, case) for variant, case in cases
    ))


def test_only_absolute_checks_are_claimed_and_no_paired_comparator_gates():
    protocol = ablation.protocol_document()
    assert protocol["comparator_gates_evaluated"] is False
    assert protocol["absolute_checks_evaluated"] == [
        "contact_ratio", "raw_peak_force", "saturation",
    ]
    assert protocol["gates_not_evaluated"] == [
        "paired_force_rmse",
        "paired_orientation_rmse",
        "paired_friction_force_rmse",
        "paired_friction_orientation_rmse",
    ]
    passed = ablation.absolute_checks(
        {"contact_ratio_pct": 99.5, "peak_force_n": 20.0, "saturation_pct": 0.0}
    )
    assert passed == {
        "failed_absolute_checks": "",
        "absolute_checks_pass": "yes",
        "comparator_gates_evaluated": "no",
        "comparator_gate_status": "not_evaluated",
    }
    failed = ablation.absolute_checks({
        "contact_ratio_pct": GATES["minimum_contact_ratio_pct"] - 1.0,
        "peak_force_n": GATES["maximum_raw_peak_force_n"] + 1.0,
        "saturation_pct": 1.5,
    })
    assert failed["failed_absolute_checks"] == "contact_ratio;raw_peak_force;saturation"
    assert failed["absolute_checks_pass"] == "no"
    assert failed["comparator_gate_status"] == "not_evaluated"
    assert not [key for key in passed if key.startswith("paired")]


def test_source_identity_pins_the_runner_and_the_frozen_case_source():
    sources = ablation.source_identity()
    for relative in ("tools/combined_residual_ablation.py", "tools/cross_surface_regression.py"):
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert sources[relative] == digest
    assert "src/compliant_control_lab/online_compensation_experiment.py" in sources


def test_existing_output_is_rejected_before_any_simulation(tmp_path, monkeypatch):
    monkeypatch.setattr(ablation, "run_diagnostic_trial", lambda case: pytest.fail("simulated"))
    output = tmp_path / "report"
    output.mkdir()
    with pytest.raises(FileExistsError, match="output already exists"):
        ablation.generate(output)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["report"]
    assert list(output.iterdir()) == []


def test_symlinked_output_path_is_rejected_before_any_simulation(tmp_path, monkeypatch):
    monkeypatch.setattr(ablation, "run_diagnostic_trial", lambda case: pytest.fail("simulated"))
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(ValueError, match="must not contain symlinks"):
        ablation.generate(tmp_path / "link" / "report")
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="must not contain symlinks"):
        ablation.generate(dangling)
    assert list((tmp_path / "real").iterdir()) == []


def test_observer_reproduces_the_executed_capped_and_slew_limited_request():
    compensation = TangentialCompensation(mode="online")
    controller = _fake_controller(
        compensation, coefficient=0.5, corrected_force_n=30.0, contact_blend=1.0
    )
    observer = ablation.CompensationObserver(controller)
    dt = 0.002
    sample = _observed_cycle(observer, controller, np.array([0.0, 0.10, 0.0]), dt)
    assert sample == {
        "diagnostic_corrected_force_n": 30.0,
        "diagnostic_compensation_active": True,
        "diagnostic_amplitude_capped": True,
        "diagnostic_slew_limited": True,
    }
    assert np.linalg.norm(compensation.last_force) == pytest.approx(
        compensation.force_slew_rate * dt
    )
    assert observer.max_reconstruction_error_n < ablation.RECONSTRUCTION_TOLERANCE_N
    assert observer.mismatch_cycles == 0


def test_observer_reports_an_unbounded_request_without_cap_or_slew_limit():
    compensation = TangentialCompensation(mode="online")
    controller = _fake_controller(
        compensation, coefficient=0.05, corrected_force_n=10.0, contact_blend=1.0
    )
    velocity = np.array([0.0, 0.10, 0.0])
    speed = float(np.linalg.norm(velocity))
    direction = velocity / np.sqrt(speed**2 + compensation.velocity_scale**2)
    settled = 0.05 * 10.0 * direction
    compensation._last_force[:] = settled
    observer = ablation.CompensationObserver(controller)
    sample = _observed_cycle(observer, controller, velocity, 0.002)
    assert sample == {
        "diagnostic_corrected_force_n": 10.0,
        "diagnostic_compensation_active": True,
        "diagnostic_amplitude_capped": False,
        "diagnostic_slew_limited": False,
    }
    assert compensation.last_force == pytest.approx(settled)
    assert (observer.mismatch_cycles, observer.max_reconstruction_error_n) == (0, 0.0)


def test_observer_counts_cycles_whose_reconstruction_disagrees():
    compensation = TangentialCompensation(mode="online")
    controller = _fake_controller(
        compensation, coefficient=0.4, corrected_force_n=12.0, contact_blend=1.0
    )
    observer = ablation.CompensationObserver(controller)
    sample = _observed_cycle(
        observer,
        controller,
        np.array([0.0, 0.10, 0.0]),
        0.002,
        executed_velocity=np.array([0.0, -0.10, 0.0]),
    )
    assert sample["diagnostic_compensation_active"] is True
    assert observer.mismatch_cycles == 1
    assert observer.max_reconstruction_error_n > ablation.RECONSTRUCTION_TOLERANCE_N


def test_observer_requires_the_online_compensation_arm():
    for mode in ("friction", "integral"):
        controller = _fake_controller(
            TangentialCompensation(mode=mode),
            coefficient=0.45,
            corrected_force_n=12.0,
            contact_blend=1.0,
        )
        with pytest.raises(ValueError, match="requires the online compensation arm"):
            ablation.CompensationObserver(controller)
    controller = _fake_controller(
        None, coefficient=0.45, corrected_force_n=12.0, contact_blend=1.0
    )
    with pytest.raises(ValueError, match="requires the online compensation arm"):
        ablation.CompensationObserver(controller)
