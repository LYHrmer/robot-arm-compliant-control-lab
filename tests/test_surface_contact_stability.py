from dataclasses import replace

import mujoco
import numpy as np
import pytest

from compliant_control_lab import surface_simulation as sim


def _noise_free_scenario() -> sim.SurfaceScenario:
    return sim.SurfaceScenario(
        name="contact_stability",
        wall_yaw_deg=0.0,
        wall_time_constant=0.012,
        tool_mass_kg=0.10,
        nominal_tool_mass_kg=0.10,
        position_noise_std_m=0.0,
        force_noise_std_n=0.0,
        torque_noise_std_nm=0.0,
        force_bias_sensor_n=(0.0, 0.0, 0.0),
        torque_bias_sensor_nm=(0.0, 0.0, 0.0),
    )


def _trial(contact_model: str, yaw: float = 0.0):
    return sim.run_surface_trial(
        sim.yaw_frame(yaw),
        scenario=replace(_noise_free_scenario(), wall_yaw_deg=yaw),
        config=sim.SurfaceSimulationConfig(
            duration=2.0, evaluation_start=1.5, seed=11, contact_model=contact_model
        ),
        task=sim.SurfaceTask(yaw_deg=yaw),
    )


def test_smooth_contact_remains_continuous_in_noise_free_case():
    metrics = _trial("smooth").metrics()

    assert metrics["contact_ratio_pct"] >= 99.0, metrics
    assert metrics["force_rmse_n"] <= 2.0, metrics
    assert metrics["peak_force_n"] <= 35.0, metrics
    assert metrics["saturation_pct"] == 0.0, metrics


def test_legacy_contact_model_retains_the_reproduced_separation_symptom():
    assert sim.SurfaceSimulationConfig().contact_model == "legacy"
    metrics = _trial("legacy").metrics()

    assert metrics["contact_ratio_pct"] < 90.0, metrics
    assert metrics["force_rmse_n"] > 5.0, metrics
    assert metrics["saturation_pct"] == 0.0, metrics


def test_unknown_contact_model_is_rejected():
    with pytest.raises(ValueError, match="contact_model"):
        sim.SurfaceSimulationConfig(contact_model="smoth")


def _capture_actual_contact(monkeypatch, contact_model: str, yaw: float = 0.0) -> dict:
    observation = {"contact": None, "sliding_force_ratios": [], "contact_samples": []}
    step2 = mujoco.mj_step2

    def capture(model, data):
        tool, wall = model.geom("tool_tip").id, model.geom("contact_wall").id
        if "model" not in observation:
            observation["geom_ids"] = (tool, wall)
            observation["model"] = {
                name: getattr(model, name).copy()
                for name in (
                    "geom_solimp",
                    "geom_solref",
                    "geom_friction",
                    "geom_condim",
                    "geom_margin",
                    "body_mass",
                    "body_inertia",
                )
            }
            observation["options"] = {
                name: getattr(model.opt, name)
                for name in ("solver", "integrator", "timestep", "iterations", "tolerance", "cone")
            }
        sample_time = data.time
        step2(model, data)
        for index in range(data.ncon):
            contact = data.contact[index]
            if set(contact.geom) != {tool, wall}:
                continue
            if observation["contact"] is None:
                observation["contact"] = {
                    "friction": contact.friction.copy(),
                    "solref": contact.solref.copy(),
                    "solimp": contact.solimp.copy(),
                    "dim": contact.dim,
                    "includemargin": contact.includemargin,
                }
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            observation["contact_samples"].append(
                (sample_time, float(contact.dist), float(np.linalg.norm(force[1:3])))
            )
            if sample_time >= 1.5 and force[0] > 0.5:
                observation["sliding_force_ratios"].append(
                    float(np.linalg.norm(force[1:3]) / force[0])
                )

    with monkeypatch.context() as context:
        context.setattr(mujoco, "mj_step2", capture)
        observation["result"] = _trial(contact_model, yaw)
    assert observation["contact"] is not None
    return observation


@pytest.mark.parametrize("yaw", [-15.0, 0.0, 15.0])
def test_smooth_contact_only_changes_impedance_at_zero_penetration(monkeypatch, yaw):
    legacy = _capture_actual_contact(monkeypatch, "legacy", yaw)
    smooth = _capture_actual_contact(monkeypatch, "smooth", yaw)
    legacy_contact, smooth_contact = legacy["contact"], smooth["contact"]

    assert legacy_contact["solimp"][0] == pytest.approx(0.925)
    assert smooth_contact["solimp"][0] == 0.0
    np.testing.assert_array_equal(smooth_contact["solimp"][1:], legacy_contact["solimp"][1:])
    assert smooth_contact["solimp"][1] == pytest.approx(0.97)
    assert smooth_contact["solimp"][2] == pytest.approx(0.0015)
    for contact in (legacy_contact, smooth_contact):
        np.testing.assert_allclose(contact["friction"][:2], [0.45, 0.45], atol=1e-14)
        assert contact["solref"][0] == pytest.approx(0.016)
        assert contact["dim"] == 3
        assert contact["includemargin"] == 0.0
    for name in ("friction", "solref", "dim", "includemargin"):
        np.testing.assert_array_equal(smooth_contact[name], legacy_contact[name])
    for name in legacy["model"]:
        expected = legacy["model"][name].copy()
        if name == "geom_solimp":
            expected[list(legacy["geom_ids"]), 0] = 0.0
        np.testing.assert_array_equal(smooth["model"][name], expected)
    assert smooth["options"] == legacy["options"]
    ratios = smooth["sliding_force_ratios"]
    assert len(ratios) > 100
    assert np.median(ratios) == pytest.approx(0.45, abs=0.01)
    for mode, observation in (("legacy", legacy), ("smooth", smooth)):
        result = observation["result"]
        trace = result.trace
        assert trace["contact_model"].item() == mode
        for sample_time, distance, tangent_force in observation["contact_samples"]:
            row = round(sample_time / result.config.timestep)
            assert trace["true_contact_gap_m"][row] == pytest.approx(distance, abs=1e-12)
            assert trace["true_tangent_force_n"][row] == pytest.approx(tangent_force, abs=1e-12)


@pytest.mark.parametrize("case_index, yaw", [(0, -15.0), (16, 15.0)])
def test_smooth_contact_stays_continuous_in_noisy_development_cases(case_index, yaw):
    scenario = sim.SurfaceScenario(
        name=f"surface_dev_{case_index:02d}",
        wall_yaw_deg=yaw,
        wall_time_constant=0.005,
        tool_mass_kg=0.10,
    )
    result = sim.run_surface_trial(
        sim.yaw_frame(yaw),
        scenario=scenario,
        config=sim.SurfaceSimulationConfig(duration=3.0, seed=11, contact_model="smooth"),
        task=sim.SurfaceTask(yaw_deg=yaw),
    )
    metrics = result.metrics()
    assert metrics["contact_ratio_pct"] >= 99.0, metrics
    assert metrics["force_rmse_n"] <= 2.0, metrics
    assert metrics["peak_force_n"] <= 35.0, metrics
    assert metrics["saturation_pct"] == 0.0, metrics


def test_full_representative_case16_retains_contact_with_sensor_noise():
    scenario = sim.SurfaceScenario(
        name="surface_dev_16",
        wall_yaw_deg=15.0,
        wall_time_constant=0.005,
        tool_mass_kg=0.10,
    )
    result = sim.run_surface_trial(
        sim.yaw_frame(15.0),
        scenario=scenario,
        config=replace(sim.SurfaceSimulationConfig(seed=11), contact_model="smooth"),
        task=sim.SurfaceTask(yaw_deg=15.0),
    )
    metrics = result.metrics()
    assert metrics["contact_ratio_pct"] >= 99.0, metrics
    assert metrics["force_rmse_n"] <= 2.0, metrics
    assert metrics["peak_force_n"] <= 35.0, metrics
    assert metrics["saturation_pct"] == 0.0, metrics
