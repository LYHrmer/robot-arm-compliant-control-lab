"""Real surface-loop checks for optional classical tangential compensation."""

import pytest

from compliant_control_lab.surface_simulation import (
    SurfaceScenario,
    SurfaceSimulationConfig,
    run_surface_trial,
    yaw_frame,
)


def noiseless_trial(kind):
    return run_surface_trial(
        yaw_frame(0),
        SurfaceScenario(
            position_noise_std_m=0,
            force_noise_std_n=0,
            torque_noise_std_nm=0,
            force_bias_sensor_n=(0, 0, 0),
            wall_time_constant=0.012,
        ),
        SurfaceSimulationConfig(duration=3, contact_model="smooth"),
        controller_kind=kind,
    )


def test_friction_compensation_reduces_real_tracking_offset():
    baseline = noiseless_trial("surface_adaptive").metrics()
    corrected = noiseless_trial("surface_friction").metrics()
    assert baseline["tangent_rmse_mm"] > 10
    assert corrected["tangent_rmse_mm"] < 0.5 * baseline["tangent_rmse_mm"]
    assert corrected["contact_ratio_pct"] >= 99
    assert corrected["force_rmse_n"] <= baseline["force_rmse_n"] + 0.2
    assert corrected["peak_force_n"] <= 35
    assert corrected["saturation_pct"] == 0


def test_original_smooth_case_remains_numerically_unchanged():
    metrics = noiseless_trial("surface_adaptive").metrics()
    assert metrics["tangent_rmse_mm"] == pytest.approx(11.651333023785067, abs=1e-10)
    assert metrics["force_rmse_n"] == pytest.approx(0.224295174736721, abs=1e-10)
