"""Physical sign, TCP-frame and calibration contracts for tool-end F/T sensing."""

import mujoco
import numpy as np
import pytest

from compliant_control_lab.franka_simulation import franka_model_path
from compliant_control_lab.surface_sensing import ToolWrenchSensor, franka_surface_model_path


def _fixed_tool_model(quaternion="1 0 0 0", mass=0.10):
    return mujoco.MjModel.from_xml_string(
        f"""<mujoco>
          <option gravity="0 0 -9.81"/>
          <worldbody>
            <body name="parent">
              <body name="tool" quat="{quaternion}">
                <inertial mass="{mass}" pos="0 0 0"
                  diaginertia=".00003 .00003 .00003"/>
                <site name="ee_site"/>
              </body>
            </body>
          </worldbody>
          <sensor>
            <force name="tool_force" site="ee_site"/>
            <torque name="tool_torque" site="ee_site"/>
          </sensor>
        </mujoco>"""
    )


def test_sensor_scene_adds_six_channels_without_changing_legacy_model():
    legacy = mujoco.MjModel.from_xml_path(str(franka_model_path()))
    model = mujoco.MjModel.from_xml_path(str(franka_surface_model_path()))
    assert legacy.nsensor == 0
    assert model.nsensor == 2
    assert model.nsensordata == 6
    assert model.sensor("tool_force").dim[0] == 3
    assert model.sensor("tool_torque").dim[0] == 3
    np.testing.assert_array_equal(model.body_mass, legacy.body_mass)
    np.testing.assert_array_equal(model.site_pos, legacy.site_pos)
    ToolWrenchSensor(model)


def test_static_suspended_franka_tool_reads_weight_then_compensates_to_zero():
    model = mujoco.MjModel.from_xml_path(str(franka_surface_model_path()))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    # Support the robot so this is a static tool test, not a freely falling arm.
    data.qfrc_applied[:] = data.qfrc_bias
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qacc, 0, atol=1e-12, rtol=0)
    tool_geom_id = model.geom("tool_tip").id
    assert all(
        tool_geom_id not in (contact.geom1, contact.geom2)
        for contact in data.contact[: data.ncon]
    )

    rotation = data.site_xmat[model.site("ee_site").id].reshape(3, 3)
    np.testing.assert_allclose(
        rotation @ data.sensor("tool_force").data, [0, 0, 0.981], atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(ToolWrenchSensor(model).read_world(data), 0, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "quaternion",
    ["1 0 0 0", "0.7071067812 0 0 0.7071067812", "0.9238795325 0 0.3826834324 0"],
)
def test_external_force_and_torque_have_robot_on_environment_sign_under_rotation(quaternion):
    model = _fixed_tool_model(quaternion)
    data = mujoco.MjData(model)
    external_wrench = np.array([3.0, -4.0, 5.0, 0.6, -0.7, 0.8])
    data.xfrc_applied[model.body("tool").id] = external_wrench
    mujoco.mj_forward(model, data)
    rotation = data.site_xmat[model.site("ee_site").id].reshape(3, 3)
    expected_force_world = -external_wrench[:3] - 0.10 * model.opt.gravity
    np.testing.assert_allclose(
        data.sensor("tool_force").data, rotation.T @ expected_force_world, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        data.sensor("tool_torque").data,
        rotation.T @ -external_wrench[3:],
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        ToolWrenchSensor(model).read_world(data), -external_wrench, atol=1e-12, rtol=0
    )


def test_tool_mass_calibration_error_remains_in_measurement():
    model = _fixed_tool_model(mass=0.12)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    measured = ToolWrenchSensor(model, nominal_mass_kg=0.10).read_world(data)
    np.testing.assert_allclose(measured, [0, 0, 0.1962, 0, 0, 0], atol=1e-12, rtol=0)


def test_bias_is_in_sensor_frame_and_reading_does_not_mutate_sensordata():
    model = _fixed_tool_model("0.7071067812 0 0 0.7071067812")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    original = data.sensordata.copy()
    sensor = ToolWrenchSensor(
        model,
        force_bias_sensor_n=(1.0, 2.0, 3.0),
        torque_bias_sensor_nm=(0.1, 0.2, 0.3),
    )
    np.testing.assert_allclose(
        sensor.read_world(data), [-2, 1, 3, -0.2, 0.1, 0.3], atol=1e-12, rtol=0
    )
    np.testing.assert_array_equal(data.sensordata, original)


def test_seeded_force_and_torque_noise_are_reproducible():
    model = _fixed_tool_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def samples(seed):
        sensor = ToolWrenchSensor(
            model, force_noise_std_n=0.2, torque_noise_std_nm=0.01, rng=np.random.default_rng(seed)
        )
        return np.array([sensor.read_world(data) for _ in range(4)])

    first = samples(13)
    np.testing.assert_array_equal(first, samples(13))
    assert not np.array_equal(first, samples(14))
    assert np.all(np.isfinite(first))
    assert np.any(first[:, :3] != 0)
    assert np.any(first[:, 3:] != 0)


def test_accelerating_tool_retains_inertial_load_without_contact():
    model = mujoco.MjModel.from_xml_string(
        """<mujoco>
          <option gravity="0 0 -9.81"/>
          <worldbody>
            <body name="parent">
              <joint type="slide" axis="1 0 0"/>
              <inertial mass="1" pos="0 0 0" diaginertia=".01 .01 .01"/>
              <body name="tool">
                <inertial mass=".1" pos="0 0 0" diaginertia=".00003 .00003 .00003"/>
                <site name="ee_site"/>
              </body>
            </body>
          </worldbody>
          <sensor>
            <force name="tool_force" site="ee_site"/>
            <torque name="tool_torque" site="ee_site"/>
          </sensor>
        </mujoco>"""
    )
    data = mujoco.MjData(model)
    data.qfrc_applied[0] = 2.2  # 1.1 kg total mass, 2 m/s² acceleration.
    mujoco.mj_forward(model, data)
    assert data.ncon == 0
    np.testing.assert_allclose(
        ToolWrenchSensor(model).read_world(data), [0.2, 0, 0, 0, 0, 0], atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    "options",
    [
        {"nominal_mass_kg": -0.1},
        {"nominal_mass_kg": np.nan},
        {"nominal_mass_kg": np.inf},
        {"force_noise_std_n": -0.1},
        {"force_noise_std_n": np.nan},
        {"torque_noise_std_nm": -0.1},
        {"torque_noise_std_nm": np.inf},
        {"force_bias_sensor_n": (1, 2)},
        {"force_bias_sensor_n": (0, np.nan, 0)},
        {"torque_bias_sensor_nm": (0, 0, np.inf)},
        {"torque_bias_sensor_nm": ((0, 0, 0),)},
    ],
)
def test_invalid_calibration_and_noise_parameters_are_rejected(options):
    with pytest.raises(ValueError):
        ToolWrenchSensor(_fixed_tool_model(), **options)


def test_sensor_without_calibrated_com_origin_is_rejected():
    model = _fixed_tool_model()
    model.site_pos[model.site("ee_site").id] = [0.01, 0, 0]
    with pytest.raises(ValueError, match="center of mass"):
        ToolWrenchSensor(model)


def test_missing_sensor_channels_are_rejected():
    legacy = mujoco.MjModel.from_xml_path(str(franka_model_path()))
    with pytest.raises(ValueError, match="tool_force"):
        ToolWrenchSensor(legacy)


def test_nonfinite_measurements_are_rejected():
    model = _fixed_tool_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    data.sensordata[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ToolWrenchSensor(model).read_world(data)
