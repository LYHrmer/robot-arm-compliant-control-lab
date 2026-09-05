"""Tool-end MuJoCo force/torque sensing at the tool TCP, not the robot wrist.

The force and torque sensors share ``ee_site`` at the tool center of mass.
World conversion changes orientation only: the wrench remains about this TCP,
so no moment-arm translation is applied. Only nominal gravity is compensated;
tool inertia, mass calibration error, bias and noise remain in the measurement.
"""

from pathlib import Path

import mujoco
import numpy as np


def franka_surface_model_path() -> Path:
    """Return the sensor-equipped scene without changing the archived scene."""
    return Path(__file__).with_name("assets") / "franka_surface_scene.xml"


class ToolWrenchSensor:
    """Read [Fx, Fy, Fz, Tx, Ty, Tz] in world axes, about the tool TCP.

    Positive load follows the robot-on-environment convention. A static tool
    supported against gravity reads ``-mass * gravity`` before compensation;
    an external wrench applied to that tool contributes its opposite sign.
    Sensor-frame bias/noise are added before rotation, then nominal world-frame
    gravity is added to the force. This is not a contact-only ground-truth sensor.

    Call ``read_world`` immediately after ``mj_step2`` (or an explicit completed
    forward solve), before ``mj_step1`` refreshes the site rotation. The runner
    owns sample timestamps and caches the returned value for its next cycle.
    Each call is a new noise sample; inject an RNG for reproducible trials.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        nominal_mass_kg: float = 0.10,
        force_bias_sensor_n: tuple[float, float, float] = (0.0, 0.0, 0.0),
        torque_bias_sensor_nm: tuple[float, float, float] = (0.0, 0.0, 0.0),
        force_noise_std_n: float = 0.0,
        torque_noise_std_nm: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._model = model
        self.nominal_mass_kg = float(nominal_mass_kg)
        self._force_noise_std = float(force_noise_std_n)
        self._torque_noise_std = float(torque_noise_std_nm)
        for name, value in (
            ("nominal_mass_kg", self.nominal_mass_kg),
            ("force_noise_std_n", self._force_noise_std),
            ("torque_noise_std_nm", self._torque_noise_std),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

        self._bias = np.empty(6)
        for name, value, destination in (
            ("force_bias_sensor_n", force_bias_sensor_n, slice(0, 3)),
            ("torque_bias_sensor_nm", torque_bias_sensor_nm, slice(3, 6)),
        ):
            vector = np.asarray(value, dtype=float)
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} must be a finite three-vector")
            self._bias[destination] = vector
        self._rng = np.random.default_rng() if rng is None else rng

        self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        if self._site_id < 0:
            raise ValueError("tool wrench sensing requires ee_site")
        body_id = model.site_bodyid[self._site_id]
        if not np.allclose(
            model.site_pos[self._site_id], model.body_ipos[body_id], atol=1e-12, rtol=0
        ):
            raise ValueError("ee_site must coincide with the tool center of mass")
        self._sensor_slices = []
        for name, sensor_type in (
            ("tool_force", mujoco.mjtSensor.mjSENS_FORCE),
            ("tool_torque", mujoco.mjtSensor.mjSENS_TORQUE),
        ):
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if (
                sensor_id < 0
                or model.sensor_type[sensor_id] != sensor_type
                or model.sensor_dim[sensor_id] != 3
                or model.sensor_objtype[sensor_id] != mujoco.mjtObj.mjOBJ_SITE
                or model.sensor_objid[sensor_id] != self._site_id
            ):
                raise ValueError(f"{name} must be a three-axis {name[5:]} sensor at ee_site")
            start = model.sensor_adr[sensor_id]
            self._sensor_slices.append(slice(start, start + 3))

    def read_world(self, data: mujoco.MjData) -> np.ndarray:
        """Read solved sensordata without stepping or accessing contact-force truth."""
        local_wrench = np.concatenate([data.sensordata[item] for item in self._sensor_slices])
        local_wrench += self._bias
        local_wrench[:3] += self._rng.normal(0.0, self._force_noise_std, size=3)
        local_wrench[3:] += self._rng.normal(0.0, self._torque_noise_std, size=3)
        rotation = data.site_xmat[self._site_id].reshape(3, 3)
        world_wrench = np.concatenate(
            (
                rotation @ local_wrench[:3] + self.nominal_mass_kg * self._model.opt.gravity,
                rotation @ local_wrench[3:],
            )
        )
        if not np.all(np.isfinite(world_wrench)):
            raise ValueError("tool wrench measurement and site rotation must be finite")
        return world_wrench
