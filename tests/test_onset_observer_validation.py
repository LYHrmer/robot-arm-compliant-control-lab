import numpy as np
import pytest

from compliant_control_lab.franka_control import FrankaState, FrankaTarget
from tools.onset_observer import ObservedCompensation
from tools.onset_observer_validation import validate_observation


def _parameters(**updates):
    values = {
        "mode": "online",
        "integral_gain": 800.0,
        "nominal_mu": 0.45,
        "max_force": 6.0,
        "velocity_scale": 0.005,
        "adaptation_gain": 800.0,
        "velocity_error_time": 0.05,
        "force_regularizer": 2.0,
        "max_equivalent_mu": 0.9,
        "min_update_speed": 0.005,
        "force_slew_rate": 10_000.0,
        "coefficient_rate_limit": 0.3,
        "motion_confirm_time": 0.004,
    }
    values.update(updates)
    return values


def _trace(parameters, *, contacts=None, target_signs=None, advance=None, allow=None):
    dt = 0.002
    contacts = list(contacts or (True,) * 6)
    count = len(contacts)
    target_signs = list(target_signs or (1.0,) * count)
    advance = list(advance or (True,) * count)
    allow = list(allow or (False, True, True, False, True, True)[:count])
    angle = np.deg2rad(20.0)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    normal = np.array([1.0, 0.0, 0.0])
    measured_position_local = np.zeros((count, 3))
    raw_position_error = np.tile([0.001, 0.02, 0.0], (count, 1))
    raw_measured_velocity = np.tile([0.011, 0.02, 0.0], (count, 1))
    raw_target_velocity = np.array([[0.027, 0.03 * sign, 0.0] for sign in target_signs])
    position_error = raw_position_error.copy()
    position_error[:, 0] = 0.0
    measured_velocity = raw_measured_velocity.copy()
    measured_velocity[:, 0] = 0.0
    target_velocity = raw_target_velocity.copy()
    target_velocity[:, 0] = 0.0
    velocity_error = target_velocity - measured_velocity
    force = np.array([10.0 if contact else 0.0 for contact in contacts])
    target_force = np.full(count, 12.0)
    blend = np.array([1.0 if contact else 0.0 for contact in contacts])
    projection = np.array([1.0 if allowed else 0.8 for allowed in allow])

    observed = {name: [] for name in (
        "normal_local", "position_error_local_m", "velocity_error_local_m_s",
        "target_velocity_local_m_s", "measured_velocity_local_m_s", "direction_local",
        "previous_direction_local", "previous_force_local_n", "requested_force_local_n",
        "corrected_force_n", "contact_blend", "target_normal_force_n", "in_contact",
        "coefficient_before_force", "coefficient_after_force", "coefficient_after_advance",
        "motion_elapsed_before_s", "motion_elapsed_after_s", "active", "update_ready",
        "amplitude_capped", "slew_limited", "position_drive_m", "velocity_drive_m",
        "drive_m", "candidate_increment", "limited_increment", "advance_called",
        "allow_integration", "dt_s",
    )}
    coefficient = parameters["nominal_mu"]
    previous_force = np.zeros(3)
    previous_direction = np.zeros(3)
    motion = 0.0
    for i in range(count):
        active = bool(contacts[i] and force[i] > 1.0 and target_force[i] > 0.0)
        before = coefficient
        after_force = before if active else parameters["nominal_mu"]
        tangent_velocity = target_velocity[i].copy()
        tangent_velocity[0] = 0.0
        speed = np.linalg.norm(tangent_velocity)
        direction = (
            tangent_velocity / np.sqrt(speed**2 + parameters["velocity_scale"]**2)
            if active else np.zeros(3)
        )
        previous_tangent = previous_force.copy() if active else np.zeros(3)
        desired = blend[i] * min(after_force * force[i], parameters["max_force"]) * direction
        distance = np.linalg.norm(desired - previous_tangent)
        slew = bool(active and distance > parameters["force_slew_rate"] * dt)
        request = (
            previous_tangent
            + (desired - previous_tangent)
            * min(1.0, parameters["force_slew_rate"] * dt / max(distance, 1e-12))
            if active else np.zeros(3)
        )
        reversal = direction @ previous_direction < 0.0
        eligible = bool(
            active and blend[i] >= 0.99 and speed >= parameters["min_update_speed"]
            and measured_velocity[i] @ direction >= parameters["min_update_speed"] / 2
            and not reversal and not slew
        )
        motion_after = motion + dt if eligible else 0.0
        ready = motion_after >= parameters["motion_confirm_time"]
        position_drive = float(direction @ position_error[i])
        velocity_drive = float(parameters["velocity_error_time"] * direction @ velocity_error[i])
        drive = position_drive + velocity_drive
        candidate = (
            dt * parameters["adaptation_gain"] * force[i]
            / (force[i] ** 2 + parameters["force_regularizer"] ** 2) * drive
            if active else 0.0
        )
        limited = float(np.clip(
            candidate,
            -parameters["coefficient_rate_limit"] * dt,
            parameters["coefficient_rate_limit"] * dt,
        ))
        after_advance = after_force
        if (advance[i] and active and allow[i] and ready
                and not (limited > 0 and after_force * force[i] >= parameters["max_force"])):
            after_advance = float(np.clip(
                after_force + limited, 0.0, parameters["max_equivalent_mu"]
            ))
        values = {
            "normal_local": normal,
            "position_error_local_m": position_error[i],
            "velocity_error_local_m_s": velocity_error[i],
            "target_velocity_local_m_s": target_velocity[i],
            "measured_velocity_local_m_s": measured_velocity[i],
            "direction_local": direction,
            "previous_direction_local": previous_direction,
            "previous_force_local_n": previous_force,
            "requested_force_local_n": request,
            "corrected_force_n": force[i],
            "contact_blend": blend[i],
            "target_normal_force_n": target_force[i],
            "in_contact": contacts[i],
            "coefficient_before_force": before,
            "coefficient_after_force": after_force,
            "coefficient_after_advance": after_advance,
            "motion_elapsed_before_s": motion,
            "motion_elapsed_after_s": motion_after,
            "active": active,
            "update_ready": ready,
            "amplitude_capped": active and after_force * force[i] >= parameters["max_force"],
            "slew_limited": slew,
            "position_drive_m": position_drive,
            "velocity_drive_m": velocity_drive,
            "drive_m": drive,
            "candidate_increment": candidate,
            "limited_increment": limited,
            "advance_called": advance[i],
            "allow_integration": allow[i],
            "dt_s": dt,
        }
        for name, value in values.items():
            observed[name].append(np.asarray(value).copy())
        coefficient = after_advance
        previous_force = request
        motion = motion_after
        if not active:
            previous_direction = np.zeros(3)
        elif speed >= parameters["min_update_speed"]:
            previous_direction = direction

    measured_position = measured_position_local @ rotation.T
    target_position = (measured_position_local + raw_position_error) @ rotation.T
    measured_velocity_world = raw_measured_velocity @ rotation.T
    target_velocity_world = raw_target_velocity @ rotation.T
    measured_wrench = np.zeros((count, 6))
    measured_wrench[:, :3] = force[:, None] * rotation[:, 0]
    trace = {
        "time": np.arange(count) * dt,
        "dt": np.array(dt),
        "controller_frame_rotation": rotation,
        "measured_position": measured_position,
        "measured_linear_velocity": measured_velocity_world,
        "target_position": target_position,
        "target_linear_velocity": target_velocity_world,
        "measured_wrench_world": measured_wrench,
        "measured_normal_force": force.copy(),
        "target_normal_force": target_force,
        "contact_blend": blend,
        "governed_normal_lead_m": raw_position_error[:, 0],
        "torque_projection_scale": projection,
        "requested_tangential_force_world": np.asarray(observed["requested_force_local_n"]) @ rotation.T,
    }
    trace.update({f"obs_{name}": np.asarray(values) for name, values in observed.items()})
    return trace


def test_real_observer_schema_validates_projected_vectors_with_normal_errors():
    parameters = _parameters(motion_confirm_time=0.05)
    compensation = ObservedCompensation(**parameters)
    angle = np.deg2rad(20.0)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    normal_local = np.array([1.0, 0.0, 0.0])
    measured_position_local = np.array([0.1, -0.02, 0.01])
    target_position_local = measured_position_local + np.array([0.007, 0.02, -0.003])
    measured_velocity_local = np.array([0.015, 0.02, -0.004])
    target_velocity_local = np.array([0.04, 0.03, 0.006])
    state = FrankaState(
        position=measured_position_local,
        rotation=np.eye(3),
        linear_velocity=measured_velocity_local,
        angular_velocity=np.zeros(3),
        normal_force=10.0,
    )
    target = FrankaTarget(
        position=target_position_local,
        rotation=np.eye(3),
        linear_velocity=target_velocity_local,
        angular_velocity=np.zeros(3),
        normal_force=12.0,
    )
    compensation.force(state, target, normal_local, 10.0, 1.0, True, dt=0.002)
    compensation.advance(state, target, normal_local, 0.002, True)
    observation = compensation.observation
    assert observation["obs_position_error_local_m"][0] == 0.0
    assert observation["obs_target_velocity_local_m_s"][0] == 0.0
    assert observation["obs_measured_velocity_local_m_s"][0] == 0.0

    measured_wrench = np.zeros((1, 6))
    measured_wrench[0, :3] = 10.0 * rotation[:, 0]
    trace = {
        "time": np.array([0.0]),
        "dt": np.array(0.002),
        "controller_frame_rotation": rotation,
        "measured_position": np.asarray([measured_position_local @ rotation.T]),
        "measured_linear_velocity": np.asarray([measured_velocity_local @ rotation.T]),
        "target_position": np.asarray([target_position_local @ rotation.T]),
        "target_linear_velocity": np.asarray([target_velocity_local @ rotation.T]),
        "measured_wrench_world": measured_wrench,
        "measured_normal_force": np.array([10.0]),
        "target_normal_force": np.array([12.0]),
        "contact_blend": np.array([1.0]),
        "governed_normal_lead_m": np.array([0.007]),
        "torque_projection_scale": np.array([1.0]),
        "requested_tangential_force_world": np.asarray([
            observation["obs_requested_force_local_n"] @ rotation.T
        ]),
    }
    trace.update({f"obs_{name[4:]}": np.asarray([value]) for name, value in observation.items()})
    result = validate_observation(trace, parameters)
    assert result["validated_cycles"] == 1


def test_valid_sequence_covers_ready_rate_limit_projection_freeze_and_missing_advance():
    parameters = _parameters()
    trace = _trace(
        parameters,
        advance=(True, True, True, False, True, True),
        allow=(False, True, True, False, True, True),
    )
    result = validate_observation(trace, parameters)
    assert result["validated_cycles"] == 6
    assert result["max_request_reconstruction_error_n"] < 1e-12
    assert result["max_coefficient_transition_error"] < 1e-12
    assert trace["obs_update_ready"].tolist() == [False, True, True, True, True, True]
    assert trace["obs_limited_increment"][1] == pytest.approx(0.0006)
    assert trace["obs_coefficient_after_advance"][0] == parameters["nominal_mu"]


def test_inactive_force_resets_coefficient_motion_direction_and_force_chain():
    parameters = _parameters()
    trace = _trace(
        parameters,
        contacts=(True, True, False, True),
        advance=(True,) * 4,
        allow=(False, True, False, False),
    )
    validate_observation(trace, parameters)
    assert trace["obs_coefficient_before_force"][2] > parameters["nominal_mu"]
    assert trace["obs_coefficient_after_force"][2] == parameters["nominal_mu"]
    np.testing.assert_array_equal(trace["obs_previous_force_local_n"][3], np.zeros(3))
    np.testing.assert_array_equal(trace["obs_previous_direction_local"][3], np.zeros(3))


def test_cap_blocks_positive_update_after_ready():
    parameters = _parameters(max_force=4.0)
    trace = _trace(parameters, advance=(True,) * 6, allow=(True,) * 6)
    validate_observation(trace, parameters)
    assert np.all(trace["obs_amplitude_capped"])
    assert trace["obs_update_ready"][1]
    assert np.all(trace["obs_coefficient_after_advance"] == parameters["nominal_mu"])


def test_slew_prevents_motion_confirmation():
    parameters = _parameters(force_slew_rate=20.0)
    trace = _trace(parameters, advance=(True,) * 6, allow=(True,) * 6)
    validate_observation(trace, parameters)
    assert np.all(trace["obs_slew_limited"])
    assert not np.any(trace["obs_update_ready"])


def test_direction_reversal_resets_ready_confirmation():
    parameters = _parameters()
    trace = _trace(parameters, target_signs=(1, 1, -1, -1, -1, -1))
    validate_observation(trace, parameters)
    assert trace["obs_update_ready"].tolist()[:4] == [False, True, False, False]


@pytest.mark.parametrize(
    ("field", "index"),
    (
        ("obs_requested_force_local_n", (2, 1)),
        ("requested_tangential_force_world", (2, 1)),
        ("obs_position_drive_m", (2,)),
        ("obs_coefficient_after_advance", (2,)),
        ("obs_motion_elapsed_after_s", (2,)),
        ("obs_corrected_force_n", (2,)),
    ),
)
def test_rejects_tampered_reconstruction_or_transition(field, index):
    parameters = _parameters()
    trace = _trace(parameters)
    trace[field][index] += 0.01
    with pytest.raises(ValueError):
        validate_observation(trace, parameters)


def test_rejects_nonboolean_flag_bad_time_and_broken_chain():
    parameters = _parameters()
    trace = _trace(parameters)
    trace["obs_active"] = trace["obs_active"].astype(float)
    with pytest.raises(ValueError, match="flag must be boolean"):
        validate_observation(trace, parameters)

    trace = _trace(parameters)
    trace["time"][2] += 0.001
    with pytest.raises(ValueError, match="time grid"):
        validate_observation(trace, parameters)

    trace = _trace(parameters)
    trace["obs_previous_direction_local"][2, 1] += 0.01
    with pytest.raises(ValueError, match="previous direction chain"):
        validate_observation(trace, parameters)


def test_rejects_false_allow_claim_for_scaled_projection_and_missing_advance():
    parameters = _parameters()
    trace = _trace(parameters)
    trace["torque_projection_scale"][2] = 0.8
    with pytest.raises(ValueError, match="unchanged projection"):
        validate_observation(trace, parameters)

    trace = _trace(parameters)
    trace["obs_advance_called"][3] = False
    trace["obs_allow_integration"][3] = True
    with pytest.raises(ValueError, match="without advance"):
        validate_observation(trace, parameters)


@pytest.mark.parametrize(
    ("name", "value"),
    (("mode", "friction"), ("nominal_mu", True), ("max_force", 0.0),
     ("motion_confirm_time", np.nan), ("max_equivalent_mu", 0.1)),
)
def test_rejects_invalid_parameter_documents(name, value):
    parameters = _parameters(**{name: value})
    with pytest.raises((TypeError, ValueError)):
        validate_observation(_trace(_parameters()), parameters)
