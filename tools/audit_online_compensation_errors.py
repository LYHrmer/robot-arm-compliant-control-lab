"""Read-only audit of online-compensation public-development evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

PROTOCOL_ID = "online-compensation-error-comparison-v1"
EXPECTED_ARMS = ("baseline", "integral", "friction", "online")
EXPECTED_ARM_MAPPING = {
    "baseline": None,
    "integral": "integral",
    "friction": "friction",
    "online": "online",
}
EXPECTED_SEEDS = (11, 29)
EXPECTED_PROFILES = (
    "static_nominal",
    "static_low_friction",
    "static_high_friction",
    "friction_step_up",
    "friction_step_down",
    "friction_ramp_up",
    "stop_hold_reverse",
    "wrench_bias_step",
    "normal_calibration_error",
    "modest_combined",
)
EXPECTED_GATES = {
    "minimum_contact_ratio_pct": 99.0,
    "maximum_raw_peak_force_n": 35.0,
    "required_saturation_pct": 0.0,
    "maximum_paired_force_rmse_increase_n": 0.2,
    "maximum_paired_orientation_rmse_increase_deg": 0.1,
}
FULL_TRACE_SELECTION = {
    ("friction_step_down", 11, "friction"),
    ("friction_step_down", 11, "online"),
    ("stop_hold_reverse", 11, "friction"),
    ("stop_hold_reverse", 11, "online"),
}
FLOAT_ATOL = 1e-10
KNOWN_ARCHIVE_SOURCE_HASHES_SHA256 = frozenset(
    {
        # results/franka_online_compensation_errors/source_hashes.json
        "8332aa912b287d6b7ea40cebaf36fd9fd10c0e223cb40bcdf1cc1d4eb6ceb25d",
        # results/franka_rotation_gain_comparison/source_hashes.json
        "414b7fdb0104eee8eeb61a6c22d2d52da6ffe3fe3106fdc94d7678451083930a",
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_relative(root: Path, label: str) -> Path:
    path = Path(label)
    if path.is_absolute() or ".." in path.parts or str(path) != label:
        raise ValueError(f"unsafe artifact path: {label!r}")
    target = root / path
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"missing or symlinked artifact: {label}")
    return target


def _verify_hashes(root: Path) -> tuple[dict, dict, dict, dict]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("audit path must be a real directory")
    manifest_path = root / "manifest.json"
    complete_path = root / "COMPLETE"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise ValueError("archive requires manifest.json and COMPLETE")
    if complete_path.read_text(encoding="utf-8").strip() != _sha256(manifest_path):
        raise ValueError("COMPLETE does not bind manifest.json")
    manifest = _json(manifest_path)
    artifacts = manifest.get("artifact_sha256")
    inputs = manifest.get("input_sha256")
    if not isinstance(artifacts, dict) or not isinstance(inputs, dict):
        raise TypeError("manifest must contain artifact and input hash mappings")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "COMPLETE"}
    }
    if set(artifacts) != actual:
        raise ValueError("manifest artifact inventory does not match archive files")
    for label, digest in artifacts.items():
        if _sha256(_safe_relative(root, label)) != digest:
            raise ValueError(f"artifact hash mismatch: {label}")
    required_inputs = {"protocol.json", "configurations.json", "source_hashes.json"}
    if set(inputs) != required_inputs:
        raise ValueError("manifest input hash inventory is not the frozen protocol set")
    for label, digest in inputs.items():
        if artifacts.get(label) != digest or _sha256(_safe_relative(root, label)) != digest:
            raise ValueError(f"input hash mismatch: {label}")
    return (
        manifest,
        _json(root / "protocol.json"),
        _json(root / "configurations.json"),
        _json(root / "source_hashes.json"),
    )


def _live_source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1] / "src" / "compliant_control_lab"
    files = sorted(set(package.rglob("*.py")) | set((package / "assets").rglob("*")))
    return {str(path.relative_to(package)): _sha256(path) for path in files if path.is_file()}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _schedule_at(schedule: dict, time: np.ndarray) -> np.ndarray:
    knots = schedule.get("knots")
    mode = schedule.get("interpolation")
    if mode not in {"step", "linear"} or not isinstance(knots, list) or not knots:
        raise ValueError("invalid frozen schedule")
    knot_time = np.asarray([item[0] for item in knots], dtype=float)
    values = np.asarray([item[1] for item in knots], dtype=float)
    if knot_time[0] != 0 or np.any(np.diff(knot_time) <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("invalid frozen schedule knots")
    indices = np.searchsorted(knot_time, time, side="right") - 1
    output = values[indices].copy()
    if mode == "linear":
        interior = indices < len(knot_time) - 1
        left = indices[interior]
        weight = (time[interior] - knot_time[left]) / (knot_time[left + 1] - knot_time[left])
        if values.ndim == 1:
            output[interior] += weight * (values[left + 1] - values[left])
        else:
            output[interior] += weight[:, None] * (values[left + 1] - values[left])
    return output


def _smooth(u: np.ndarray) -> np.ndarray:
    return u * u * (3.0 - 2.0 * u)


def _trajectory_rate(name: str, time: np.ndarray) -> np.ndarray:
    rate = np.zeros_like(time)
    start = (time > 1.2) & (time < 1.7)
    rate[start] = _smooth((time[start] - 1.2) / 0.5)
    rate[time >= 1.7] = 1.0
    if name == "standard":
        return rate
    if name != "stop_hold_reverse":
        raise ValueError(f"unknown frozen trajectory: {name}")
    decel = (time >= 4.5) & (time < 5.5)
    rate[decel] = 1.0 - _smooth(time[decel] - 4.5)
    rate[(time >= 5.5) & (time < 7.0)] = 0.0
    reverse_ramp = (time >= 7.0) & (time < 8.0)
    rate[reverse_ramp] = -_smooth(time[reverse_ramp] - 7.0)
    rate[time >= 8.0] = -1.0
    return rate


def _assert_close(actual, expected, context: str) -> None:
    if not np.allclose(actual, expected, rtol=1e-10, atol=FLOAT_ATOL):
        difference = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        raise ValueError(f"{context} mismatch (max_abs={difference})")


def _positive_finite_float(value, context: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{context} must be a positive finite number")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{context} must be a positive finite number")
    return result


def _rotation_gain_scale(manifest: dict, protocol: dict, configurations: dict) -> float:
    documents = (manifest, protocol, configurations)
    present = tuple("rotation_gain_scale" in document for document in documents)
    if any(present) and not all(present):
        raise ValueError("rotation gain scale metadata differs across frozen JSON files")
    if not any(present):
        scale = 1.0
    else:
        scales = tuple(
            _positive_finite_float(document["rotation_gain_scale"], "rotation_gain_scale")
            for document in documents
        )
        if scales[1:] != scales[:-1]:
            raise ValueError("rotation gain scale metadata differs across frozen JSON files")
        scale = scales[0]

    controllers = protocol.get("controller_constructor_configurations")
    if not isinstance(controllers, dict) or not set(EXPECTED_ARMS) <= set(controllers):
        raise ValueError("controller configurations do not contain every frozen arm")
    expected_stiffness = np.full(3, 20.0 * scale)
    expected_damping = np.full(3, 5.0 * np.sqrt(scale))
    for arm in EXPECTED_ARMS:
        try:
            base = controllers[arm]["safe_adaptive_base"]["base"]["base"]
            stiffness = np.asarray(base["rotational_stiffness"])
            damping = np.asarray(base["rotational_damping"])
        except (KeyError, TypeError) as error:
            raise ValueError(f"{arm} controller rotation gains are missing") from error
        for values, expected, label in (
            (stiffness, expected_stiffness, "rotational stiffness"),
            (damping, expected_damping, "rotational damping"),
        ):
            if values.shape != (3,) or values.dtype.kind not in "iuf" or not np.all(
                np.isfinite(values)
            ):
                raise ValueError(f"{arm} controller {label} must be three finite numbers")
            _assert_close(values, expected, f"{arm} controller {label}")
    return scale


def _full_trace_rotation_gain_scale(path: Path) -> float:
    with np.load(path, allow_pickle=False) as loaded:
        if "rotation_gain_scale" not in loaded.files:
            return 1.0
        value = loaded["rotation_gain_scale"]
    if value.shape != ():
        raise ValueError(f"full trace rotation_gain_scale must be scalar: {path.name}")
    return _positive_finite_float(value.item(), f"full trace rotation_gain_scale: {path.name}")


def _load_compact(path: Path, case: dict, arm: str, controller: dict) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        trace = {name: loaded[name] for name in loaded.files}
    required = {
        "time",
        "true_normal_force",
        "target_normal_force",
        "measured_normal_force",
        "orientation_error_rad",
        "torque_projection_scale",
        "contact_blend",
        "controller_coefficient_before_compute",
        "controller_coefficient_after_compute",
        "controller_update_ready_before_compute",
        "controller_update_ready_after_compute",
        "applied_wall_friction",
        "applied_tool_friction",
        "applied_raw_wrench_bias_world",
        "feedback_raw_wrench_bias_world",
        "controller_yaw_error_deg",
        "trajectory_rate_scale",
        "tangent_position_error_world",
        "tangent_position_error_norm_m",
        "tangent_velocity_error_world",
        "tangent_velocity_error_norm_m_s",
        "actuator_clipped",
        "requested_tangential_force_norm_n",
        "requested_tangential_force_delta_norm_n",
    }
    if set(trace) != required:
        raise ValueError(f"compact trace schema mismatch: {path.name}")
    time = trace["time"]
    config = case["config"]
    expected_count = round(config["duration"] / config["timestep"])
    if time.shape != (expected_count,):
        raise ValueError(f"compact trace sample count mismatch: {path.name}")
    _assert_close(time, np.arange(expected_count) * config["timestep"], "control time")
    for name, value in trace.items():
        if value.shape[0] != expected_count:
            raise ValueError(f"compact trace alignment mismatch: {name}")
        if value.dtype.kind not in "biuf" or not np.all(np.isfinite(value)):
            raise ValueError(f"compact trace must be finite numeric data: {name}")

    friction = _schedule_at(case["friction_schedule"], time)
    bias = _schedule_at(case["additional_world_wrench_bias_schedule"], time)
    _assert_close(trace["applied_wall_friction"], friction, "wall friction schedule")
    _assert_close(trace["applied_tool_friction"], friction, "tool friction schedule")
    _assert_close(trace["applied_raw_wrench_bias_world"], bias, "raw wrench-bias schedule")
    expected_feedback_bias = np.concatenate((bias[:1], bias[:-1]))
    _assert_close(
        trace["feedback_raw_wrench_bias_world"],
        expected_feedback_bias,
        "causal feedback wrench-bias schedule",
    )
    _assert_close(
        trace["controller_yaw_error_deg"],
        case["controller_yaw_error_deg"],
        "controller yaw-error schedule",
    )
    _assert_close(
        trace["trajectory_rate_scale"],
        _trajectory_rate(case["task"]["trajectory"], time),
        "trajectory rate schedule",
    )
    _assert_close(
        np.linalg.norm(trace["tangent_position_error_world"], axis=1),
        trace["tangent_position_error_norm_m"],
        "tangent-position norm",
    )
    _assert_close(
        np.linalg.norm(trace["tangent_velocity_error_world"], axis=1),
        trace["tangent_velocity_error_norm_m_s"],
        "tangent-velocity norm",
    )
    if trace["requested_tangential_force_delta_norm_n"][0] != 0:
        raise ValueError("first requested-force delta must be zero")
    before = trace["controller_coefficient_before_compute"]
    after = trace["controller_coefficient_after_compute"]
    _assert_close(before[1:], after[:-1], "coefficient before/after chronology")
    tangential = controller["safe_adaptive_base"]["tangential"]
    nominal = 0.0 if tangential is None else tangential["nominal_mu"]
    _assert_close(before[0], nominal, "initial controller coefficient")
    ready = trace["controller_update_ready_after_compute"].astype(bool)
    if arm != "online":
        _assert_close(after, nominal, "unchanged non-online coefficient")
        if np.any(ready) or np.any(trace["controller_update_ready_before_compute"]):
            raise ValueError("non-online arm cannot report online update readiness")
    else:
        online = tangential
        if np.any(after < -FLOAT_ATOL) or np.any(after > online["max_equivalent_mu"] + FLOAT_ATOL):
            raise ValueError("online coefficient violates frozen bounds")
        rates = np.diff(after) / config["timestep"]
        reset = np.abs(rates) > online["coefficient_rate_limit"] + FLOAT_ATOL
        if np.any(reset):
            _assert_close(after[1:][reset], online["nominal_mu"], "coefficient reset exception")
            reset_indices = np.flatnonzero(reset) + 1
            legitimate = (
                ~ready[reset_indices]
                & (
                    (trace["measured_normal_force"][reset_indices] <= 1.0)
                    | (trace["target_normal_force"][reset_indices] <= 0.0)
                    | (trace["contact_blend"][reset_indices] < 0.99)
                )
                & (trace["requested_tangential_force_norm_n"][reset_indices] <= FLOAT_ATOL)
            )
            if not np.all(legitimate):
                raise ValueError("illegitimate coefficient reset during active/ready control")
        force_reset = trace["requested_tangential_force_delta_norm_n"][1:] > (
            online["force_slew_rate"] * config["timestep"] + FLOAT_ATOL
        )
        if np.any(force_reset):
            reset_indices = np.flatnonzero(force_reset) + 1
            _assert_close(
                trace["requested_tangential_force_norm_n"][reset_indices],
                0.0,
                "force reset exception",
            )
            legitimate = ~ready[reset_indices] & (
                (trace["measured_normal_force"][reset_indices] <= 1.0)
                | (trace["target_normal_force"][reset_indices] <= 0.0)
                | (trace["contact_blend"][reset_indices] < 0.99)
            )
            if not np.all(legitimate):
                raise ValueError("illegitimate force reset during active/ready control")
    return trace


def _window(trace: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    force_error = trace["true_normal_force"][mask] - trace["target_normal_force"][mask]
    return {
        "sample_count": int(np.count_nonzero(mask)),
        "force_rmse_n": float(np.sqrt(np.mean(force_error**2))),
        "peak_force_n": float(np.max(trace["true_normal_force"][mask])),
        "contact_ratio_pct": float(100 * np.mean(trace["true_normal_force"][mask] > 0.5)),
        "tangent_rmse_mm": float(
            1000 * np.sqrt(np.mean(trace["tangent_position_error_norm_m"][mask] ** 2))
        ),
        "tangent_velocity_error_rms_m_s": float(
            np.sqrt(np.mean(trace["tangent_velocity_error_norm_m_s"][mask] ** 2))
        ),
        "orientation_rmse_deg": float(
            np.rad2deg(np.sqrt(np.mean(trace["orientation_error_rad"][mask] ** 2)))
        ),
        "saturation_pct": float(100 * np.mean(trace["actuator_clipped"][mask])),
    }


def _recovery(trace: dict[str, np.ndarray], case: dict, protocol: dict) -> dict:
    start_s = case["recovery_start_s"]
    if start_s is None:
        return {
            "force_recovery_status": "not_applicable",
            "force_recovered": None,
            "force_recovery_time_s": None,
            "tangent_recovery_status": "not_applicable",
            "tangent_recovered": None,
            "tangent_recovery_time_s": None,
        }
    time = trace["time"]
    config = case["config"]
    recovery = protocol["recovery"]
    window = max(1, round(recovery["window_s"] / config["timestep"]))
    force_error = trace["true_normal_force"] - trace["target_normal_force"]
    force_time = None
    tangent_time = None
    for index in np.flatnonzero(time >= start_s):
        stop = index + window
        if stop > len(time):
            break
        elapsed = float(time[index] - start_s)
        if force_time is None and np.sqrt(np.mean(force_error[index:stop] ** 2)) <= recovery[
            "absolute_force_error_n"
        ]:
            force_time = elapsed
        if tangent_time is None and 1000 * np.sqrt(
            np.mean(trace["tangent_position_error_norm_m"][index:stop] ** 2)
        ) <= recovery["tangent_rmse_mm"]:
            tangent_time = elapsed
        if force_time is not None and tangent_time is not None:
            break
    return {
        "force_recovery_status": "recovered" if force_time is not None else "not_recovered",
        "force_recovered": force_time is not None,
        "force_recovery_time_s": force_time,
        "tangent_recovery_status": "recovered" if tangent_time is not None else "not_recovered",
        "tangent_recovered": tangent_time is not None,
        "tangent_recovery_time_s": tangent_time,
    }


def _summarize(trace: dict[str, np.ndarray], case: dict, arm: str, protocol: dict) -> tuple[dict, list[dict]]:
    time = trace["time"]
    evaluation = time >= case["config"]["evaluation_start"]
    phases = []
    for phase in case["phases"]:
        mask = (time >= phase["start_s"]) & (time < phase["end_s"])
        phases.append({"phase": phase["name"], **_window(trace, mask)})
    overall = _window(trace, evaluation)
    force_worst = max(phases, key=lambda row: row["force_rmse_n"])
    tangent_worst = max(phases, key=lambda row: row["tangent_rmse_mm"])
    orientation_worst = max(phases, key=lambda row: row["orientation_rmse_deg"])
    online = protocol["controller_constructor_configurations"]["online"]["safe_adaptive_base"][
        "tangential"
    ]
    coefficient = trace["controller_coefficient_after_compute"]
    requested = trace["requested_tangential_force_norm_n"]
    overall.update(
        peak_force_n=float(np.max(trace["true_normal_force"])),
        saturation_pct=float(100 * np.mean(trace["actuator_clipped"])),
        worst_force_phase=force_worst["phase"],
        worst_phase_force_rmse_n=force_worst["force_rmse_n"],
        worst_tangent_phase=tangent_worst["phase"],
        worst_phase_tangent_rmse_mm=tangent_worst["tangent_rmse_mm"],
        worst_orientation_phase=orientation_worst["phase"],
        worst_phase_orientation_rmse_deg=orientation_worst["orientation_rmse_deg"],
        **_recovery(trace, case, protocol),
        tangential_update_ready_pct=float(
            100 * np.mean(trace["controller_update_ready_after_compute"][evaluation])
        ),
        coefficient_min=float(np.min(coefficient)),
        coefficient_max=float(np.max(coefficient)),
        max_requested_tangential_force_n=float(np.max(requested)),
    )
    overall["compensation_limit_observed"] = bool(
        overall["coefficient_max"] >= online["max_equivalent_mu"] - 1e-12
        or overall["max_requested_tangential_force_n"] >= online["max_force"] - 1e-12
    )
    if (
        overall["tangent_recovery_status"] == "not_recovered"
        and overall["compensation_limit_observed"]
    ):
        overall["tangent_recovery_status"] = "not_recovered_with_limit_active"
    dt = case["config"]["timestep"]
    rates = np.diff(coefficient) / dt
    delta = trace["requested_tangential_force_delta_norm_n"][1:]
    active = (trace["measured_normal_force"] > 1) & (trace["target_normal_force"] > 0)
    normal = (
        (trace["contact_blend"][1:] >= 0.99)
        & (trace["contact_blend"][:-1] >= 0.99)
        & active[1:]
        & active[:-1]
    )
    coefficient_reset = (
        np.abs(rates) > online["coefficient_rate_limit"] + 1e-10
        if arm == "online"
        else np.zeros_like(rates, dtype=bool)
    )
    force_reset = (
        delta > online["force_slew_rate"] * dt + 1e-10
        if arm == "online"
        else np.zeros_like(delta, dtype=bool)
    )
    coefficient_normal = normal & trace["controller_update_ready_after_compute"][1:].astype(bool)
    coefficient_normal &= ~coefficient_reset
    force_normal = normal & ~force_reset
    overall.update(
        max_normal_operation_coefficient_rate_s=float(
            np.max(np.abs(rates[coefficient_normal])) if np.any(coefficient_normal) else 0
        ),
        max_coefficient_reset_jump=float(
            np.max(np.abs(np.diff(coefficient)[coefficient_reset]))
            if np.any(coefficient_reset)
            else 0
        ),
        max_normal_operation_force_slew_n_s=float(
            np.max(delta[force_normal] / dt) if np.any(force_normal) else 0
        ),
        max_force_reset_jump_n=float(np.max(delta[force_reset]) if np.any(force_reset) else 0),
    )
    return overall, phases


def _failures(row: dict, baseline: dict, friction: dict | None, arm: str, gates: dict) -> tuple[str, ...]:
    failures = [
        name
        for passed, name in (
            (row["contact_ratio_pct"] >= gates["minimum_contact_ratio_pct"], "contact_ratio"),
            (row["peak_force_n"] <= gates["maximum_raw_peak_force_n"], "raw_peak_force"),
            (row["saturation_pct"] == gates["required_saturation_pct"], "saturation"),
            (
                row["force_rmse_n"] - baseline["force_rmse_n"]
                <= gates["maximum_paired_force_rmse_increase_n"],
                "paired_force_rmse",
            ),
            (
                row["orientation_rmse_deg"] - baseline["orientation_rmse_deg"]
                <= gates["maximum_paired_orientation_rmse_increase_deg"],
                "paired_orientation_rmse",
            ),
        )
        if not passed
    ]
    if arm == "online":
        if friction is None:
            raise ValueError("online rows require fixed-friction comparator rows")
        if row["force_rmse_n"] - friction["force_rmse_n"] > gates[
            "maximum_paired_force_rmse_increase_n"
        ]:
            failures.append("paired_friction_force_rmse")
        if row["orientation_rmse_deg"] - friction["orientation_rmse_deg"] > gates[
            "maximum_paired_orientation_rmse_increase_deg"
        ]:
            failures.append("paired_friction_orientation_rmse")
    return tuple(failures)


def _cell(row: dict[str, str], name: str, expected, context: str) -> None:
    if name not in row:
        raise ValueError(f"published column missing: {name}")
    actual = row[name]
    if expected is None:
        if actual != "":
            raise ValueError(f"reconstructed metric mismatch: {context}.{name}")
    elif isinstance(expected, (bool, np.bool_)):
        if actual != str(bool(expected)):
            raise ValueError(f"reconstructed metric mismatch: {context}.{name}")
    elif isinstance(expected, str):
        if actual != expected:
            raise ValueError(f"reconstructed metric mismatch: {context}.{name}")
    else:
        try:
            value = float(actual)
        except ValueError as error:
            raise ValueError(f"reconstructed metric mismatch: {context}.{name}") from error
        if not np.isclose(value, float(expected), rtol=1e-10, atol=FLOAT_ATOL):
            raise ValueError(f"reconstructed metric mismatch: {context}.{name}")


def _validate_protocol(manifest: dict, protocol: dict, configurations: dict) -> tuple[list[dict], list[str], list[int], list[str]]:
    if (
        manifest.get("experiment_identity") != PROTOCOL_ID
        or protocol.get("identity") != PROTOCOL_ID
        or manifest.get("new_holdout") is not False
        or protocol.get("new_holdout") is not False
        or protocol.get("public_development") is not True
        or protocol.get("comparison_defined_before_execution") is not True
    ):
        raise ValueError("archive is not the frozen public-development protocol")
    if protocol.get("gates") != EXPECTED_GATES:
        raise ValueError("frozen gates changed")
    if tuple(protocol.get("predetermined_noise_seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("predetermined seeds changed")
    if protocol.get("arms") != EXPECTED_ARM_MAPPING:
        raise ValueError("arm definitions changed")
    cases = protocol.get("all_cases")
    if not isinstance(cases, list) or len(cases) != len(EXPECTED_PROFILES) * len(EXPECTED_SEEDS):
        raise ValueError("frozen physical-profile/seed grid changed")
    case_grid = {(case["name"], case["config"]["seed"]) for case in cases}
    if case_grid != {(name, seed) for name in EXPECTED_PROFILES for seed in EXPECTED_SEEDS}:
        raise ValueError("frozen physical-profile/seed grid changed")
    names = configurations.get("selected_case_names")
    seeds = configurations.get("selected_seeds")
    arms = configurations.get("selected_arms")
    if (
        not isinstance(names, list)
        or not isinstance(seeds, list)
        or not isinstance(arms, list)
        or len(set(names)) != len(names)
        or len(set(seeds)) != len(seeds)
        or len(set(arms)) != len(arms)
        or not set(names) <= set(EXPECTED_PROFILES)
        or not set(seeds) <= set(EXPECTED_SEEDS)
        or not set(arms) <= set(EXPECTED_ARMS)
        or "baseline" not in arms
        or ("online" in arms and "friction" not in arms)
    ):
        raise ValueError("invalid selected subset declaration")
    expected_subset = (
        set(names) != set(EXPECTED_PROFILES)
        or set(seeds) != set(EXPECTED_SEEDS)
        or set(arms) != set(EXPECTED_ARMS)
    )
    if configurations.get("is_subset") is not expected_subset or manifest.get("is_subset") is not expected_subset:
        raise ValueError("subset status is not explicit and consistent")
    if manifest.get("selected_case_names") != names or manifest.get("selected_seeds") != seeds or manifest.get("selected_arms") != arms:
        raise ValueError("manifest/configuration selections differ")
    selected_cases = [case for case in cases if case["name"] in names and case["config"]["seed"] in seeds]
    executions = configurations.get("executions")
    expected_executions = {
        (case["name"], case["config"]["seed"], arm) for case in selected_cases for arm in arms
    }
    if not isinstance(executions, list) or {
        (item["case"]["name"], item["paired_seed"], item["arm"]) for item in executions
    } != expected_executions:
        raise ValueError("execution configuration grid mismatch")
    return selected_cases, names, seeds, arms


def audit_archive(directory: Path | str) -> dict:
    root = Path(directory).resolve()
    manifest, protocol, configurations, source_hashes = _verify_hashes(root)
    current_source_matches_archive = source_hashes == _live_source_hashes()
    if current_source_matches_archive:
        source_identity_status = "current"
    elif _sha256(root / "source_hashes.json") in KNOWN_ARCHIVE_SOURCE_HASHES_SHA256:
        source_identity_status = "known_archive"
    else:
        raise ValueError("archive source identity is neither current nor a known archive")
    rotation_gain_scale = _rotation_gain_scale(manifest, protocol, configurations)
    cases, _, _, arms = _validate_protocol(manifest, protocol, configurations)
    case_map = {(case["name"], case["config"]["seed"]): case for case in cases}
    expected_grid = {(name, seed, arm) for name, seed in case_map for arm in arms}
    _, comparison = _read_csv(root / "comparison.csv")
    comparison_grid = {(row["case"], int(row["simulation_seed"]), row["arm"]) for row in comparison}
    if len(comparison) != len(expected_grid) or comparison_grid != expected_grid:
        raise ValueError("comparison grid does not match the selected protocol")
    _, published_phases = _read_csv(root / "phase_metrics.csv")
    phase_grid = {
        (name, seed, arm, phase["name"])
        for (name, seed), case in case_map.items()
        for arm in arms
        for phase in case["phases"]
    }
    actual_phase_grid = {
        (row["case"], int(row["simulation_seed"]), row["arm"], row["phase"])
        for row in published_phases
    }
    if len(published_phases) != len(phase_grid) or actual_phase_grid != phase_grid:
        raise ValueError("phase grid does not match the selected protocol")

    controllers = protocol["controller_constructor_configurations"]
    summaries = {}
    phases = {}
    compact_paths = {}
    for key in sorted(expected_grid):
        name, seed, arm = key
        filename = f"traces/{name}__seed_{seed}__{arm}__compact.npz"
        path = root / filename
        if filename not in manifest["artifact_sha256"]:
            raise ValueError(f"missing compact trace: {filename}")
        trace = _load_compact(path, case_map[name, seed], arm, controllers[arm])
        summaries[key], phase_values = _summarize(trace, case_map[name, seed], arm, protocol)
        phases[key] = {row["phase"]: row for row in phase_values}
        compact_paths[key] = path

    comparison_by_key = {
        (row["case"], int(row["simulation_seed"]), row["arm"]): row for row in comparison
    }
    for key, summary in summaries.items():
        name, seed, arm = key
        baseline = summaries[name, seed, "baseline"]
        friction = summaries.get((name, seed, "friction"))
        failures = _failures(summary, baseline, friction, arm, protocol["gates"])
        expected = {
            "case": name,
            "arm": arm,
            "simulation_seed": seed,
            "paired_baseline_seed": seed,
            "controller_yaw_error_deg": case_map[name, seed]["controller_yaw_error_deg"],
            **summary,
            "paired_force_rmse_increase_n": summary["force_rmse_n"] - baseline["force_rmse_n"],
            "paired_orientation_rmse_increase_deg": summary["orientation_rmse_deg"]
            - baseline["orientation_rmse_deg"],
            "paired_friction_force_rmse_increase_n": None
            if friction is None
            else summary["force_rmse_n"] - friction["force_rmse_n"],
            "paired_friction_orientation_rmse_increase_deg": None
            if friction is None
            else summary["orientation_rmse_deg"] - friction["orientation_rmse_deg"],
            "failed_gates": ";".join(failures),
            "all_gates_pass": "yes" if not failures else "no",
        }
        row = comparison_by_key[key]
        if set(row) != set(expected):
            raise ValueError("comparison columns do not match reconstructed metric schema")
        for name_, value in expected.items():
            _cell(row, name_, value, f"comparison[{key}]")

    phase_by_key = {
        (row["case"], int(row["simulation_seed"]), row["arm"], row["phase"]): row
        for row in published_phases
    }
    for name, seed, arm, phase_name in phase_grid:
        phase = phases[name, seed, arm][phase_name]
        baseline = phases[name, seed, "baseline"][phase_name]
        friction = phases.get((name, seed, "friction"), {}).get(phase_name)
        failures = _failures(phase, baseline, friction, arm, protocol["gates"])
        expected = {
            "case": name,
            "simulation_seed": seed,
            "arm": arm,
            **phase,
            "paired_baseline_force_rmse_increase_n": phase["force_rmse_n"]
            - baseline["force_rmse_n"],
            "paired_baseline_orientation_rmse_increase_deg": phase["orientation_rmse_deg"]
            - baseline["orientation_rmse_deg"],
            "paired_friction_force_rmse_increase_n": None
            if friction is None
            else phase["force_rmse_n"] - friction["force_rmse_n"],
            "paired_friction_orientation_rmse_increase_deg": None
            if friction is None
            else phase["orientation_rmse_deg"] - friction["orientation_rmse_deg"],
            "failed_gates": ";".join(failures),
            "all_gates_pass": "yes" if not failures else "no",
        }
        row = phase_by_key[name, seed, arm, phase_name]
        if set(row) != set(expected):
            raise ValueError("phase columns do not match reconstructed metric schema")
        for name_, value in expected.items():
            _cell(row, name_, value, f"phase[{name, seed, arm, phase_name}]")

    compact_files = set(root.glob("traces/*__compact.npz"))
    if compact_files != set(compact_paths.values()):
        raise ValueError("compact trace inventory contains unexpected files")
    full_files = set(root.glob("traces/*__full.npz"))
    full_override = configurations["trace_retention"]["full_traces_override"]
    expected_full_keys = expected_grid if full_override else expected_grid & FULL_TRACE_SELECTION
    expected_full_files = {
        root / f"traces/{name}__seed_{seed}__{arm}__full.npz"
        for name, seed, arm in expected_full_keys
    }
    if full_files != expected_full_files:
        raise ValueError("full trace inventory differs from the predeclared retention policy")
    for path in full_files:
        if _full_trace_rotation_gain_scale(path) != rotation_gain_scale:
            raise ValueError(f"full trace rotation_gain_scale differs from archive: {path.name}")
    return {
        "archive": str(root),
        "identity": PROTOCOL_ID,
        "current_source_matches_archive": current_source_matches_archive,
        "source_identity_status": source_identity_status,
        "is_subset": manifest["is_subset"],
        "comparison_rows": len(comparison),
        "phase_rows": len(published_phases),
        "compact_traces": len(compact_files),
        "full_traces": len(full_files),
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = audit_archive(arguments.audit)
    except Exception as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        raise SystemExit(1) from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
