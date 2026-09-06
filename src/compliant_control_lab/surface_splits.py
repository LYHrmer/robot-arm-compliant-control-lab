"""Pure grouped development splits; published cases never become a new blind test."""

import hashlib
import json
from copy import deepcopy
from dataclasses import fields
from numbers import Integral, Real

import numpy as np

from compliant_control_lab.surface_control import SurfaceFrame
from compliant_control_lab.surface_simulation import (
    SurfaceScenario,
    SurfaceSimulationConfig,
    SurfaceTask,
)

GROUP_SCHEMA = "surface_physical_task_group"
GROUP_SCHEMA_VERSION = 1
GROUP_CONFIG_FIELDS = (
    "contact_model",
    "timestep",
    "target_force",
    "force_filter_time_constant",
)
IGNORED_CONFIG_FIELDS = ("seed", "duration", "evaluation_start")
SPLIT_NAMES = ("train", "validation", "development_test")
_SCENARIO_FIELDS = frozenset(field.name for field in fields(SurfaceScenario))
_CONFIG_FIELDS = frozenset((*GROUP_CONFIG_FIELDS, *IGNORED_CONFIG_FIELDS))
_TASK_REQUIRED = frozenset(("yaw_deg", "nominal_plane_x_m"))
_TASK_NUMERIC = _TASK_REQUIRED | {
    "tangent_amplitude_m",
    "vertical_amplitude_m",
    "frequency_hz",
    "phase_rad",
    "direction",
}


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number, not boolean")
    return 0.0 if value == 0 else float(value)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, not boolean")
    return int(value)


def _json_value(value):
    """Normalize finite JSON, including equivalent numeric and tuple/list spellings."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = _number(value, "JSON numeric value")
        return int(numeric) if numeric.is_integer() else numeric
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_value(item) for key, item in value.items()}
    raise ValueError("configuration must contain finite JSON values with string keys")


def _complete_mapping(case, name, required, *, extra=False):
    value = case.get(name)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an explicit mapping")
    missing, unknown = required - value.keys(), value.keys() - required
    if missing or (unknown and not extra):
        raise ValueError(
            f"{name} missing fields {sorted(missing)} or unsupported fields {sorted(unknown)}"
        )
    return deepcopy(value)


def case_group_id(case: dict) -> str:
    """Hash all physical/sensor/task parameters, not method, noise seed or calibration.

    scenario/config must explicitly include every supported field; defaults are
    never inferred. Unknown config/scenario fields fail instead of being silently
    excluded. Extra finite task fields are all retained, so new trajectory fields
    cannot collide with an old task. Config-schema changes require a version bump.
    Top-level identifiers and controller choices are metadata, excluded from the
    physical group; a supplied controller_frame_rotation is still validated as SO(3).
    """
    if not isinstance(case, dict):
        raise TypeError("case must be a mapping")
    _json_value(
        {name: value for name, value in case.items() if name != "controller_frame_rotation"}
    )
    if {field.name for field in fields(SurfaceSimulationConfig)} != _CONFIG_FIELDS:
        raise ValueError("simulation config changed: explicitly update the group schema")
    scenario = _complete_mapping(case, "scenario", _SCENARIO_FIELDS)
    config = _complete_mapping(case, "config", _CONFIG_FIELDS)
    task = _complete_mapping(case, "task", _TASK_REQUIRED, extra=True)
    if "controller_frame_rotation" in case:
        SurfaceFrame(case["controller_frame_rotation"])
    if not isinstance(scenario["name"], str):
        raise TypeError("scenario name must be a string")
    for name, value in scenario.items():
        if name == "name":
            continue
        if name == "delay_steps":
            scenario[name] = _integer(value, name)
        elif name in {"force_bias_sensor_n", "torque_bias_sensor_nm"}:
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise ValueError(f"{name} must contain three numbers")
            scenario[name] = [_number(item, name) for item in value]
        else:
            scenario[name] = _number(value, name)
    SurfaceScenario(**scenario)
    for name in config:
        if name == "seed":
            config[name] = _integer(config[name], name)
        elif name != "contact_model":
            config[name] = _number(config[name], name)
    if not isinstance(config["contact_model"], str):
        raise TypeError("contact_model must be a string")
    SurfaceSimulationConfig(**config)
    for name in task.keys() & _TASK_NUMERIC:
        task[name] = _number(task[name], name)
    SurfaceTask(**{name: task[name] for name in _TASK_REQUIRED})
    scenario.pop("name")
    payload = {
        "schema": GROUP_SCHEMA,
        "schema_version": GROUP_SCHEMA_VERSION,
        "scenario": scenario,
        "config": {name: config[name] for name in GROUP_CONFIG_FIELDS},
        "task": _json_value(task),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def split_cases(cases, seed=11, train_fraction=0.7, validation_fraction=0.15) -> list[dict]:
    """Assign whole groups deterministically, returning deep copies in input order.

    Fractions are targets over GROUP counts, not sample counts. Floors are clamped
    to keep each split nonempty; at least three physical/task groups are required.
    Existing labels are replaced. No train normalization, windows or data are made.
    """
    seed = _integer(seed, "seed")
    train_fraction = _number(train_fraction, "train_fraction")
    validation_fraction = _number(validation_fraction, "validation_fraction")
    if min(train_fraction, validation_fraction) <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("fractions must be positive with a positive remaining test fraction")
    cases = list(cases)
    identifiers = [case_group_id(case) for case in cases]
    groups = sorted(set(identifiers))
    count = len(groups)
    if count < 3:
        raise ValueError("at least three distinct physical/task groups are required")
    order = np.random.default_rng(seed).permutation(count)
    train_count = max(1, min(count - 2, int(count * train_fraction)))
    validation_count = max(1, min(count - train_count - 1, int(count * validation_fraction)))
    assignment = {
        groups[index]: SPLIT_NAMES[
            0 if rank < train_count else 1 if rank < train_count + validation_count else 2
        ]
        for rank, index in enumerate(order)
    }
    return [
        {**deepcopy(case), "group_id": group, "split": assignment[group]}
        for case, group in zip(cases, identifiers)
    ]
