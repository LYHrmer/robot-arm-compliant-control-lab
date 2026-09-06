"""Physical/task groups are indivisible across public development data splits."""

import re
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from compliant_control_lab.surface_simulation import (
    SurfaceScenario,
    SurfaceSimulationConfig,
    SurfaceTask,
)
from compliant_control_lab.surface_splits import (
    GROUP_CONFIG_FIELDS,
    GROUP_SCHEMA,
    GROUP_SCHEMA_VERSION,
    IGNORED_CONFIG_FIELDS,
    SPLIT_NAMES,
    case_group_id,
    split_cases,
)


def case():
    return {
        "scenario": asdict(SurfaceScenario()),
        "config": asdict(SurfaceSimulationConfig(contact_model="smooth")),
        "task": asdict(SurfaceTask()),
        "controller_frame_rotation": np.eye(3).tolist(),
        "controller_kind": "surface_friction",
    }


def cases(count=10):
    output = []
    for index in range(count):
        current = case()
        current["scenario"]["wall_sliding_friction"] = 0.2 + index * 0.05
        for seed in (11, 29):
            row = deepcopy(current)
            row["config"]["seed"] = seed
            row["episode_id"] = f"physical_{index}_seed_{seed}"
            output.append(row)
    return output


def assignment(records):
    return {row["group_id"]: row["split"] for row in records}


def test_versioned_hash_ignores_only_nuisance_and_calibration_variants():
    original, variant = case(), case()
    variant.update(
        method="learned",
        controller_kind="surface_adaptive",
        name="second",
        case_index=17,
        episode_id="extra",
        split="train",
        group_id="stale",
    )
    angle = np.deg2rad(15)
    variant["controller_frame_rotation"] = [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ]
    variant["scenario"]["name"] = "different_label"
    variant["config"].update(seed=234, duration=12.0, evaluation_start=2.0)
    assert case_group_id(original) == case_group_id(variant)
    assert re.fullmatch(r"[0-9a-f]{64}", case_group_id(original))
    assert GROUP_SCHEMA == "surface_physical_task_group" and GROUP_SCHEMA_VERSION == 1
    assert set(GROUP_CONFIG_FIELDS) == {
        "contact_model",
        "timestep",
        "target_force",
        "force_filter_time_constant",
    }
    assert set(IGNORED_CONFIG_FIELDS) == {"seed", "duration", "evaluation_start"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("wall_yaw_deg", 15.0),
        ("wall_time_constant", 0.008),
        ("wall_sliding_friction", 0.65),
        ("tool_sliding_friction", 0.25),
        ("tool_mass_kg", 0.13),
        ("nominal_tool_mass_kg", 0.12),
        ("position_noise_std_m", 0.0001),
        ("force_noise_std_n", 0.2),
        ("torque_noise_std_nm", 0.004),
        ("force_bias_sensor_n", [0.3, 0, 0]),
        ("torque_bias_sensor_nm", [0, 0.01, 0]),
        ("delay_steps", 1),
        ("bias_compensation_scale", 0.9),
    ],
)
def test_every_physical_and_sensor_field_changes_group(field, value):
    original, changed = case(), case()
    changed["scenario"][field] = value
    assert case_group_id(original) != case_group_id(changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("contact_model", "legacy"),
        ("timestep", 0.001),
        ("target_force", 6.0),
        ("force_filter_time_constant", 0.04),
    ],
)
def test_every_grouped_config_field_changes_group(field, value):
    original, changed = case(), case()
    changed["config"][field] = value
    assert case_group_id(original) != case_group_id(changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("yaw_deg", 15.0),
        ("nominal_plane_x_m", 0.41),
        ("tangent_amplitude_m", 0.05),
        ("vertical_amplitude_m", 0.03),
        ("frequency_hz", 0.2),
        ("phase_rad", 1.0),
        ("direction", -1.0),
        ("trajectory_extension", {"enabled": True, "knots": [[0, 1], [1, 2]]}),
    ],
)
def test_all_task_parameters_including_extensions_participate(field, value):
    original, changed = case(), case()
    changed["task"][field] = value
    assert case_group_id(original) != case_group_id(changed)


def test_canonical_order_numbers_signed_zero_and_tuple_list_equivalence():
    original, changed = case(), case()
    changed["task"] = dict(reversed(list(changed["task"].items())))
    changed["task"]["yaw_deg"] = -0.0
    changed["config"]["target_force"] = 12
    changed["scenario"]["force_bias_sensor_n"] = [0.2, 0, 0]
    changed["controller_frame_rotation"] = np.eye(3)
    assert case_group_id(original) == case_group_id(changed)


def test_splits_are_grouped_reproducible_order_independent_and_not_blind():
    source = cases()
    source.append(deepcopy(source[0]))
    first = split_cases(source, seed=12)
    repeated = split_cases(source, seed=12)
    reversed_rows = split_cases(list(reversed(source)), seed=12)
    assert first == repeated
    assert assignment(first) == assignment(reversed_rows)
    assert assignment(first) != assignment(split_cases(source, seed=29))
    assert len(first) == len(source)
    assert len(assignment(first)) == 10
    assert (
        set(assignment(first).values())
        == set(SPLIT_NAMES)
        == {"train", "validation", "development_test"}
    )
    assert [list(assignment(first).values()).count(name) for name in SPLIT_NAMES] == [7, 1, 2]
    for group in assignment(first):
        assert len({row["split"] for row in first if row["group_id"] == group}) == 1
    assert [row["episode_id"] for row in first] == [row["episode_id"] for row in source]


def test_three_groups_keep_every_split_nonempty_even_for_extreme_valid_fractions():
    records = split_cases(cases(3), train_fraction=0.98, validation_fraction=0.01)
    assert set(assignment(records).values()) == set(SPLIT_NAMES)
    assert [list(assignment(records).values()).count(name) for name in SPLIT_NAMES] == [1, 1, 1]


def test_output_and_input_copies_do_not_alias():
    source = cases(3)
    frozen = deepcopy(source)
    output = split_cases(source)
    assert source == frozen
    output[0]["scenario"]["force_bias_sensor_n"] = [8, 9, 10]
    output[0]["controller_frame_rotation"][0][0] = 100
    output[0]["task"]["yaw_deg"] = 90
    assert source == frozen
    assert output[1]["controller_frame_rotation"][0][0] == 1


@pytest.mark.parametrize("section", ["scenario", "config", "task"])
def test_explicit_complete_mappings_required(section):
    malformed = case()
    del malformed[section]
    with pytest.raises(ValueError, match="mapping"):
        case_group_id(malformed)
    malformed = case()
    malformed[section].pop(next(iter(malformed[section])))
    with pytest.raises(ValueError, match="missing"):
        case_group_id(malformed)


@pytest.mark.parametrize("section", ["scenario", "config"])
def test_unsupported_physical_configuration_fields_are_not_silently_dropped(section):
    malformed = case()
    malformed[section]["future_physical_parameter"] = 2.0
    with pytest.raises(ValueError, match="unsupported"):
        case_group_id(malformed)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("scenario", "wall_time_constant", np.nan),
        ("scenario", "tool_mass_kg", True),
        ("scenario", "force_bias_sensor_n", [0, False, 0]),
        ("scenario", "delay_steps", 1.0),
        ("config", "seed", True),
        ("config", "duration", np.inf),
        ("config", "target_force", "12"),
        ("config", "contact_model", []),
        ("task", "yaw_deg", False),
        ("task", "frequency_hz", True),
        ("task", "extension", {"nested": [np.inf]}),
        ("task", "extension", object()),
    ],
)
def test_nonfinite_and_inappropriate_json_types_are_rejected(section, field, value):
    malformed = case()
    malformed[section][field] = value
    with pytest.raises((ValueError, TypeError)):
        case_group_id(malformed)


@pytest.mark.parametrize(
    "rotation", [np.zeros((3, 3)), np.diag([1.0, 1.0, -1.0]), np.full((3, 3), np.nan)]
)
def test_ignored_calibration_frame_must_still_be_valid(rotation):
    malformed = case()
    malformed["controller_frame_rotation"] = rotation
    with pytest.raises(ValueError, match="rotation"):
        case_group_id(malformed)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"seed": True},
        {"seed": 1.2},
        {"train_fraction": 0},
        {"train_fraction": np.nan},
        {"validation_fraction": np.inf},
        {"validation_fraction": False},
        {"validation_fraction": -0.1},
        {"train_fraction": 0.8, "validation_fraction": 0.2},
    ],
)
def test_invalid_seed_and_fractions(kwargs):
    with pytest.raises(ValueError):
        split_cases(cases(3), **kwargs)


@pytest.mark.parametrize("source", [[], [case()], [case(), case()]])
def test_too_few_distinct_groups_rejected(source):
    with pytest.raises(ValueError, match="three distinct"):
        split_cases(source)
