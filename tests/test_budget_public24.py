import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import budget_public24 as budget
from tools import rotation_gain_regression as rotation

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / budget.REFERENCE["directory"]


def _protocol():
    return json.loads((ARCHIVE / "protocol.json").read_text(encoding="utf-8"))


def _published():
    with (ARCHIVE / "comparison.csv").open(newline="", encoding="utf-8") as handle:
        return {
            (float(row["scale"]), row["method"], int(row["case_index"])): row
            for row in csv.DictReader(handle)
        }


def _archived_trace(case, scale):
    path = ARCHIVE / budget.reference_trace(case, scale)
    assert not path.is_symlink()
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def test_archive_resolves_in_this_worktree_and_still_matches_the_pinned_manifest():
    assert ARCHIVE.is_dir() and not ARCHIVE.is_symlink()
    assert ARCHIVE.parents[1] == REPO
    digest = hashlib.sha256((ARCHIVE / "manifest.json").read_bytes()).hexdigest()
    assert digest == budget.REFERENCE["manifest_sha256"]
    assert (ARCHIVE / "COMPLETE").read_text(encoding="utf-8").strip() == digest


def test_cases_are_the_original_twenty_four_archived_case_documents():
    cases = budget.cases()
    assert [case["case_index"] for case in cases] == list(range(24))
    assert [budget.case_document(case) for case in cases] == _protocol()["configurations"]

    groups = {}
    for case in cases:
        scenario, config = case["scenario"], case["config"]
        key = (scenario.wall_yaw_deg, scenario.wall_time_constant, scenario.tool_mass_kg)
        groups.setdefault(key, set()).add(config.seed)
        assert (config.contact_model, config.duration) == ("smooth", 4.5)
        assert case["task"].yaw_deg == scenario.wall_yaw_deg
    assert len(groups) == 12
    assert all(seeds == {11, 29} for seeds in groups.values())


def test_case_document_is_plain_json():
    document = budget.case_document(budget.cases()[0])
    assert json.loads(json.dumps(document)) == document
    assert set(document) == {"case_index", "scenario", "config", "task"}


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_default_budget_document_is_exactly_the_archived_online_constructor(scale):
    assert budget.DEFAULT_MAX_FORCE_N == 6.0
    archived = _protocol()["controller_parameters"][str(int(scale))]["online"]
    assert archived["tangential"]["max_force"] == 6.0
    assert budget.controller_document(scale, 6.0) == archived


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_eight_newton_document_changes_only_the_compensation_force_budget(scale):
    default = budget.controller_document(scale, 6.0)
    raised = budget.controller_document(scale, 8.0)
    assert raised["tangential"]["max_force"] == 8.0
    assert raised["tangential"] | {"max_force": 6.0} == default["tangential"]
    assert {k: v for k, v in raised.items() if k != "tangential"} == {
        k: v for k, v in default.items() if k != "tangential"
    }


@pytest.mark.parametrize("limit", budget.BUDGETS_N)
def test_scale_two_changes_only_the_rotational_gains_at_a_fixed_budget(limit):
    gains = ("rotational_stiffness", "rotational_damping")
    one = budget.controller_document(1.0, limit)["base"]["base"]
    two = budget.controller_document(2.0, limit)["base"]["base"]
    np.testing.assert_allclose(two[gains[0]], [40.0] * 3)
    np.testing.assert_allclose(two[gains[1]], [5.0 * np.sqrt(2.0)] * 3)
    assert {k: v for k, v in two.items() if k not in gains} == {
        k: v for k, v in one.items() if k not in gains
    }


def test_controller_is_a_fresh_online_arm_bound_to_the_requested_budget():
    control = budget.controller(2.0, 8.0)
    compensation = control._base.tangential
    assert compensation.mode == "online"
    assert compensation.max_force == 8.0
    assert control.equivalent_tangential_coefficient == compensation.nominal_mu
    np.testing.assert_array_equal(control.requested_tangential_force_world, np.zeros(3))
    assert compensation is not budget.controller(2.0, 8.0)._base.tangential


def test_budget_replacement_requires_the_unchanged_six_newton_default(monkeypatch):
    monkeypatch.setattr(budget, "DEFAULT_MAX_FORCE_N", 7.0)
    with pytest.raises(ValueError, match="no longer 7.0 N"):
        budget.controller(1.0, 6.0)


@pytest.mark.parametrize("scale", [0.0, -1.0, 1.5, 3.0, np.nan, np.inf, True, "1", None])
def test_invalid_rotation_gain_scales_are_rejected_before_any_trial(scale):
    with pytest.raises((TypeError, ValueError)):
        budget.controller(scale, 6.0)
    with pytest.raises((TypeError, ValueError)):
        budget.run_trial(None, scale, 6.0)
    with pytest.raises((TypeError, ValueError)):
        budget.reference_trace(None, scale)


@pytest.mark.parametrize("limit", [0.0, -6.0, 5.0, 7.0, 12.0, np.nan, np.inf, True, "6", None])
def test_invalid_force_budgets_are_rejected_before_any_trial(limit):
    with pytest.raises((TypeError, ValueError)):
        budget.controller(1.0, limit)
    with pytest.raises((TypeError, ValueError)):
        budget.run_trial(None, 1.0, limit)


def test_reference_trace_names_the_archived_default_budget_compact_traces():
    for case in budget.cases():
        for scale in budget.SCALES:
            name = budget.reference_trace(case, scale)
            index = case["case_index"]
            assert name == f"traces/s{int(scale)}__online__case_{index:02d}__compact.npz"
            assert (ARCHIVE / name).is_file()


def test_compact_keeps_the_frozen_fields_plus_the_executed_budget_metadata():
    case = budget.cases()[0]
    trace = {**_archived_trace(case, 2.0), "extra": np.zeros(3), "max_force_n": np.array(8.0)}
    result = budget.compact(trace)
    assert budget.TRACE_METADATA == ("rotation_gain_scale", "case_index", "method", "max_force_n")
    assert set(result) == {*rotation.COMPACT_FIELDS, *budget.TRACE_METADATA}
    for name, expected, kinds in (
        ("rotation_gain_scale", 2.0, "f"),
        ("case_index", 0, "iu"),
        ("max_force_n", 8.0, "f"),
    ):
        assert result[name].shape == () and result[name].dtype.kind in kinds
        assert result[name].item() == expected
    assert result["method"].item() == "online"
    for name in rotation.COMPACT_FIELDS:
        np.testing.assert_array_equal(result[name], trace[name])


def test_compact_rejects_a_trace_without_the_executed_budget():
    with pytest.raises(ValueError, match="max_force_n"):
        budget.compact(_archived_trace(budget.cases()[0], 1.0))


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_archived_case_zero_metrics_are_reconstructed_at_both_gains(scale):
    case = budget.cases()[0]
    values = budget.metrics(_archived_trace(case, scale), case)
    row = _published()[scale, "online", 0]
    assert set(values) == {"has_raw_contact", *rotation.ALL_METRICS}
    assert str(values["has_raw_contact"]) == row["has_raw_contact"]
    for name in rotation.ALL_METRICS:
        expected = float(row[name]) if row[name] else None
        if expected is None:
            assert values[name] is None
        else:
            assert abs(values[name] - expected) <= 1e-10
