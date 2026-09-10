"""Unit tests for the rotation-gain by force-budget interaction adapter.

The archived gain-1 traces must reproduce the published force-budget comparison,
phase and diagnostic metrics through the independent array summary. The gain-2
controller, its constructor document and the input rejections are checked without
running the physics loop.
"""

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from compliant_control_lab.online_compensation_experiment import (
    _case_document,
    _constructor_config,
)
from compliant_control_lab.surface_simulation import yaw_frame
from tools import audit_online_compensation_errors as audit
from tools import budget_gain_interaction as interaction
from tools import combined_residual_ablation as old
from tools import compensation_budget_study as study
from tools.cross_surface_regression import read_csv

ARCHIVE = Path(__file__).resolve().parents[1] / interaction.REFERENCE["directory"]
VARIANT, YAW = "combined", 0


@pytest.fixture(scope="module")
def archive():
    """Return the pinned gain-1 archive root after checking its manifest digest."""
    digest = hashlib.sha256((ARCHIVE / "manifest.json").read_bytes()).hexdigest()
    assert digest == interaction.REFERENCE["manifest_sha256"]
    assert (ARCHIVE / "COMPLETE").read_text().strip() == digest
    return ARCHIVE


@pytest.fixture(scope="module")
def study_case():
    matches = [
        case for variant, case in interaction.cases()
        if variant == VARIANT and case.scenario.wall_yaw_deg == YAW
    ]
    assert len(matches) == 1
    return matches[0]


def _published(archive, table, case, budget):
    return [
        row for row in read_csv(archive / f"{table}.csv")
        if row["variant"] == VARIANT
        and float(row["surface_yaw_deg"]) == case.scenario.wall_yaw_deg
        and float(row["max_force_n"]) == budget
    ]


def _archived_trace(archive, case, budget):
    with np.load(archive / interaction.reference_trace(VARIANT, case, budget),
                 allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def test_cases_and_documents_come_from_the_frozen_budget_screen():
    frozen = study.study_cases()
    assert len(frozen) == 6
    assert [variant for variant, _ in interaction.cases()] == [v for v, _ in frozen]
    assert [interaction.case_document(variant, case)
            for variant, case in interaction.cases()] == [
        {"variant": variant, **_case_document(case)} for variant, case in frozen
    ]


def test_reference_traces_are_relative_paths_inside_the_pinned_archive(archive):
    for variant, case in interaction.cases():
        for budget in study.BUDGETS:
            name = interaction.reference_trace(variant, case, budget)
            assert not Path(name).is_absolute()
            assert name == f"traces/{study.stem(variant, case, budget)}__diagnostic.npz"
            assert (archive / name).is_file()


@pytest.mark.parametrize("budget", study.BUDGETS)
def test_archived_gain_one_metrics_reproduce_the_published_tables(archive, study_case, budget):
    trace = _archived_trace(archive, study_case, budget)
    assert float(trace["rotation_gain_scale"]) == 1.
    assert float(trace["max_force_n"]) == budget
    overall, phases, diagnostics = interaction.metrics(trace, study_case, budget)

    published = _published(archive, "comparison", study_case, budget)
    assert len(published) == 1
    # The screened summary drops the flags that assume the shipped 6 N budget.
    assert set(study.DROPPED_METRICS).isdisjoint(overall)
    assert set(study.DROPPED_METRICS).isdisjoint(published[0])
    for name, value in overall.items():
        audit._cell(published[0], name, value, "overall")

    rows = {row["phase"]: row for row in _published(archive, "phase_metrics", study_case, budget)}
    assert [row["phase"] for row in phases] == [phase.name for phase in study_case.phases]
    assert set(rows) == {row["phase"] for row in phases}
    for row in phases:
        for name, value in row.items():
            audit._cell(rows[row["phase"]], name, value, "phase")

    windows = {row["window"]: row
               for row in _published(archive, "diagnostics", study_case, budget)}
    assert [row["window"] for row in diagnostics] == [name for name, _, _ in old.DIAGNOSTIC_WINDOWS]
    assert set(windows) == {row["window"] for row in diagnostics}
    for row in diagnostics:
        for name, value in row.items():
            audit._cell(windows[row["window"]], name, value, "diagnostic")


def test_overall_metrics_add_the_independent_array_diagnostics(archive, study_case):
    trace = _archived_trace(archive, study_case, 6.)
    overall, _, _ = interaction.metrics(trace, study_case, 6.)
    assert overall["max_uncapped_tangential_amplitude_n"] == float(np.max(
        trace["controller_coefficient_before_compute"] * trace["diagnostic_corrected_force_n"]))
    assert "minimum_reserved_torque_headroom_nm" in overall
    assert "diagnostic_amplitude_capped_pct" in overall
    assert overall["comparator_gates_evaluated"] == "no"
    # The archived diagnostic traces carry no observer scalars, and the matched
    # request difference is left to the caller.
    assert set(old.AUDIT_FIELDS).isdisjoint(overall)
    assert "max_matched_request_difference_n" not in overall


def test_compact_keeps_the_diagnostic_fields_and_needs_the_executed_budget(archive, study_case):
    trace = _archived_trace(archive, study_case, 8.)
    assert set(interaction.compact(trace)) == {*old.TRACE_FIELDS, "max_force_n"}
    assert float(interaction.compact(trace)["max_force_n"]) == 8.
    with pytest.raises(ValueError):
        interaction.compact({name: value for name, value in trace.items()
                             if name != "max_force_n"})


@pytest.mark.parametrize("budget", study.BUDGETS)
def test_gain_two_controller_scales_only_the_rotational_gains(study_case, budget):
    control = interaction.controller(study_case, 2., budget)
    document = interaction.controller_document(2., budget)
    assert document == _constructor_config(control._base)
    assert np.array_equal(control.frame.rotation, yaw_frame(
        study_case.task.yaw_deg + study_case.controller_yaw_error_deg).rotation)
    reference = interaction.controller_document(1., budget)
    nominal, scaled = reference["base"]["base"], document["base"]["base"]
    assert [name for name in reference if reference[name] != document[name]] == ["base"]
    assert [name for name in nominal if nominal[name] != scaled[name]] == [
        "rotational_stiffness", "rotational_damping"]
    assert scaled["rotational_stiffness"] == [
        2. * value for value in nominal["rotational_stiffness"]]
    assert scaled["rotational_damping"] == pytest.approx(
        [math.sqrt(2.) * value for value in nominal["rotational_damping"]])
    assert document["tangential"]["max_force"] == budget
    assert document["tangential"]["mode"] == study.ARM
    assert reference["tangential"] == document["tangential"]


def test_summary_protocol_reports_the_actual_gain_and_budget_bounds():
    for scale, budget in ((1., 6.), (2., 8.)):
        protocol = interaction._summary_protocol(scale, budget)
        online = protocol["controller_constructor_configurations"]["online"]
        assert online["safe_adaptive_base"] == interaction.controller_document(scale, budget)
        assert online["safe_adaptive_base"]["tangential"]["max_force"] == budget
        assert protocol["recovery"] == {
            "window_s": 0.25, "absolute_force_error_n": 1., "tangent_rmse_mm": 3.}
        assert protocol["identity"] == old.IDENTITY


@pytest.mark.parametrize("scale", [0., -1., 1.5, 3., 4., float("nan"), float("inf")])
def test_unlisted_gain_scales_are_rejected(study_case, scale):
    with pytest.raises(ValueError):
        interaction.controller(study_case, scale, 6.)
    with pytest.raises(ValueError):
        interaction.controller_document(scale, 6.)
    with pytest.raises(ValueError):
        interaction.run_trial(study_case, scale, 6.)


@pytest.mark.parametrize("scale", [True, False, "1", None, np.array([1.]), (1.,)])
def test_non_real_gain_scales_are_rejected(study_case, scale):
    with pytest.raises(TypeError):
        interaction.controller(study_case, scale, 6.)
    with pytest.raises(TypeError):
        interaction.run_trial(study_case, scale, 6.)


@pytest.mark.parametrize("budget", [0., 7., 9., float("nan"), float("inf")])
def test_unlisted_force_budgets_are_rejected(study_case, budget):
    with pytest.raises(ValueError):
        interaction.controller(study_case, 2., budget)
    with pytest.raises(ValueError):
        interaction.controller_document(2., budget)
    with pytest.raises(ValueError):
        interaction.run_trial(study_case, 2., budget)


@pytest.mark.parametrize("budget", [True, "6", None])
def test_non_real_force_budgets_are_rejected(study_case, budget):
    with pytest.raises(TypeError):
        interaction.controller(study_case, 2., budget)
    with pytest.raises(TypeError):
        interaction.run_trial(study_case, 2., budget)


def test_metrics_reject_a_trace_gain_outside_the_screen(archive, study_case):
    trace = _archived_trace(archive, study_case, 6.)
    trace["rotation_gain_scale"] = np.array(1.5)
    with pytest.raises(ValueError):
        interaction.metrics(trace, study_case, 6.)
    with pytest.raises(ValueError):
        interaction.metrics(_archived_trace(archive, study_case, 6.), study_case, 7.)
