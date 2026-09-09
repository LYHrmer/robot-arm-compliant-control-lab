"""Structural tests for the bounded-compensation force-budget screen.

These cover the fixed protocol, the controller construction scope, the paired
table helper and the early output rejections; no test runs the physics loop.
"""

import pytest

from tools import combined_residual_ablation as old
from tools import compensation_budget_study as study

WINDOWS = [name for name, _, _ in old.DIAGNOSTIC_WINDOWS]


def _diagnostic_rows(offset=1.0):
    rows = []
    for yaw in study.YAWS:
        for variant in study.VARIANTS:
            for window in WINDOWS:
                for budget in study.BUDGETS:
                    metrics = {
                        name: float(index) + (offset if budget == 8. else 0.)
                        for index, name in enumerate(old.PAIRED_DIAGNOSTICS)
                    }
                    rows.append({"surface_yaw_deg": float(yaw), "variant": variant,
                                 "window": window, "max_force_n": budget, **metrics})
    return rows


def test_twelve_runs_have_unique_budget_aware_identities():
    runs = study.trials()
    assert len(runs) == 12
    identities = [study.identity(variant, case, budget) for variant, case, budget in runs]
    assert all(tuple(row) == study.KEYS for row in identities)
    keys = [tuple(row.values()) for row in identities]
    assert len(set(keys)) == 12
    assert {row["max_force_n"] for row in identities} == set(study.BUDGETS)
    assert {row["variant"] for row in identities} == set(study.VARIANTS)
    assert {int(row["surface_yaw_deg"]) for row in identities} == set(study.YAWS)
    assert [row["max_force_n"] for row in identities[:2]] == [6., 8.]
    stems = [study.stem(variant, case, budget) for variant, case, budget in runs]
    assert len(set(stems)) == 12
    assert stems[0].endswith("__f6n") and stems[1].endswith("__f8n")


def test_only_the_force_budget_changes_in_the_controller_config():
    low, high = (study.controller_document(budget) for budget in study.BUDGETS)
    assert low["max_force_n"] == 6. and high["max_force_n"] == 8.
    base, other = low["safe_adaptive_base"], high["safe_adaptive_base"]
    assert set(base) == set(other)
    assert [key for key in base if base[key] != other[key]] == ["tangential"]
    assert base["tangential"]["max_force"] == 6. and other["tangential"]["max_force"] == 8.
    assert [
        key for key in base["tangential"]
        if base["tangential"][key] != other["tangential"][key]
    ] == ["max_force"]
    assert low["rotation_gain_scale"] == high["rotation_gain_scale"] == 1.
    assert low["tangential_mode"] == high["tangential_mode"] == old.ARMS[study.ARM]


def test_shipped_default_budget_is_still_six_newtons():
    _, case = study.study_cases()[0]
    controller = study.make_controller(case, 8.)
    assert study.DEFAULT_MAX_FORCE_N == 6.
    assert controller._base.tangential.max_force == 8.
    assert study.make_controller(case, 6.)._base.tangential.max_force == 6.
    fresh = old.SurfaceAdaptiveController(
        old.yaw_frame(0.0), tangential_mode=old.ARMS[study.ARM], rotation_gain_scale=1.,
    )
    assert fresh._base.tangential.max_force == 6.


@pytest.mark.parametrize("budget", [True, False, "6", None, 7., 0., float("nan"), float("inf"), -6.])
def test_invalid_force_budgets_are_rejected(budget):
    variant, case = study.study_cases()[0]
    with pytest.raises(TypeError if isinstance(budget, (bool, str)) or budget is None else ValueError):
        study.make_controller(case, budget)
    with pytest.raises(TypeError if isinstance(budget, (bool, str)) or budget is None else ValueError):
        study.controller_document(budget)
    with pytest.raises(TypeError if isinstance(budget, (bool, str)) or budget is None else ValueError):
        study.stem(variant, case, budget)


def test_protocol_document_freezes_the_twelve_cases_and_both_controllers():
    protocol = study.protocol_document()
    assert protocol["identity"] == "compensation-budget-screen-v1"
    assert protocol["budgets_n"] == [6., 8.] and protocol["variants"] == ["combined", "no_bias"]
    assert protocol["new_holdout"] is False and protocol["default_changed"] is False
    assert protocol["comparator_gates_evaluated"] is False
    assert protocol["gates_not_evaluated"] == list(old.NOT_EVALUATED_GATES)
    assert protocol["reference"]["directory"] == "results/franka_combined_residual_ablation"
    assert [window["name"] for window in protocol["diagnostic_windows"]] == WINDOWS
    assert [c["max_force_n"] for c in protocol["controllers"]] == [6., 8.]
    assert len(protocol["cases"]) == 12
    assert len({tuple(case[key] for key in study.KEYS) for case in protocol["cases"]}) == 12
    assert protocol["legacy_metrics_dropped"] == ["compensation_limit_observed"]
    assert "max_force_n" in protocol["trace_fields"]


def test_legacy_summary_fields_are_dropped_and_normalized():
    metrics = study.screen_metrics({
        "compensation_limit_observed": "yes",
        "tangent_recovery_status": "not_recovered_with_limit_active",
        "tangent_rmse_mm": 1.5,
    })
    assert "compensation_limit_observed" not in metrics
    assert metrics == {"tangent_recovery_status": "not_recovered", "tangent_rmse_mm": 1.5}
    assert study.screen_metrics({"tangent_recovery_status": "recovered"}) == {
        "tangent_recovery_status": "recovered"
    }


def test_case_inputs_screen_and_source_identity_are_predeclared():
    protocol = study.protocol_document()
    assert protocol["co_primary_late_window"] == "late_post"
    assert protocol["screening_criteria"]["maximum_phase_force_rmse_increase_n"] == .2
    assert protocol["screening_criteria"]["maximum_phase_orientation_rmse_increase_deg"] == .1
    assert protocol["screening_criteria"]["minimum_post_tangent_reduction_pct"] == 20.
    assert protocol["screening_criteria"]["minimum_reserved_headroom_nm"] == 1.
    for variant, case in study.study_cases():
        original = next(c for v, c in old.ablation_cases()
                        if v == variant and c.scenario.wall_yaw_deg == case.scenario.wall_yaw_deg)
        assert old._case_document(case) == old._case_document(original)
    hashes = study.source_identity()
    assert all(hashes[name] == digest for name, digest in old.source_identity().items())
    for name in ("compensation_budget_study.py", "compensation_budget_validation.py",
                 "compensation_budget_screen.py"):
        assert len(hashes[f"tools/{name}"]) == 64


def test_paired_rows_report_eight_minus_six_newton_deltas():
    pairs = study.diagnostic_pairs(_diagnostic_rows(offset=2.0))
    assert len(pairs) == len(study.YAWS) * len(study.VARIANTS) * len(WINDOWS)
    assert {pair["delta_direction"] for pair in pairs} == {"budget_8n_minus_6n"}
    for pair in pairs:
        assert pair["max_force_n"] == 8. and pair["reference_max_force_n"] == 6.
        for name in old.PAIRED_DIAGNOSTICS:
            assert pair[f"delta_{name}"] == pytest.approx(2.0)
    negative = study.diagnostic_pairs(_diagnostic_rows(offset=-3.0))
    assert all(
        pair[f"delta_{name}"] == pytest.approx(-3.0)
        for pair in negative for name in old.PAIRED_DIAGNOSTICS
    )


def test_paired_helper_rejects_missing_rows():
    rows = _diagnostic_rows()
    assert len(rows) == 36
    with pytest.raises(ValueError, match="missing diagnostic rows"):
        study.diagnostic_pairs(rows[:-1])


def test_paired_helper_rejects_duplicate_and_unexpected_rows():
    rows = _diagnostic_rows()
    with pytest.raises(ValueError, match="duplicate diagnostic row"):
        study.diagnostic_pairs([*rows, rows[0]])
    with pytest.raises(ValueError, match="unexpected diagnostic row"):
        study.diagnostic_pairs([*rows, {**rows[0], "max_force_n": 7.}])
    with pytest.raises(ValueError, match="unexpected diagnostic row"):
        study.diagnostic_pairs([*rows, {**rows[0], "variant": "no_yaw_error"}])


def test_generate_rejects_existing_output_and_symlinked_paths_before_physics(tmp_path):
    existing = tmp_path / "report"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        study.generate(existing)
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        study.generate(link / "screen")
    assert not (tmp_path / "screen").exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["link", "report"]
