"""Predeclared engineering screen for matched 8 N versus 6 N development runs.

Passing supports a scoped simulation configuration, not a global default change,
an equal-budget algorithm comparison, hardware safety or unseen-data robustness.
"""

from itertools import product

import numpy as np

CRITERIA = {
    "minimum_contact_ratio_pct": 99.0,
    "maximum_full_raw_peak_n": 35.0,
    "required_clipping_pct": 0.0,
    "required_projection_pct": 0.0,
    "minimum_reserved_headroom_nm": 1.0,
    "maximum_post_tangent_rmse_mm": 3.0,
    "minimum_post_tangent_reduction_pct": 20.0,
    "maximum_late_tangent_rmse_mm": 3.0,
    "maximum_phase_force_rmse_increase_n": 0.2,
    "maximum_phase_orientation_rmse_increase_deg": 0.1,
    "maximum_full_raw_peak_increase_n": 1.0,
    "minimum_candidate_uncapped_amplitude_exclusive_n": 6.0,
}
YAWS = (-15, 0, 15)
VARIANTS = ("combined", "no_bias")
PHASES = ("pre_change", "combined_transition", "combined_post")
WINDOWS = ("post", "early_post", "late_post")


def _number(row, name):
    value = float(row[name])
    if not np.isfinite(value):
        raise ValueError(f"nonfinite screening metric: {name}")
    return value


def _index(rows, suffix=None):
    indexed = {}
    for row in rows:
        yaw = _number(row, "surface_yaw_deg")
        budget = _number(row, "max_force_n")
        key = (yaw, row["variant"], budget)
        if suffix:
            key += (row[suffix],)
        if key in indexed:
            raise ValueError("duplicate screening identity")
        indexed[key] = row
    extra = (PHASES if suffix == "phase" else WINDOWS,) if suffix else ()
    expected = set(product(YAWS, VARIANTS, (6., 8.), *extra))
    if set(indexed) != expected:
        raise ValueError("incomplete or unexpected screening identities")
    return indexed


def screen(rows, phases, diagnostics):
    """Return all six decisions; malformed or missing data never count as a pass."""
    whole = _index(rows)
    phase_rows = _index(phases, "phase")
    windows = _index(diagnostics, "window")
    decisions = []
    for yaw, variant in product(YAWS, VARIANTS):
        ref, candidate = (whole[yaw, variant, b] for b in (6., 8.))
        original, post = (windows[yaw, variant, b, "post"] for b in (6., 8.))
        late = windows[yaw, variant, 8., "late_post"]
        reference_error = _number(original, "tangent_rmse_mm")
        if reference_error <= 0:
            raise ValueError("positive reference RMSE required for relative reduction")
        improvement = 100 * (1 - _number(post, "tangent_rmse_mm") / reference_error)
        force_cost = max(_number(phase_rows[yaw, variant, 8., p], "force_rmse_n")
                         - _number(phase_rows[yaw, variant, 6., p], "force_rmse_n")
                         for p in PHASES)
        pose_cost = max(_number(phase_rows[yaw, variant, 8., p], "orientation_rmse_deg")
                        - _number(phase_rows[yaw, variant, 6., p], "orientation_rmse_deg")
                        for p in PHASES)
        peak_cost = _number(candidate, "peak_force_n") - _number(ref, "peak_force_n")
        failures = []
        for budget in (6., 8.):
            row = whole[yaw, variant, budget]
            checks = {
                "contact": _number(row, "contact_ratio_pct") >= CRITERIA["minimum_contact_ratio_pct"],
                "raw_peak": _number(row, "peak_force_n") <= CRITERIA["maximum_full_raw_peak_n"],
                "clipping": _number(row, "saturation_pct") == CRITERIA["required_clipping_pct"],
                "projection": _number(row, "projection_pct") == CRITERIA["required_projection_pct"],
                "reserved_headroom": _number(row, "minimum_reserved_torque_headroom_nm")
                >= CRITERIA["minimum_reserved_headroom_nm"],
            }
            failures.extend(f"{int(budget)}n_{name}" for name, passed in checks.items() if not passed)
        checks = {
            "post_accuracy": _number(post, "tangent_rmse_mm")
            <= CRITERIA["maximum_post_tangent_rmse_mm"],
            "post_reduction": improvement >= CRITERIA["minimum_post_tangent_reduction_pct"],
            "late_accuracy": _number(late, "tangent_rmse_mm")
            <= CRITERIA["maximum_late_tangent_rmse_mm"],
            "phase_force_cost": force_cost <= CRITERIA["maximum_phase_force_rmse_increase_n"],
            "phase_pose_cost": pose_cost <= CRITERIA["maximum_phase_orientation_rmse_increase_deg"],
            "full_peak_cost": peak_cost <= CRITERIA["maximum_full_raw_peak_increase_n"],
            "budget_exercised": _number(candidate, "max_uncapped_tangential_amplitude_n")
            > CRITERIA["minimum_candidate_uncapped_amplitude_exclusive_n"]
            and _number(candidate, "max_matched_request_difference_n") > 1e-9,
        }
        failures.extend(name for name, passed in checks.items() if not passed)
        decisions.append({
            "surface_yaw_deg": yaw, "variant": variant,
            "post_tangent_reduction_pct": improvement,
            "reference_post_tangent_rmse_mm": reference_error,
            "candidate_post_tangent_rmse_mm": _number(post, "tangent_rmse_mm"),
            "candidate_late_tangent_rmse_mm": _number(late, "tangent_rmse_mm"),
            "worst_phase_force_rmse_increase_n": force_cost,
            "worst_phase_orientation_rmse_increase_deg": pose_cost,
            "full_raw_peak_increase_n": peak_cost,
            "failed_checks": failures,
            "status": "FAIL" if failures else "PASS",
        })
    return {
        "criteria": CRITERIA,
        "pairs": decisions,
        "passed_pairs": sum(row["status"] == "PASS" for row in decisions),
        "status": "PASS" if all(row["status"] == "PASS" for row in decisions) else "FAIL",
        "default_changed": False,
        "scope": "public-development parameter-budget feasibility only",
    }
