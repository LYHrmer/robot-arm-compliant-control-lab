"""Predeclared compatibility decisions; no tuning or global-default promotion."""

from itertools import product

import numpy as np

from tools import compensation_budget_screen as budget_screen
from tools import compensation_budget_study as budget_study
from tools import rotation_gain_regression as rotation

PUBLIC_LIMITS = {
    "tangent_rmse_mm": .10,
    "tangent_velocity_error_rms_m_s": .0002,
    "force_rmse_n": .20,
    "orientation_rmse_deg": .20,
    "peak_force_n": 1.,
}
GAIN_LIMITS = {
    "post_tangent_rmse_mm": .10,
    "worst_phase_force_rmse_increase_n": .20,
    "worst_phase_orientation_rmse_increase_deg": .10,
    "full_raw_peak_increase_n": 1.,
}
ABSOLUTE = {"contact_ratio_pct": 99., "peak_force_n": 35.,
            "saturation_pct": 0., "projection_pct": 0.,
            "minimum_reserved_torque_headroom_nm": 1.}


def number(row, name):
    value = float(row[name])
    if not np.isfinite(value):
        raise ValueError(f"nonfinite metric: {name}")
    return value


def index(rows, fields, expected):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in fields)
        if key in result:
            raise ValueError("duplicate transfer identity")
        result[key] = row
    if set(result) != set(expected):
        raise ValueError("incomplete or unexpected transfer identities")
    return result


def absolute_failures(row):
    values = {key: number(row, key) for key in ABSOLUTE}
    checks = {
        "contact": values["contact_ratio_pct"] >= ABSOLUTE["contact_ratio_pct"],
        "peak": values["peak_force_n"] <= ABSOLUTE["peak_force_n"],
        "clipping": values["saturation_pct"] == 0.,
        "projection": values["projection_pct"] == 0.,
        "reserve": values["minimum_reserved_torque_headroom_nm"] >= 1.,
    }
    return [key for key, passed in checks.items() if not passed]


def decision(context, deltas, limits, failures=()):
    failed = [*failures, *(key for key, limit in limits.items() if deltas[key] > limit)]
    return {**context, **deltas, "failed_checks": failed,
            "status": "FAIL" if failed else "PASS"}


def public_screen(rows):
    lookup = index(rows, ("case_index", "rotation_gain_scale", "max_force_n"),
                   product(range(24), (1., 2.), (6., 8.)))
    pairs = []
    for case, scale in product(range(24), (1., 2.)):
        ref, candidate = (lookup[case, scale, b] for b in (6., 8.))
        deltas = {m: number(candidate, m) - number(ref, m) for m in PUBLIC_LIMITS}
        # Retain every numeric metric delta, including costs without a pass/fail gate.
        other = {f"delta_{m}": (number(candidate, m) - number(ref, m)
                                if candidate[m] is not None and ref[m] is not None else None)
                 for m in rotation.ALL_METRICS if m not in PUBLIC_LIMITS}
        pairs.append(decision({"case_index": case, "rotation_gain_scale": scale,
                               "delta_direction": "8n_minus_6n", **other}, deltas,
                              PUBLIC_LIMITS, absolute_failures(candidate)))
    groups = []
    # Original index pairs differ only in the measurement-noise seed.
    for physical, scale in product(range(12), (1., 2.)):
        selected = [p for p in pairs if p["case_index"] // 2 == physical
                    and p["rotation_gain_scale"] == scale]
        first = lookup[physical * 2, scale, 6.]
        groups.append({"physical_group": physical, "rotation_gain_scale": scale,
                       **{k: first[k] for k in ("wall_yaw_deg", "wall_time_constant_s", "tool_mass_kg")},
                       "case_indices": [p["case_index"] for p in selected],
                       "worst_deltas": {m: max(p[m] for p in selected) for m in PUBLIC_LIMITS},
                       "status": "PASS" if all(p["status"] == "PASS" for p in selected) else "FAIL"})
    return {"pairs": pairs, "physical_groups": groups,
            "passed_pairs": sum(p["status"] == "PASS" for p in pairs),
            "status": "PASS" if all(p["status"] == "PASS" for p in pairs) else "FAIL"}


def budget_decisions(rows, phases, diagnostics):
    result = budget_screen.screen(rows, phases, diagnostics)
    # Keep the frozen historical implementation untouched. For this new screen,
    # compare the inclusive reduction boundary without cancellation in 1 - a/b.
    # E.g. 2.0 versus 2.5 is exactly 20%, but the displayed ratio rounds below 20.
    minimum = budget_screen.CRITERIA["minimum_post_tangent_reduction_pct"]
    for pair in result["pairs"]:
        passes = pair["candidate_post_tangent_rmse_mm"] * 100 <= (
            pair["reference_post_tangent_rmse_mm"] * (100 - minimum))
        failures = [f for f in pair["failed_checks"] if f != "post_reduction"]
        if not passes:
            failures.append("post_reduction")
        pair["failed_checks"] = failures
        pair["status"] = "FAIL" if failures else "PASS"
    result["passed_pairs"] = sum(p["status"] == "PASS" for p in result["pairs"])
    result["status"] = "PASS" if result["passed_pairs"] == 6 else "FAIL"
    return result


def dynamic_screen(rows, phases, diagnostics):
    grid = tuple(product((-15, 0, 15), ("combined", "no_bias"), (1., 2.), (6., 8.)))
    keys = ("surface_yaw_deg", "variant", "rotation_gain_scale", "max_force_n")
    whole = index(rows, keys, grid)
    phase = index(phases, (*keys, "phase"),
                  (k + (p,) for k in grid for p in budget_screen.PHASES))
    windows = index(diagnostics, (*keys, "window"),
                    (k + (w,) for k in grid for w in budget_screen.WINDOWS))
    budgets = {str(int(s)): budget_decisions(
        [r for r in rows if r["rotation_gain_scale"] == s],
        [r for r in phases if r["rotation_gain_scale"] == s],
        [r for r in diagnostics if r["rotation_gain_scale"] == s]) for s in (1., 2.)}
    gains, interactions = [], []
    for yaw, variant in product((-15, 0, 15), ("combined", "no_bias")):
        for budget in (6., 8.):
            before, after = (whole[yaw, variant, s, budget] for s in (1., 2.))
            costs = {"post_tangent_rmse_mm":
                     number(windows[yaw, variant, 2., budget, "post"], "tangent_rmse_mm")
                     - number(windows[yaw, variant, 1., budget, "post"], "tangent_rmse_mm"),
                     "full_raw_peak_increase_n": number(after, "peak_force_n") - number(before, "peak_force_n")}
            for label, metric in (("force_rmse_increase_n", "force_rmse_n"),
                                  ("orientation_rmse_increase_deg", "orientation_rmse_deg")):
                costs[f"worst_phase_{label}"] = max(
                    number(phase[yaw, variant, 2., budget, p], metric)
                    - number(phase[yaw, variant, 1., budget, p], metric)
                    for p in budget_screen.PHASES)
            gains.append(decision({"surface_yaw_deg": yaw, "variant": variant,
                                   "max_force_n": budget, "delta_direction": "gain2_minus_gain1"},
                                  costs, GAIN_LIMITS, absolute_failures(after)))
        effects = {}
        for window, metric in product(budget_screen.WINDOWS,
                                      ("tangent_rmse_mm", "normal_force_error_rms_n", "orientation_error_rms_deg")):
            values = {(s, b): number(windows[yaw, variant, s, b, window], metric)
                      for s in (1., 2.) for b in (6., 8.)}
            effects[f"{window}_{metric}"] = ((values[2., 8.] - values[2., 6.])
                                             - (values[1., 8.] - values[1., 6.]))
        interactions.append({"surface_yaw_deg": yaw, "variant": variant,
                             "definition": "(8n-6n)_gain2 - (8n-6n)_gain1", **effects})
    paired = [{"rotation_gain_scale": s, **row} for s in (1., 2.)
              for row in budget_study.diagnostic_pairs([r for r in diagnostics if r["rotation_gain_scale"] == s])]
    return {"budget_by_gain": budgets, "paired_diagnostics": paired, "gain_pairs": gains,
            "passed_gain_pairs": sum(p["status"] == "PASS" for p in gains),
            "descriptive_interactions": interactions,
            "status": "PASS" if all(s["status"] == "PASS" for s in budgets.values())
            and all(p["status"] == "PASS" for p in gains) else "FAIL"}


def screen(public, dynamic, phases, diagnostics):
    p, d = public_screen(public), dynamic_screen(dynamic, phases, diagnostics)
    return {"public24": p, "dynamic": d,
            "status": "PASS" if p["status"] == d["status"] == "PASS" else "FAIL",
            "default_changed": False, "new_holdout": False}
