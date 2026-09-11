"""Predeclared fail-closed screen for the measured-load budget pilot."""

from itertools import product

import numpy as np

from tools import budget_transfer_screen as transfer
from tools import compensation_budget_screen as budget

YAWS = (-15.0, 0.0, 15.0)
SCALES = (1.0, 2.0)
PUBLIC_CASES = (6, 7, 14, 15, 22, 23)
CRITERIA = budget.CRITERIA
PHASES = budget.PHASES
WINDOWS = budget.WINDOWS
PUBLIC_LIMITS = transfer.PUBLIC_LIMITS
GAIN_LIMITS = transfer.GAIN_LIMITS
ABSOLUTE = transfer.ABSOLUTE


def _number(row, name):
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid screening metric: {name}") from error
    if not np.isfinite(value):
        raise ValueError(f"nonfinite screening metric: {name}")
    return value


def _absolute(row, label):
    try:
        return [f"{label}_{name}" for name in transfer.absolute_failures(row)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {label} absolute metric") from error


def _named(rows, field, expected, label):
    try:
        sequence = list(rows)
    except TypeError as error:
        raise ValueError(f"{label} rows must be a sequence") from error
    indexed = {}
    for row in sequence:
        try:
            key = row[field]
        except (KeyError, TypeError) as error:
            raise ValueError(f"missing {label} identity: {field}") from error
        if key in indexed:
            raise ValueError(f"duplicate {label} identity")
        indexed[key] = row
    if set(indexed) != set(expected):
        raise ValueError(f"incomplete or unexpected {label} identities")
    return indexed


def _dynamic_rows(rows):
    indexed = {}
    for row in rows:
        try:
            if isinstance(row["surface_yaw_deg"], (bool, np.bool_)) or isinstance(
                row["rotation_gain_scale"], (bool, np.bool_)
            ):
                raise TypeError("dynamic identities must be numeric, not boolean")
            yaw = _number(row, "surface_yaw_deg")
            scale = _number(row, "rotation_gain_scale")
            variant = row["variant"]
            candidate, reference = row["candidate"], row["reference"]
            exercised = row["budget_exercised"]
        except (KeyError, TypeError) as error:
            raise ValueError("missing dynamic identity or payload") from error
        key = (yaw, scale)
        if key in indexed:
            raise ValueError("duplicate dynamic identity")
        if variant != "combined" or not isinstance(exercised, (bool, np.bool_)):
            raise ValueError("unexpected dynamic variant or budget flag")
        for label, payload in (("candidate", candidate), ("reference", reference)):
            try:
                overall = payload["overall"]
                phases = _named(payload["phases"], "phase", PHASES, f"{label} phase")
                windows = _named(payload["windows"], "window", WINDOWS, f"{label} window")
            except (KeyError, TypeError) as error:
                raise ValueError(f"missing {label} dynamic payload") from error
            if not isinstance(overall, dict):
                raise TypeError(f"invalid {label} overall payload")
            payload = {"overall": overall, "phases": phases, "windows": windows}
            if label == "candidate":
                candidate = payload
            else:
                reference = payload
        indexed[key] = {**row, "candidate": candidate, "reference": reference}
    if set(indexed) != set(product(YAWS, SCALES)):
        raise ValueError("incomplete or unexpected dynamic identities")
    return indexed


def _costs(candidate, reference):
    candidate_post = _number(candidate["windows"]["post"], "tangent_rmse_mm")
    reference_post = _number(reference["windows"]["post"], "tangent_rmse_mm")
    if reference_post <= 0.0:
        raise ValueError("positive reference post tangent RMSE required")
    force_cost = max(
        _number(candidate["phases"][phase], "force_rmse_n")
        - _number(reference["phases"][phase], "force_rmse_n")
        for phase in PHASES
    )
    pose_cost = max(
        _number(candidate["phases"][phase], "orientation_rmse_deg")
        - _number(reference["phases"][phase], "orientation_rmse_deg")
        for phase in PHASES
    )
    return {
        "candidate_post_tangent_rmse_mm": candidate_post,
        "reference_post_tangent_rmse_mm": reference_post,
        "post_tangent_reduction_pct": 100.0 * (1.0 - candidate_post / reference_post),
        "candidate_late_tangent_rmse_mm": _number(
            candidate["windows"]["late_post"], "tangent_rmse_mm"
        ),
        "worst_phase_force_rmse_increase_n": force_cost,
        "worst_phase_orientation_rmse_increase_deg": pose_cost,
        "full_raw_peak_increase_n": _number(candidate["overall"], "peak_force_n")
        - _number(reference["overall"], "peak_force_n"),
    }


def dynamic_screen(rows):
    """Screen six matched pilot pairs and three candidate-only gain interactions."""
    indexed = _dynamic_rows(rows)
    pairs = []
    minimum = CRITERIA["minimum_post_tangent_reduction_pct"]
    for yaw, scale in product(YAWS, SCALES):
        row = indexed[yaw, scale]
        candidate, reference = row["candidate"], row["reference"]
        costs = _costs(candidate, reference)
        failures = [
            *_absolute(reference["overall"], "reference"),
            *_absolute(candidate["overall"], "candidate"),
        ]
        checks = {
            "post_accuracy": costs["candidate_post_tangent_rmse_mm"]
            <= CRITERIA["maximum_post_tangent_rmse_mm"],
            "post_reduction": 100.0 * costs["candidate_post_tangent_rmse_mm"]
            <= (100.0 - minimum) * costs["reference_post_tangent_rmse_mm"],
            "late_accuracy": costs["candidate_late_tangent_rmse_mm"]
            <= CRITERIA["maximum_late_tangent_rmse_mm"],
            "phase_force_cost": costs["worst_phase_force_rmse_increase_n"]
            <= CRITERIA["maximum_phase_force_rmse_increase_n"],
            "phase_pose_cost": costs["worst_phase_orientation_rmse_increase_deg"]
            <= CRITERIA["maximum_phase_orientation_rmse_increase_deg"],
            "full_peak_cost": costs["full_raw_peak_increase_n"]
            <= CRITERIA["maximum_full_raw_peak_increase_n"],
            "budget_exercised": bool(row["budget_exercised"]),
        }
        failures.extend(name for name, passed in checks.items() if not passed)
        pairs.append(
            {
                "surface_yaw_deg": yaw,
                "rotation_gain_scale": scale,
                "variant": "combined",
                **costs,
                "failed_checks": failures,
                "status": "FAIL" if failures else "PASS",
            }
        )

    gain_pairs = []
    for yaw in YAWS:
        before, after = (indexed[yaw, scale]["candidate"] for scale in SCALES)
        costs = _costs(after, before)
        deltas = {
            "post_tangent_rmse_mm": costs["candidate_post_tangent_rmse_mm"]
            - costs["reference_post_tangent_rmse_mm"],
            "worst_phase_force_rmse_increase_n": costs[
                "worst_phase_force_rmse_increase_n"
            ],
            "worst_phase_orientation_rmse_increase_deg": costs[
                "worst_phase_orientation_rmse_increase_deg"
            ],
            "full_raw_peak_increase_n": costs["full_raw_peak_increase_n"],
        }
        failures = _absolute(after["overall"], "candidate_gain2")
        failures.extend(
            name for name, limit in GAIN_LIMITS.items() if deltas[name] > limit
        )
        gain_pairs.append(
            {
                "surface_yaw_deg": yaw,
                "variant": "combined",
                "delta_direction": "gain2_minus_gain1",
                **deltas,
                "failed_checks": failures,
                "status": "FAIL" if failures else "PASS",
            }
        )
    passed = sum(row["status"] == "PASS" for row in pairs)
    return {
        "criteria": dict(CRITERIA),
        "gain_limits": dict(GAIN_LIMITS),
        "absolute_limits": dict(ABSOLUTE),
        "pairs": pairs,
        "gain_pairs": gain_pairs,
        "passed_pairs": passed,
        "status": "PASS"
        if passed == 6 and all(row["status"] == "PASS" for row in gain_pairs)
        else "FAIL",
        "default_changed": False,
        "new_holdout": False,
    }


def public_screen(rows):
    """Screen the exact twelve public compatibility pairs against their old 6 N runs."""
    indexed = {}
    for row in rows:
        try:
            case = row["case_index"]
            if isinstance(case, (bool, np.bool_)) or isinstance(
                row["rotation_gain_scale"], (bool, np.bool_)
            ):
                raise TypeError("public identities must be numeric, not boolean")
            scale = _number(row, "rotation_gain_scale")
            candidate, reference = row["candidate"], row["reference"]
        except (KeyError, TypeError) as error:
            raise ValueError("missing public identity or payload") from error
        key = (case, scale)
        if key in indexed:
            raise ValueError("duplicate public identity")
        indexed[key] = (candidate, reference)
    if set(indexed) != set(product(PUBLIC_CASES, SCALES)):
        raise ValueError("incomplete or unexpected public identities")

    pairs = []
    for case, scale in product(PUBLIC_CASES, SCALES):
        candidate, reference = indexed[case, scale]
        deltas = {
            name: _number(candidate, name) - _number(reference, name)
            for name in PUBLIC_LIMITS
        }
        failures = [
            *_absolute(reference, "reference"),
            *_absolute(candidate, "candidate"),
            *(name for name, limit in PUBLIC_LIMITS.items() if deltas[name] > limit),
        ]
        pairs.append(
            {
                "case_index": case,
                "rotation_gain_scale": scale,
                "delta_direction": "candidate_minus_old_6n",
                **deltas,
                "failed_checks": failures,
                "status": "FAIL" if failures else "PASS",
            }
        )
    passed = sum(row["status"] == "PASS" for row in pairs)
    return {
        "public_limits": dict(PUBLIC_LIMITS),
        "absolute_limits": dict(ABSOLUTE),
        "pairs": pairs,
        "passed_pairs": passed,
        "status": "PASS" if passed == 12 else "FAIL",
        "default_changed": False,
        "new_holdout": False,
    }
