"""Full regression screen for scoped adoption of the measured-load preset."""

from itertools import product

import numpy as np

from tools import load_budget_screen as pilot

YAWS = pilot.YAWS
SCALES = pilot.SCALES
VARIANTS = ("combined", "no_bias")
PUBLIC_CASES = tuple(range(24))
CRITERIA = pilot.CRITERIA
GAIN_LIMITS = pilot.GAIN_LIMITS
PUBLIC_LIMITS = pilot.PUBLIC_LIMITS
ABSOLUTE = pilot.ABSOLUTE


def _dynamic_identity(row):
    try:
        if any(
            isinstance(row[name], (bool, np.bool_))
            for name in ("surface_yaw_deg", "rotation_gain_scale")
        ):
            raise TypeError("boolean dynamic identity")
        return (
            pilot._number(row, "surface_yaw_deg"),
            row["variant"],
            pilot._number(row, "rotation_gain_scale"),
        )
    except (KeyError, TypeError) as error:
        raise ValueError("missing or invalid dynamic identity") from error


def dynamic_screen(rows):
    """Apply the unchanged six-pair pilot screen independently to both variants."""
    indexed = {}
    for row in rows:
        key = _dynamic_identity(row)
        if key in indexed:
            raise ValueError("duplicate dynamic identity")
        indexed[key] = row
    expected = set(product(YAWS, VARIANTS, SCALES))
    if set(indexed) != expected:
        raise ValueError("incomplete or unexpected dynamic identities")

    pairs, gain_pairs = [], []
    variant_status = []
    for variant in VARIANTS:
        normalized = [
            {**indexed[yaw, variant, scale], "variant": "combined"}
            for yaw, scale in product(YAWS, SCALES)
        ]
        result = pilot.dynamic_screen(normalized)
        pairs.extend({**row, "variant": variant} for row in result["pairs"])
        gain_pairs.extend({**row, "variant": variant} for row in result["gain_pairs"])
        variant_status.append(result["status"])
    passed = sum(row["status"] == "PASS" for row in pairs)
    passed_gain = sum(row["status"] == "PASS" for row in gain_pairs)
    return {
        "criteria": dict(CRITERIA),
        "gain_limits": dict(GAIN_LIMITS),
        "absolute_limits": dict(ABSOLUTE),
        "pairs": pairs,
        "gain_pairs": gain_pairs,
        "passed_pairs": passed,
        "passed_gain_pairs": passed_gain,
        "status": "PASS"
        if passed == 12 and passed_gain == 6 and all(value == "PASS" for value in variant_status)
        else "FAIL",
        "default_changed": False,
        "new_holdout": False,
    }


def _public_identity(row):
    try:
        case = row["case_index"]
        raw_scale = row["rotation_gain_scale"]
        if (
            isinstance(case, (bool, np.bool_))
            or not isinstance(case, (int, np.integer))
            or isinstance(raw_scale, (bool, np.bool_))
        ):
            raise TypeError("invalid public identity type")
        return int(case), pilot._number(row, "rotation_gain_scale")
    except (KeyError, TypeError) as error:
        raise ValueError("missing or invalid public identity") from error


def public_screen(rows):
    """Apply unchanged public limits and two-arm absolutes to all 48 pairs."""
    indexed = {}
    for row in rows:
        key = _public_identity(row)
        if key in indexed:
            raise ValueError("duplicate public identity")
        try:
            indexed[key] = (row["candidate"], row["reference"])
        except (KeyError, TypeError) as error:
            raise ValueError("missing public candidate or reference") from error
    if set(indexed) != set(product(PUBLIC_CASES, SCALES)):
        raise ValueError("incomplete or unexpected public identities")

    pairs = []
    for case, scale in product(PUBLIC_CASES, SCALES):
        candidate, reference = indexed[case, scale]
        deltas = {
            name: pilot._number(candidate, name) - pilot._number(reference, name)
            for name in PUBLIC_LIMITS
        }
        failures = [
            *pilot._absolute(reference, "reference"),
            *pilot._absolute(candidate, "candidate"),
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
        "status": "PASS" if passed == 48 else "FAIL",
        "default_changed": False,
        "new_holdout": False,
    }


def screen(dynamic_rows, public_rows):
    """Require every full-regression pair before scoped experimental adoption."""
    dynamic = dynamic_screen(dynamic_rows)
    public = public_screen(public_rows)
    passed = dynamic["status"] == public["status"] == "PASS"
    return {
        "dynamic": dynamic,
        "public24": public,
        "status": "PASS" if passed else "FAIL",
        "eligible_for_scoped_adoption": passed,
        "global_default_changed": False,
        "new_holdout": False,
    }
