"""Bounded-compensation force-budget screen of the online modest_combined arm.

The twelve runs reuse the frozen input-ablation tooling: the same preselected
seed-11 cases at the three surface yaws, the same observed control loop, window
diagnostics and per-run absolute physical checks. The only input that changes is
the tangential compensation force budget (6 N, the current default, and 8 N), so
the original paired-baseline and paired-friction comparator gates are again not
evaluated here and no default is changed; the tables report raw metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
import os
import platform
import tempfile
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import numpy as np

from compliant_control_lab.online_compensation_experiment import (
    ARMS,
    GATES,
    ScheduledSurfaceSimulator,
    _case_document,
    _constructor_config,
    _save_trace,
    _sha256,
    _write_csv,
    _write_json,
    summarize_trial,
)
from compliant_control_lab.surface_control import SurfaceAdaptiveController
from compliant_control_lab.surface_simulation import yaw_frame
from tools import combined_residual_ablation as old
from tools import (
    combined_residual_diagnostics,
    compensation_budget_screen,
    cross_surface_regression,
)
from tools import compensation_budget_validation as validator

IDENTITY = "compensation-budget-screen-v1"
BUDGETS = (6., 8.)
VARIANTS = ("combined", "no_bias")
YAWS = (-15, 0, 15)
ARM = old.ARM
SEED = old.SEED
ROTATION_GAIN_SCALE = old.ROTATION_GAIN_SCALE
DEFAULT_MAX_FORCE_N = 6.
BASELINE_BUDGET_N = 6.
KEYS = (*old.KEYS, "max_force_n")
TRACE_FIELDS = (*old.TRACE_FIELDS, "max_force_n")
DELTA_DIRECTION = "budget_8n_minus_6n"
# summarize_trial builds these two fields against the 6 N default budget, so they
# would silently describe the wrong limit on the 8 N runs.
DROPPED_METRICS = ("compensation_limit_observed",)
NORMALIZED_STATUS = {"not_recovered_with_limit_active": "not_recovered"}
REFERENCE = {
    "directory": "results/franka_combined_residual_ablation",
    "manifest_sha256": "95f91febee9f4ff59b6cb159618d529835f7d8a86c7a73371a67b34b7fd36132",
}


def verify_reference():
    root = Path(__file__).resolve().parents[1] / REFERENCE["directory"]
    manifest = root / "manifest.json"
    digest = REFERENCE["manifest_sha256"]
    if _sha256(manifest) != digest or (root / "COMPLETE").read_text().strip() != digest:
        raise ValueError("reference manifest differs")
    for name, expected in json.loads(manifest.read_text())["artifact_sha256"].items():
        relative = Path(name)
        path = root / relative
        if relative.is_absolute() or ".." in relative.parts or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("unsafe reference artifact")
        if _sha256(path) != expected:
            raise ValueError(f"reference artifact differs: {name}")
    return root


def check_reference(variant: str, case, trace: dict, reference) -> None:
    """Require the 6 N run to reproduce the archived ablation trace field by field."""
    path = reference / "traces" / f"{old.stem(variant, case)}__diagnostic.npz"
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in old.TRACE_FIELDS if name not in archive.files]
        if missing:
            raise ValueError(f"reference trace missing fields: {missing}")
        for name in old.TRACE_FIELDS:
            if not np.array_equal(np.asarray(trace[name]), archive[name]):
                raise ValueError(f"default-budget run differs from reference: {name}")


def _budget(value) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"force budget must be a real number: {value!r}")
    number = float(value)
    if not math.isfinite(number) or number not in BUDGETS:
        raise ValueError(f"force budget must be one of {BUDGETS} N: {value!r}")
    return number


def _with_budget(controller, max_force_n: float):
    """Replace only max_force on the fresh online compensation of this controller."""
    compensation = controller._base.tangential
    if compensation is None or compensation.mode != ARM:
        raise ValueError("the force-budget screen requires the online compensation arm")
    if float(compensation.max_force) != DEFAULT_MAX_FORCE_N:
        raise ValueError("the default compensation force budget is no longer 6 N")
    controller._base.tangential = replace(compensation, max_force=max_force_n)
    return controller


def make_controller(case, max_force_n):
    budget = _budget(max_force_n)
    return _with_budget(
        SurfaceAdaptiveController(
            yaw_frame(case.task.yaw_deg + case.controller_yaw_error_deg),
            tangential_mode=ARMS[ARM],
            rotation_gain_scale=ROTATION_GAIN_SCALE,
        ),
        budget,
    )


def controller_document(max_force_n) -> dict:
    budget = _budget(max_force_n)
    control = _with_budget(
        SurfaceAdaptiveController(
            yaw_frame(0.0), tangential_mode=ARMS[ARM], rotation_gain_scale=ROTATION_GAIN_SCALE,
        ),
        budget,
    )
    return {
        "arm": ARM,
        "tangential_mode": ARMS[ARM],
        "rotation_gain_scale": ROTATION_GAIN_SCALE,
        "max_force_n": budget,
        "default_max_force_n": DEFAULT_MAX_FORCE_N,
        "safe_adaptive_base": _constructor_config(control._base),
    }


def study_cases() -> list[tuple[str, object]]:
    """Return the six (variant, ProtocolCase) pairs kept from the frozen ablation list."""
    cases = [(variant, case) for variant, case in old.ablation_cases() if variant in VARIANTS]
    identities = {(case.scenario.wall_yaw_deg, variant) for variant, case in cases}
    if len(cases) != len(YAWS) * len(VARIANTS) or identities != {
        (yaw, variant) for yaw in YAWS for variant in VARIANTS
    }:
        raise ValueError("expected one combined and one no_bias case per surface yaw")
    return cases


def trials() -> list[tuple[str, object, float]]:
    """Return the fixed run order: surface yaw, then variant, then 6 N before 8 N."""
    return [(variant, case, budget) for variant, case in study_cases() for budget in BUDGETS]


def identity(variant: str, case, max_force_n) -> dict:
    return {**old.identity(variant, case), "max_force_n": _budget(max_force_n)}


def stem(variant: str, case, max_force_n) -> str:
    return f"{old.stem(variant, case)}__f{int(_budget(max_force_n))}n"


def run_trial(case, max_force_n):
    """Run one case at one force budget on the fixed online arm at gain scale 1."""
    budget = _budget(max_force_n)
    frame = yaw_frame(case.task.yaw_deg + case.controller_yaw_error_deg)
    simulator = ScheduledSurfaceSimulator(case, frame, ARM)
    result = old.run_observed_loop(case, make_controller(case, budget), simulator)
    result.trace["max_force_n"] = np.array(budget)
    return result


def diagnostic_trace(trace: dict) -> dict:
    if "max_force_n" not in trace:
        raise ValueError("trace missing the executed force budget")
    return {**old.diagnostic_trace(trace), "max_force_n": np.asarray(trace["max_force_n"])}


def screen_metrics(metrics: dict) -> dict:
    """Drop and normalize the legacy summary fields that assume the 6 N budget."""
    row = {name: value for name, value in metrics.items() if name not in DROPPED_METRICS}
    if "tangent_recovery_status" in row:
        status = row["tangent_recovery_status"]
        row["tangent_recovery_status"] = NORMALIZED_STATUS.get(status, status)
    return row


def diagnostic_pairs(rows: list[dict]) -> list[dict]:
    """Pair the two budgets of the same yaw, variant and window as 8 N minus 6 N."""
    expected = {
        (yaw, variant, window, budget)
        for yaw in YAWS for variant in VARIANTS
        for window, _, _ in old.DIAGNOSTIC_WINDOWS for budget in BUDGETS
    }
    lookup: dict[tuple, dict] = {}
    for row in rows:
        key = (row["surface_yaw_deg"], row["variant"], row["window"], row["max_force_n"])
        if key not in expected:
            raise ValueError(f"unexpected diagnostic row identity: {key}")
        if key in lookup:
            raise ValueError(f"duplicate diagnostic row identity: {key}")
        lookup[key] = row
    missing = sorted(map(str, expected - set(lookup)))
    if missing:
        raise ValueError(f"missing diagnostic rows: {missing}")
    pairs = []
    for yaw in YAWS:
        for variant in VARIANTS:
            for window, _, _ in old.DIAGNOSTIC_WINDOWS:
                row = lookup[yaw, variant, window, BUDGETS[1]]
                reference = lookup[yaw, variant, window, BUDGETS[0]]
                pairs.append({
                    "surface_yaw_deg": yaw, "variant": variant, "window": window,
                    "max_force_n": BUDGETS[1], "reference_max_force_n": BUDGETS[0],
                    "delta_direction": DELTA_DIRECTION,
                    **{f"delta_{m}": row[m] - reference[m] for m in old.PAIRED_DIAGNOSTICS},
                })
    return pairs


def protocol_document() -> dict:
    """Return the frozen screen definition together with all twelve case documents."""
    cases = trials()
    config = cases[0][1].config
    return {
        "identity": IDENTITY,
        "goal": (
            "Screen the bounded tangential force budget on the online arm; "
            "a diagnostic screen, not a default change or a new holdout."
        ),
        "public_development": True,
        "new_holdout": False,
        "default_changed": False,
        "comparison_defined_before_execution": True,
        "reference": REFERENCE,
        "primary_window": "post",
        "co_primary_late_window": "late_post",
        "screening_criteria": compensation_budget_screen.CRITERIA,
        "screening_scope": (
            "Predeclared public-development engineering feasibility checks, not old paired-method "
            "gates, a hardware safety guarantee or authority to promote a global default. "
            "The late-window RMSE criterion is not a pointwise or all-rolling-window guarantee."
        ),
        "diagnostic_windows": [{"name": name, "start_s": start, "end_s": end}
                               for name, start, end in old.DIAGNOSTIC_WINDOWS],
        "paired_diagnostics": list(old.PAIRED_DIAGNOSTICS),
        "paired_delta_direction": DELTA_DIRECTION,
        "profile": old.PROFILE,
        "yaws": list(YAWS),
        "variants": list(VARIANTS),
        "budgets_n": list(BUDGETS),
        "baseline_max_force_n": BASELINE_BUDGET_N,
        "default_max_force_n": DEFAULT_MAX_FORCE_N,
        "arms_executed": [ARM],
        "seed": SEED,
        "rotation_gain_scale": ROTATION_GAIN_SCALE,
        "duration_s": config.duration,
        "timestep_s": config.timestep,
        "evaluation_start_s": config.evaluation_start,
        "case_source": "tools.combined_residual_ablation.ablation_cases",
        "variant_definitions": {
            "combined": "The original seed-11 modest_combined case object at this yaw; nothing replaced.",
            "no_bias": "The ablation case that replaces only wrench_bias_world with zero.",
        },
        "fixed_input_scope": [
            "Within each matched budget pair, only max_force on the online compensation changes.",
            "Case objects come unmodified from the frozen ablation list, filtered to two variants.",
            "Surface and task yaw, noise seed, payload, schedules, phases and clock are unchanged.",
            "Every other controller and compensation parameter keeps its constructor default.",
            f"{DEFAULT_MAX_FORCE_N} N remains the shipped default; {BUDGETS[1]} N is executed as a counterfactual only.",
        ],
        "interpretation": [
            "Each pair changes one bound conditional on all other inputs held at their original values.",
            "Only the online arm runs; no paired baseline or fixed-friction comparator exists here.",
            "A larger budget changes the closed-loop trajectory; deltas are not error budgets.",
        ],
        "gates": GATES,
        "absolute_checks_evaluated": list(old.ABSOLUTE_CHECKS),
        "comparator_gates_evaluated": False,
        "gates_not_evaluated": list(old.NOT_EVALUATED_GATES),
        "legacy_metrics_dropped": list(DROPPED_METRICS),
        "legacy_status_normalized": dict(NORMALIZED_STATUS),
        "diagnostic_observers": {
            **old.protocol_document()["diagnostic_observers"],
            "budget_awareness": (
                "The cap and slew observers read max_force and force_slew_rate from the "
                "compensation object that executed the cycle, so both flags follow the "
                "budget of the run rather than the default."
            ),
        },
        "trace_fields": list(TRACE_FIELDS),
        "controllers": [controller_document(budget) for budget in BUDGETS],
        "limitations": [
            "One preselected noise seed and fixed payload; not statistical robustness or a new holdout.",
            "Online-arm counterfactuals only; the frozen paired comparator gates are not evaluated.",
            "Two budgets at two variants; not a continuous sweep or a bound-selection result.",
            "Observed flags describe the executed command path, not a stability or safety proof.",
        ],
        "cases": [
            {
                "variant": variant,
                **identity(variant, case, budget),
                "controller_frame_yaw_deg": case.task.yaw_deg + case.controller_yaw_error_deg,
                "controller_frame_rotation": yaw_frame(
                    case.task.yaw_deg + case.controller_yaw_error_deg
                ).rotation.tolist(),
                **_case_document(case),
            }
            for variant, case, budget in cases
        ],
    }


def source_identity() -> dict:
    sources = dict(old.source_identity())
    for module in (__file__, validator.__file__, compensation_budget_screen.__file__):
        path = Path(module).resolve()
        sources[f"tools/{path.name}"] = _sha256(path)
    return sources


def generate(output_dir) -> Path:
    output = Path(output_dir).absolute()
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError("output path must not contain symlinks")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    cases = trials()
    protocol = protocol_document()
    sources = source_identity()
    reference = verify_reference()
    rows: list[dict] = []
    phases: list[dict] = []
    diagnostics: list[dict] = []
    reference_requests = {}
    combined_traces = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".compensation-budget-", dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        traces = staging / "traces"
        traces.mkdir(parents=True)
        _write_json(staging / "protocol.json", protocol)
        _write_json(staging / "source_hashes.json", sources)
        for index, (variant, case, budget) in enumerate(cases, start=1):
            label = stem(variant, case, budget)
            result = run_trial(case, budget)
            trace = result.trace
            validator.validate_trace(trace, case, budget)
            if budget == BASELINE_BUDGET_N:
                check_reference(variant, case, trace, reference)
                reference_requests[case.scenario.wall_yaw_deg, variant] = (
                    trace["requested_tangential_force_world"].copy()
                )
            if variant == "combined":
                combined_traces[case.scenario.wall_yaw_deg, budget] = diagnostic_trace(trace)
            else:
                old.combined_residual_validation.check_unchanged_prefix(
                    combined_traces[case.scenario.wall_yaw_deg, budget], diagnostic_trace(trace), 6.
                )
            summary, windows = summarize_trial(result, case)
            overall = screen_metrics(summary)
            context = identity(variant, case, budget)
            audit = {name: float(trace[name]) for name in old.AUDIT_FIELDS}
            time = np.asarray(trace["time"])
            rows.append({
                **context,
                "controller_yaw_error_deg": case.controller_yaw_error_deg,
                **overall,
                **cross_surface_regression.extra_metrics(trace, slice(None)),
                **old.diagnostic_rates(trace, time >= case.config.evaluation_start),
                **audit,
                **old.absolute_checks(overall),
                "max_uncapped_tangential_amplitude_n": float(np.max(
                    trace["controller_coefficient_before_compute"] * trace["diagnostic_corrected_force_n"])),
                "max_matched_request_difference_n": float(np.max(np.linalg.norm(
                    trace["requested_tangential_force_world"]
                    - reference_requests[case.scenario.wall_yaw_deg, variant], axis=1))),
            })
            for window, start, end in old.DIAGNOSTIC_WINDOWS:
                diagnostics.append({
                    **context, "window": window,
                    **combined_residual_diagnostics.analyze_trace(
                        trace, surface_yaw_deg=case.scenario.wall_yaw_deg,
                        controller_yaw_error_deg=case.controller_yaw_error_deg,
                        start_s=start, end_s=end),
                })
            for window, phase in zip(windows, case.phases, strict=True):
                metrics = screen_metrics(window)
                mask = (time >= phase.start_s) & (time < phase.end_s)
                phases.append({
                    **context,
                    **metrics,
                    **old.diagnostic_rates(trace, mask),
                    **old.absolute_checks(metrics),
                })
            _save_trace(traces / f"{label}__diagnostic.npz", diagnostic_trace(trace))
            print(f"{index:02d}/{len(cases)} {label}", flush=True)
        if sources != source_identity():
            raise ValueError("source/assets changed during execution")
        verify_reference()
        _write_csv(staging / "comparison.csv", rows)
        _write_csv(staging / "phase_metrics.csv", phases)
        _write_csv(staging / "diagnostics.csv", diagnostics)
        _write_csv(staging / "paired_diagnostics.csv", diagnostic_pairs(diagnostics))
        _write_json(staging / "screening.json", compensation_budget_screen.screen(rows, phases, diagnostics))
        manifest = {
            "schema_version": 1,
            "identity": IDENTITY,
            "new_holdout": False,
            "default_changed": False,
            "comparator_gates_evaluated": False,
            "runs": len(rows),
            "budgets_n": list(BUDGETS),
            "versions": {
                "python": platform.python_version(),
                **{name: version(name) for name in ("numpy", "mujoco")},
            },
            "input_sha256": {
                name: _sha256(staging / name)
                for name in ("protocol.json", "source_hashes.json")
            },
            "artifact_sha256": {
                str(path.relative_to(staging)): _sha256(path)
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            },
        }
        _write_json(staging / "manifest.json", manifest)
        (staging / "COMPLETE").write_text(
            _sha256(staging / "manifest.json") + "\n", encoding="utf-8"
        )
        if output.exists():
            raise FileExistsError(f"output appeared during execution: {output}")
        os.rename(staging, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(f"Compensation force-budget screen complete: {generate(arguments.output)}")


if __name__ == "__main__":
    main()
