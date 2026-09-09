"""One-factor input ablation of the modest_combined 8-12 s residual on the online arm.

Each variant removes exactly one measured-input factor from the original seed-11
modest_combined case at one surface yaw and keeps every other input, the fixed
controller and its bounds. The runs are online-arm counterfactuals only, so the
original paired-baseline and paired-friction comparator gates are not evaluated
here; the tables report raw metrics plus per-run absolute physical checks.
"""

from __future__ import annotations

import argparse
import json
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
    _compact_trace,
    _constant,
    _constructor_config,
    _save_trace,
    _sha256,
    _source_hashes,
    _write_csv,
    _write_json,
    _zero_bias,
    summarize_trial,
)
from compliant_control_lab.surface_control import SurfaceAdaptiveController
from compliant_control_lab.surface_simulation import yaw_frame
from compliant_control_lab.tangential_compensation import _tangent
from tools import (
    combined_residual_diagnostics,
    combined_residual_validation,
    cross_surface_regression,
)

IDENTITY = "modest-combined-input-ablation-v1"
PROFILE = "modest_combined"
YAWS = (-15, 0, 15)
VARIANTS = ("combined", "no_yaw_error", "no_bias", "constant_initial_friction")
ARM = "online"
SEED = 11
ROTATION_GAIN_SCALE = 1.0
CONSTANT_FRICTION = 0.30
RECONSTRUCTION_TOLERANCE_N = 1e-9
REPLACED_FIELD = {
    "combined": None,
    "no_yaw_error": "controller_yaw_error_deg",
    "no_bias": "wrench_bias_world",
    "constant_initial_friction": "friction",
}
DIAGNOSTIC_FIELDS = (
    "diagnostic_corrected_force_n",
    "diagnostic_compensation_active",
    "diagnostic_amplitude_capped",
    "diagnostic_slew_limited",
)
AUDIT_FIELDS = ("slew_reconstruction_max_error_n", "slew_reconstruction_mismatch_cycles")
TRACE_FIELDS = (
    "time",
    "position",
    "target_position",
    "linear_velocity",
    "target_linear_velocity",
    "true_normal_force",
    "target_normal_force",
    "measured_normal_force",
    "orientation_error_rad",
    "torque_projection_scale",
    "contact_blend",
    "commanded_torque",
    "applied_torque",
    "lower_torque_limit",
    "upper_torque_limit",
    "requested_tangential_force_world",
    "controller_coefficient_before_compute",
    "controller_coefficient_after_compute",
    "controller_update_ready_before_compute",
    "controller_update_ready_after_compute",
    "applied_wall_friction",
    "applied_tool_friction",
    "applied_raw_wrench_bias_world",
    "feedback_raw_wrench_bias_world",
    "controller_yaw_error_deg",
    "trajectory_rate_scale",
    "rotation_gain_scale",
    "controller_kind",
    "true_contact_gap_m",
    *DIAGNOSTIC_FIELDS,
)
KEYS = ("surface_yaw_deg", "variant", "case", "arm", "simulation_seed", "rotation_gain_scale")
ABSOLUTE_CHECKS = ("contact_ratio", "raw_peak_force", "saturation")
NOT_EVALUATED_GATES = (
    "paired_force_rmse",
    "paired_orientation_rmse",
    "paired_friction_force_rmse",
    "paired_friction_orientation_rmse",
)
DIAGNOSTIC_WINDOWS = (("post", 8., 12.), ("early_post", 8., 8.5), ("late_post", 10., 12.))
PAIRED_DIAGNOSTICS = (
    "tangent_rmse_mm", "along_track_rmse_mm", "cross_track_rmse_mm",
    "tangent_velocity_error_rms_m_s", "amplitude_capped_pct", "slew_limited_pct",
    "normal_force_error_rms_n", "orientation_error_rms_deg", "minimum_reserved_torque_headroom_nm",
)
REFERENCE = {
    "directory": "results/franka_cross_surface_dynamic",
    "manifest_sha256": "a3d96b00769741f75fe320a047c086cdb59f9bfe3e5024782490121aa86ffe9b",
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


def check_reference(trace, case, reference):
    label = f"yaw_{int(case.scenario.wall_yaw_deg)}__modest_combined__seed_11__online__s1"
    for suffix, reconstructed in (("compact", _compact_trace(trace, case)), ("inputs", trace)):
        with np.load(reference / "traces" / f"{label}__{suffix}.npz", allow_pickle=False) as old:
            for name in set(reconstructed) & set(old.files):
                if not np.array_equal(reconstructed[name], old[name]):
                    raise ValueError(f"original combined differs: {suffix}/{name}")


def diagnostic_pairs(rows):
    lookup = {(r["surface_yaw_deg"], r["variant"], r["window"]): r for r in rows}
    pairs = []
    for yaw in YAWS:
        for variant in VARIANTS[1:]:
            for label, _, _ in DIAGNOSTIC_WINDOWS:
                row, reference = lookup[yaw, variant, label], lookup[yaw, "combined", label]
                pairs.append({"surface_yaw_deg": yaw, "variant": variant, "window": label,
                              "reference_variant": "combined", "delta_direction": "variant_minus_combined",
                              **{f"delta_{m}": row[m] - reference[m] for m in PAIRED_DIAGNOSTICS}})
    return pairs


def _original_cases() -> dict[int, object]:
    originals = {
        int(c.scenario.wall_yaw_deg): c
        for c in cross_surface_regression.make_cases()
        if c.name == PROFILE
    }
    if sorted(originals) != sorted(YAWS):
        raise ValueError("expected one modest_combined case per surface yaw")
    for case in originals.values():
        if case.config.seed != SEED:
            raise ValueError("ablation requires the original preselected noise seed")
        # constant_initial_friction only drops the schedule, never the scenario setup.
        if (
            case.friction.at(0.0) != CONSTANT_FRICTION
            or case.scenario.wall_sliding_friction != CONSTANT_FRICTION
            or case.scenario.tool_sliding_friction != CONSTANT_FRICTION
        ):
            raise ValueError("scenario initial friction no longer equals the constant variant")
    return originals


def _variant_case(original, variant: str):
    if variant == "combined":
        return original
    if variant == "no_yaw_error":
        return replace(original, controller_yaw_error_deg=0.0)
    if variant == "no_bias":
        return replace(original, wrench_bias_world=_zero_bias())
    if variant == "constant_initial_friction":
        return replace(original, friction=_constant(CONSTANT_FRICTION))
    raise ValueError(f"unknown ablation variant: {variant}")


def ablation_cases() -> list[tuple[str, object]]:
    """Return the fixed 3 yaw x 4 variant list of (variant, ProtocolCase) pairs."""
    originals = _original_cases()
    return [
        (variant, _variant_case(originals[yaw], variant)) for yaw in YAWS for variant in VARIANTS
    ]


def identity(variant: str, case) -> dict:
    values = (
        case.scenario.wall_yaw_deg, variant, case.name, ARM, case.config.seed, ROTATION_GAIN_SCALE,
    )
    return dict(zip(KEYS, values))


def stem(variant: str, case) -> str:
    yaw = int(case.scenario.wall_yaw_deg)
    return f"yaw_{yaw}__{case.name}__{variant}__seed_{SEED}__{ARM}__s{int(ROTATION_GAIN_SCALE)}"


def controller_document() -> dict:
    control = SurfaceAdaptiveController(
        yaw_frame(0.0), tangential_mode=ARMS[ARM], rotation_gain_scale=ROTATION_GAIN_SCALE,
    )
    return {
        "arm": ARM,
        "tangential_mode": ARMS[ARM],
        "rotation_gain_scale": ROTATION_GAIN_SCALE,
        "safe_adaptive_base": _constructor_config(control._base),
    }


def protocol_document() -> dict:
    """Return the frozen ablation definition together with all twelve case documents."""
    cases = ablation_cases()
    config = cases[0][1].config
    return {
        "identity": IDENTITY,
        "goal": "Diagnose the modest_combined 8-12 s residual, not tune the controller.",
        "public_development": True,
        "new_holdout": False,
        "default_changed": False,
        "comparison_defined_before_execution": True,
        "reference": REFERENCE,
        "primary_window": "post",
        "diagnostic_windows": [{"name": name, "start_s": start, "end_s": end}
                               for name, start, end in DIAGNOSTIC_WINDOWS],
        "paired_diagnostics": list(PAIRED_DIAGNOSTICS),
        "profile": PROFILE,
        "yaws": list(YAWS),
        "variants": list(VARIANTS),
        "arms_executed": [ARM],
        "seed": SEED,
        "rotation_gain_scale": ROTATION_GAIN_SCALE,
        "duration_s": config.duration,
        "timestep_s": config.timestep,
        "evaluation_start_s": config.evaluation_start,
        "case_source": "tools.cross_surface_regression.make_cases",
        "variant_definitions": {
            "combined": (
                "The original seed-11 modest_combined case object at this yaw; nothing replaced."
            ),
            "no_yaw_error": "Replaces only controller_yaw_error_deg with 0.0.",
            "no_bias": "Replaces only wrench_bias_world with a constant zero world six-vector.",
            "constant_initial_friction": (
                f"Replaces only the friction schedule with the constant initial value "
                f"{CONSTANT_FRICTION}; the scenario already starts at that friction."
            ),
        },
        "replaced_field": REPLACED_FIELD,
        "preserved_inputs": [
            "Wall and task yaw, including the rotated world wrench-bias direction.",
            "Tool mass, contact model, sensor/position noise seed and noise magnitudes.",
            "Phase windows, recovery start, reverse trajectory clock and target pose.",
            "Every schedule that the selected variant does not replace.",
            "Case name; the variant is separate identity metadata.",
        ],
        "interpretation": [
            "Each row removes one factor conditional on the other factors held at their original values.",
            "These are not additive attributions and not a factorial interaction design.",
            "Only the online arm runs; no paired baseline or fixed-friction comparator exists here.",
        ],
        "gates": GATES,
        "absolute_checks_evaluated": list(ABSOLUTE_CHECKS),
        "comparator_gates_evaluated": False,
        "gates_not_evaluated": list(NOT_EVALUATED_GATES),
        "diagnostic_observers": {
            "diagnostic_corrected_force_n": (
                "Controller bias-corrected normal force after compute; the force the bounded "
                "tangential request consumed in the same cycle."
            ),
            "diagnostic_compensation_active": "base.tangential._active after compute.",
            "diagnostic_amplitude_capped": (
                "Before-compute equivalent coefficient times the actual corrected force reaches "
                "max_force, evaluated only while the compensation is active."
            ),
            "diagnostic_slew_limited": (
                "Reconstructed bounded and direction-smoothed desired force versus the previously "
                "stored projected force exceeds force_slew_rate times the actual dt, evaluated "
                "only while the compensation is active."
            ),
            "sampling": "One sample per control cycle; observation only, no extra sensor read.",
            "reconstruction_uncertainty": (
                "The slew reconstruction reads the commanded target twist. The reference governor "
                "changes only the normal component, so the tangential projection agrees up to "
                "floating-point rounding rather than exactly. Every active cycle re-checks the "
                "reconstructed bounded force against the stored force; "
                f"{AUDIT_FIELDS[0]} and {AUDIT_FIELDS[1]} report the residual and the number of "
                f"cycles above {RECONSTRUCTION_TOLERANCE_N} N, above which the slew flag of those "
                "cycles is uncertain."
            ),
        },
        "trace_fields": list(TRACE_FIELDS),
        "controller": controller_document(),
        "limitations": [
            "One preselected noise seed and fixed payload; not statistical robustness or a new holdout.",
            "Online-arm counterfactuals only; the frozen paired comparator gates are not evaluated.",
            "Observed flags describe the executed command path, not a stability or safety proof.",
            "Removing an input changes the closed-loop trajectory; residual differences are not error budgets.",
        ],
        "cases": [
            {
                "variant": variant,
                **identity(variant, case),
                "controller_frame_yaw_deg": case.task.yaw_deg + case.controller_yaw_error_deg,
                "controller_frame_rotation": yaw_frame(
                    case.task.yaw_deg + case.controller_yaw_error_deg
                ).rotation.tolist(),
                **_case_document(case),
            }
            for variant, case in cases
        ],
    }


class CompensationObserver:
    """Read-only reconstruction of the bounded tangential request already computed.

    Every value comes from controller telemetry and the compensation parameters that
    produced the current cycle. FrankaSafeAdaptiveController.compute updates the
    corrected force and the contact blend before it calls the compensation, so the
    post-compute reads are the same numbers the request consumed. Nothing here reads
    the simulator, integrates or evaluates contact.
    """

    def __init__(self, controller) -> None:
        base = controller._base
        compensation = base.tangential
        if compensation is None or compensation.mode != ARM:
            raise ValueError("diagnostic observation requires the online compensation arm")
        normal = np.asarray(base.base.base.normal, dtype=float)
        self._controller = controller
        self._compensation = compensation
        self._normal = normal / np.linalg.norm(normal)
        self._frame = controller.frame
        self.mismatch_cycles = 0
        self.max_reconstruction_error_n = 0.0

    def before_compute(self, target, dt: float) -> dict:
        """Snapshot the coefficient and stored force that this cycle's request will use."""
        return {
            "coefficient": float(self._controller.equivalent_tangential_coefficient),
            "previous_force": _tangent(self._normal, self._compensation.last_force),
            "target_velocity": self._frame.vector_to_local(target.linear_velocity),
            "dt": float(dt),
        }

    def after_compute(self, snapshot: dict) -> dict:
        compensation = self._compensation
        active = bool(compensation._active)
        corrected_force_n = float(self._controller.corrected_force_n)
        load = snapshot["coefficient"] * corrected_force_n
        amplitude = min(load, compensation.max_force)
        velocity = _tangent(self._normal, snapshot["target_velocity"])
        speed = float(np.linalg.norm(velocity))
        direction = velocity / np.sqrt(speed**2 + compensation.velocity_scale**2)
        requested = float(self._controller.contact_blend) * amplitude * direction
        delta = requested - snapshot["previous_force"]
        delta_norm = float(np.linalg.norm(delta))
        limit = compensation.force_slew_rate * snapshot["dt"]
        if active:
            applied = snapshot["previous_force"] + delta * min(1.0, limit / max(delta_norm, 1e-12))
            stored = _tangent(self._normal, compensation.last_force)
            error = float(np.max(np.abs(applied - stored)))
            self.max_reconstruction_error_n = max(self.max_reconstruction_error_n, error)
            self.mismatch_cycles += int(error > RECONSTRUCTION_TOLERANCE_N)
        return {
            "diagnostic_corrected_force_n": corrected_force_n,
            "diagnostic_compensation_active": active,
            "diagnostic_amplitude_capped": bool(active and load >= compensation.max_force),
            "diagnostic_slew_limited": bool(active and delta_norm > limit),
        }


def run_observed_loop(case, controller, simulator, observer=None):
    """Run the existing protocol control loop and attach the read-only observations."""
    observer = CompensationObserver(controller) if observer is None else observer
    dt = case.config.timestep
    samples: dict[str, list] = {name: [] for name in DIAGNOSTIC_FIELDS}
    for step in range(round(case.config.duration / dt)):
        sample = simulator.sample()
        if step == 0:
            controller.reset(sample.state)
        before = (
            controller.equivalent_tangential_coefficient,
            controller.tangential_update_ready,
        )
        snapshot = observer.before_compute(sample.target, dt)
        wrench = controller.compute(sample.state, sample.target, dt)
        simulator.step(wrench, controller, telemetry_before=before)
        for name, value in observer.after_compute(snapshot).items():
            samples[name].append(value)
    result = simulator.result()
    result.trace["rotation_gain_scale"] = np.array(float(ROTATION_GAIN_SCALE))
    result.trace.update({name: np.asarray(values) for name, values in samples.items()})
    result.trace[AUDIT_FIELDS[0]] = np.array(observer.max_reconstruction_error_n)
    result.trace[AUDIT_FIELDS[1]] = np.array(observer.mismatch_cycles)
    return result


def run_diagnostic_trial(case):
    """Run one ablation case on the fixed online arm at rotation_gain_scale 1."""
    frame = yaw_frame(case.task.yaw_deg + case.controller_yaw_error_deg)
    controller = SurfaceAdaptiveController(
        frame, tangential_mode=ARMS[ARM], rotation_gain_scale=ROTATION_GAIN_SCALE,
    )
    simulator = ScheduledSurfaceSimulator(case, frame, ARM)
    return run_observed_loop(case, controller, simulator)


def diagnostic_trace(trace: dict) -> dict:
    missing = [name for name in TRACE_FIELDS if name not in trace]
    if missing:
        raise ValueError(f"trace missing ablation fields: {missing}")
    return {name: np.asarray(trace[name]) for name in TRACE_FIELDS}


def diagnostic_rates(trace: dict, mask) -> dict:
    return {
        f"{name}_pct": float(100.0 * np.mean(np.asarray(trace[name], dtype=bool)[mask]))
        for name in DIAGNOSTIC_FIELDS[1:]
    }


def absolute_checks(row: dict) -> dict:
    """Report only the per-run physical checks; comparator gates need arms not run here."""
    failures = [
        label
        for passed, label in (
            (row["contact_ratio_pct"] >= GATES["minimum_contact_ratio_pct"], "contact_ratio"),
            (row["peak_force_n"] <= GATES["maximum_raw_peak_force_n"], "raw_peak_force"),
            (row["saturation_pct"] == GATES["required_saturation_pct"], "saturation"),
        )
        if not passed
    ]
    return {
        "failed_absolute_checks": ";".join(failures),
        "absolute_checks_pass": "no" if failures else "yes",
        "comparator_gates_evaluated": "no",
        "comparator_gate_status": "not_evaluated",
    }


def source_identity() -> dict:
    sources = {f"src/compliant_control_lab/{k}": v for k, v in _source_hashes().items()}
    # make_cases defines the twelve ablation inputs, so its module is pinned too.
    for module in (__file__, cross_surface_regression.__file__,
                   combined_residual_diagnostics.__file__, combined_residual_validation.__file__):
        path = Path(module).resolve()
        sources[f"tools/{path.name}"] = _sha256(path)
    return sources


def generate(output_dir) -> Path:
    output = Path(output_dir).absolute()
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError("output path must not contain symlinks")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    cases = ablation_cases()
    protocol = protocol_document()
    sources = source_identity()
    reference = verify_reference()
    rows: list[dict] = []
    phases: list[dict] = []
    diagnostics: list[dict] = []
    combined_trace = None
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".combined-ablation-", dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        traces = staging / "traces"
        traces.mkdir(parents=True)
        _write_json(staging / "protocol.json", protocol)
        _write_json(staging / "source_hashes.json", sources)
        for index, (variant, case) in enumerate(cases, start=1):
            label = stem(variant, case)
            result = run_diagnostic_trial(case)
            trace = result.trace
            combined_residual_validation.validate_trace(trace, case)
            if variant == "combined":
                check_reference(trace, case, reference)
                combined_trace = trace
            elif variant in {"no_bias", "constant_initial_friction"}:
                onset = 6. if variant == "no_bias" else 4.
                combined_residual_validation.check_unchanged_prefix(combined_trace, trace, onset)
            overall, windows = summarize_trial(result, case)
            context = identity(variant, case)
            audit = {name: float(trace[name]) for name in AUDIT_FIELDS}
            time = np.asarray(trace["time"])
            rows.append({
                **context,
                "controller_yaw_error_deg": case.controller_yaw_error_deg,
                **overall,
                **cross_surface_regression.extra_metrics(trace, slice(None)),
                **diagnostic_rates(trace, time >= case.config.evaluation_start),
                **audit,
                **absolute_checks(overall),
            })
            for window, start, end in DIAGNOSTIC_WINDOWS:
                diagnostics.append({
                    **context, "window": window,
                    **combined_residual_diagnostics.analyze_trace(
                        trace, surface_yaw_deg=case.scenario.wall_yaw_deg,
                        controller_yaw_error_deg=case.controller_yaw_error_deg,
                        start_s=start, end_s=end),
                })
            for window, phase in zip(windows, case.phases, strict=True):
                mask = (time >= phase.start_s) & (time < phase.end_s)
                phases.append({
                    **context,
                    **window,
                    **diagnostic_rates(trace, mask),
                    **absolute_checks(window),
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
        manifest = {
            "schema_version": 1,
            "identity": IDENTITY,
            "new_holdout": False,
            "default_changed": False,
            "comparator_gates_evaluated": False,
            "runs": len(rows),
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
    print(f"Combined-residual input ablation complete: {generate(arguments.output)}")


if __name__ == "__main__":
    main()
