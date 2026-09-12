"""Replay schema-2 measured-budget traces through Python and C++ controllers.

This is deterministic full-input controller replay.  It does not rerun MuJoCo
and makes no live-physics or hard-real-time claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from compliant_control_lab.franka_control import FrankaTarget
from compliant_control_lab.surface_control import SurfaceAdaptiveController, SurfaceFrame
from compliant_control_lab.surface_replay import _state, _validate_trace
from tools.load_budget_trial import LoadBudgetObserver, _load_aware
from tools.measured_budget_robustness import (
    FULL_TRACE_FIELDS,
    STATIC_TRACE_FIELDS,
)
from tools.measured_budget_robustness import (
    source_identity as baseline_source_identity,
)
from tools.measured_budget_trial import (
    MAX_MEASUREMENT_AGE_S,
    METHOD_BOUNDS,
    LoadPacketGate,
    RawLoadPacket,
)
from tools.verify_cpp_surface_replay import _compiler_identity, _numbers, _sha256

TOLERANCE = 1e-8
PROBE_PROTOCOL = "surface-control-measured-v1"
PACKET_STATUS_NAMES = {
    0: "missing",
    1: "accepted",
    2: "stale",
    3: "future",
    4: "reordered",
    5: "nonfinite",
}
REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_DEMO = REPOSITORY / "results/franka_measured_budget_demo/trace.npz"
DEFAULT_ROBUSTNESS = REPOSITORY / "results/franka_measured_budget_robustness/traces"
DEFAULT_PROBE = REPOSITORY / "build/compliant_control_surface_probe"
DEFAULT_OUTPUT = REPOSITORY / "results/franka_measured_budget_cpp_replay"

SOURCE_PATHS = (
    REPOSITORY / "CMakeLists.txt",
    REPOSITORY / "cpp/include/compliant_control_lab/franka_control.hpp",
    REPOSITORY / "cpp/include/compliant_control_lab/surface_control.hpp",
    REPOSITORY / "cpp/include/compliant_control_lab/torque_safety.hpp",
    REPOSITORY / "cpp/src/franka_control.cpp",
    REPOSITORY / "cpp/src/surface_control.cpp",
    REPOSITORY / "cpp/src/torque_safety.cpp",
    REPOSITORY / "cpp/tools/surface_controller_probe.cpp",
    REPOSITORY / "tools/verify_cpp_surface_replay.py",
    Path(__file__).resolve(),
)

NUMERIC_FIELDS = (
    "commanded_wrench",
    "commanded_torque",
    "contact_blend",
    "governed_normal_lead_m",
    "torque_projection_scale",
    "requested_tangential_force_world",
    "load_budget_applied_n",
    "load_budget_next_n",
    "load_estimate_n",
    "load_projected_n",
    "load_force_local",
    "load_measurement_time_s",
    "load_measurement_age_s",
    "load_compensation_force_local",
    "controller_coefficient_before_compute",
    "controller_coefficient_after_compute",
    "diagnostic_corrected_force_n",
)
BOOLEAN_FIELDS = (
    "load_budget_updated",
    "load_projection_accepted",
    "controller_update_ready_before_compute",
    "controller_update_ready_after_compute",
    "diagnostic_compensation_active",
    "diagnostic_amplitude_capped",
    "diagnostic_slew_limited",
    "load_measurement_available",
    "measured_in_contact",
)
INTEGER_FIELDS = ("load_packet_status",)
REPLAY_FIELDS = (*NUMERIC_FIELDS, *BOOLEAN_FIELDS, *INTEGER_FIELDS)
CYCLE_SHAPES = {
    **{name: () for name in NUMERIC_FIELDS},
    **{name: () for name in BOOLEAN_FIELDS},
    **{name: () for name in INTEGER_FIELDS},
    "commanded_wrench": (6,),
    "commanded_torque": (7,),
    "requested_tangential_force_world": (3,),
    "load_force_local": (3,),
    "load_compensation_force_local": (3,),
    "raw_load_packet_present": (),
    "raw_load_packet_force": (3,),
    "raw_load_packet_stamp": (),
}


def _scalar_number(arrays: dict[str, np.ndarray], name: str) -> float:
    value = np.asarray(arrays.get(name))
    if value.shape != () or value.dtype.kind not in "iuf" or not np.isfinite(value.item()):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(value)


def _scalar_string(arrays: dict[str, np.ndarray], name: str) -> str:
    value = np.asarray(arrays.get(name))
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"{name} must be a Unicode scalar")
    return str(value.item())


def _metadata(arrays: dict[str, np.ndarray]) -> dict:
    schema = np.asarray(arrays.get("trace_schema_version"))
    if schema.shape != () or schema.dtype.kind not in "iu" or int(schema) != 2:
        raise ValueError("unsupported trace_schema_version; expected schema 2")
    if _scalar_string(arrays, "controller_kind") != "surface_online":
        raise ValueError("controller_kind must be surface_online")
    method = _scalar_string(arrays, "method")
    if method not in METHOD_BOUNDS:
        raise ValueError(f"unsupported measured-budget method: {method}")
    minimum = _scalar_number(arrays, "minimum_force_n")
    maximum = _scalar_number(arrays, "max_force_n")
    if (minimum, maximum) != METHOD_BOUNDS[method]:
        raise ValueError("budget metadata does not match method")
    gain = _scalar_number(arrays, "rotation_gain_scale")
    if gain <= 0.0 or gain > np.finfo(float).max / 20.0:
        raise ValueError("rotation_gain_scale must be a finite positive real scalar")
    maximum_age = _scalar_number(arrays, "load_max_measurement_age_s")
    if maximum_age != MAX_MEASUREMENT_AGE_S:
        raise ValueError("load_max_measurement_age_s differs from the frozen runner")
    fault_name = _scalar_string(arrays, "fault_name")
    fault_kind = _scalar_string(arrays, "fault_kind")
    fault_start = _scalar_number(arrays, "fault_start_s")
    fault_end = _scalar_number(arrays, "fault_end_s")
    fault_value = _scalar_number(arrays, "fault_value")
    if (
        not fault_name
        or fault_kind not in {"fresh", "bias", "scale", "missing", "stale"}
        or fault_end < fault_start
        or (fault_kind == "scale" and fault_value <= 0.0)
    ):
        raise ValueError("invalid measured-budget fault metadata")
    return {
        "trace_schema_version": 2,
        "controller_kind": "surface_online",
        "method": method,
        "minimum_force_n": minimum,
        "maximum_force_n": maximum,
        "rotation_gain_scale": gain,
        "maximum_packet_age_s": maximum_age,
        "fault": {
            "name": fault_name,
            "kind": fault_kind,
            "start_s": fault_start,
            "end_s": fault_end,
            "value": fault_value,
        },
    }


def _validate_measured_trace(arrays: dict[str, np.ndarray]) -> tuple[int, dict]:
    if set(arrays) != FULL_TRACE_FIELDS:
        missing = sorted(FULL_TRACE_FIELDS - set(arrays))
        extra = sorted(set(arrays) - FULL_TRACE_FIELDS)
        raise ValueError(f"full trace schema mismatch; missing={missing}, extra={extra}")
    metadata = _metadata(arrays)
    count = _validate_trace(arrays)
    for name, raw in arrays.items():
        value = np.asarray(raw)
        if value.dtype.kind not in "biufUS":
            raise ValueError(f"unsupported full trace field: {name}")
        if value.dtype.kind in "biuf" and not np.all(np.isfinite(value)):
            raise ValueError(f"nonfinite full trace field: {name}")
        if name not in STATIC_TRACE_FIELDS and (value.ndim == 0 or value.shape[0] != count):
            raise ValueError(f"full trace cycle count mismatch: {name}")
    for name, shape in CYCLE_SHAPES.items():
        value = np.asarray(arrays[name])
        if value.shape != (count, *shape):
            raise ValueError(f"invalid full-state shape: {name}")
    for name in (*BOOLEAN_FIELDS, "raw_load_packet_present"):
        if np.asarray(arrays[name]).dtype.kind != "b":
            raise ValueError(f"{name} must contain boolean flags")
    for name in INTEGER_FIELDS:
        if np.asarray(arrays[name]).dtype.kind not in "iu":
            raise ValueError(f"{name} must contain integer codes")
    time = np.asarray(arrays["time"])
    expected_time = np.arange(count, dtype=float) * float(arrays["dt"])
    if time.shape != (count,) or not np.allclose(time, expected_time, rtol=0, atol=1e-12):
        raise ValueError("time must be the zero-based dt grid")
    present = np.asarray(arrays["raw_load_packet_present"])
    status = np.asarray(arrays["load_packet_status"])
    if present.shape != (count,) or present.dtype.kind != "b":
        raise ValueError("raw_load_packet_present must be a boolean cycle vector")
    if status.shape != (count,) or status.dtype.kind not in "iu" or np.any(status > 5):
        raise ValueError("load_packet_status must contain packet status codes 0 through 5")
    return count, metadata


def _load_trace(path: Path) -> tuple[dict[str, np.ndarray], int, dict]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    count, metadata = _validate_measured_trace(arrays)
    return arrays, count, metadata


def _repository_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"replay input must be inside the repository: {path}") from error


def _target(arrays: dict[str, np.ndarray], index: int) -> FrankaTarget:
    return FrankaTarget(
        **{
            name: arrays[f"target_{name}"][index].copy()
            for name in ("position", "rotation", "linear_velocity", "angular_velocity")
        },
        normal_force=float(arrays["target_normal_force"][index]),
    )


def _python_replay(
    arrays: dict[str, np.ndarray], count: int, metadata: dict
) -> dict[str, np.ndarray]:
    frame = SurfaceFrame(arrays["controller_frame_rotation"])
    controller = _load_aware(
        SurfaceAdaptiveController(
            frame,
            tangential_mode="online",
            rotation_gain_scale=metadata["rotation_gain_scale"],
        ),
        metadata["minimum_force_n"],
        metadata["maximum_force_n"],
    )
    controller.reset(_state(arrays, 0))
    compensation = controller._base.tangential
    observer = LoadBudgetObserver(controller)
    gate = LoadPacketGate(metadata["maximum_packet_age_s"])
    rows = {name: [] for name in REPLAY_FIELDS}
    dt = float(arrays["dt"])
    for index in range(count):
        state = _state(arrays, index)
        target = _target(arrays, index)
        packet = RawLoadPacket(
            bool(arrays["raw_load_packet_present"][index]),
            arrays["raw_load_packet_force"][index],
            float(arrays["raw_load_packet_stamp"][index]),
        )
        decision = gate.evaluate(packet, float(arrays["time"][index]))
        if decision.available:
            compensation.set_force_measurement(decision.force_local)
        before_coefficient = controller.equivalent_tangential_coefficient
        before_ready = controller.tangential_update_ready
        snapshot = observer.before_compute(target, dt)
        projection_before = controller.torque_projection_fallback_count
        wrench = np.asarray(controller.compute(state, target, dt), dtype=float)
        diagnostics = observer.after_compute(snapshot)
        assert state.actuation is not None
        values = {
            "commanded_wrench": wrench,
            "commanded_torque": state.actuation.joint_torque(wrench),
            "contact_blend": controller.contact_blend,
            "governed_normal_lead_m": controller.last_governed_normal_lead_m,
            "torque_projection_scale": controller.last_torque_projection_scale,
            "requested_tangential_force_world": controller.requested_tangential_force_world,
            "load_budget_applied_n": compensation.applied_budget_n,
            "load_budget_next_n": compensation.next_budget_n,
            "load_estimate_n": compensation.load_estimate_n,
            "load_budget_updated": compensation.measurement_used_for_budget,
            "load_projected_n": compensation.projected_load_n,
            "load_force_local": decision.force_local,
            "load_measurement_time_s": decision.stamp_s,
            "load_measurement_age_s": decision.age_s,
            "load_compensation_force_local": compensation.last_force,
            "load_projection_accepted": bool(
                controller.last_torque_projection_scale == 1.0
                and controller.torque_projection_fallback_count == projection_before
            ),
            "controller_update_ready_before_compute": before_ready,
            "controller_update_ready_after_compute": controller.tangential_update_ready,
            "controller_coefficient_before_compute": before_coefficient,
            "controller_coefficient_after_compute": (controller.equivalent_tangential_coefficient),
            "load_packet_status": int(decision.status),
            "load_measurement_available": decision.available,
            "measured_in_contact": controller._base.base.base.in_contact,
            **diagnostics,
        }
        for name in REPLAY_FIELDS:
            rows[name].append(values[name])
    return {name: np.asarray(values) for name, values in rows.items()}


def _compare(actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> dict:
    numeric_errors = {}
    exact_mismatches = {}
    for name in NUMERIC_FIELDS:
        left, right = np.asarray(actual[name]), np.asarray(expected[name])
        if left.shape != right.shape or not np.all(np.isfinite(left)):
            raise ValueError(f"invalid replay numeric field: {name}")
        numeric_errors[name] = float(np.max(np.abs(left - right), initial=0.0))
    for name in (*BOOLEAN_FIELDS, *INTEGER_FIELDS):
        left, right = np.asarray(actual[name]), np.asarray(expected[name])
        if left.shape != right.shape:
            raise ValueError(f"invalid replay exact field shape: {name}")
        exact_mismatches[name] = int(np.count_nonzero(left != right))
    return {
        "max_abs_errors": numeric_errors,
        "exact_mismatch_counts": exact_mismatches,
        "matches": bool(
            all(error <= TOLERANCE for error in numeric_errors.values())
            and not any(exact_mismatches.values())
        ),
    }


def _encode_probe_input(arrays: dict[str, np.ndarray], count: int) -> str:
    lines = [" ".join(_numbers(arrays["controller_frame_rotation"]))]
    dt = float(arrays["dt"])
    for index in range(count):
        state = _state(arrays, index)
        sample_time = float(arrays["time"][index])
        fields = [
            str(index),
            str(int(index == 0)),
            format(sample_time, ".17g"),
            format(sample_time, ".17g"),
            format(dt, ".17g"),
            "1",
        ]
        for value in (
            state.position,
            state.rotation,
            state.linear_velocity,
            state.angular_velocity,
            [state.normal_force],
            arrays["target_position"][index],
            arrays["target_rotation"][index],
            arrays["target_linear_velocity"][index],
            arrays["target_angular_velocity"][index],
            [arrays["target_normal_force"][index]],
            arrays["cartesian_jacobian"][index],
            arrays["joint_torque_offset"][index],
            arrays["lower_torque_limit"][index],
            arrays["upper_torque_limit"][index],
        ):
            fields += _numbers(value)
        fields.append(str(int(arrays["raw_load_packet_present"][index])))
        fields += _numbers(arrays["raw_load_packet_force"][index])
        fields += _numbers([arrays["raw_load_packet_stamp"][index]])
        lines.append(" ".join(fields))
    return "\n".join(lines)


def _default_traces() -> list[Path]:
    robustness = sorted(DEFAULT_ROBUSTNESS.glob("*.npz")) if DEFAULT_ROBUSTNESS.is_dir() else []
    if not DEFAULT_DEMO.is_file() or len(robustness) != 27:
        raise FileNotFoundError("default replay requires the demo and all 27 robustness traces")
    return [DEFAULT_DEMO, *robustness]


def _source_hashes() -> dict[str, str]:
    missing = [path for path in SOURCE_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"replay source missing: {missing[0]}")
    hashes = dict(baseline_source_identity())
    hashes.update({path.relative_to(REPOSITORY).as_posix(): _sha256(path) for path in SOURCE_PATHS})
    return dict(sorted(hashes.items()))


def _write_archive(output: Path, report: dict, frozen: dict) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        report_path = staging / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "identity": "measured-budget-cpp-replay-v1",
            "trace_count": report["trace_count"],
            "independent_trace_count": report["independent_trace_count"],
            "demo_trace_count": report["demo_trace_count"],
            "sample_count": report["sample_count"],
            "error_field_count": len(NUMERIC_FIELDS),
            "exact_field_count": len(BOOLEAN_FIELDS) + len(INTEGER_FIELDS),
            "input_sha256": frozen["traces"],
            "source_sha256": frozen["sources"],
            "probe_run_path": frozen["probe_run_path"],
            "binary_sha256": frozen["probe_binary"],
            "aggregate_results": report["aggregate_results"],
            "artifact_sha256": {"report.json": _sha256(report_path)},
            "artifact_count": 1,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / "COMPLETE").write_text(_sha256(manifest_path) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"output appeared during verification: {output}")
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _parse_probe_output(output: str, count: int) -> dict[str, np.ndarray]:
    """Parse the measured-mode CSV contract emitted by the C++ probe."""
    rows = list(csv.reader(output.splitlines()))
    if len(rows) != count:
        raise ValueError(f"C++ measured probe returned {len(rows)} rows for {count} samples")
    parsed = {name: [] for name in REPLAY_FIELDS}
    boolean_columns = (4, 5, 23, 25, 34, 39, 41, 42, 43, 44, 52)
    projection_statuses = {
        "unchanged",
        "scaled",
        "nominal_outside",
        "nonfinite",
        "verification_failed",
    }
    for index, row in enumerate(rows):
        if len(row) != 53 or row[:2] != ["surface_case", str(index)]:
            raise ValueError(f"malformed C++ measured probe row {index}")
        if row[2] != "accepted":
            raise ValueError(f"C++ watchdog rejected recorded row {index}: {row[2]}")
        if row[3] not in projection_statuses:
            raise ValueError(f"invalid projection status in C++ measured probe row {index}")
        if any(row[column] not in {"0", "1"} for column in boolean_columns):
            raise ValueError(f"invalid boolean in C++ measured probe row {index}")
        try:
            packet_status = int(row[24])
            numeric = np.asarray(
                [
                    value
                    for column, value in enumerate(row[6:], start=6)
                    if column not in boolean_columns and column != 24
                ],
                dtype=float,
            )
            if packet_status not in range(6) or not np.all(np.isfinite(numeric)):
                raise ValueError
            values = {
                "commanded_wrench": np.asarray(row[6:12], dtype=float),
                "commanded_torque": np.asarray(row[45:52], dtype=float),
                "contact_blend": float(row[12]),
                "governed_normal_lead_m": float(row[19]),
                "torque_projection_scale": float(row[20]),
                "requested_tangential_force_world": np.asarray(row[16:19], dtype=float),
                "load_budget_applied_n": float(row[31]),
                "load_budget_next_n": float(row[32]),
                "load_estimate_n": float(row[33]),
                "load_projected_n": float(row[35]),
                "load_force_local": np.asarray(row[26:29], dtype=float),
                "load_measurement_time_s": float(row[29]),
                "load_measurement_age_s": float(row[30]),
                "load_compensation_force_local": np.asarray(row[36:39], dtype=float),
                "controller_coefficient_before_compute": float(row[40]),
                "controller_coefficient_after_compute": float(row[15]),
                "diagnostic_corrected_force_n": float(row[13]),
                "load_budget_updated": bool(int(row[34])),
                "load_projection_accepted": bool(int(row[39])),
                "controller_update_ready_before_compute": bool(int(row[41])),
                "controller_update_ready_after_compute": bool(int(row[23])),
                "diagnostic_compensation_active": bool(int(row[42])),
                "diagnostic_amplitude_capped": bool(int(row[43])),
                "diagnostic_slew_limited": bool(int(row[44])),
                "load_measurement_available": bool(int(row[25])),
                "measured_in_contact": bool(int(row[52])),
                "load_packet_status": packet_status,
            }
        except (TypeError, ValueError) as error:
            raise ValueError(f"non-numeric C++ measured probe row {index}") from error
        for name in REPLAY_FIELDS:
            parsed[name].append(values[name])
    return {name: np.asarray(values) for name, values in parsed.items()}


def _run_trace(path: Path, probe: Path) -> tuple[dict, dict]:
    arrays, count, metadata = _load_trace(path)
    python = _python_replay(arrays, count, metadata)
    command = [
        str(probe),
        "--mode",
        "online",
        "--rotation-gain-scale",
        format(metadata["rotation_gain_scale"], ".17g"),
        "--load-budget",
        format(metadata["minimum_force_n"], ".17g"),
        format(metadata["maximum_force_n"], ".17g"),
        "--maximum-packet-age",
        format(metadata["maximum_packet_age_s"], ".17g"),
    ]
    completed = subprocess.run(
        command,
        input=_encode_probe_input(arrays, count),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise ValueError(f"C++ measured probe failed for {path}: {detail}")
    cpp = _parse_probe_output(completed.stdout, count)
    python_vs_trace = _compare(python, arrays)
    cpp_vs_python = _compare(cpp, python)
    cpp_vs_trace = _compare(cpp, arrays)
    packet_status = np.asarray(python["load_packet_status"], dtype=int)
    summary = {
        "trace": _repository_path(path),
        "sample_count": count,
        "metadata": metadata,
        "matches": bool(
            python_vs_trace["matches"] and cpp_vs_python["matches"] and cpp_vs_trace["matches"]
        ),
        "python_vs_trace": python_vs_trace,
        "cpp_vs_python": cpp_vs_python,
        "cpp_vs_trace": cpp_vs_trace,
        "gate_effect": {
            "accepted_samples": int(np.count_nonzero(packet_status == 1)),
            "rejected_samples": int(np.count_nonzero(packet_status != 1)),
            "status_counts": {
                PACKET_STATUS_NAMES[code]: int(np.count_nonzero(packet_status == code))
                for code in PACKET_STATUS_NAMES
            },
            "budget_update_samples": int(np.count_nonzero(python["load_budget_updated"])),
        },
    }
    return summary, cpp


def _run_trace_task(arguments: tuple[Path, Path]) -> dict:
    return _run_trace(*arguments)[0]


def _aggregate_results(summaries: list[dict]) -> dict:
    result = {}
    for comparison_name in ("python_vs_trace", "cpp_vs_python", "cpp_vs_trace"):
        comparisons = [row[comparison_name] for row in summaries]
        result[comparison_name] = {
            "max_abs_errors": {
                name: max(item["max_abs_errors"][name] for item in comparisons)
                for name in NUMERIC_FIELDS
            },
            "exact_mismatch_counts": {
                name: sum(item["exact_mismatch_counts"][name] for item in comparisons)
                for name in (*BOOLEAN_FIELDS, *INTEGER_FIELDS)
            },
            "matches": all(item["matches"] for item in comparisons),
        }
    return result


def verify(traces: list[Path], probe: Path, output: Path) -> Path:
    output = output.absolute()
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError("output path must not contain symlinks")
    traces = [path.resolve() for path in traces]
    probe = probe.resolve()
    output = output.resolve()
    if not traces or len(set(traces)) != len(traces):
        raise ValueError("traces must be a nonempty list of unique paths")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    if not probe.is_file():
        raise FileNotFoundError(f"C++ surface probe not found: {probe}")
    for trace in traces:
        if not trace.is_file():
            raise FileNotFoundError(f"measured-budget trace not found: {trace}")
    frozen = {
        "traces": {_repository_path(path): _sha256(path) for path in traces},
        "sources": _source_hashes(),
        "probe_run_path": str(probe),
        "probe_binary": _sha256(probe),
    }
    worker_count = min(4, len(traces), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        summaries = list(executor.map(_run_trace_task, ((path, probe) for path in traces)))
    current = {
        "traces": {_repository_path(path): _sha256(path) for path in traces},
        "sources": _source_hashes(),
        "probe_run_path": str(probe),
        "probe_binary": _sha256(probe),
    }
    if current != frozen:
        raise RuntimeError("trace, source, or executable changed during replay")
    report = {
        "schema_version": 1,
        "identity": "measured-budget-cpp-replay-v1",
        "scope": "deterministic_full_input_controller_replay_not_live_physics",
        "timestamp_policy": {
            "state_watchdog": "trace time is supplied as both sample time and watchdog now",
            "auxiliary_packet": "recorded raw packet stamp is gated against trace time",
        },
        "tolerance": TOLERANCE,
        "protocol": {
            "identity": PROBE_PROTOCOL,
            "trace_schema_version": 2,
            "reset": "one controller and packet gate reset before trace row zero",
            "numeric_tolerance_fields": list(NUMERIC_FIELDS),
            "exact_fields": [*BOOLEAN_FIELDS, *INTEGER_FIELDS],
            "packet_status_codes": {str(code): name for code, name in PACKET_STATUS_NAMES.items()},
            "binary_audit": (
                "the run records the probe path and hash; portable audit checks the binary "
                "only when --probe is supplied"
            ),
        },
        "trace_count": len(summaries),
        "demo_trace_count": sum(path == DEFAULT_DEMO.resolve() for path in traces),
        "independent_trace_count": sum(path != DEFAULT_DEMO.resolve() for path in traces),
        "sample_count": sum(row["sample_count"] for row in summaries),
        "replay_workers": worker_count,
        "matches": all(row["matches"] for row in summaries),
        "traces": summaries,
        "aggregate_results": _aggregate_results(summaries),
        "frozen_sha256": frozen,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "hostname": platform.node(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "compiler": _compiler_identity(probe),
        },
    }
    return _write_archive(output, report, frozen)


def audit_archive(directory: Path, probe: Path | None = None) -> dict:
    directory = directory.absolute()
    if any(path.is_symlink() for path in (directory, *directory.parents)):
        raise ValueError("replay archive path must not contain symlinks")
    directory = directory.resolve()
    manifest_path = directory / "manifest.json"
    report_path = directory / "report.json"
    if not all(path.is_file() for path in (manifest_path, report_path, directory / "COMPLETE")):
        raise ValueError("incomplete measured-budget C++ replay archive")
    if (directory / "COMPLETE").read_text(encoding="utf-8").strip() != _sha256(manifest_path):
        raise ValueError("COMPLETE does not match manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("identity") != "measured-budget-cpp-replay-v1":
        raise ValueError("replay manifest identity differs")
    if (
        manifest.get("artifact_count") != 1
        or manifest.get("error_field_count") != len(NUMERIC_FIELDS)
        or manifest.get("exact_field_count") != len(BOOLEAN_FIELDS) + len(INTEGER_FIELDS)
        or manifest.get("artifact_sha256") != {"report.json": _sha256(report_path)}
    ):
        raise ValueError("replay artifact hashes differ")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frozen = report.get("frozen_sha256", {})
    if (
        report.get("identity") != manifest["identity"]
        or report.get("tolerance") != TOLERANCE
        or report.get("protocol", {}).get("identity") != PROBE_PROTOCOL
        or report.get("protocol", {}).get("numeric_tolerance_fields") != list(NUMERIC_FIELDS)
        or report.get("protocol", {}).get("exact_fields") != [*BOOLEAN_FIELDS, *INTEGER_FIELDS]
        or report.get("trace_count") != manifest.get("trace_count")
        or report.get("independent_trace_count") != manifest.get("independent_trace_count")
        or report.get("demo_trace_count") != manifest.get("demo_trace_count")
        or report.get("independent_trace_count", -1) + report.get("demo_trace_count", -1)
        != report.get("trace_count")
        or report.get("sample_count") != manifest.get("sample_count")
        or len(report.get("traces", [])) != manifest.get("trace_count")
        or sum(row.get("sample_count", -1) for row in report.get("traces", []))
        != report.get("sample_count")
        or report.get("matches") != all(row.get("matches") for row in report.get("traces", []))
        or manifest.get("input_sha256") != frozen.get("traces")
        or manifest.get("source_sha256") != frozen.get("sources")
        or manifest.get("probe_run_path") != frozen.get("probe_run_path")
        or manifest.get("binary_sha256") != frozen.get("probe_binary")
    ):
        raise ValueError("replay report counts or result summary differ")
    if report.get("aggregate_results") != _aggregate_results(report["traces"]) or manifest.get(
        "aggregate_results"
    ) != report.get("aggregate_results"):
        raise ValueError("aggregate replay errors differ")
    trace_names = [row.get("trace") for row in report["traces"]]
    if len(set(trace_names)) != len(trace_names) or set(trace_names) != set(frozen["traces"]):
        raise ValueError("replay trace inventory differs from frozen inputs")
    for row in report["traces"]:
        comparisons = [
            row.get(name, {}) for name in ("python_vs_trace", "cpp_vs_python", "cpp_vs_trace")
        ]
        for comparison in comparisons:
            errors = comparison.get("max_abs_errors", {})
            mismatches = comparison.get("exact_mismatch_counts", {})
            if (
                set(errors) != set(NUMERIC_FIELDS)
                or set(mismatches) != set(BOOLEAN_FIELDS) | set(INTEGER_FIELDS)
                or any(not np.isfinite(value) or value < 0 for value in errors.values())
                or any(not isinstance(value, int) or value < 0 for value in mismatches.values())
            ):
                raise ValueError("invalid replay error inventory")
            expected_match = all(
                value <= report["tolerance"] for value in errors.values()
            ) and not any(mismatches.values())
            if comparison.get("matches") is not expected_match:
                raise ValueError("replay comparison status disagrees with errors")
        if row.get("matches") is not all(item["matches"] for item in comparisons):
            raise ValueError("trace replay status disagrees with comparisons")
        gate = row.get("gate_effect", {})
        status_counts = gate.get("status_counts", {})
        if (
            set(status_counts) != set(PACKET_STATUS_NAMES.values())
            or any(type(value) is not int or value < 0 for value in status_counts.values())
            or type(gate.get("accepted_samples")) is not int
            or type(gate.get("rejected_samples")) is not int
            or type(gate.get("budget_update_samples")) is not int
            or not 0 <= gate["budget_update_samples"] <= row["sample_count"]
            or sum(status_counts.values()) != row["sample_count"]
            or gate.get("accepted_samples") != status_counts["accepted"]
            or gate.get("rejected_samples") != row["sample_count"] - status_counts["accepted"]
        ):
            raise ValueError("trace gate-effect counts differ")
    for name, expected in frozen["traces"].items():
        relative = Path(name)
        path = REPOSITORY / relative
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe replay input path: {name}")
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"replay input changed: {name}")
    if _source_hashes() != frozen["sources"]:
        raise ValueError("replay source changed")
    binary_verification = "not_checked"
    if probe is not None:
        probe = probe.resolve()
        if not probe.is_file() or _sha256(probe) != frozen["probe_binary"]:
            raise ValueError("specified replay binary differs")
        binary_verification = "verified"
    audited = dict(report)
    audited["audit"] = {"binary_verification": binary_verification}
    return audited


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "audit"), nargs="?", default="run")
    parser.add_argument("directory", type=Path, nargs="?")
    parser.add_argument("--trace", type=Path, nargs="+")
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    if arguments.action == "audit":
        report = audit_archive(arguments.directory or arguments.output, arguments.probe)
        print(
            json.dumps(
                {
                    "matches": report["matches"],
                    "binary_verification": report["audit"]["binary_verification"],
                },
                allow_nan=False,
            )
        )
        return
    if arguments.directory is not None:
        parser.error("a positional directory is supported only by the audit action")
    traces = arguments.trace if arguments.trace else _default_traces()
    output = verify(traces, arguments.probe or DEFAULT_PROBE, arguments.output)
    report = audit_archive(output)
    print(output)
    if not report["matches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
