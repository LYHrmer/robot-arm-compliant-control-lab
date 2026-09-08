"""Verify the C++ surface loop against recorded full-input NPZ traces.

This is deterministic controller replay, not live-physics or hard-real-time evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from compliant_control_lab.surface_replay import (
    _state,
    _validate_trace,
    replay_surface_trace,
)

TOLERANCE = 1e-8
REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    REPOSITORY / "CMakeLists.txt",
    REPOSITORY / "cpp/include/compliant_control_lab/franka_control.hpp",
    REPOSITORY / "cpp/include/compliant_control_lab/surface_control.hpp",
    REPOSITORY / "cpp/include/compliant_control_lab/torque_safety.hpp",
    REPOSITORY / "cpp/src/franka_control.cpp",
    REPOSITORY / "cpp/src/surface_control.cpp",
    REPOSITORY / "cpp/src/torque_safety.cpp",
    REPOSITORY / "cpp/tools/surface_control_benchmark.cpp",
    REPOSITORY / "cpp/tools/surface_controller_probe.cpp",
    REPOSITORY / "src/compliant_control_lab/franka_adaptive.py",
    REPOSITORY / "src/compliant_control_lab/franka_control.py",
    REPOSITORY / "src/compliant_control_lab/surface_control.py",
    REPOSITORY / "src/compliant_control_lab/surface_replay.py",
    REPOSITORY / "src/compliant_control_lab/tangential_compensation.py",
    Path(__file__).resolve(),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numbers(values) -> list[str]:
    return [format(float(value), ".17g") for value in np.asarray(values).reshape(-1)]


def _mode(controller_kind: str) -> str:
    modes = {
        "surface_adaptive": "none",
        "world_safe_adaptive": "none",
        "surface_integral": "integral",
        "surface_friction": "friction",
        "surface_online": "online",
    }
    try:
        return modes[controller_kind]
    except KeyError as error:
        raise ValueError(f"unsupported controller kind: {controller_kind}") from error


def _encode_probe_input(arrays: dict[str, np.ndarray], count: int) -> str:
    lines = [" ".join(_numbers(arrays["controller_frame_rotation"]))]
    dt = float(arrays["dt"])
    for index in range(count):
        state = _state(arrays, index)
        fields = [
            str(index),
            str(int(index == 0)),
            format((index + 1) * dt, ".17g"),
            format((index + 1) * dt, ".17g"),
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
        lines.append(" ".join(fields))
    return "\n".join(lines)


def _parse_probe_output(output: str, count: int) -> list[dict]:
    rows = list(csv.reader(output.splitlines()))
    if len(rows) != count:
        raise ValueError(f"C++ probe returned {len(rows)} rows for {count} samples")
    parsed = []
    for index, row in enumerate(rows):
        if len(row) != 24 or row[:2] != ["surface_case", str(index)]:
            raise ValueError(f"malformed C++ probe row {index}")
        if row[2] != "accepted":
            raise ValueError(f"C++ watchdog rejected recorded row {index}: {row[2]}")
        if row[3] not in {"unchanged", "scaled", "nominal_outside", "nonfinite",
                          "verification_failed"} or any(row[i] not in {"0", "1"} for i in (4, 5, 23)):
            raise ValueError(f"invalid status or boolean in C++ probe row {index}")
        try:
            parsed.append(
                {
                    "projection_status": row[3],
                    "fallback": bool(int(row[4])),
                    "feasible": bool(int(row[5])),
                    "wrench": np.asarray(row[6:12], dtype=float),
                    "contact_blend": float(row[12]),
                    "corrected_force": float(row[13]),
                    "force_rate": float(row[14]),
                    "equivalent_mu": float(row[15]),
                    "tangential_force": np.asarray(row[16:19], dtype=float),
                    "governed_lead": float(row[19]),
                    "projection_scale": float(row[20]),
                    "contact_stiffness": float(row[21]),
                    "gain_scale": float(row[22]),
                    "update_ready": bool(int(row[23])),
                }
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"non-numeric C++ probe row {index}") from error
        if not all(
            np.all(np.isfinite(value))
            for value in parsed[-1].values()
            if not isinstance(value, (str, bool))
        ):
            raise ValueError(f"nonfinite C++ probe row {index}")
    return parsed


def _load_trace(path: Path) -> tuple[dict[str, np.ndarray], int]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    count = _validate_trace(arrays)
    return arrays, count


def _run_trace(path: Path, probe: Path) -> tuple[dict, list[dict]]:
    arrays, count = _load_trace(path)
    kind = str(arrays["controller_kind"].item())
    python_result = replay_surface_trace(path)
    completed = subprocess.run(
        [str(probe), "--mode", _mode(kind)],
        input=_encode_probe_input(arrays, count),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise ValueError(f"C++ probe failed for {path}: {detail}")
    cpp = _parse_probe_output(completed.stdout, count)
    cpp_wrench = np.asarray([row["wrench"] for row in cpp])
    cpp_torque = np.einsum(
        "nij,nj->ni", np.swapaxes(arrays["cartesian_jacobian"], 1, 2), cpp_wrench
    ) + arrays["joint_torque_offset"]
    wrench_error = np.max(np.abs(cpp_wrench - arrays["commanded_wrench"]), axis=1)
    torque_error = np.max(np.abs(cpp_torque - arrays["commanded_torque"]), axis=1)
    coefficients = np.asarray([row["equivalent_mu"] for row in cpp])
    ready = np.asarray([row["update_ready"] for row in cpp], dtype=bool)
    learning_updates = ready & (np.abs(np.diff(coefficients, prepend=coefficients[0])) > 1e-15)
    telemetry_errors = {}
    for field, cpp_field in (
        ("contact_blend", "contact_blend"),
        ("torque_projection_scale", "projection_scale"),
        ("requested_tangential_force_world", "tangential_force"),
    ):
        if field in arrays:
            values = np.asarray([row[cpp_field] for row in cpp])
            telemetry_errors[field] = float(np.max(np.abs(values - arrays[field])))

    recorded_mu_error = None
    if "controller_coefficient_after_compute" in arrays:
        recorded_mu_error = float(np.max(np.abs(
            coefficients - arrays["controller_coefficient_after_compute"]
        )))
    recorded_ready_mismatches = None
    if "controller_update_ready_after_compute" in arrays:
        recorded_ready_mismatches = int(np.count_nonzero(
            ready != arrays["controller_update_ready_after_compute"].astype(bool)
        ))
    matches = bool(
        python_result.matches
        and np.max(wrench_error) <= TOLERANCE
        and np.max(torque_error) <= TOLERANCE
        and (recorded_mu_error is None or recorded_mu_error <= TOLERANCE)
        and (recorded_ready_mismatches is None or recorded_ready_mismatches == 0)
        and all(error <= TOLERANCE for error in telemetry_errors.values())
    )
    summary = {
        "trace": str(path),
        "controller_kind": kind,
        "sample_count": count,
        "matches": matches,
        "cpp_max_wrench_error": float(np.max(wrench_error)),
        "cpp_max_torque_error": float(np.max(torque_error)),
        "recorded_telemetry_max_errors": telemetry_errors,
        "python_cross_check": {
            "matches": python_result.matches,
            "max_wrench_error": python_result.max_wrench_error,
            "max_torque_error": python_result.max_torque_error,
        },
        "equivalent_mu": {
            "initial": float(coefficients[0]),
            "final": float(coefficients[-1]),
            "minimum": float(np.min(coefficients)),
            "maximum": float(np.max(coefficients)),
            "learning_update_count": int(np.count_nonzero(learning_updates)),
            "ready_sample_count": int(np.count_nonzero(ready)),
            "recorded_max_error": recorded_mu_error,
            "recorded_ready_mismatch_count": recorded_ready_mismatches,
        },
    }
    per_step = [
        {
            "index": index,
            "wrench_max_abs_error": float(wrench_error[index]),
            "torque_max_abs_error": float(torque_error[index]),
            "equivalent_mu": float(coefficients[index]),
            "update_ready": bool(ready[index]),
            "projection_status": cpp[index]["projection_status"],
            "projection_scale": cpp[index]["projection_scale"],
        }
        for index in range(count)
    ]
    return summary, per_step


def _compiler_identity(probe: Path) -> dict:
    cache = probe.parent / "CMakeCache.txt"
    compiler = None
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.startswith("CMAKE_CXX_COMPILER:FILEPATH="):
                compiler = line.split("=", 1)[1]
                break
    identity = {"path": compiler, "version": None}
    if compiler:
        completed = subprocess.run(
            [compiler, "--version"], capture_output=True, text=True, check=False
        )
        if completed.returncode == 0 and completed.stdout:
            identity["version"] = completed.stdout.splitlines()[0]
    return identity


def _benchmark(probe: Path) -> dict:
    executable = probe.with_name("compliant_control_surface_benchmark")
    if not executable.is_file():
        raise FileNotFoundError(f"surface benchmark not found: {executable}")
    completed = subprocess.run([str(executable)], capture_output=True, text=True, check=True)
    rows = list(csv.reader(completed.stdout.splitlines()))
    expected = ["samples", "p50_us", "p99_us", "max_us", "overruns_2ms"]
    if [row[0] for row in rows] != expected or any(len(row) != 2 for row in rows):
        raise ValueError("malformed surface benchmark output")
    return {
        "executable": str(executable),
        "binary_sha256": _sha256(executable),
        **{row[0]: float(row[1]) for row in rows},
    }


def verify(traces: list[Path], probe: Path, output: Path) -> Path:
    output = output.absolute()
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError("output path must not contain symlinks")
    traces = [path.resolve() for path in traces]
    probe = probe.resolve()
    output = output.resolve()
    if not traces or len(set(traces)) != len(traces):
        raise ValueError("--trace must contain unique paths")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    if not probe.is_file():
        raise FileNotFoundError(f"C++ surface probe not found: {probe}")
    for trace in traces:
        if not trace.is_file():
            raise FileNotFoundError(f"surface trace not found: {trace}")
    frozen = {
        "traces": {str(path): _sha256(path) for path in traces},
        "sources": {str(path.relative_to(REPOSITORY)): _sha256(path) for path in SOURCE_PATHS},
        "probe_binary": _sha256(probe),
    }
    benchmark_executable = probe.with_name("compliant_control_surface_benchmark")
    if not benchmark_executable.is_file():
        raise FileNotFoundError(f"surface benchmark not found: {benchmark_executable}")
    frozen["benchmark_binary"] = _sha256(benchmark_executable)

    summaries = []
    per_step_rows = []
    for trace in traces:
        summary, steps = _run_trace(trace, probe)
        summaries.append(summary)
        per_step_rows.extend({"trace": str(trace), **row} for row in steps)
    benchmark = _benchmark(probe)
    current = {
        "traces": {str(path): _sha256(path) for path in traces},
        "sources": {str(path.relative_to(REPOSITORY)): _sha256(path) for path in SOURCE_PATHS},
        "probe_binary": _sha256(probe),
        "benchmark_binary": _sha256(benchmark_executable),
    }
    if current != frozen:
        raise RuntimeError("trace, source or executable changed during replay")

    report = {
        "schema_version": 1,
        "scope": "deterministic_full_input_controller_replay_not_live_physics",
        "watchdog_timestamp_policy": (
            "fresh strictly increasing timestamps are synthesized as (index+1)*dt; "
            "this replay does not validate archived sensor transport age"
        ),
        "tolerance": TOLERANCE,
        "matches": all(item["matches"] for item in summaries),
        "traces": summaries,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "hostname": platform.node(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "compiler": _compiler_identity(probe),
        },
        "benchmark": benchmark,
        "frozen_sha256": frozen,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        report_path = staging / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (staging / "per_step.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=per_step_rows[0].keys())
            writer.writeheader()
            writer.writerows(per_step_rows)
        manifest = {
            "report.json": _sha256(report_path),
            "per_step.csv": _sha256(staging / "per_step.csv"),
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / "COMPLETE").write_text(_sha256(manifest_path) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"output appeared during verification: {output}")
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, nargs="+", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = verify(arguments.trace, arguments.probe, arguments.output)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    print(output)
    if not report["matches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
