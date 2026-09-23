"""Read-only four-process diagnostic; not an archive verifier or compatibility fix."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from tools.ci.diagnose_recovery_replay import differences

CONTROLS = ("OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES", "GLIBC_TUNABLES")
PROFILES = ("native", "blas_haswell", "blas_sandybridge", "numpy_portable")


def dump(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def blas_runtime():
    paths = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                    if "openblas" in line.lower() and line.split()[-1].startswith("/")})
    found = []
    for path in paths:
        lib = ctypes.CDLL(path)
        row = {"library": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for name, result_type in (("get_corename", ctypes.c_char_p),
                                  ("get_config", ctypes.c_char_p),
                                  ("get_num_threads", ctypes.c_int)):
            for prefix in ("scipy_openblas_", "openblas_"):
                for suffix in ("64_", "", "_"):
                    symbol = prefix + name + suffix
                    if hasattr(lib, symbol):
                        func = getattr(lib, symbol)
                        func.restype, func.argtypes = result_type, []
                        result = func()
                        row[name] = result.decode() if isinstance(result, bytes) else result
                        row[name + "_symbol"] = symbol
                        break
                if name in row:
                    break
            if name not in row:
                raise RuntimeError(f"cannot inspect OpenBLAS {name}: {path}")
        found.append(row)
    if not found:
        raise RuntimeError("no loaded OpenBLAS library found; this probe requires OpenBLAS")
    return found


def runtime(np):
    cpu = np._core._multiarray_umath
    # These are the compiled dispatch targets, not individual CPUID aliases.
    targets = [name for name in cpu.__cpu_dispatch__
               if name == "X86_V4" or name.startswith("AVX512")]
    info = np.lib.introspect.opt_func_info(
        func_name="^(add|subtract|multiply|divide|sqrt|square|expm1|tanh)$")
    models = sorted({line.split(":", 1)[1].strip()
                     for line in Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name")})
    return {"python": platform.python_version(), "platform": platform.platform(),
            "cpu_models": models, "packages": {name: importlib.metadata.version(name)
            for name in ("numpy", "mujoco")}, "environment": {name: os.environ.get(name)
            for name in (*CONTROLS, "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "numpy_baseline": cpu.__cpu_baseline__, "numpy_dispatch": cpu.__cpu_dispatch__,
            "numpy_features": cpu.__cpu_features__, "portable_targets": targets,
            "numpy_selected_functions": info, "blas": blas_runtime()}


def child(repo, profile):
    import numpy as np

    from tools.ci.recovery_first_divergence import first_divergence
    from tools.reversal_recovery import study

    actual = runtime(np)
    print(dump({"profile": profile, "runtime": actual}), flush=True)
    control_errors = []
    if profile.startswith("blas_"):
        expected = profile.removeprefix("blas_").lower()
        if any(row["get_corename"].lower() != expected for row in actual["blas"]):
            control_errors.append(f"requested {expected} OpenBLAS core was not selected by name")
    if profile == "numpy_portable":
        active = [name for name in actual["portable_targets"]
                  if actual["numpy_features"].get(name)]
        selected = [(name, signature, row["current"])
                    for name, signatures in actual["numpy_selected_functions"].items()
                    for signature, row in signatures.items()
                    if "AVX512" in row["current"] or "X86_V4" in row["current"]]
        if active or selected:
            control_errors.append(f"AVX512 dispatch is still active: {active!r}; {selected!r}")
    archive = repo / "results/franka_reversal_recovery"
    def stamp():
        return {str(p.relative_to(archive)): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in archive.rglob("*") if p.is_file()}

    before = stamp()
    manifest = study.previous.archive.verify_archive(archive)
    source_check = study.verify_sources(json.loads((archive / "source_hashes.json").read_text()),
                                        manifest["source_commit"])
    protocol = study.protocol_document()
    study.verify_comparison(json.loads((archive / "protocol.json").read_text()), protocol)
    saved = json.loads((archive / "comparison.json").read_text())
    rebuilt = json.loads(dump(study.collect(archive, protocol, execute=False)))
    changes = list(differences(saved, rebuilt))
    arithmetic = first_divergence()
    if stamp() != before:
        raise RuntimeError("archive files changed during read-only probe")
    print(dump({"profile": profile, "source_check": source_check,
                "control_errors": control_errors,
                "manifest_sha256": hashlib.sha256((archive / "manifest.json").read_bytes()).hexdigest(),
                "exact_difference_count": len(changes), "differences": changes,
                "first_trace_arithmetic": arithmetic,
                "same_status": saved["status"] == rebuilt["status"],
                "same_pair_comparisons": dump(saved["comparisons"]) == dump(rebuilt["comparisons"]),
                "archive_stamp_unchanged": True}), flush=True)
    return int(bool(changes or control_errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--child", choices=PROFILES)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    if args.child:
        return child(repo, args.child)
    base = os.environ.copy()
    for name in (*CONTROLS, "PYTHONPATH"):
        base.pop(name, None)
    base.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", MUJOCO_GL="disable",
                OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    results, native_runtime = [], None
    for profile in PROFILES:
        env = base.copy()
        if profile.startswith("blas_"):
            env["OPENBLAS_CORETYPE"] = profile.removeprefix("blas_").title()
        if profile == "numpy_portable":
            if native_runtime is None or not native_runtime["portable_targets"]:
                results.append({"profile": profile, "error": "native dispatch metadata unavailable"})
                continue
            env["NPY_DISABLE_CPU_FEATURES"] = ",".join(native_runtime["portable_targets"])
        command = [sys.executable, "-m", "tools.ci.recovery_runtime_matrix",
                   "--repo", str(repo), "--child", profile]
        try:
            proc = subprocess.run(command, cwd=repo, env=env, text=True, capture_output=True,
                                  timeout=180, check=False)
        except subprocess.TimeoutExpired as error:
            print(dump({"profile": profile, "error": "timeout", "stdout": str(error.stdout),
                        "stderr": str(error.stderr)}), flush=True)
            results.append({"profile": profile, "error": "timeout"})
            continue
        # Preserve every diagnostic leaf, warning and traceback; never truncate.
        print(proc.stdout, end="", flush=True)
        print(proc.stderr, end="", file=sys.stderr, flush=True)
        records = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
        actual = next((row["runtime"] for row in records if "runtime" in row), None)
        completed = any("exact_difference_count" in row for row in records)
        if profile == "native":
            native_runtime = actual
        results.append({"profile": profile, "returncode": proc.returncode, "completed": completed})
        if native_runtime is not None and actual is not None:
            print(dump({"profile": profile, "runtime_differences_from_native":
                        list(differences(native_runtime, actual))}), flush=True)
    failed = any(row.get("returncode") != 0 or not row.get("completed") for row in results)
    print(dump({"profile_results": results, "diagnostic_status": "FAIL" if failed else "PASS",
                "note": "No compatibility claim; all original exact differences remain failures."}), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
