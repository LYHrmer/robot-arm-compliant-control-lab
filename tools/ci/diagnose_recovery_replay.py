"""Print exact old-summary differences after CI failure; never repair an archive."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
from pathlib import Path


def differences(saved, rebuilt, path="root"):
    """Find differences with the frozen comparator's JSON semantics, no tolerance."""
    if isinstance(saved, dict) and isinstance(rebuilt, dict):
        for key in sorted(saved.keys() | rebuilt.keys()):
            if key not in saved or key not in rebuilt:
                yield {"path": f"{path}/{key}", "kind": "missing_key",
                       "in_saved": key in saved, "in_rebuilt": key in rebuilt}
            else:
                yield from differences(saved[key], rebuilt[key], f"{path}/{key}")
    elif isinstance(saved, list) and isinstance(rebuilt, list):
        if len(saved) != len(rebuilt):
            yield {"path": path, "kind": "length", "saved": len(saved), "rebuilt": len(rebuilt)}
        for index, (left, right) in enumerate(zip(saved, rebuilt)):
            yield from differences(left, right, f"{path}/{index}")
    elif json.dumps(saved, sort_keys=True, allow_nan=False) != json.dumps(
        rebuilt, sort_keys=True, allow_nan=False,
    ):
        yield {"path": path, "kind": "value", "saved": repr(saved), "rebuilt": repr(rebuilt),
               "saved_type": type(saved).__name__, "rebuilt_type": type(rebuilt).__name__,
               "saved_hex": saved.hex() if isinstance(saved, float) else None,
               "rebuilt_hex": rebuilt.hex() if isinstance(rebuilt, float) else None,
               "absolute_difference": abs(saved - rebuilt)
               if type(saved) in (int, float) and type(rebuilt) in (int, float) else None}


def main():
    import numpy as np

    from tools.reversal_recovery import study

    cpu = getattr(getattr(np, "_core", None), "_multiarray_umath", None)
    features = getattr(cpu, "__cpu_features__", {})
    print(json.dumps({
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "mujoco")},
        "enabled_numpy_cpu_features": sorted(name for name, enabled in features.items() if enabled),
        "numeric_environment": {name: os.environ.get(name) for name in (
            "OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NPY_DISABLE_CPU_FEATURES",
        )},
    }, sort_keys=True), flush=True)
    np.show_runtime()
    archive = Path(__file__).resolve().parents[2] / "results/franka_reversal_recovery"
    study.previous.archive.verify_archive(archive)
    saved = json.loads((archive / "comparison.json").read_text())
    # Do not call audit_archive: its generic ValueError is the condition diagnosed.
    rebuilt = study.collect(archive, study.protocol_document(), execute=False)
    rebuilt = json.loads(json.dumps(rebuilt, sort_keys=True, allow_nan=False))
    changed = list(differences(saved, rebuilt))
    print(json.dumps({"exact_difference_count": len(changed), "first_differences": changed[:20],
                      "same_status": saved["status"] == rebuilt["status"],
                      "same_pair_comparisons": json.dumps(saved["comparisons"], sort_keys=True)
                      == json.dumps(rebuilt["comparisons"], sort_keys=True)},
                     sort_keys=True), flush=True)
    return int(bool(changed))


if __name__ == "__main__":
    raise SystemExit(main())
