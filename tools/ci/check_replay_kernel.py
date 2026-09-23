"""Require the qualified BLAS path for exact frozen-summary replay; never set it."""

from __future__ import annotations

import json

from tools.ci.recovery_runtime_matrix import blas_runtime


def main():
    # NumPy must load its bundled BLAS before the runtime getter inspects it.
    import numpy as np

    features = np._core._multiarray_umath.__cpu_features__
    if not all(features.get(name) for name in ("AVX2", "FMA3")):
        raise RuntimeError("The qualified Haswell path requires AVX2 and FMA3 CPU support.")
    libraries = blas_runtime()
    print(json.dumps({"numpy_version": np.__version__, "loaded_blas": libraries},
                     sort_keys=True, allow_nan=False), flush=True)
    if not libraries or any(
        row.get("get_corename") != "Haswell"
        or type(row.get("get_num_threads")) is not int
        or row["get_num_threads"] != 1
        for row in libraries
    ):
        raise RuntimeError(
            "Exact frozen-summary replay requires every loaded OpenBLAS library "
            "to report Haswell with 1 thread; runtime settings were not changed."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
