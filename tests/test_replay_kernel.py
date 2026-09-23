"""A replay qualification check reports the loaded kernel; it never selects one."""

import json
import os

import pytest

from tools.ci import check_replay_kernel as checker


def _library(core="Haswell", threads=1, name="/example/numpy-openblas.so"):
    return {"library": name, "get_corename": core, "get_num_threads": threads,
            "get_config": "example runtime configuration"}


def test_reports_every_qualified_library_without_changing_environment(monkeypatch, capsys):
    records = [_library(), _library(name="/example/other-openblas.so")]
    monkeypatch.setattr(checker, "blas_runtime", lambda: records)
    monkeypatch.setenv("OPENBLAS_CORETYPE", "do-not-change")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "do-not-change")
    before = dict(os.environ)
    assert checker.main() == 0
    assert json.loads(capsys.readouterr().out)["loaded_blas"] == records
    assert dict(os.environ) == before


@pytest.mark.parametrize("records", [
    [], [_library(core="SkylakeX")], [_library(core="Sandybridge")],
    [_library(threads=2)], [_library(threads=0)], [_library(threads=True)],
    [_library(), _library(core="SkylakeX", name="/example/other-openblas.so")],
    [_library(), _library(threads=2, name="/example/other-openblas.so")],
    [{"library": "/example/incomplete-openblas.so"}],
])
def test_rejects_empty_unqualified_or_incomplete_runtime_without_fallback(monkeypatch, capsys, records):
    monkeypatch.setattr(checker, "blas_runtime", lambda: records)
    before = dict(os.environ)
    with pytest.raises(RuntimeError, match="Haswell.*1"):
        checker.main()
    assert json.loads(capsys.readouterr().out)["loaded_blas"] == records
    assert dict(os.environ) == before


def test_unsupported_runtime_error_is_not_swallowed(monkeypatch):
    def unsupported():
        raise RuntimeError("OpenBLAS getter unavailable")

    monkeypatch.setattr(checker, "blas_runtime", unsupported)
    with pytest.raises(RuntimeError, match="getter unavailable"):
        checker.main()


@pytest.mark.parametrize("missing", ["AVX2", "FMA3"])
def test_requires_cpu_instruction_support_before_inspecting_blas(monkeypatch, missing):
    import numpy as np

    monkeypatch.setitem(np._core._multiarray_umath.__cpu_features__, missing, False)
    monkeypatch.setattr(checker, "blas_runtime", lambda: pytest.fail("unsupported CPU reached BLAS"))
    with pytest.raises(RuntimeError, match="AVX2 and FMA3"):
        checker.main()
