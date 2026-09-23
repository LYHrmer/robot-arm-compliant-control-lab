"""Bind the published runtime diagnosis to its recorded controls and scope."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "0ce5c97daaa3cdd814ec5b21e116a0ee4270da29"
PROFILES = ("native", "blas_haswell", "blas_sandybridge", "numpy_portable")
BLAS_PATHS = {
    "root/blas/0/get_config", "root/blas/0/get_corename",
    "root/environment/OPENBLAS_CORETYPE",
}


def test_runtime_evidence_preserves_four_profile_results_and_controls():
    evidence = json.loads((ROOT / "docs/evidence/recovery-runtime-2026-09-23.json").read_text())
    assert evidence["source_commit"] == COMMIT
    runners = evidence["runners"]
    assert len(runners) == 2
    assert {runner["artifact"] for runner in runners} == {
        f"recovery-runtime-{name}-{COMMIT}" for name in ("locked-core", "reported-failure")
    }
    manifest = hashlib.sha256((ROOT / "results/franka_reversal_recovery/manifest.json").read_bytes()).hexdigest()
    assert manifest == "e8310dca906626267a8d8884aea4a3c6f619e66cc79a8627acbff220604746e5"
    for runner in runners:
        profiles = runner["profiles"]
        assert [profile["profile"] for profile in profiles] == list(PROFILES)
        assert [profile["result"]["exact_difference_count"] for profile in profiles] == [16, 0, 0, 16]
        assert runner["aggregate"]["diagnostic_status"] == "FAIL"
        for profile, name in zip(profiles, PROFILES, strict=True):
            result = profile["result"]
            assert result["profile"] == name
            assert result["control_errors"] == []
            assert result["manifest_sha256"] == manifest
            assert result["source_check"] == "serialization-finalization-only compatibility"
            assert all(result[key] is True for key in (
                "same_status", "same_pair_comparisons", "archive_stamp_unchanged",
            ))
            assert len(result["differences"]) == result["exact_difference_count"]
        for profile, kernel in zip(profiles[1:3], ("Haswell", "Sandybridge"), strict=True):
            changes = profile["runtime_differences_from_native"]
            assert len(changes) == 3
            assert {change["path"] for change in changes} == BLAS_PATHS
            assert profile["blas"][0]["get_corename"] == kernel
        assert profiles[3]["result"]["differences"] == profiles[0]["result"]["differences"]


def test_documentation_retains_performance_failure_and_bitwise_replay_limits():
    environment = (ROOT / "docs/reproducible_environment.md").read_text()
    transfer = (ROOT / "docs/reversal_recovery_transfer.md").read_text()
    assert "evidence/recovery-runtime-2026-09-23.json" in environment
    assert "并没有让所有逐拍字段变成逐位相同" in environment
    assert "也不是跨所有 CPU 的逐位一致性保证" in environment
    for document in (environment, transfer):
        assert "34/36" in document
        assert "整体 `FAIL`" in document
