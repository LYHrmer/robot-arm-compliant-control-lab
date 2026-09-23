"""The manual diagnostic must preserve failures and isolate numeric controls."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from tools.ci import recovery_runtime_matrix as matrix

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PROFILES = ["native", "blas_haswell", "blas_sandybridge", "numpy_portable"]


@pytest.fixture
def run_matrix(monkeypatch, tmp_path, capsys):
    """Only replace the process boundary; run the real parent CLI and JSON handling."""
    def run(behaviors=None):
        calls = []
        behaviors = behaviors or {}

        def launch(command, **options):
            profile = command[-1]
            calls.append((command, options))
            behavior = behaviors.get(profile, 0)
            if behavior == "timeout":
                raise subprocess.TimeoutExpired(command, 180, output="partial evidence", stderr="timed out")
            runtime = {"portable_targets": ["X86_V4", "AVX512_ICL", "AVX512_SPR"]}
            rows = [{"profile": profile, "runtime": runtime}]
            if behavior != "missing_completion":
                rows.append({"profile": profile, "exact_difference_count": 0})
            return subprocess.CompletedProcess(
                command, behavior if isinstance(behavior, int) else 0,
                stdout="\n".join(json.dumps(row) for row in rows) + "\n",
                stderr=f"{profile}: retained diagnostic stderr\n",
            )

        monkeypatch.setattr(matrix.sys, "argv", ["recovery_runtime_matrix", "--repo", str(tmp_path)])
        monkeypatch.setattr(matrix.subprocess, "run", launch)
        code = matrix.main()
        output = capsys.readouterr()
        rows = [json.loads(line) for line in output.out.splitlines() if line.startswith("{")]
        return code, calls, rows, output

    return run


def test_profiles_start_fresh_and_change_only_one_numeric_control(monkeypatch, run_matrix):
    ambient = {"OPENBLAS_CORETYPE": "ambient-core", "NPY_DISABLE_CPU_FEATURES": "ambient-numpy",
               "GLIBC_TUNABLES": "ambient-glibc", "PYTHONPATH": "/old/ros/site-packages"}
    for name, value in ambient.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("RECOVERY_TEST_SENTINEL", "preserved")
    code, calls, rows, _ = run_matrix()
    assert code == 0 and rows[-1]["diagnostic_status"] == "PASS"
    assert [command[-1] for command, _ in calls] == EXPECTED_PROFILES
    environments = [options["env"] for _, options in calls]
    assert len({id(env) for env in environments}) == 4
    base = environments[0]
    assert not set(ambient) & base.keys()
    assert base["RECOVERY_TEST_SENTINEL"] == "preserved"
    assert {name: base[name] for name in (
        "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "MUJOCO_GL",
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
    )} == {"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "MUJOCO_GL": "disable",
          "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    assert environments[1] == base | {"OPENBLAS_CORETYPE": "Haswell"}
    assert environments[2] == base | {"OPENBLAS_CORETYPE": "Sandybridge"}
    assert environments[3] == base | {"NPY_DISABLE_CPU_FEATURES": "X86_V4,AVX512_ICL,AVX512_SPR"}
    assert all(matrix.os.environ[name] == value for name, value in ambient.items())
    for command, options in calls:
        assert command[:4] == [matrix.sys.executable, "-m", "tools.ci.recovery_runtime_matrix", "--repo"]
        assert command[-2] == "--child"
        assert options["cwd"] == Path(command[4])
        assert options["timeout"] == 180
        assert options["check"] is False
        assert options["capture_output"] is True
        assert options["text"] is True


@pytest.mark.parametrize("failed_profile", EXPECTED_PROFILES)
def test_nonzero_child_stays_failed_and_does_not_prevent_other_profiles(run_matrix, failed_profile):
    code, calls, rows, output = run_matrix({failed_profile: 7})
    assert code == 1
    assert [command[-1] for command, _ in calls] == EXPECTED_PROFILES
    assert rows[-1]["diagnostic_status"] == "FAIL"
    failed = next(row for row in rows[-1]["profile_results"] if row["profile"] == failed_profile)
    assert failed["returncode"] == 7 and failed["completed"] is True
    assert all(f"{profile}: retained diagnostic stderr" in output.err for profile in EXPECTED_PROFILES)


def test_timeout_preserves_partial_evidence_and_still_runs_remaining_profiles(run_matrix):
    code, calls, rows, _ = run_matrix({"blas_haswell": "timeout"})
    assert code == 1
    assert [command[-1] for command, _ in calls] == EXPECTED_PROFILES
    timeout = next(row for row in rows if row.get("error") == "timeout")
    assert timeout["stdout"] == "partial evidence" and timeout["stderr"] == "timed out"
    assert rows[-1]["diagnostic_status"] == "FAIL"


def test_native_timeout_cannot_make_unexecuted_portable_profile_pass(run_matrix):
    code, calls, rows, _ = run_matrix({"native": "timeout"})
    assert code == 1
    assert [command[-1] for command, _ in calls] == EXPECTED_PROFILES[:3]
    portable = rows[-1]["profile_results"][-1]
    assert portable == {"profile": "numpy_portable", "error": "native dispatch metadata unavailable"}
    assert rows[-1]["diagnostic_status"] == "FAIL"


@pytest.mark.parametrize("missing_profile", EXPECTED_PROFILES)
def test_zero_exit_without_completion_is_not_a_success(run_matrix, missing_profile):
    code, calls, rows, _ = run_matrix({missing_profile: "missing_completion"})
    assert code == 1 and len(calls) == 4
    missing = next(row for row in rows[-1]["profile_results"] if row["profile"] == missing_profile)
    assert missing["returncode"] == 0 and missing["completed"] is False
    assert rows[-1]["diagnostic_status"] == "FAIL"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("nested", [False, True])
def test_json_output_rejects_nonfinite_numbers(value, nested):
    with pytest.raises(ValueError):
        matrix.dump({"primitive": [value]} if nested else value)


def test_json_output_preserves_primitive_types_and_signed_zero():
    assert matrix.dump([False, 0, 0.0, -0.0, None]) == "[false, 0, 0.0, -0.0, null]"


def test_forensic_workflow_is_manual_read_only_and_cannot_hide_probe_failure():
    workflow = (ROOT / ".github/workflows/recovery-runtime.yml").read_text()
    trigger = re.search(r"(?ms)^on:\n(.*?)(?=^\S|\Z)", workflow)
    assert trigger is not None and trigger.group(1).strip() == "workflow_dispatch:"
    assert workflow.count("permissions:") == 1
    permission = re.search(r"(?ms)^permissions:\n(.*?)(?=^\S|\Z)", workflow)
    assert permission is not None and permission.group(1).strip() == "contents: read"
    assert "timeout-minutes: 15" in workflow
    assert "fail-fast: false" in workflow
    assert "profile: [locked-core, reported-failure]" in workflow
    assert "python-version-file: environment/python-version" in workflow
    assert 'python-version: "3.11.16"' in workflow
    assert "numpy==2.4.6 mujoco==3.13.0" in workflow
    # An explicit bash shell gives GitHub Actions -e -o pipefail for the tee pipe.
    assert "shell: bash\n        run: python -m tools.ci.recovery_runtime_matrix --repo . | tee recovery-runtime.log" in workflow
    assert "continue-on-error" not in workflow and "|| true" not in workflow
    assert "if: always()\n        uses: actions/upload-artifact@v4" in workflow
    assert "path: recovery-runtime.log" in workflow and "if-no-files-found: error" in workflow
