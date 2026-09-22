"""Check worked answers, source rejection and physical-plane error calculation."""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools import budget_transfer
from tools.tutorials import reversal_recovery as lab


@pytest.fixture(scope="module")
def experiment():
    return lab.load_experiment()


@pytest.mark.parametrize(
    "method,positions,ready",
    [
        ("adaptive6_8", [2.392606, 1.771404, 0.801344, 1.374777], 7.352),
        ("fixed6", [1.637491, 1.032627, 0.796044, 0.921956], 7.338),
        ("fixed8", [2.562208, 1.185542, 0.797234, 1.010221], 7.374),
    ],
)
def test_published_recovery_answers(experiment, method, positions, ready):
    result = lab.analyze(*experiment[method])
    assert [row["samples"] for row in result["windows"]] == [500, 1000, 1000, 2000]
    np.testing.assert_allclose(
        [row["position_rmse_mm"] for row in result["windows"]], positions, rtol=0, atol=5e-7
    )
    assert result["first_motion_s"] == pytest.approx(7.002)
    assert result["first_ready_s"] == pytest.approx(ready)
    assert result["replay"]["validated_cycles"] == 6000


def test_hold_does_not_mean_high_budget_or_nonzero_request(experiment):
    adaptive = lab.analyze(*experiment["adaptive6_8"])
    fixed = lab.analyze(*experiment["fixed6"])
    hold = adaptive["snapshots"][1]
    assert hold["time_s"] == 7.0 and hold["budget_n"] == 6.0
    assert hold["request_n"] == 0.0 and not hold["ready"]
    assert hold["coefficient"] == adaptive["snapshots"][0]["coefficient"]
    assert hold["coefficient"] > fixed["snapshots"][1]["coefficient"]
    assert fixed["snapshots"][1]["budget_n"] == 6.0
    # Equal-length windows combine squared errors, not the two RMSE values.
    early, late, total = adaptive["windows"][1:]
    assert total["position_rmse_mm"] ** 2 == pytest.approx(
        (early["position_rmse_mm"] ** 2 + late["position_rmse_mm"] ** 2) / 2
    )


def test_physical_tangent_plane_not_controller_frame_and_units():
    trace = {
        "time": np.array([0.0, 0.002]), "dt": np.array(0.002),
        "position": np.array([[0.003, 100, 0.004]] * 2),
        "target_position": np.zeros((2, 3)),
        "linear_velocity": np.array([[0.03, 100, 0.04]] * 2),
        "target_linear_velocity": np.zeros((2, 3)),
        "controller_frame_rotation": np.eye(3),
    }
    result = lab.window_metrics(trace, 90.0, 0.0, 0.004)
    assert result["position_rmse_mm"] == pytest.approx(5.0, abs=1e-9)
    assert result["velocity_rms_m_s"] == pytest.approx(0.05, abs=1e-12)
    with pytest.raises(ValueError, match="every scheduled sample"):
        lab.window_metrics(trace, 90.0, 0.0, 0.006)
    trace["position"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        lab.window_metrics(trace, 90.0, 0.0, 0.004)


def test_corrupted_manifest_rejected_before_loading(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="pinned reference manifest differs"):
        lab.load_experiment(tmp_path)


def test_corrupted_trace_hash_rejected_without_modifying_archive(monkeypatch):
    original_hash = budget_transfer._sha256
    target = f"{lab.SCENARIO}__adaptive6_8.npz"
    monkeypatch.setattr(
        budget_transfer, "_sha256",
        lambda path: "0" * 64 if path.name == target else original_hash(path),
    )
    with pytest.raises(ValueError, match="artifact hash differs"):
        lab.load_experiment()


def test_changed_tracking_data_rejected_by_archived_metric(experiment):
    trace, row = experiment["adaptive6_8"]
    changed = dict(trace)
    changed["position"] = trace["position"].copy()
    changed["position"][4000:6000, 1] += 0.01
    with pytest.raises(ValueError, match="recomputed metric differs"):
        lab.analyze(changed, row)


def test_documented_stdout_and_events_write_no_files(tmp_path):
    document = (lab.ROOT / "docs/tutorial/labs/03_reversal_recovery.md").read_text()
    expected = re.search(r"```text\n(.*?)\n```", document, re.DOTALL).group(1)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    inherited_paths = [
        str(Path(entry).resolve())
        for entry in environment.get("PYTHONPATH", "").split(os.pathsep) if entry
    ]
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(lab.ROOT / "src"), str(lab.ROOT), *inherited_paths)
    )
    command = [sys.executable, "-m", "tools.tutorials.reversal_recovery"]
    completed = subprocess.run(
        command, cwd=tmp_path, env=environment, capture_output=True, text=True, check=True, timeout=60,
    )
    assert completed.stdout.strip() == expected
    events = subprocess.run(
        [*command, "--events"], cwd=tmp_path, env=environment,
        capture_output=True, text=True, check=True, timeout=60,
    )
    assert "first reverse target=7.002 s; first ready=7.352 s" in events.stdout
    assert "t=7.000 budget=6.000000 N mu=0.616769473 request=0.000000 N" in events.stdout
    assert list(tmp_path.iterdir()) == []
