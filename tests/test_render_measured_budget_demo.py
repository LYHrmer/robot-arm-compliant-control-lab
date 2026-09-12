import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/diagnostics/render_measured_budget_demo.py"
SPEC = importlib.util.spec_from_file_location("render_measured_budget_demo", SCRIPT)
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


def _case(count=4):
    config = SimpleNamespace(duration=count * 0.002, timestep=0.002, seed=11)
    scenario = SimpleNamespace(wall_yaw_deg=0.0)
    return SimpleNamespace(config=config, scenario=scenario)


def _trace(count=4):
    return {
        "time": np.arange(count) * 0.002,
        "q": np.zeros((count, 7)),
        "position": np.column_stack((np.zeros(count), np.arange(count) * 0.001, np.zeros(count))),
        "target_position": np.zeros((count, 3)),
        "true_normal_force": np.arange(count, dtype=float),
        "target_normal_force": np.full(count, 12.0),
        "load_compensation_force_local": np.column_stack(
            (np.zeros(count), np.arange(count, dtype=float), np.zeros(count))
        ),
        "load_budget_applied_n": np.linspace(6.0, 8.0, count),
        "load_budget_next_n": np.linspace(6.0, 8.0, count),
    }


def test_display_signals_use_surface_tangent_and_local_tangent_force():
    trace = _trace()
    signals = demo.validate_trace(trace, _case())
    np.testing.assert_allclose(signals["tangent_error_mm"], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(signals["tangent_compensation_n"], [0.0, 1.0, 2.0, 3.0])


def test_trace_rejects_bad_clock_and_budget():
    trace = _trace()
    trace["time"][2] += 1e-4
    with pytest.raises(ValueError, match="500 Hz clock"):
        demo.validate_trace(trace, _case())
    trace = _trace()
    trace["load_budget_next_n"][1] = 8.1
    with pytest.raises(ValueError, match="6-8 N preset"):
        demo.validate_trace(trace, _case())


def test_representative_case_is_named_not_positional():
    variant, case = demo.representative_case()
    assert variant == "combined"
    assert case.scenario.wall_yaw_deg == 0
    assert (case.config.duration, case.config.timestep, case.config.seed) == (12.0, 0.002, 11)


def test_published_demo_is_complete_and_replays_as_the_declared_case():
    root = SCRIPT.parents[2] / "results/franka_measured_budget_demo"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["identity"] == "franka-measured-budget-representative-demo-v2"
    assert (root / "COMPLETE").read_text().strip() == demo._sha256(root / "manifest.json")
    assert set(manifest["artifact_sha256"]) == {
        "demo.mp4",
        "overview.png",
        "trace.npz",
        "validation_report.json",
    }
    for name, digest in manifest["artifact_sha256"].items():
        assert demo._sha256(root / name) == digest
    assert (root / "overview.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert b"ftyp" in (root / "demo.mp4").read_bytes()[:32]
    with np.load(root / "trace.npz", allow_pickle=False) as archive:
        trace = {name: archive[name] for name in archive.files}
    _, case = demo.representative_case()
    demo.validate_trace(trace, case)
    report = json.loads((root / "validation_report.json").read_text())
    assert report["validated_cycles"] == 6000
    assert report["coefficient_transition_checked"] is True
    assert report["readiness_transition_checked"] is True
    assert manifest["source_identity"]["legacy_measured_budget_v1"]["source_count"] == 153
    assert manifest["source_identity"]["v2_runtime"] == {
        "tools/measured_budget_trial.py": demo._sha256(
            SCRIPT.parents[1] / "measured_budget_trial.py"
        ),
        "tools/measured_budget_validation.py": demo._sha256(
            SCRIPT.parents[1] / "measured_budget_validation.py"
        ),
    }


def test_mock_generation_hashes_trace_video_and_overview(tmp_path, monkeypatch):
    trace, case = _trace(), _case()
    validation = {"validated_cycles": len(trace["time"])}
    monkeypatch.setattr(demo, "run_case", lambda: (trace, "combined", case, validation))
    monkeypatch.setattr(demo, "source_identity", lambda: {"v2_runtime": {}})
    monkeypatch.setattr(demo.shutil, "which", lambda _: "/fake/ffmpeg")
    monkeypatch.setattr(demo.subprocess, "check_output", lambda *_a, **_k: "ffmpeg test\n")
    monkeypatch.setattr(demo.interaction, "case_document", lambda *_: {"variant": "combined"})

    def render(path, *_args):
        (path / "demo.mp4").write_bytes(b"video")
        (path / "overview.png").write_bytes(b"preview")

    monkeypatch.setattr(demo, "_render", render)
    output = demo.generate_demo(tmp_path / "output")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["preset"] == {
        "minimum_force_n": 6.0,
        "maximum_force_n": 8.0,
        "rotation_gain_scale": 1.0,
        "velocity_error_time_s": 0.05,
    }
    assert set(manifest["artifact_sha256"]) == {
        "demo.mp4",
        "overview.png",
        "trace.npz",
        "validation_report.json",
    }
    for name, digest in manifest["artifact_sha256"].items():
        assert demo._sha256(output / name) == digest
    assert (output / "COMPLETE").read_text().strip() == demo._sha256(output / "manifest.json")
    with pytest.raises(ValueError, match="absent or empty"):
        demo.generate_demo(output)


def test_missing_ffmpeg_leaves_no_output(tmp_path, monkeypatch):
    monkeypatch.setattr(demo.shutil, "which", lambda _: None)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="ffmpeg"):
        demo.generate_demo(output)
    assert not output.exists()
