import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/diagnostics/render_tangential_demo.py"
spec = importlib.util.spec_from_file_location("render_tangential_demo", SCRIPT)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_tangent_error_uses_wall_frame_not_world_yz():
    normal = np.array([0.6, 0.8, 0])
    trace = {
        "position": np.array([normal * 0.1, [-0.008, 0.006, 0]]),
        "target_position": np.zeros((2, 3)),
    }
    np.testing.assert_allclose(
        demo.tangent_error_mm(trace, np.rad2deg(np.arctan2(0.8, 0.6))), [0, 10], atol=1e-12
    )


def test_source_archive_has_required_aligned_fields():
    trace, case, identity = demo.load_source(
        SCRIPT.parents[2] / "results/franka_tangential_development"
    )
    assert len(trace["q"]) == 6000 and len(range(0, 6000, demo.STRIDE)) == 300
    assert case["config"]["duration"] == 12
    assert identity["trace"] == demo.TRACE_NAME


def test_published_video_hashes_and_logged_source_identity():
    root = SCRIPT.parents[2] / "results/franka_tangential_demo"
    manifest = json.loads((root / "manifest.json").read_text())
    assert (root / "COMPLETE").read_text().strip() == demo._sha256(root / "manifest.json")
    assert set(manifest["artifact_sha256"]) == {"demo.mp4", "preview.png"}
    for name, digest in manifest["artifact_sha256"].items():
        assert demo._sha256(root / name) == digest
    assert (root / "preview.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert b"ftyp" in (root / "demo.mp4").read_bytes()[:32]
    assert manifest["disclaimer"] == demo.DISCLAIMER
    assert (manifest["frame_count"], manifest["fps"], manifest["duration_s"]) == (300, 25, 12)
    source = SCRIPT.parents[2] / manifest["source"]["directory"]
    assert manifest["source"]["manifest_sha256"] == demo._sha256(source / "manifest.json")
    assert manifest["source"]["trace_sha256"] == demo._sha256(source / demo.TRACE_NAME)


@pytest.mark.parametrize("change", ("manifest", "trace"))
def test_changed_source_bytes_rejected(tmp_path, change):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_identity": "tangential-compensation-public24-v1",
                "artifact_sha256": {demo.TRACE_NAME: "0" * 64},
            }
        )
    )
    (tmp_path / "COMPLETE").write_text(
        demo._sha256(tmp_path / "manifest.json") if change == "trace" else "wrong"
    )
    (tmp_path / demo.TRACE_NAME).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="mismatch"):
        demo.load_source(tmp_path)


def test_mock_render_publishes_complete_hashed_new_artifacts(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(
        demo, "load_source", lambda _: ({}, {"config": {}}, {"trace_sha256": "fixed"})
    )
    monkeypatch.setattr(demo.shutil, "which", lambda _: "/fake/ffmpeg")
    monkeypatch.setattr(demo.subprocess, "check_output", lambda *_args, **_kwargs: "ffmpeg test\n")

    def render(path, *_args):
        (path / "demo.mp4").write_bytes(b"video")
        (path / "preview.png").write_bytes(b"preview")

    monkeypatch.setattr(demo, "_render", render)
    output = demo.render_demo(source, tmp_path / "output")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["disclaimer"] == demo.DISCLAIMER
    assert manifest["frame_count"] / manifest["fps"] == 12
    assert manifest["raw_curve_sample_count"] == 6000
    for name, digest in manifest["artifact_sha256"].items():
        assert demo._sha256(output / name) == digest
    assert (output / "COMPLETE").read_text().strip() == demo._sha256(output / "manifest.json")
    with pytest.raises(ValueError):
        demo.render_demo(source, output)


def test_fresh_path_and_failure_leave_no_partial_report(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(demo, "load_source", lambda _: ({}, {}, {}))
    monkeypatch.setattr(demo.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="ffmpeg unavailable"):
        demo.render_demo(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "target")
    for path in (empty, link, source):
        with pytest.raises(ValueError):
            demo.render_demo(source, path)
