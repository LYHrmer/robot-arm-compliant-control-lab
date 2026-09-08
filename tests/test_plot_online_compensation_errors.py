import json

import numpy as np
import pytest

import tools.plot_online_compensation_errors as plotting


def _trace(time, *, online, step_down):
    friction = np.where(time < 6.0, 0.65, 0.25) if step_down else np.full_like(time, 0.45)
    return {
        "time": time,
        "tangent_position_error_norm_m": 0.001 + 0.0002 * np.sin(time),
        "true_normal_force": 12.0 + 0.2 * np.cos(time),
        "controller_coefficient_after_compute": (
            0.45 + 0.04 * np.sin(0.5 * time) if online else np.full_like(time, 0.45)
        ),
        "applied_wall_friction": friction,
    }


@pytest.fixture
def archive(tmp_path, monkeypatch):
    root = tmp_path / "archive"
    traces = root / "traces"
    traces.mkdir(parents=True)
    (root / "manifest.json").write_text('{"identity": "test"}\n', encoding="utf-8")
    (root / "COMPLETE").write_text("complete\n", encoding="utf-8")
    time = np.linspace(0.0, 12.0, 121)
    for case in plotting.CASES:
        for arm in plotting.ARMS:
            np.savez_compressed(
                plotting._trace_path(root, case, arm),
                **_trace(
                    time,
                    online=arm == "online",
                    step_down=case == "friction_step_down",
                ),
            )
    monkeypatch.setattr(
        plotting,
        "audit_archive",
        lambda path: {
            "status": "PASS",
            "identity": "online-compensation-error-comparison-v1",
            "is_subset": False,
        },
    )
    return root


def test_plot_writes_one_png_and_hashed_provenance(archive, tmp_path):
    output, sidecar = plotting.plot_archive(archive, tmp_path / "figure.png")
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    assert provenance["figure_sha256"] == plotting._sha256(output)
    assert len(provenance["input_trace_sha256"]) == 4
    assert provenance["cases"] == list(plotting.CASES)
    assert "not an estimator target" in provenance["applied_friction_note"]
    assert not list(tmp_path.glob("*.pdf")) and not list(tmp_path.glob("*.svg"))


def test_incomplete_or_subset_archive_is_rejected_before_reading(tmp_path, monkeypatch):
    archive = tmp_path / "incomplete"
    archive.mkdir()
    output = tmp_path / "figure.png"
    monkeypatch.setattr(
        plotting,
        "audit_archive",
        lambda _: (_ for _ in ()).throw(ValueError("archive requires manifest.json and COMPLETE")),
    )
    with pytest.raises(ValueError, match="requires manifest"):
        plotting.plot_archive(archive, output)
    assert not output.exists()

    monkeypatch.setattr(
        plotting,
        "audit_archive",
        lambda _: {"status": "PASS", "identity": "test", "is_subset": True},
    )
    with pytest.raises(ValueError, match="full archive"):
        plotting.plot_archive(archive, output)


@pytest.mark.parametrize("occupied", ("figure", "sidecar"))
def test_existing_output_or_sidecar_is_never_overwritten(archive, tmp_path, occupied):
    output = tmp_path / "figure.png"
    sidecar = tmp_path / "figure.png.provenance.json"
    target = output if occupied == "figure" else sidecar
    target.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        plotting.plot_archive(archive, output)
    assert target.read_text(encoding="utf-8") == "keep"


def test_motion_panel_marks_reversal_not_a_fictitious_six_second_event(archive):
    traces = {
        (case, arm): plotting._load_trace(plotting._trace_path(archive, case, arm))
        for case in plotting.CASES for arm in plotting.ARMS
    }
    figure = plotting._draw(traces)
    try:
        left_labels = [text.get_text() for text in figure.axes[0].texts]
        right_labels = [text.get_text() for text in figure.axes[1].texts]
        assert "6 s friction step" in left_labels
        assert "7 s reverse ramp" in right_labels and "Hold" in right_labels
        assert not any("6 s" in label for label in right_labels)
    finally:
        plotting.plt.close(figure)
