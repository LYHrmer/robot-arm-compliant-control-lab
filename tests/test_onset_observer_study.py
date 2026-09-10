"""Synthetic publisher/audit tests; these never execute robot physics or alter archives."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from tools import onset_observer_study as study


def _trace(case_index=7, scale=1.0, budget=6.0):
    time = np.arange(2_250, dtype=float) * 0.002
    zeros = np.zeros(time.size)
    vectors = np.zeros((time.size, 3))
    trace = {
        "time": time,
        "marker": np.array([case_index, scale, budget]),
        "obs_coefficient_after_advance": zeros.copy(),
        "obs_coefficient_after_force": zeros.copy(),
        "obs_coefficient_before_force": zeros.copy(),
        "obs_position_drive_m": zeros.copy(),
        "obs_velocity_drive_m": zeros.copy(),
        "obs_drive_m": zeros.copy(),
        "obs_candidate_increment": zeros.copy(),
        "obs_limited_increment": zeros.copy(),
        "obs_corrected_force_n": zeros.copy(),
        "obs_update_ready": np.zeros(time.size, dtype=bool),
        "obs_active": np.zeros(time.size, dtype=bool),
        "obs_advance_called": np.zeros(time.size, dtype=bool),
        "obs_allow_integration": np.zeros(time.size, dtype=bool),
        "obs_amplitude_capped": np.zeros(time.size, dtype=bool),
        "obs_slew_limited": np.zeros(time.size, dtype=bool),
        "requested_tangential_force_world": vectors.copy(),
        "measured_linear_velocity": vectors.copy(),
        "measured_position": vectors.copy(),
        "linear_velocity": vectors.copy(),
        "position": vectors.copy(),
        "target_linear_velocity": vectors.copy(),
        "target_position": vectors.copy(),
    }
    return trace


def _event_pair():
    before = _trace()
    after = deepcopy(before)
    before["obs_amplitude_capped"][750:] = True
    for field in (
        "obs_active",
        "obs_update_ready",
        "obs_advance_called",
        "obs_allow_integration",
    ):
        before[field][760:] = True
    before["obs_limited_increment"][760:] = 0.01
    after["obs_amplitude_capped"][750:] = False
    after["obs_coefficient_after_advance"][775:] = 0.01
    after["requested_tangential_force_world"][800:, 0] = 0.1
    after["measured_linear_velocity"][801:, 0] = 0.01
    after["measured_position"][802:, 0] = 0.001
    after["linear_velocity"][803:, 0] = 0.01
    after["position"][804:, 0] = 0.001
    return before, after


def _tables():
    return {
        "comparison": [
            {"case_index": index, "validated_cycles": 2_250}
            for index in range(8)
        ],
        "windows": [
            {"case_index": index // 4, "window": study.WINDOWS[index % 4][0]}
            for index in range(32)
        ],
        "events": [
            {"case_index": index, "first_request_difference_s": 1.6}
            for index in range(4)
        ],
    }


def _fake_collect(directory, *, execute):
    if execute:
        for _, _, _, name in study.specifications():
            path = Path(directory) / name
            np.savez_compressed(path, marker=np.array([1.0]))
    return deepcopy(_tables())


@pytest.fixture
def generated_archive(tmp_path, monkeypatch):
    sources = {"tools/onset_observer_study.py": "a" * 64}
    monkeypatch.setattr(study, "input_archives", dict)
    monkeypatch.setattr(study, "source_identity", lambda: sources.copy())
    monkeypatch.setattr(study, "collect", _fake_collect)
    monkeypatch.setattr(
        study.observer,
        "run_trial",
        lambda *args, **kwargs: pytest.fail("synthetic publication must not run physics"),
    )
    return study.generate(tmp_path / "onset-observer")


def _reseal(root):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"] = {
        name: study._sha256(root / name) for name in manifest["artifact_sha256"]
    }
    study._write_json(manifest_path, manifest)
    (root / "COMPLETE").write_text(
        study._sha256(manifest_path) + "\n", encoding="utf-8"
    )


def test_specifications_freeze_four_pairs_and_eight_runs():
    specs = list(study.specifications())

    assert [(case["case_index"], scale) for case, scale, budget, _ in specs[::2]] == list(
        study.SELECTED
    )
    assert [(case["case_index"], scale, budget) for case, scale, budget, _ in specs] == [
        (case, scale, budget)
        for case, scale in study.SELECTED
        for budget in (6.0, 8.0)
    ]
    assert len({name for *_, name in specs}) == 8


def test_protocol_freezes_windows_timing_and_controller_parameters():
    protocol = study.protocol_document()

    assert protocol["selected_pairs"] == [list(pair) for pair in study.SELECTED]
    assert protocol["windows"] == [list(window) for window in study.WINDOWS]
    assert (protocol["duration_s"], protocol["timestep_s"], protocol["evaluation_start_s"]) == (
        4.5,
        0.002,
        1.5,
    )
    assert len(protocol["controllers"]) == 8
    assert protocol["parameters"]["6"]["max_force"] == 6.0
    assert protocol["parameters"]["8"]["max_force"] == 8.0
    assert set(protocol["parameters"]["6"]) == set(protocol["parameters"]["8"])


def test_input_archives_checks_all_pins_and_frozen_parent_inputs(tmp_path, monkeypatch):
    roots = {name: tmp_path / name for name in study.PARENTS}
    for root in roots.values():
        root.mkdir()
        (root / "source_hashes.json").write_text(
            json.dumps({"src/controller.py": "a" * 64}), encoding="utf-8"
        )
    cases = [study.public.case_document(case) for case in study.public.cases()]
    controllers = [
        {
            "rotation_gain_scale": scale,
            "max_force_n": budget,
            "public24": study.public.controller_document(scale, budget),
        }
        for scale in (1.0, 2.0)
        for budget in (6.0, 8.0)
    ]
    (roots["transfer"] / "protocol.json").write_text(
        json.dumps({"public24": {"cases": cases}, "controllers": controllers}),
        encoding="utf-8",
    )
    calls = []
    by_directory = {
        record["directory"]: roots[name] for name, record in study.PARENTS.items()
    }
    monkeypatch.setattr(
        study.transfer, "safe_path", lambda root, relative: by_directory[relative]
    )
    monkeypatch.setattr(
        study.transfer,
        "verify_archive",
        lambda root, digest: calls.append((root, digest)),
    )
    monkeypatch.setattr(study, "source_identity", lambda: {"src/controller.py": "a" * 64})

    assert study.input_archives() == roots
    assert calls == [
        (roots[name], record["manifest_sha256"])
        for name, record in study.PARENTS.items()
    ]


@pytest.mark.parametrize("ndim", (1, 2))
def test_first_difference_handles_scalar_and_vector_threshold(ndim):
    time = np.array([0.0, 0.1, 0.2])
    left = np.zeros((3, ndim)) if ndim == 2 else np.zeros(3)
    right = left.copy()
    if ndim == 2:
        right[1, 0] = 1e-12
        right[2, 0] = 1.01e-12
    else:
        right[1] = 1e-12
        right[2] = 1.01e-12

    assert study.first_difference(time, left, right) == pytest.approx(0.2)
    assert study.first_difference(time, left, left) is None


def test_window_rows_have_exact_counts_signs_and_gate_intersection():
    trace = _trace()
    onset = (trace["time"] >= 1.5) & (trace["time"] < 2.0)
    middle = (trace["time"] >= 2.0) & (trace["time"] < 3.5)
    trace["obs_active"][onset | middle] = True
    trace["obs_update_ready"][onset | middle] = True
    trace["obs_advance_called"][onset | middle] = True
    trace["obs_allow_integration"][onset] = True
    trace["obs_amplitude_capped"][onset | middle] = True
    trace["obs_limited_increment"][onset | middle] = 0.01
    trace["obs_candidate_increment"][onset | middle] = 0.02
    trace["obs_coefficient_after_advance"][onset] = 0.01
    trace["obs_coefficient_after_advance"][middle] = -0.01
    trace["obs_position_drive_m"][onset] = 0.002
    trace["obs_velocity_drive_m"][onset] = -0.001

    rows = {row["window"]: row for row in study.window_rows(trace)}

    assert {name: row["samples"] for name, row in rows.items()} == {
        "motion_ramp": 150,
        "onset": 250,
        "middle": 750,
        "late": 500,
    }
    assert rows["onset"]["cap_blocks_positive_update_pct"] == 100.0
    assert rows["middle"]["cap_blocks_positive_update_pct"] == 0.0
    assert rows["onset"]["positive_update_lag_ahead_pct"] == 100.0
    assert rows["onset"]["accepted_positive_coefficient_sum"] > 0
    assert rows["middle"]["accepted_negative_coefficient_sum"] < 0


def test_pair_events_records_internal_request_and_later_state_timing():
    before, after = _event_pair()

    events = study.pair_events(before, after)

    assert events == {
        "first_cap_difference_s": 1.5,
        "first_coefficient_difference_s": 1.55,
        "first_request_difference_s": 1.6,
        "first_measured_velocity_difference_s": 1.602,
        "first_measured_position_difference_s": 1.604,
        "first_velocity_difference_s": 1.606,
        "first_position_difference_s": 1.608,
        "first_6n_cap_s": 1.5,
        "first_6n_cap_blocks_positive_update_s": 1.52,
    }


@pytest.mark.parametrize("field", ("measured_linear_velocity", "target_position"))
def test_pair_events_rejects_actor_difference_at_or_before_request(field):
    before, after = _event_pair()
    after[field][800, 0] = 0.01

    with pytest.raises(ValueError, match="actor(?:/state input| prefix) differs"):
        study.pair_events(before, after)


def test_pair_events_rejects_state_change_without_request_change():
    before = _trace()
    after = deepcopy(before)
    after["position"][900, 0] = 0.001

    with pytest.raises(ValueError, match="actor/state differs without a changed request"):
        study.pair_events(before, after)


def _install_collect_fixture(monkeypatch, traces):
    monkeypatch.setattr(study, "input_archives", dict)
    monkeypatch.setattr(
        study,
        "parent_trace",
        lambda roots, case, scale, budget: (
            {"marker": np.array([case["case_index"], scale, budget])},
            f"parent/{case['case_index']}-{scale}-{budget}.npz",
        ),
    )
    monkeypatch.setattr(study.transfer, "load_trace", lambda path: deepcopy(traces[Path(path).name]))
    monkeypatch.setattr(
        study.public,
        "compact",
        lambda trace: {"marker": trace["marker"]} if "marker" in trace else {},
    )
    monkeypatch.setattr(study.transfer.validation, "validate_public", lambda *args: None)
    monkeypatch.setattr(study.validation, "validate_observation", lambda *args: {"validated_cycles": 2_250})
    monkeypatch.setattr(study.public, "metrics", lambda *args: {"metric": 1.0})
    monkeypatch.setattr(
        study.observer,
        "run_trial",
        lambda *args, **kwargs: pytest.fail("collect fixture must not execute physics"),
    )


def test_collect_parent_matches_eight_traces_and_four_pairs(tmp_path, monkeypatch):
    traces = {}
    for case, scale, budget, name in study.specifications():
        trace = _trace(case["case_index"], scale, budget)
        if budget == 8.0:
            before = traces[Path(name.replace("f8n", "f6n")).name]
            _, trace = _event_pair()
            trace["marker"] = np.array([case["case_index"], scale, budget])
            trace["obs_amplitude_capped"] = before["obs_amplitude_capped"].copy()
        traces[Path(name).name] = trace
    _install_collect_fixture(monkeypatch, traces)

    tables = study.collect(tmp_path, execute=False)

    assert {name: len(rows) for name, rows in tables.items()} == {
        "comparison": 8,
        "windows": 32,
        "events": 4,
    }
    assert {(row["case_index"], row["rotation_gain_scale"]) for row in tables["events"]} == set(
        study.SELECTED
    )


@pytest.mark.parametrize("tamper", ("value", "dtype", "missing"))
def test_collect_rejects_changed_or_missing_parent_field(tmp_path, monkeypatch, tamper):
    traces = {
        Path(name).name: _trace(case["case_index"], scale, budget)
        for case, scale, budget, name in study.specifications()
    }
    first = next(iter(traces.values()))
    if tamper == "value":
        first["marker"][0] += 1
    elif tamper == "dtype":
        first["marker"] = first["marker"].astype(np.float32)
    else:
        del first["marker"]
    _install_collect_fixture(monkeypatch, traces)

    with pytest.raises(ValueError, match="instrumented run differs from pinned parent"):
        study.collect(tmp_path, execute=False)


def test_generate_rejects_occupied_and_symlinked_output_before_parent_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(
        study,
        "input_archives",
        lambda: pytest.fail("unsafe output must be rejected before reading parents"),
    )
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(FileExistsError, match="output already exists"):
        study.generate(occupied)

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match=r"output (?:path )?must not contain symlinks"):
        study.generate(link / "archive")


def test_generate_source_mutation_fails_without_partial_publication(tmp_path, monkeypatch):
    output = tmp_path / "onset-observer"
    sources = iter(({"source.py": "a" * 64}, {"source.py": "b" * 64}))
    monkeypatch.setattr(study, "input_archives", dict)
    monkeypatch.setattr(study, "source_identity", lambda: next(sources))
    monkeypatch.setattr(study, "collect", _fake_collect)

    with pytest.raises(ValueError, match="sources changed during instrumented execution"):
        study.generate(output)

    assert not output.exists()
    assert not list(tmp_path.glob(".onset-observer-*"))


def test_audit_binds_complete_to_manifest(generated_archive):
    (generated_archive / "COMPLETE").write_text("0" * 64 + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete manifest"):
        study.audit_archive(generated_archive)


@pytest.mark.parametrize("tamper", ("identity", "inventory", "parents"))
def test_audit_rejects_resealed_manifest_contract_tamper(generated_archive, tamper):
    path = generated_archive / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if tamper == "identity":
        manifest["identity"] = "different-study"
    elif tamper == "inventory":
        manifest["artifact_sha256"].pop("events.csv")
    else:
        manifest["parents"]["transfer"]["manifest_sha256"] = "0" * 64
    study._write_json(path, manifest)
    (generated_archive / "COMPLETE").write_text(
        study._sha256(path) + "\n", encoding="utf-8"
    )

    with pytest.raises(
        ValueError, match="identity/count|parents/inventory|unexpected/missing archive files"
    ):
        study.audit_archive(generated_archive)


@pytest.mark.parametrize("artifact", ("protocol.json", "source_hashes.json", "windows.csv"))
def test_audit_rejects_resealed_frozen_input_or_recompute_tamper(generated_archive, artifact):
    path = generated_archive / artifact
    if artifact.endswith(".json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        document["tampered"] = True
        study._write_json(path, document)
    else:
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("motion_ramp", "tampered", 1), encoding="utf-8")
    _reseal(generated_archive)

    with pytest.raises(ValueError, match="frozen input differs|reconstructed metric mismatch"):
        study.audit_archive(generated_archive)
