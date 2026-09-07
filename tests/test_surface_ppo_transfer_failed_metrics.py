"""Missing rollout metrics must veto a comparison, never improve its average."""

from copy import deepcopy

import pytest

from tools import surface_learning_pilot as pilot
from tools import surface_ppo_transfer as protocol


def _report(tangent=3.0):
    rows = []
    for index in range(4):
        row = {name: rule["value"] for name, rule in pilot.THRESHOLDS.items()}
        row.update(
            case_id=f"case{index}", group_id=f"group{index // 2}", split="validation",
            simulation_seed=index, tangent_rmse_mm=tangent, max_contact_loss_s=0.0,
            episode_success=True, independent_physical_gates_met=True,
        )
        rows.append(row)
    return {
        "runs": rows, "evaluation_case_ids": [row["case_id"] for row in rows],
        "expected_case_count": 4, "evaluation_split": "validation",
        "acceptance_met": True, "all_metrics_available": True,
        "all_episodes_succeeded": True,
    }


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True])
def test_unavailable_pair_is_retained_and_invalidates_whole_mean(value):
    transfer = _report()
    transfer["runs"][1]["tangent_rmse_mm"] = value
    paired = protocol._paired_tangent_delta(transfer, _report(4.0))
    assert paired["per_case"] == [-1.0, None, -1.0, -1.0]
    assert paired["mean"] is None
    assert not paired["all_metrics_available"]
    assert paired["unavailable_case_ids"] == ["case1"]


@pytest.mark.parametrize("arm", ["transfer", "fresh"])
def test_same_length_duplicate_cannot_replace_a_missing_case(arm):
    transfer, fresh = _report(), _report(4.0)
    report = transfer if arm == "transfer" else fresh
    report["runs"][1] = deepcopy(report["runs"][0])
    with pytest.raises(ValueError, match=f"duplicate {arm}"):
        protocol._paired_tangent_delta(transfer, fresh)


def test_missing_metric_freezes_veto_without_development(tmp_path, monkeypatch):
    output, parent = tmp_path / "output", tmp_path / "parent"
    output.mkdir()
    parent.mkdir()
    pilot._write_new(output / "plan.json", {"frozen": True})
    controls = []
    for seed in protocol.SEEDS:
        candidate = f"fresh{seed}.json"
        pilot._write_new(parent / candidate, {"seed": seed})
        directory = output / f"transfer_seed{seed}"
        directory.mkdir()
        pilot._write_new(directory / "checkpoint_ep032.json", {"seed": seed})
        controls.append({"seed": seed, "candidate": candidate, "validation_report": _report(4)})
    plan = {
        "seeds": list(protocol.SEEDS), "cases": _report()["runs"],
        "thresholds": pilot.THRESHOLDS, "fresh_controls": controls,
    }
    failed = _report()
    failed["runs"][0]["tangent_rmse_mm"] = None
    failed["all_metrics_available"] = False
    failed["acceptance_met"] = False
    monkeypatch.setattr(protocol, "load", lambda *args: plan)
    monkeypatch.setattr(protocol, "train_one", lambda *args: None)
    monkeypatch.setattr(protocol, "validate_one", lambda *args: failed)
    monkeypatch.setattr(
        protocol, "_training_manifest", lambda *args: ({"failed_episodes": 0}, {})
    )
    monkeypatch.setattr(
        pilot, "_evaluate", lambda *args: pytest.fail("veto must not open development")
    )
    result = protocol.run(output, parent, "bc", "dataset", "benchmark", "plan-sha")
    record = pilot._read(result)
    assert record["retained_arm"] == "fresh_ep32"
    assert not record["consistent_transfer_gain"]
    assert not record["validation_safe_for_development"]
    assert (output / "VALIDATION_VETO").read_text().strip() == pilot._sha256(result)
    assert not (output / "COMPLETE").exists()
