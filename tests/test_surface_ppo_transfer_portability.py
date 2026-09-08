"""Archived arithmetic is portable; live policy execution remains version-pinned."""

from pathlib import Path

import pytest

from compliant_control_lab.surface_policy_artifact import load_policy_artifact, policy_contract
from tools import publish_surface_ppo_transfer as publication
from tools import surface_learning_pilot as pilot
from tools import surface_mlp_actor

ARCHIVE = Path(__file__).resolve().parents[1] / "results" / "franka_surface_ppo_transfer"


@pytest.mark.parametrize(
    "field", ["python_version", "numpy_version", "mujoco_version", "gymnasium_version",
              "package_source_and_assets_sha256"]
)
def test_offline_audit_accepts_changed_host_or_package_but_live_runner_does_not(
    monkeypatch, field
):
    identity = surface_mlp_actor.current_runner_identity()
    monkeypatch.setattr(
        surface_mlp_actor, "current_runner_identity",
        lambda: {**identity, field: "different-audit-host-version"},
    )
    plan = pilot._read(ARCHIVE / "plan.json")
    contract = policy_contract(
        pilot.pilot_cases(plan, "friction"), purpose="rl_residual", nominal_kind="friction"
    )
    artifact = load_policy_artifact(
        ARCHIVE / "transfer_seed11" / "checkpoint_ep000.json", expected_contract=contract
    )
    with pytest.raises(ValueError, match="runner identity differs"):
        surface_mlp_actor.actor_from_artifact(artifact)
    result = publication.audit(ARCHIVE)
    assert result["comparison_recomputed"]
    assert result["representative_physics_traces_recomputed"] == 2
    # The offline path must not temporarily weaken or replace the live loader.
    with pytest.raises(ValueError, match="runner identity differs"):
        surface_mlp_actor.actor_from_artifact(artifact)


@pytest.mark.parametrize("field", ["runner_sha256", "evaluator_sha256"])
def test_offline_audit_still_rejects_changed_execution_source(monkeypatch, field):
    identity = surface_mlp_actor.current_runner_identity()
    monkeypatch.setattr(
        surface_mlp_actor, "current_runner_identity", lambda: {**identity, field: "0" * 64}
    )
    with pytest.raises(ValueError, match="identity|source"):
        publication.audit(ARCHIVE)
