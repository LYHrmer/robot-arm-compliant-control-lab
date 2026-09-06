"""Audited NumPy transition views; no fitting, optimization or training framework."""

import hashlib
import json
from numbers import Real
from pathlib import Path

import numpy as np

from compliant_control_lab.surface_dataset import audit_dataset
from compliant_control_lab.surface_env import POLICY_SUBSTEPS
from compliant_control_lab.surface_splits import SPLIT_NAMES


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_transition_batch(path, split, *, gamma_per_policy_step=0.99, allow_examples=False):
    """Load one declared split from an independently audited teacher dataset.

    ``gamma_per_policy_step`` is the discount for one ordinary 50 Hz transition,
    not one 500 Hz physics step. A partial final block uses the corresponding
    fractional exponent. Valid time-limit truncations may bootstrap; safety
    terminations and invalid terminal observations never do.

    Only stored 49-vector observations and 3-vector actions are actor inputs.
    Evaluator info, trace truth and case parameters remain outside those arrays.
    Example subsets require explicit opt-in and are only suitable for smoke tests.
    """
    audit = audit_dataset(path)
    path = Path(path).absolute()
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be exactly one of {SPLIT_NAMES}")
    if (
        isinstance(gamma_per_policy_step, (bool, np.bool_))
        or not isinstance(gamma_per_policy_step, Real)
        or not np.isfinite(gamma_per_policy_step)
        or not 0 <= gamma_per_policy_step <= 1
    ):
        raise ValueError("gamma_per_policy_step must be a finite real number in [0, 1]")
    if not isinstance(allow_examples, (bool, np.bool_)):
        raise TypeError("allow_examples must be boolean")

    manifest_path = path / "manifest.json"
    manifest_sha256 = _sha256(manifest_path)
    if (path / "COMPLETE").read_text().strip() != manifest_sha256:
        raise ValueError("dataset changed after audit")
    manifest = json.loads(manifest_path.read_text())
    scope = manifest["corpus_scope"]
    if scope == "example_subset" and not allow_examples:
        raise ValueError("example_subset is smoke-test material, not a complete training corpus")

    entries = [entry for entry in manifest["episodes"] if entry["split"] == split]
    if not entries:
        raise ValueError(f"dataset contains no episodes for split {split!r}")
    fields = (
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "terminated",
        "truncated",
        "terminal_observation_valid",
        "physics_substeps",
    )
    columns = {name: [] for name in fields}
    episode_ids, episode_start = [], []
    for entry in entries:
        episode_path = path / entry["file"]
        if _sha256(episode_path) != entry["sha256"]:
            raise ValueError("dataset episode changed after audit")
        with np.load(episode_path, allow_pickle=False) as stored:
            arrays = {name: stored[name].copy() for name in fields}
        if _sha256(episode_path) != entry["sha256"]:
            raise ValueError("dataset episode changed while loading")
        count = len(arrays["observations"])
        for name, value in arrays.items():
            columns[name].append(value)
        identity = entry["case_id"]
        episode_ids.append(np.full(count, identity, dtype=f"U{len(identity)}"))
        start = np.zeros(count, dtype=bool)
        start[0] = True
        episode_start.append(start)

    batch = {name: np.concatenate(values) for name, values in columns.items()}
    valid = batch["terminal_observation_valid"]
    batch["next_observations"] = batch["next_observations"].copy()
    batch["next_observations"][~valid] = 0.0
    bootstrap = (~batch["terminated"]) & valid
    gamma = float(gamma_per_policy_step)
    batch["bootstrap_mask"] = bootstrap
    batch["discounts"] = gamma ** (batch["physics_substeps"] / POLICY_SUBSTEPS) * bootstrap
    batch["episode_id"] = np.concatenate(episode_ids)
    batch["episode_start"] = np.concatenate(episode_start)
    group_ids = sorted({entry["group_id"] for entry in entries})
    batch["metadata"] = {
        "split": split,
        "group_ids": group_ids,
        "case_ids": [entry["case_id"] for entry in entries],
        "corpus_scope": scope,
        "example_smoke_test_only": scope == "example_subset",
        "source_dataset": str(path),
        "source_manifest_sha256": manifest_sha256,
        "source_and_assets_sha256": manifest["source_and_assets_sha256"],
        "gamma_per_policy_step": gamma,
        "discount_clock": "50_hz_policy_transition_fractional_for_partial_physics_block",
        "audit": audit,
    }
    return batch
