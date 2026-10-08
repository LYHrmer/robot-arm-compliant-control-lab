"""Keep the one-time migration baseline bounded, honest and self-consistent."""

import hashlib
import json
from pathlib import Path

import pytest

from tools import (
    onset_observer_study,
    provenance,
    provenance_migration,
    velocity_cost_study,
    velocity_time_study,
)

ROOT = Path(__file__).resolve().parents[1]

# Each migrated study and the archives sealed before its source_identity changed.
MIGRATED = {
    "tools/velocity_time_study.py": ("results/franka_velocity_time",),
    "tools/onset_observer_study.py": ("results/franka_onset_observer",
                                      "results/franka_velocity_time"),
    "tools/velocity_cost_study.py": ("results/franka_velocity_cost",
                                     "results/franka_onset_observer",
                                     "results/franka_velocity_time"),
}


def test_baseline_stays_bounded():
    """The table is closed: a new entry means the granularity is still wrong."""
    assert len(provenance_migration.MIGRATION_BASELINE) <= provenance_migration.MAXIMUM_ENTRIES
    assert set(provenance_migration.MIGRATION_BASELINE) == set(MIGRATED)


def test_every_entry_is_the_digest_some_archive_actually_recorded():
    """A baseline value must come from a sealed archive, not from guesswork."""
    for study, archives in MIGRATED.items():
        expected, _ = provenance_migration.MIGRATION_BASELINE[study]
        for archive in archives:
            saved = json.loads((ROOT / archive / "source_hashes.json").read_text())
            assert saved[study] == expected, f"{archive} does not record {study} at {expected}"


def test_every_entry_describes_a_file_that_really_changed():
    """A stale entry would silently widen what the audit accepts."""
    for study in MIGRATED:
        current = hashlib.sha256((ROOT / study).read_bytes()).hexdigest()
        old, migrated = provenance_migration.MIGRATION_BASELINE[study]
        assert current == migrated, f"{study} changed after its reviewed migration"
        assert current != old, (
            f"{study} matches its baseline again; drop the entry"
        )


def test_baseline_for_refuses_unrecorded_paths():
    with pytest.raises(KeyError, match="no migration baseline recorded"):
        provenance_migration.baseline_for("tools/not_migrated.py")


def test_baseline_for_scopes_each_study_to_what_it_names():
    single = provenance_migration.baseline_for("tools/velocity_time_study.py")
    assert set(single) == {"tools/velocity_time_study.py"}


@pytest.mark.parametrize("module,archive", (
    (velocity_cost_study, "results/franka_velocity_cost"),
    (onset_observer_study, "results/franka_onset_observer"),
    (velocity_time_study, "results/franka_velocity_time"),
))
def test_migrated_studies_still_audit_their_published_archives(module, archive):
    result = module.audit_archive(ROOT / archive)
    assert result["audit_status"] == "PASS"
    assert result["source_identity_check"].startswith("migrated: ")


def test_migration_rejects_subsequent_source_changes():
    """Old evidence cannot vouch for a subsequent edit to a migrated study."""
    archive = ROOT / "results/franka_velocity_time"
    saved = json.loads((archive / "source_hashes.json").read_text())
    live = velocity_time_study.source_identity()
    baseline = provenance_migration.baseline_for(*MIGRATED)
    assert provenance.verify_source_identity(saved, live, baseline=baseline).startswith("migrated")
    for name in MIGRATED:
        changed = {**live, name: "0" * 64}
        with pytest.raises(ValueError, match="replay-relevant source differs"):
            provenance.verify_source_identity(saved, changed, baseline=baseline)
