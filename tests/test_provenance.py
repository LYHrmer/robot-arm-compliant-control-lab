"""Check dependency selection and strict archive migration boundaries."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import provenance

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "src/compliant_control_lab"
# Entries outside this prefix are rejected as injected, so fixtures live under it.
PKG = "src/compliant_control_lab/"

# Entry points spanning a flat study, a package study and a transfer chain.
ENTRIES = (
    ("tools/combined_residual_ablation.py", "results/franka_combined_residual_ablation"),
    ("tools/budget_transfer.py", "results/franka_budget_transfer"),
    ("tools/reversal_recovery/study.py", "results/franka_reversal_recovery"),
    ("tools/velocity_cost_study.py", "results/franka_velocity_cost"),
    ("tools/reversible_recovery/study.py", "results/franka_reversible_recovery_transfer"),
)

# Imported in a clean interpreter so that modules pulled in by unrelated test
# files cannot inflate the runtime closure this is compared against.
RUNTIME_CLOSURE = """
import importlib, json, pathlib, sys
package = pathlib.Path(sys.argv[1]).resolve()
root = package.parents[1]
importlib.import_module(sys.argv[2])
print(json.dumps(sorted(
    pathlib.Path(module.__file__).resolve().relative_to(root).as_posix()
    for module in list(sys.modules.values())
    if getattr(module, "__file__", None)
    and pathlib.Path(module.__file__).resolve().is_relative_to(package)
    and pathlib.Path(module.__file__).suffix == ".py"
)))
"""


def runtime_closure(module_name):
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "src")))}
    completed = subprocess.run(
        [sys.executable, "-c", RUNTIME_CLOSURE, str(PACKAGE_DIR), module_name],
        capture_output=True, text=True, cwd=ROOT, env=environment, check=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.parametrize("entry,_archive", ENTRIES)
def test_static_closure_matches_a_clean_interpreter(entry, _archive):
    """The AST graph must agree with what importing actually loads."""
    module = entry.removesuffix(".py").replace("/", ".")
    assert sorted(provenance.closure_identity(ROOT / entry)) == runtime_closure(module)


@pytest.mark.parametrize("entry,archive", ENTRIES)
def test_published_archives_satisfy_the_closure_without_being_rewritten(entry, archive):
    """Whole-package archives record a superset, so they pass as a closure subset.

    `live` mirrors what a migrated study reports: the closure of package modules
    plus everything it still pins verbatim (assets, tools files, lock files).
    """
    saved = json.loads((ROOT / archive / "source_hashes.json").read_text())
    closure = provenance.closure_identity(ROOT / entry)
    pinned_verbatim = {name: digest for name, digest in saved.items()
                       if not name.startswith("src/compliant_control_lab/")
                       or "/assets/" in name}
    live = {**pinned_verbatim, **closure}
    assert provenance.verify_source_identity(saved, live) == "closure-subset"
    assert len(closure) < len(saved)


def test_closure_excludes_modules_that_cannot_change_the_replay():
    closure = provenance.closure_identity(ROOT / "tools/reversal_recovery/study.py")
    unrelated = {
        "surface_policy.py", "surface_env.py", "surface_dataset.py", "plotting.py",
        "kinematics.py", "simulation.py", "smoke.py", "surface_splits.py",
    }
    assert unrelated.isdisjoint({Path(name).name for name in closure})
    assert "src/compliant_control_lab/tangential_compensation.py" in closure


def test_package_init_is_always_reached():
    """Importing any package module runs __init__, which an entry need not mention."""
    closure = provenance.closure_identity(ROOT / "tools/combined_residual_ablation.py")
    assert "src/compliant_control_lab/__init__.py" in closure


def test_identity_is_deterministic_and_repository_relative():
    first = provenance.closure_identity(ROOT / "tools/budget_transfer.py")
    assert first == provenance.closure_identity(ROOT / "tools/budget_transfer.py")
    assert all(name.startswith("src/compliant_control_lab/") for name in first)
    assert all(len(digest) == 64 for digest in first.values())


def test_identity_covers_no_assets():
    """Assets stay whole-package; narrowing them would need the MuJoCo include graph."""
    closure = provenance.closure_identity(ROOT / "tools/combined_residual_ablation.py")
    assert all(name.endswith(".py") for name in closure)


def test_equal_identity_reports_exact():
    closure = provenance.closure_identity(ROOT / "tools/budget_transfer.py")
    assert provenance.verify_source_identity(dict(closure), closure) == "exact"


def test_unrecorded_closure_member_is_rejected():
    closure = provenance.closure_identity(ROOT / "tools/budget_transfer.py")
    saved = dict(closure)
    dropped = saved.pop(min(saved))
    assert dropped
    with pytest.raises(ValueError, match="does not record replay-relevant source"):
        provenance.verify_source_identity(saved, closure)


def test_changed_closure_member_is_rejected():
    closure = provenance.closure_identity(ROOT / "tools/budget_transfer.py")
    saved = {**closure, min(closure): "0" * 64}
    with pytest.raises(ValueError, match="replay-relevant source differs"):
        provenance.verify_source_identity(saved, closure)


def test_changes_outside_the_closure_are_ignored():
    """The point of the change: an unrelated module must not fail the audit."""
    closure = provenance.closure_identity(ROOT / "tools/reversal_recovery/study.py")
    saved = {**closure, "src/compliant_control_lab/surface_policy.py": "0" * 64}
    assert provenance.verify_source_identity(saved, closure) == "closure-subset"


def test_relative_imports_fail_loudly_instead_of_vanishing(tmp_path):
    """A skipped relative import would drop a real dependency from the closure."""
    module = tmp_path / "relative.py"
    module.write_text("from . import sibling\n")
    with pytest.raises(ValueError, match="relative import"):
        provenance._referenced_modules(module)


def test_repository_has_no_relative_imports():
    """Guards the assumption the closure resolver relies on."""
    for path in (*ROOT.glob("src/**/*.py"), *ROOT.glob("tools/**/*.py")):
        provenance._referenced_modules(path)


def test_missing_entry_point_is_reported():
    with pytest.raises(FileNotFoundError, match="entry point is not a file"):
        provenance.closure_identity(ROOT / "tools/does_not_exist.py")


def test_external_and_repository_modules_are_separated():
    assert provenance._module_file("numpy") is None
    assert provenance._module_file("compliant_control_lab.controllers") == (
        PACKAGE_DIR / "controllers.py"
    )
    assert provenance._module_file("tools.reversal_recovery.study") == (
        ROOT / "tools/reversal_recovery/study.py"
    )
    # Package directories resolve through their __init__.
    assert provenance._module_file("compliant_control_lab") == PACKAGE_DIR / "__init__.py"


def test_symbol_imports_do_not_invent_modules():
    """`from a.b import c` offers both names; only existing files survive."""
    assert provenance._module_file("compliant_control_lab.controllers.ImpedanceController") is None


MIGRATED_STUDIES = {"tools/velocity_time_study.py", "tools/onset_observer_study.py",
                    "tools/velocity_cost_study.py"}


def test_only_migrated_studies_depend_on_this_module():
    """Keep the blast radius visible: each new dependant needs a baseline entry."""
    users = set()
    for path in (*ROOT.glob("tools/**/*.py"), *ROOT.glob("src/**/*.py")):
        if path.name in {"provenance.py", "provenance_migration.py"}:
            continue
        # Match import statements, not the English word, which several modules use.
        if "tools.provenance" in provenance._referenced_modules(path):
            users.add(path.relative_to(ROOT).as_posix())
    assert users == MIGRATED_STUDIES


def test_inherited_identity_only_judges_the_overlap():
    shared_a, shared_b = f"{PKG}a.py", f"{PKG}b.py"
    live = {shared_a: "a" * 64, shared_b: "b" * 64, "tools/own.py": "c" * 64}
    # Parent pinned an extra package module and never saw the child's own file.
    parent = {shared_a: "a" * 64, shared_b: "b" * 64, f"{PKG}unrelated.py": "9" * 64}
    assert provenance.verify_inherited_identity(parent, live) == "closure-subset"


def test_inherited_identity_rejects_a_changed_shared_file():
    live = {f"{PKG}a.py": "a" * 64, "tools/own.py": "c" * 64}
    with pytest.raises(ValueError, match="replay-relevant source differs"):
        provenance.verify_inherited_identity({f"{PKG}a.py": "f" * 64}, live)


def test_inherited_identity_rejects_an_unrelated_parent():
    with pytest.raises(ValueError, match="share no source files"):
        provenance.verify_inherited_identity(
            {f"{PKG}other.py": "a" * 64}, {f"{PKG}a.py": "a" * 64})


def test_baseline_admits_only_the_recorded_predecessor():
    live, old, other = {"tools/study.py": "a" * 64}, "0" * 64, "e" * 64
    baseline = {"tools/study.py": (old, live["tools/study.py"])}
    assert provenance.verify_source_identity(
        {"tools/study.py": old}, live, baseline=baseline
    ) == "migrated: tools/study.py"
    # A different old digest is still a real difference.
    with pytest.raises(ValueError, match="replay-relevant source differs"):
        provenance.verify_source_identity(
            {"tools/study.py": other}, live, baseline=baseline
        )
    # A later behavioral edit cannot inherit an old migration exception.
    with pytest.raises(ValueError, match="replay-relevant source differs"):
        provenance.verify_source_identity(
            {"tools/study.py": old}, {"tools/study.py": other}, baseline=baseline
        )


def test_baseline_cannot_excuse_a_file_it_does_not_name():
    live = {"tools/study.py": "a" * 64, "tools/core.py": "a" * 64}
    saved = {"tools/study.py": "0" * 64, "tools/core.py": "0" * 64}
    with pytest.raises(ValueError, match=r"differs: \['tools/core.py'\]"):
        provenance.verify_source_identity(
            saved, live, baseline={"tools/study.py": ("0" * 64, "a" * 64)}
        )


@pytest.mark.parametrize("extra", (
    "tools/fabricated.py", f"{PKG}../outside.py", f"{PKG}./alias.py",
    f"{PKG}assets/removed.xml", f"{PKG}nested//alias.py",
))
def test_only_canonical_extra_package_python_files_are_allowed(extra):
    live = {f"{PKG}core.py": "a" * 64}
    with pytest.raises(ValueError, match="unexpected records"):
        provenance.verify_source_identity({**live, extra: "b" * 64}, live)


@pytest.mark.parametrize("saved", ([], None, "not an identity"))
def test_non_object_identity_is_rejected(saved):
    with pytest.raises(ValueError, match="must be an object"):
        provenance.verify_source_identity(saved, {})


def test_subpackage_initializers_and_function_imports_are_reached(tmp_path, monkeypatch):
    package = tmp_path / "compliant_control_lab"
    nested = package / "nested"
    nested.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (nested / "__init__.py").write_text("import compliant_control_lab.initialized\n")
    (nested / "leaf.py").write_text("def f():\n    import compliant_control_lab.delayed\n")
    (package / "initialized.py").write_text("")
    (package / "delayed.py").write_text("")
    entry = tmp_path / "entry.py"
    entry.write_text("import compliant_control_lab.nested.leaf\n")
    monkeypatch.setattr(provenance, "PACKAGE_DIR", package)
    monkeypatch.setattr(provenance, "SEARCH_ROOTS", ((provenance.PACKAGE, tmp_path),))
    assert provenance.closure_files(entry) == {entry, *package.rglob("*.py")}


def test_malformed_archived_records_are_rejected():
    """An injected non-digest entry must not slip through as an ignored extra."""
    good = f"{PKG}a.py"
    live = {good: "a" * 64}
    for junk in ({good: "a" * 64, "tampered": True},
                 {good: "a" * 64, f"{PKG}b.py": "short"},
                 {good: "a" * 64, f"{PKG}c.py": "Z" * 64}):
        with pytest.raises(ValueError, match="malformed records"):
            provenance.verify_source_identity(junk, live)


def test_frozen_source_keeps_the_established_wording(tmp_path):
    name = f"{PKG}a.py"
    (tmp_path / "source_hashes.json").write_text(json.dumps({name: "a" * 64}))
    assert provenance.verify_frozen_source(tmp_path, {name: "a" * 64}) == "exact"
    with pytest.raises(ValueError, match="frozen input differs: source_hashes.json"):
        provenance.verify_frozen_source(tmp_path, {name: "f" * 64})
