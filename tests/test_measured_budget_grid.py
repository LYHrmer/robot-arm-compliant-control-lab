"""Contract tests for tools.measured_budget_study specification enumeration."""

import pathlib

import pytest

from tools.measured_budget_study import reference_path, specifications

PUBLIC_REUSED_INDEXES = {6, 7, 14, 15, 22, 23}
SCALES = (1.0, 2.0)
YAWS = (-15.0, 0.0, 15.0)
VARIANTS = ("combined", "no_bias")


@pytest.fixture(scope="module")
def specs():
    return specifications()


def _public(specs):
    return [s for s in specs if s["suite"] == "public"]


def _dynamic(specs):
    return [s for s in specs if s["suite"] == "dynamic"]


def _identity(spec):
    return tuple(sorted((k, v) for k, v in spec.items() if k != "case"))


def test_grids_are_exact_and_unique(specs):
    assert len(specs) == 60
    public, dynamic = _public(specs), _dynamic(specs)
    assert (len(public), len(dynamic)) == (48, 12)
    assert {(s["case_index"], s["scale"]) for s in public} == {
        (i, sc) for i in range(24) for sc in SCALES
    }
    assert {(s["variant"], float(s["surface_yaw_deg"]), s["scale"]) for s in dynamic} == {
        (v, y, sc) for v in VARIANTS for y in YAWS for sc in SCALES
    }
    for s in public:
        assert isinstance(s["case_index"], int) and not isinstance(s["case_index"], bool)
        assert isinstance(s["scale"], float) and isinstance(s["case"], dict)
    for s in dynamic:
        assert isinstance(s["scale"], float) and not isinstance(s["case"], dict)


def test_new_and_reused_counts(specs):
    assert sum(1 for s in specs if not s["reused"]) == 42
    assert sum(1 for s in specs if s["reused"]) == 18
    public, dynamic = _public(specs), _dynamic(specs)
    assert sum(1 for s in public if not s["reused"]) == 36
    assert sum(1 for s in dynamic if not s["reused"]) == 6
    assert sum(1 for s in public if s["reused"]) == 12
    assert sum(1 for s in dynamic if s["reused"]) == 6
    for s in public:
        assert s["reused"] is (s["case_index"] in PUBLIC_REUSED_INDEXES)
    for s in dynamic:
        assert s["reused"] is (s["variant"] == "combined")


def test_trace_names_unique_and_traversal_free(specs):
    names = [s["trace_name"] for s in specs]
    assert len(set(names)) == 60
    for name in names:
        assert name.endswith(".npz")
        assert "/" not in name and "\\" not in name and ".." not in name
        assert pathlib.PurePosixPath(name).name == name


def test_trace_names_match_deterministic_formula(specs):
    for s in _public(specs):
        assert s["trace_name"] == f"public__case{s['case_index']}__s{int(s['scale'])}.npz"
    for s in _dynamic(specs):
        assert s["trace_name"] == (
            f"dynamic__yaw{int(s['surface_yaw_deg'])}__{s['variant']}__s{int(s['scale'])}.npz"
        )


def test_reference_paths_exist_and_dynamic_parents(specs):
    expected_parent = {1: "franka_compensation_budget", 2: "franka_budget_transfer"}
    for s in specs:
        path = reference_path(s)
        assert isinstance(path, pathlib.Path)
        assert path.is_absolute() and path.is_file()
        assert path.suffix == ".npz"
    for s in _dynamic(specs):
        assert reference_path(s).parent.parent.name == expected_parent[int(s["scale"])]
        assert "f6n" in reference_path(s).name


def test_repeat_calls_yield_fresh_but_identical_specs(specs):
    again = specifications()
    assert again is not specs and len(again) == 60
    assert all(a is not b for a, b in zip(again, specs))
    assert sorted(map(_identity, again)) == sorted(map(_identity, specs))
