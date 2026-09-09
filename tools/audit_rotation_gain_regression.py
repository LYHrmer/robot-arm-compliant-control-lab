"""Read-only arithmetic audit of the matched public-24 rotation-gain archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

IDENTITY = "rotation-gain-public24-v1"
SCALES = (1.0, 2.0)
METHODS = {
    "baseline": "surface_adaptive",
    "integral": "surface_integral",
    "friction": "surface_friction",
    "online": "surface_online",
}
METRICS = (
    "force_rmse_n",
    "peak_force_n",
    "tangent_rmse_mm",
    "contact_ratio_pct",
    "saturation_pct",
    "orientation_rmse_deg",
    "measurement_rmse_n",
    "seconds_over_35_n",
    "first_raw_contact_time_s",
)
AUXILIARY_METRICS = (
    "max_penetration_mm",
    "evaluation_median_penetration_mm",
    "projection_pct",
    "max_compensation_force_n",
)
EXTRA_METRICS = (
    "tangent_velocity_error_rms_m_s",
    "minimum_torque_headroom_nm",
    "minimum_reserved_torque_headroom_nm",
)
ALL_METRICS = (*METRICS, *AUXILIARY_METRICS, *EXTRA_METRICS)
COMPACT_FIELDS = (
    "time",
    "true_normal_force",
    "target_normal_force",
    "position",
    "target_position",
    "linear_velocity",
    "target_linear_velocity",
    "orientation_error_rad",
    "measured_normal_force",
    "commanded_torque",
    "applied_torque",
    "lower_torque_limit",
    "upper_torque_limit",
    "torque_projection_scale",
    "requested_tangential_force_world",
    "true_contact_gap_m",
)
TRACE_METADATA = ("rotation_gain_scale", "case_index", "method")
FULL_CASES = (0, 16, 23)
CHECKS = {
    "minimum_contact_ratio_pct": 99.0,
    "maximum_raw_peak_force_n": 35.0,
    "required_saturation_pct": 0.0,
    "maximum_paired_force_rmse_increase_n": 0.2,
    "maximum_paired_orientation_rmse_increase_deg": 0.2,
    "maximum_projection_pct": 0.0,
    "minimum_reserved_torque_headroom_nm": 0.0,
    "numeric_tolerance": 1e-10,
}
REFERENCE_MANIFEST_SHA256 = "ddd65fd48fd078603c399aa1a9f7485c57ab018610139800e233c3c7764ef543"
KNOWN_ARCHIVE_SOURCE_HASHES_SHA256 = frozenset(
    {
        # results/franka_rotation_gain_public24/source_hashes.json (also used
        # by the matching frozen-source one-case smoke archive).
        "ecccee0e6987f56f15139131912cda9c659edf3fbb89ccfeb43d13bb0650cd7d",
    }
)
FLOAT_ATOL = 1e-10


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _safe_file(root: Path, label: str) -> Path:
    relative = Path(label)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != label:
        raise ValueError(f"unsafe artifact path: {label!r}")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked artifact: {label}")
    return path


def _verify_manifest(root: Path) -> dict:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("audit path must be a real directory")
    manifest_path, complete_path = root / "manifest.json", root / "COMPLETE"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise ValueError("archive requires manifest.json and COMPLETE")
    if complete_path.read_text(encoding="utf-8").strip() != _sha256(manifest_path):
        raise ValueError("COMPLETE does not bind manifest.json")
    manifest = _json(manifest_path)
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict):
        raise TypeError("manifest artifact_sha256 must be a mapping")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "COMPLETE"}
    }
    if set(hashes) != actual:
        raise ValueError("manifest artifact inventory does not match archive files")
    for label, digest in hashes.items():
        if _sha256(_safe_file(root, label)) != digest:
            raise ValueError(f"artifact hash mismatch: {label}")
    return manifest


def _reference() -> tuple[dict, dict, dict]:
    reference = Path(__file__).resolve().parents[1] / "results" / "franka_online_compensation_regression"
    manifest_path = reference / "manifest.json"
    if _sha256(manifest_path) != REFERENCE_MANIFEST_SHA256:
        raise ValueError("reference does not match the pinned public96 manifest")
    if (reference / "COMPLETE").read_text().strip() != REFERENCE_MANIFEST_SHA256:
        raise ValueError("reference COMPLETE does not bind its manifest")
    manifest = _json(manifest_path)
    for name, digest in manifest.get("artifact_sha256", {}).items():
        if Path(name).name != name or _sha256(_safe_file(reference, name)) != digest:
            raise ValueError(f"reference artifact mismatch: {name}")
    _, rows = _read_csv(reference / "comparison.csv")
    configurations = _json(reference / "configurations.json")
    keys = {(method, index) for method in METHODS for index in range(24)}
    row_map = {(row["method"], int(row["case_index"])): row for row in rows}
    config_map = {(row["method"], int(row["case_index"])): row for row in configurations}
    if len(rows) != 96 or len(configurations) != 96 or set(row_map) != keys or set(config_map) != keys:
        raise ValueError("reference must contain the unique four-method public24 grid")
    identity = {
        "directory": "results/franka_online_compensation_regression",
        "manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "artifact_sha256": manifest["artifact_sha256"],
    }
    return row_map, config_map, {"identity": identity, "manifest": manifest}


def _live_source_identity() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[1]
    package = repo / "src" / "compliant_control_lab"
    files = sorted(set(package.rglob("*.py")) | set((package / "assets").rglob("*")))
    sources = {
        f"src/compliant_control_lab/{path.relative_to(package)}": _sha256(path)
        for path in files
        if path.is_file()
    }
    for name in ("rotation_gain_regression.py", "online_compensation_regression.py"):
        sources[f"tools/{name}"] = _sha256(repo / "tools" / name)
    return sources


def _expected_controller_parameters(reference_manifest: dict) -> dict:
    scale_one = deepcopy(reference_manifest["controller_parameters"])
    expected = {"1": scale_one, "2": deepcopy(scale_one)}
    for config in expected["2"].values():
        base = config["base"]["base"]
        base["rotational_stiffness"] = [40.0] * 3
        base["rotational_damping"] = [5.0 * np.sqrt(2.0)] * 3
    return expected


def _validate_protocol(protocol: dict, reference_configs: dict, reference_manifest: dict):
    indices = protocol.get("selected_case_indices")
    if (
        protocol.get("identity") != IDENTITY
        or protocol.get("scales") != list(SCALES)
        or protocol.get("methods") != METHODS
        or protocol.get("new_holdout") is not False
        or not isinstance(indices, list)
        or not indices
        or any(isinstance(i, bool) or not isinstance(i, int) or i not in range(24) for i in indices)
        or len(set(indices)) != len(indices)
        or protocol.get("is_subset") is not (set(indices) != set(range(24)))
        or protocol.get("checks") != CHECKS
        or protocol.get("gate_scope")
        != "Same-scale friction reference for every method; new descriptive engineering checks."
        or protocol.get("full_trace_case_indices") != list(FULL_CASES)
        or protocol.get("full_trace_methods") != ["friction", "online"]
        or protocol.get("compact_fields") != list(COMPACT_FIELDS)
        or protocol.get("pair_metrics") != list(ALL_METRICS)
        or protocol.get("torque_reserve_fraction") != 0.10
        or protocol.get("decision_policy")
        != "No new tuning; report all paired and physical-group costs before adoption."
    ):
        raise ValueError("protocol does not match the frozen rotation-gain contract")
    configurations = protocol.get("configurations")
    if not isinstance(configurations, list) or len(configurations) != len(indices):
        raise ValueError("protocol configuration grid is incomplete")
    mapping = {row.get("case_index"): row for row in configurations}
    if len(mapping) != len(configurations) or set(mapping) != set(indices):
        raise ValueError("protocol configuration grid differs from selected cases")
    for index, row in mapping.items():
        expected = reference_configs["baseline", index]
        if row != {
            "case_index": index,
            "scenario": expected["scenario"],
            "config": expected["config"],
            "task": expected["task"],
        }:
            raise ValueError(f"protocol case {index} differs from pinned public24 reference")
    if protocol.get("controller_parameters") != _expected_controller_parameters(reference_manifest):
        raise ValueError("protocol controller parameters differ from the declared gain variants")
    return indices, mapping


def _number(row: dict[str, str], name: str, *, optional: bool = False):
    value = row.get(name)
    if optional and value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"published numeric cell is invalid: {name}") from error
    if not np.isfinite(result):
        raise ValueError(f"published numeric cell is invalid: {name}")
    return result


def _assert_close(actual, expected, context: str) -> None:
    if expected is None:
        if actual is not None:
            raise ValueError(f"{context} mismatch")
    elif actual is None or not np.allclose(actual, expected, rtol=1e-10, atol=FLOAT_ATOL):
        raise ValueError(f"{context} mismatch")


def _trace_identity(loaded, path: Path, scale: float, method: str, index: int) -> None:
    try:
        trace_scale = loaded["rotation_gain_scale"]
        trace_index = loaded["case_index"]
        trace_method = loaded["method"]
    except KeyError as error:
        raise ValueError(f"trace scalar identity is missing: {path.name}") from error
    if (
        trace_scale.shape != ()
        or trace_scale.dtype.kind not in "iuf"
        or float(trace_scale) != scale
        or trace_index.shape != ()
        or trace_index.dtype.kind not in "iu"
        or int(trace_index) != index
        or trace_method.shape != ()
        or trace_method.dtype.kind not in "US"
        or str(trace_method.item()) != method
    ):
        raise ValueError(f"trace scalar identity mismatch: {path.name}")


def _load_compact(path: Path, scale: float, method: str, index: int, case: dict) -> dict:
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != {*COMPACT_FIELDS, *TRACE_METADATA}:
            raise ValueError(f"compact trace schema mismatch: {path.name}")
        _trace_identity(loaded, path, scale, method, index)
        trace = {name: loaded[name] for name in COMPACT_FIELDS}
    timestep = case["config"]["timestep"]
    count = round(case["config"]["duration"] / timestep)
    shapes = {
        **dict.fromkeys(
            (
                "time",
                "true_normal_force",
                "target_normal_force",
                "orientation_error_rad",
                "measured_normal_force",
                "torque_projection_scale",
                "true_contact_gap_m",
            ),
            (count,),
        ),
        **dict.fromkeys(
            (
                "position",
                "target_position",
                "linear_velocity",
                "target_linear_velocity",
                "requested_tangential_force_world",
            ),
            (count, 3),
        ),
        **dict.fromkeys(
            ("commanded_torque", "applied_torque", "lower_torque_limit", "upper_torque_limit"),
            (count, 7),
        ),
    }
    for name, expected in shapes.items():
        values = trace[name]
        if values.shape != expected or values.dtype.kind not in "iuf" or not np.all(np.isfinite(values)):
            raise ValueError(f"compact trace field is not finite numeric {expected}: {name}")
    _assert_close(trace["time"], np.arange(count) * timestep, "control time")
    low, high = trace["lower_torque_limit"], trace["upper_torque_limit"]
    if np.any(low >= high) or not np.allclose(
        trace["applied_torque"], np.clip(trace["commanded_torque"], low, high), rtol=0, atol=1e-10
    ):
        raise ValueError("recorded torque limits or clipping chronology is invalid")
    angle = np.deg2rad(case["scenario"]["wall_yaw_deg"])
    normal = np.array([np.cos(angle), np.sin(angle), 0.0])
    gap = (np.array([0.4, 0.0, 0.0]) - trace["position"]) @ normal - 0.025
    if not np.allclose(trace["true_contact_gap_m"], gap, rtol=0, atol=1e-12):
        raise ValueError("recorded contact gap differs from geometry")
    return trace


def _metrics(trace: dict, case: dict) -> dict:
    time = trace["time"]
    mask = time >= case["config"]["evaluation_start"]
    if not np.any(mask):
        raise ValueError("compact trace does not observe evaluation window")
    force = trace["true_normal_force"]
    contact = np.flatnonzero(force > 0)
    angle = np.deg2rad(case["scenario"]["wall_yaw_deg"])
    normal = np.array([np.cos(angle), np.sin(angle), 0.0])
    position_error = trace["position"][mask] - trace["target_position"][mask]
    position_error -= np.outer(position_error @ normal, normal)
    velocity_error = trace["linear_velocity"][mask] - trace["target_linear_velocity"][mask]
    velocity_error -= np.outer(velocity_error @ normal, normal)
    commanded = trace["commanded_torque"]
    low, high = trace["lower_torque_limit"], trace["upper_torque_limit"]
    reserve = 0.05 * (high - low)
    penetration = np.maximum(-trace["true_contact_gap_m"], 0) * 1000
    return {
        "has_raw_contact": bool(contact.size),
        "force_rmse_n": float(
            np.sqrt(np.mean((force[mask] - trace["target_normal_force"][mask]) ** 2))
        ),
        "peak_force_n": float(np.max(force)),
        "tangent_rmse_mm": float(1000 * np.sqrt(np.mean(np.sum(position_error**2, axis=1)))),
        "contact_ratio_pct": float(100 * np.mean(force[mask] > 0.5)),
        "saturation_pct": float(
            100
            * np.mean(
                np.any(np.abs(commanded - trace["applied_torque"]) > 1e-9, axis=1)
            )
        ),
        "orientation_rmse_deg": float(
            np.rad2deg(np.sqrt(np.mean(trace["orientation_error_rad"][mask] ** 2)))
        ),
        "measurement_rmse_n": float(
            np.sqrt(np.mean((trace["measured_normal_force"][mask] - force[mask]) ** 2))
        ),
        "seconds_over_35_n": float(
            np.count_nonzero(force > 35) * case["config"]["timestep"]
        ),
        "first_raw_contact_time_s": float(time[contact[0]]) if contact.size else None,
        "max_penetration_mm": float(np.max(penetration)),
        "evaluation_median_penetration_mm": float(np.median(penetration[mask])),
        "projection_pct": float(100 * np.mean(trace["torque_projection_scale"] < 1 - 1e-12)),
        "max_compensation_force_n": float(
            np.max(np.linalg.norm(trace["requested_tangential_force_world"], axis=1))
        ),
        "tangent_velocity_error_rms_m_s": float(
            np.sqrt(np.mean(np.sum(velocity_error**2, axis=1)))
        ),
        "minimum_torque_headroom_nm": float(
            np.min(np.minimum(commanded - low, high - commanded))
        ),
        "minimum_reserved_torque_headroom_nm": float(
            np.min(np.minimum(commanded - low - reserve, high - reserve - commanded))
        ),
    }


def _failures(row: dict, friction: dict) -> tuple[str, ...]:
    return tuple(
        name
        for passed, name in (
            (
                row["has_raw_contact"]
                and row["contact_ratio_pct"] >= CHECKS["minimum_contact_ratio_pct"],
                "contact_ratio",
            ),
            (row["peak_force_n"] <= CHECKS["maximum_raw_peak_force_n"], "raw_peak_force"),
            (row["saturation_pct"] == CHECKS["required_saturation_pct"], "saturation"),
            (
                row["force_rmse_n"] - friction["force_rmse_n"]
                <= CHECKS["maximum_paired_force_rmse_increase_n"] + 1e-12,
                "paired_force_rmse",
            ),
            (
                row["orientation_rmse_deg"] - friction["orientation_rmse_deg"]
                <= CHECKS["maximum_paired_orientation_rmse_increase_deg"] + 1e-12,
                "paired_orientation_rmse",
            ),
            (row["projection_pct"] <= FLOAT_ATOL, "torque_projection"),
            (row["minimum_reserved_torque_headroom_nm"] >= -FLOAT_ATOL, "reserved_torque_headroom"),
        )
        if not passed
    )


def _summary(rows: list[dict], pairs: list[dict], indices: list[int]) -> dict:
    effects = {}
    for method in METHODS:
        group = [row for row in pairs if row["method"] == method]
        effects[method] = {}
        for name in ALL_METRICS:
            values = [row[name] for row in group if row[name] is not None]
            effects[method][name] = {
                "count": len(values),
                "mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None,
                "minimum": min(values) if values else None,
                "maximum": max(values) if values else None,
                "negative_count": sum(value < -FLOAT_ATOL for value in values),
                "positive_count": sum(value > FLOAT_ATOL for value in values),
            }
    group_keys = ("method", "wall_yaw_deg", "wall_time_constant_s", "tool_mass_kg")
    groups = []
    for key in sorted({tuple(row[name] for name in group_keys) for row in pairs}):
        group = [row for row in pairs if tuple(row[name] for name in group_keys) == key]
        groups.append(
            {
                **dict(zip(group_keys, key, strict=True)),
                "noise_seeds": sorted(row["simulation_seed"] for row in group),
                "mean_deltas": {
                    name: float(np.mean([row[name] for row in group]))
                    if all(row[name] is not None for row in group)
                    else None
                    for name in ALL_METRICS
                },
            }
        )
    return {
        "identity": IDENTITY,
        "is_subset": len(indices) != 24,
        "new_holdout": False,
        "comparison_rows": len(rows),
        "paired_rows": len(pairs),
        "gate_counts": {
            str(int(scale)): {
                method: sum(
                    row["all_gates_pass"] == "yes"
                    for row in rows
                    if row["scale"] == scale and row["method"] == method
                )
                for method in METHODS
            }
            for scale in SCALES
        },
        "failures": [
            {name: row[name] for name in ("scale", "method", "case_index", "failed_gates")}
            for row in rows
            if row["all_gates_pass"] != "yes"
        ],
        "frozen_metric_max_abs_error": max(
            row["frozen_metric_max_abs_error"] for row in rows if row["scale"] == 1
        ),
        "delta_direction": "scale2_minus_scale1_same_method_case_seed",
        "paired_effects": effects,
        "physical_groups": groups,
        "default_changed": False,
        "decision_scope": "Engineering checks and tradeoffs; no automatic global-default promotion.",
    }


def _json_close(actual, expected, context="summary") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{context} schema mismatch")
        for key in expected:
            _json_close(actual[key], expected[key], f"{context}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{context} schema mismatch")
        for index, value in enumerate(expected):
            _json_close(actual[index], value, f"{context}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise TypeError(f"{context} mismatch")
        _assert_close(float(actual), float(expected), context)
    elif actual != expected:
        raise ValueError(f"{context} mismatch")


def audit_archive(directory: Path | str) -> dict:
    root = Path(directory).resolve()
    manifest = _verify_manifest(root)
    reference_rows, reference_configs, reference = _reference()
    if (
        manifest.get("identity") != IDENTITY
        or manifest.get("new_holdout") is not False
        or manifest.get("reference_archive") != reference["identity"]
    ):
        raise ValueError("manifest identity or pinned reference differs")
    protocol = _json(root / "protocol.json")
    indices, cases = _validate_protocol(protocol, reference_configs, reference["manifest"])
    if manifest.get("is_subset") is not protocol["is_subset"]:
        raise ValueError("manifest and protocol subset status differ")
    source_hashes = _json(root / "source_hashes.json")
    current_source_matches_archive = source_hashes == _live_source_identity()
    if current_source_matches_archive:
        source_identity_status = "current"
    elif _sha256(root / "source_hashes.json") in KNOWN_ARCHIVE_SOURCE_HASHES_SHA256:
        source_identity_status = "known_archive"
    else:
        raise ValueError("archive source identity is neither current nor a known archive")

    fields, published = _read_csv(root / "comparison.csv")
    expected_fields = {
        "scale",
        "method",
        "case_index",
        "scenario",
        "simulation_seed",
        "wall_yaw_deg",
        "wall_time_constant_s",
        "tool_mass_kg",
        "duration_s",
        "has_raw_contact",
        *ALL_METRICS,
        "frozen_metric_max_abs_error",
        "failed_gates",
        "all_gates_pass",
    }
    expected_grid = {(scale, method, index) for scale in SCALES for index in indices for method in METHODS}
    if set(fields) != expected_fields or len(published) != len(expected_grid):
        raise ValueError("comparison schema or row count differs from protocol")
    published_map = {
        (_number(row, "scale"), row["method"], int(_number(row, "case_index"))): row
        for row in published
    }
    if len(published_map) != len(published) or set(published_map) != expected_grid:
        raise ValueError("comparison grid differs from protocol")

    rows = []
    compact_paths = set()
    for scale, method, index in sorted(expected_grid):
        row = published_map[scale, method, index]
        case = cases[index]
        expected_identity = {
            "scenario": case["scenario"]["name"],
            "simulation_seed": case["config"]["seed"],
            "wall_yaw_deg": case["scenario"]["wall_yaw_deg"],
            "wall_time_constant_s": case["scenario"]["wall_time_constant"],
            "tool_mass_kg": case["scenario"]["tool_mass_kg"],
            "duration_s": case["config"]["duration"],
        }
        if row["scenario"] != expected_identity.pop("scenario"):
            raise ValueError("comparison scenario identity mismatch")
        for name, value in expected_identity.items():
            _assert_close(_number(row, name), value, f"comparison.{name}")
        path = root / "traces" / f"s{int(scale)}__{method}__case_{index:02d}__compact.npz"
        trace = _load_compact(path, scale, method, index, case)
        compact_paths.add(path)
        values = _metrics(trace, case)
        if row["has_raw_contact"] != str(values["has_raw_contact"]):
            raise ValueError("comparison has_raw_contact mismatch")
        for name in ALL_METRICS:
            _assert_close(
                _number(row, name, optional=name == "first_raw_contact_time_s"),
                values[name],
                f"comparison[{scale},{method},{index}].{name}",
            )
        if scale == 1:
            errors = []
            frozen = reference_rows[method, index]
            for name in (*METRICS, *AUXILIARY_METRICS):
                expected = float(frozen[name]) if frozen[name] else None
                actual = values[name]
                errors.append(0.0 if actual is expected else abs(actual - expected))
            frozen_error = max(errors)
            if not np.isfinite(frozen_error) or frozen_error > FLOAT_ATOL:
                raise ValueError("scale-1 trace does not reproduce the pinned public96 metrics")
        else:
            frozen_error = None
        _assert_close(
            _number(row, "frozen_metric_max_abs_error", optional=True),
            frozen_error,
            "frozen metric error",
        )
        rows.append({"scale": scale, "method": method, "case_index": index, **expected_identity, **values})

    friction = {(row["scale"], row["case_index"]): row for row in rows if row["method"] == "friction"}
    for row in rows:
        published_row = published_map[row["scale"], row["method"], row["case_index"]]
        failed = _failures(row, friction[row["scale"], row["case_index"]])
        row.update(
            frozen_metric_max_abs_error=_number(
                published_row, "frozen_metric_max_abs_error", optional=True
            ),
            failed_gates=";".join(failed),
            all_gates_pass="no" if failed else "yes",
        )
        if (
            published_row["failed_gates"] != row["failed_gates"]
            or published_row["all_gates_pass"] != row["all_gates_pass"]
        ):
            raise ValueError("published gate result differs from reconstruction")

    actual_compact = set(root.glob("traces/*__compact.npz"))
    if actual_compact != compact_paths:
        raise ValueError("compact trace inventory differs from protocol")
    expected_full = {
        root / "traces" / f"s{int(scale)}__{method}__case_{index:02d}__full.npz"
        for scale in SCALES
        for index in set(indices) & set(FULL_CASES)
        for method in ("friction", "online")
    }
    actual_full = set(root.glob("traces/*__full.npz"))
    if actual_full != expected_full:
        raise ValueError("full trace inventory differs from predeclared retention")
    for path in actual_full:
        parts = path.stem.split("__")
        scale, method, index = float(parts[0][1:]), parts[1], int(parts[2].removeprefix("case_"))
        compact = root / "traces" / f"s{int(scale)}__{method}__case_{index:02d}__compact.npz"
        with np.load(path, allow_pickle=False) as loaded, np.load(
            compact, allow_pickle=False
        ) as compact_loaded:
            _trace_identity(loaded, path, scale, method, index)
            for name in COMPACT_FIELDS:
                if name not in loaded.files or not np.array_equal(loaded[name], compact_loaded[name]):
                    raise ValueError(f"full trace differs from compact field {name}: {path.name}")

    pair_fields, published_pairs = _read_csv(root / "paired.csv")
    context = (
        "method",
        "case_index",
        "simulation_seed",
        "wall_yaw_deg",
        "wall_time_constant_s",
        "tool_mass_kg",
    )
    if set(pair_fields) != {*context, "from_scale", "to_scale", *ALL_METRICS}:
        raise ValueError("paired.csv schema mismatch")
    old = {(row["method"], row["case_index"]): row for row in rows if row["scale"] == 1}
    pairs = []
    for row in (row for row in rows if row["scale"] == 2):
        baseline = old[row["method"], row["case_index"]]
        pair = {
            **{name: row[name] for name in context},
            "from_scale": 1.0,
            "to_scale": 2.0,
            **{
                name: row[name] - baseline[name]
                if row[name] is not None and baseline[name] is not None
                else None
                for name in ALL_METRICS
            },
        }
        pairs.append(pair)
    published_pair_map = {
        (row["method"], int(_number(row, "case_index"))): row for row in published_pairs
    }
    if len(published_pairs) != len(pairs) or set(published_pair_map) != set(old):
        raise ValueError("paired.csv grid mismatch")
    for pair in pairs:
        row = published_pair_map[pair["method"], pair["case_index"]]
        if row["method"] != pair["method"]:
            raise ValueError("paired method mismatch")
        for name in context[1:]:
            _assert_close(_number(row, name), pair[name], f"paired.{name}")
        for name in ("from_scale", "to_scale", *ALL_METRICS):
            _assert_close(
                _number(row, name, optional=name == "first_raw_contact_time_s"),
                pair[name],
                f"paired.{name}",
            )
    _json_close(_json(root / "summary.json"), _summary(rows, pairs, indices))
    return {
        "archive": str(root),
        "identity": IDENTITY,
        "current_source_matches_archive": current_source_matches_archive,
        "source_identity_status": source_identity_status,
        "is_subset": protocol["is_subset"],
        "comparison_rows": len(rows),
        "paired_rows": len(pairs),
        "compact_traces": len(compact_paths),
        "full_traces": len(actual_full),
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = audit_archive(arguments.audit)
    except Exception as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        raise SystemExit(1) from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
