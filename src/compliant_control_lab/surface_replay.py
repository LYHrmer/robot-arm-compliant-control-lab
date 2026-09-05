"""Replay recorded controller inputs, without simulating new robot dynamics."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from compliant_control_lab.franka_adaptive import FrankaSafeAdaptiveController
from compliant_control_lab.franka_control import (
    FrankaActuationContext,
    FrankaController,
    FrankaState,
    FrankaTarget,
)
from compliant_control_lab.surface_control import SurfaceAdaptiveController, SurfaceFrame

SCHEMA_VERSION = 1
REPLAY_ABSOLUTE_TOLERANCE = 1e-10
SAMPLE_SHAPES = {
    "measured_position": (3,),
    "measured_rotation": (3, 3),
    "measured_linear_velocity": (3,),
    "measured_angular_velocity": (3,),
    "measured_normal_force": (),
    "cartesian_jacobian": (6, 7),
    "joint_torque_offset": (7,),
    "lower_torque_limit": (7,),
    "upper_torque_limit": (7,),
    "target_position": (3,),
    "target_rotation": (3, 3),
    "target_linear_velocity": (3,),
    "target_angular_velocity": (3,),
    "target_normal_force": (),
    "commanded_wrench": (6,),
    "commanded_torque": (7,),
    "applied_torque": (7,),
}


@dataclass(frozen=True)
class SurfaceReplayResult:
    """Absolute mathematical replay errors; matches makes no closed-loop claim."""

    sample_count: int
    max_wrench_error: float
    max_torque_error: float
    matches: bool
    controller_kind: str
    controller_name: str
    controller_supplied: bool


def _validate_trace(arrays: dict[str, np.ndarray]) -> int:
    required = {"schema_version", "dt", "controller_frame_rotation", "controller_kind"}
    missing = (required | SAMPLE_SHAPES.keys()) - arrays.keys()
    if missing:
        raise ValueError(f"surface trace missing fields: {sorted(missing)}")
    for name, values in arrays.items():
        if values.dtype.kind not in "biufUS":
            raise ValueError(f"{name} must contain real numbers or strings, never objects")
        if values.dtype.kind in "biuf" and not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
    schema = arrays["schema_version"]
    if schema.shape != () or schema.dtype.kind not in "iu" or schema.item() != SCHEMA_VERSION:
        raise ValueError("unsupported surface trace schema_version")
    dt = arrays["dt"]
    if dt.shape != () or dt.dtype.kind not in "iuf" or float(dt) <= 0:
        raise ValueError("dt must be a finite positive real scalar")
    kind = arrays["controller_kind"]
    if (
        kind.shape != ()
        or kind.dtype.kind != "U"
        or kind.item() not in {"surface_adaptive", "world_safe_adaptive"}
    ):
        raise ValueError("controller_kind must identify surface_adaptive or world_safe_adaptive")
    frame = SurfaceFrame(arrays["controller_frame_rotation"])
    if kind.item() == "world_safe_adaptive" and not np.allclose(
        frame.rotation, np.eye(3), rtol=0, atol=REPLAY_ABSOLUTE_TOLERANCE
    ):
        raise ValueError("world_safe_adaptive requires an identity controller frame")
    force = arrays["measured_normal_force"]
    if force.ndim != 1 or len(force) == 0:
        raise ValueError("surface trace must contain a nonempty sequence")
    count = len(force)
    for name, shape in SAMPLE_SHAPES.items():
        values = arrays[name]
        if values.shape != (count, *shape) or values.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be real numeric data of shape {(count, *shape)}")
    for name in ("measured_rotation", "target_rotation"):
        rotations = arrays[name]
        if not np.allclose(
            np.swapaxes(rotations, -1, -2) @ rotations, np.eye(3), rtol=0, atol=1e-10
        ) or not np.allclose(np.linalg.det(rotations), 1.0, rtol=0, atol=1e-10):
            raise ValueError(f"{name} must contain proper orthonormal rotations")
    lower, upper = arrays["lower_torque_limit"], arrays["upper_torque_limit"]
    if np.any(lower >= upper):
        raise ValueError("lower torque limits must be below upper torque limits")
    clipped = np.clip(arrays["commanded_torque"], lower, upper)
    if not np.allclose(arrays["applied_torque"], clipped, rtol=0, atol=1e-10):
        raise ValueError("applied_torque must equal commanded_torque clipped to recorded limits")
    return count


def _new_trace_path(path: Path | str) -> Path:
    target = Path(path).absolute()
    if any(item.is_symlink() for item in (target, *target.parents)):
        raise ValueError("surface trace path must not contain symlinks")
    if target.exists():
        raise FileExistsError(f"surface trace already exists: {target}")
    return target


def save_surface_trace(path: Path | str, arrays: dict[str, np.ndarray]) -> Path:
    """Validate and atomically publish a compressed NPZ; never replace an existing path.

    Schema version defaults to 1. Controller kind describes the default controller
    configuration; a replay with altered parameters must supply that controller.
    """
    target = _new_trace_path(path)
    payload = {name: np.asarray(values) for name, values in arrays.items()}
    payload.setdefault("schema_version", np.array(SCHEMA_VERSION))
    _validate_trace(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".surface-trace-", dir=target.parent) as temporary:
        staged = Path(temporary) / "trace.npz"
        with staged.open("wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        _new_trace_path(target)
        # link is atomic and fails if a competing writer has already published.
        os.link(staged, target)
    return target


def _state(arrays: dict[str, np.ndarray], index: int) -> FrankaState:
    context = FrankaActuationContext(
        **{
            name: arrays[name][index]
            for name in (
                "cartesian_jacobian",
                "joint_torque_offset",
                "lower_torque_limit",
                "upper_torque_limit",
            )
        }
    )
    return FrankaState(
        **{
            name: arrays[f"measured_{name}"][index].copy()
            for name in ("position", "rotation", "linear_velocity", "angular_velocity")
        },
        normal_force=float(arrays["measured_normal_force"][index]),
        actuation=context,
    )


def replay_surface_trace(
    path: Path | str,
    controller: FrankaController | None = None,
) -> SurfaceReplayResult:
    """Reset from row 0 and replay every recorded input through one stateful controller.

    The stored torque is the UNCLIPPED J.T @ wrench + offset request. Changed
    targets or output commands are valid inputs to this check and produce a
    mismatch rather than an archive-format error. No plant states are integrated.
    """
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    count = _validate_trace(arrays)
    supplied = controller is not None
    kind = str(arrays["controller_kind"].item())
    if controller is None:
        controller = (
            SurfaceAdaptiveController(SurfaceFrame(arrays["controller_frame_rotation"]))
            if kind == "surface_adaptive"
            else FrankaSafeAdaptiveController()
        )
    controller.reset(_state(arrays, 0))
    max_wrench_error = max_torque_error = 0.0
    for index in range(count):
        state = _state(arrays, index)
        target = FrankaTarget(
            **{
                name: arrays[f"target_{name}"][index].copy()
                for name in ("position", "rotation", "linear_velocity", "angular_velocity")
            },
            normal_force=float(arrays["target_normal_force"][index]),
        )
        wrench = np.asarray(controller.compute(state, target, float(arrays["dt"])), dtype=float)
        assert state.actuation is not None
        torque = state.actuation.joint_torque(wrench)
        max_wrench_error = max(
            max_wrench_error, float(np.max(np.abs(wrench - arrays["commanded_wrench"][index])))
        )
        max_torque_error = max(
            max_torque_error, float(np.max(np.abs(torque - arrays["commanded_torque"][index])))
        )
    return SurfaceReplayResult(
        sample_count=count,
        max_wrench_error=max_wrench_error,
        max_torque_error=max_torque_error,
        matches=max(max_wrench_error, max_torque_error) <= REPLAY_ABSOLUTE_TOLERANCE,
        controller_kind=kind,
        controller_name=controller.name,
        controller_supplied=supplied,
    )
