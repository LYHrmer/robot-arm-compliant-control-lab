"""Same-case gain differences; missing or duplicate counterparts are errors."""

from __future__ import annotations

import math

METRICS = (
    "force_rmse_n", "peak_force_n", "contact_ratio_pct", "tangent_rmse_mm",
    "tangent_velocity_error_rms_m_s", "orientation_rmse_deg", "saturation_pct",
    "projection_pct", "minimum_torque_headroom_nm", "minimum_reserved_torque_headroom_nm",
)
RECOVERY_METRICS = ("force_recovery_time_s", "tangent_recovery_time_s")
IDENTITY_FIELDS = ("case", "surface_yaw_deg", "simulation_seed", "arm", "phase")


def pair_gains(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        scale = row["scale"]
        if isinstance(scale, bool) or scale not in (1, 2):
            raise ValueError("expected fixed scale 1 or 2")
        key = tuple(row.get(name, "overall") if name == "phase" else row[name]
                    for name in IDENTITY_FIELDS)
        group = groups.setdefault(key, {})
        if scale in group:
            raise ValueError("duplicate gain counterpart")
        group[scale] = row
    pairs = []
    for key, group in sorted(groups.items()):
        if set(group) != {1, 2}:
            raise ValueError("missing gain counterpart")
        pair = {**dict(zip(IDENTITY_FIELDS, key)), "from_scale": 1, "to_scale": 2}
        for name in (*METRICS, *RECOVERY_METRICS):
            values = [group[s].get(name) if name in RECOVERY_METRICS else group[s][name]
                      for s in (1, 2)]
            for value in values:
                if value is None and name in RECOVERY_METRICS:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"nonfinite or nonnumeric metric: {name}")
            pair[f"delta_{name}"] = None if None in values else values[1] - values[0]
        pairs.append(pair)
    return pairs
