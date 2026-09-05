# Safe-adaptive contact-event diagnosis

This is a post-reveal descriptive replay of the published v0.5 torque-safe adaptive baseline. It does not alter the frozen first-reveal result or create new holdout evidence.

## Replay check

All 48 published cases were replayed. The table reports the largest absolute difference from the seven frozen metrics. The report writer requires each absolute replay error to be <= 1e-12.

| Frozen metric | Maximum absolute replay error |
|---|---:|
| `force_rmse_n` | 0.000e+00 |
| `peak_force_n` | 0.000e+00 |
| `tangent_rmse_mm` | 0.000e+00 |
| `orientation_rmse_deg` | 0.000e+00 |
| `contact_ratio_pct` | 0.000e+00 |
| `torque_rms_nm` | 0.000e+00 |
| `saturation_pct` | 0.000e+00 |

## Where the raw-force peak occurred

The frozen 35 N gate failed in 18/48 safe-adaptive cases. Delay is measured from the first sample with positive raw normal force to the trial's global raw-force peak.

| Cohort | Cases | Delay median [P25, P75] (s) | Delay range (s) | Early <= 0.5 s | Late >= 1.0 s |
|---|---:|---:|---:|---:|---:|
| all replayed cases | 48 | 1.976 [1.308, 2.306] | 0.000 to 2.802 | 9 | 39 |
| frozen peak-force failures | 18 | 1.774 [0.312, 1.983] | 0.000 to 2.484 | 7 | 11 |

Peak contact phase:

- All cases: raw_contact=4, confirmed_transition=5, blended_contact=39.
- Frozen peak-force failures: raw_contact=2, confirmed_transition=5, blended_contact=11.

Peak motion phase:

- All cases: pre_wiping=8, wiping=40.
- Frozen peak-force failures: pre_wiping=6, wiping=12.

## Peak-sample context

These values share the log row of each failed case's global raw-force peak. They describe the recorded motion and torque context; they are not evidence that any listed variable caused the peak.

The frozen loop reads contact and Jacobian caches from the previous MuJoCo forward evaluation together with already integrated qvel. The wrench is the current controller return, before its next mj_step. The row does not represent strictly synchronized physical measurements.

Velocity and commanded-force components below use world-x, the controller approach axis. Yawed walls have a different surface normal. The `normal` names in peak_context.csv are retained for compatibility and refer to these world-x components.

| Metric | Early <= 0.5 s (n=7) | Late >= 1.0 s (n=11) |
|---|---:|---:|
| Actual world-x velocity median [range] (m/s) | 0.1311 [0.0451, 0.1972] | 0.0053 [0.0015, 0.0080] |
| Target world-x velocity max abs (m/s) | 0.0000 | 0.0000 |
| Commanded world-x force median [range] (N) | 11.302 [5.932, 14.380] | 11.615 [9.348, 14.818] |
| Minimum torque headroom min / median (Nm) | 5.084 / 6.568 | 7.507 / 7.935 |
| Torque-projection scale min | 1.0000 | 1.0000 |

The per-case event table is [`safe_adaptive_contact_events.csv`](safe_adaptive_contact_events.csv). The two-row derived table is [`peak_context.csv`](peak_context.csv); the plot uses the same validated case rows.

Frozen source: [`../../franka_safety_blind/comparison.csv`](../../franka_safety_blind/comparison.csv).
