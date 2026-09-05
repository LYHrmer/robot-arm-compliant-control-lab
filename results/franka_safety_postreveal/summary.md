# Post-reveal gate diagnosis

This report is derived from the published first-reveal CSV. The 48 cases are now public validation data; this is diagnostic work, not new blind evidence.
The frozen primary threshold was 44/48 passes per policy.
Failure columns are non-exclusive; one case may trigger more than one failure flag.

| Method | Pass | Force RMSE failures | Peak-force failures | Tangent failures | Saturation failures |
|---|---:|---:|---:|---:|---:|
| fixed hybrid | 17/48 | 3 | 18 | 18 | 3 |
| adaptive hybrid | 23/48 | 3 | 19 | 10 | 3 |
| torque-safe adaptive | 24/48 | 5 | 18 | 10 | 0 |
| residual 00 | 22/48 | 3 | 24 | 4 | 0 |
| residual 01 | 25/48 | 3 | 20 | 5 | 0 |
| residual 02 | 26/48 | 3 | 20 | 4 | 0 |
| residual 03 | 24/48 | 3 | 22 | 5 | 0 |
| residual 04 | 25/48 | 3 | 21 | 5 | 0 |

Across 5 residual policies, peak force failed in 20–24 cases and tangential tracking failed in 4–5 cases. The combined saturation-failure count was 0. Peak timing and controller-state telemetry are needed to distinguish entry and later in-contact failures.

## Paired residual effect

Each difference is residual minus torque-safe adaptive on the same case; negative values favor the residual policy. The interval is a deterministic 20,000-sample percentile bootstrap of the paired median. Win/tie/loss uses the stored values without a tolerance.
These are post-reveal descriptive estimates, not confirmatory intervals.

| Policy | Metric | Median difference | Bootstrap 95% CI | Win/tie/loss |
|---|---|---:|---:|---:|
| residual 00 | force RMSE (N) | 0.111 | 0.088 to 0.135 | 7/0/41 |
| residual 00 | raw peak force (N) | 0.586 | 0.226 to 0.959 | 6/9/33 |
| residual 00 | tangent RMSE (mm) | -1.925 | -2.343 to -0.842 | 35/0/13 |
| residual 01 | force RMSE (N) | 0.050 | 0.031 to 0.064 | 9/0/39 |
| residual 01 | raw peak force (N) | 0.359 | 0.014 to 0.639 | 8/9/31 |
| residual 01 | tangent RMSE (mm) | -1.838 | -2.297 to -0.731 | 35/0/13 |
| residual 02 | force RMSE (N) | 0.101 | 0.083 to 0.125 | 8/0/40 |
| residual 02 | raw peak force (N) | 0.498 | 0.050 to 0.760 | 8/8/32 |
| residual 02 | tangent RMSE (mm) | -2.093 | -2.402 to -1.286 | 35/0/13 |
| residual 03 | force RMSE (N) | 0.094 | 0.062 to 0.114 | 8/0/40 |
| residual 03 | raw peak force (N) | 0.481 | 0.203 to 0.968 | 6/9/33 |
| residual 03 | tangent RMSE (mm) | -1.474 | -1.852 to -0.728 | 35/0/13 |
| residual 04 | force RMSE (N) | 0.100 | 0.079 to 0.126 | 8/0/40 |
| residual 04 | raw peak force (N) | 0.502 | 0.094 to 1.051 | 6/9/33 |
| residual 04 | tangent RMSE (mm) | -1.516 | -1.891 to -0.476 | 34/0/14 |

## Exploratory gate sensitivity (post-hoc)

Each column recounts passes after omitting only the named gate and keeping the other four. This was computed after reveal and does not alter the frozen primary result.

| Method | Original | No force RMSE | No contact ratio | No peak force | No tangent RMSE | No saturation |
|---|---:|---:|---:|---:|---:|---:|
| fixed hybrid | 17 | 18 | 17 | 26 | 29 | 17 |
| adaptive hybrid | 23 | 24 | 23 | 33 | 28 | 23 |
| torque-safe adaptive | 24 | 25 | 24 | 33 | 29 | 24 |
| residual 00 | 22 | 22 | 22 | 41 | 24 | 22 |
| residual 01 | 25 | 25 | 25 | 40 | 28 | 25 |
| residual 02 | 26 | 26 | 26 | 41 | 28 | 26 |
| residual 03 | 24 | 24 | 24 | 40 | 26 | 24 |
| residual 04 | 25 | 25 | 25 | 40 | 27 | 25 |

Removing only the peak-force gate raises the residual-policy counts to 40–41/48, still below the frozen 44/48 threshold. The raw-peak gate is the largest single source of gate failures; event timing is needed before assigning a physical cause.

Source: [`../franka_safety_blind/comparison.csv`](../franka_safety_blind/comparison.csv).
