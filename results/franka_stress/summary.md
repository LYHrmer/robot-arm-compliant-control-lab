# Franka hybrid controller — randomized holdout stress test

## Pre-declared Residual RL gate

- Force RMSE: <= 2.0 N
- Contact ratio: >= 95.0%
- Raw peak force: <= 35.0 N
- Tangential RMSE: <= 15.0 mm
- Torque saturation: <= 1.0%
- Required case pass rate: >= 90.0%

## Holdout result

- Cases: 24
- Case pass rate: 25.0%
- Force RMSE P50 / P95 / worst: 1.55 / 2.32 / 5.87 N
- Contact ratio P05 / worst: 100.0 / 100.0%
- Raw peak force P95 / worst: 57.74 / 65.75 N
- Tangential RMSE P95 / worst: 21.98 / 29.16 mm
- Saturation worst: 12.53%

## Decision

GO TO EXPERIMENT: the impact-aware fixed-gain baseline misses the pre-declared robustness target, so evaluating a bounded residual policy is justified. This result does not by itself prove that RL will outperform adaptive classical control.

The randomized variables and every per-case metric are preserved in `metrics.csv`.
