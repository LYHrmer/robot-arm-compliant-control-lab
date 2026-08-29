# Franka adaptive control and bounded Residual RL comparison

All methods use the same frozen 24 cases and the same per-case simulation seeds.
The seed-29 set is retained for continuity with v0.3 and is not described as a blind test.

| Method | Pass rate | Force RMSE P95 / worst [N] | Raw peak P95 / worst [N] | Tangent P95 / worst [mm] | Contact worst | Saturation worst |
|---|---:|---:|---:|---:|---:|---:|
| fixed_hybrid | 25.0% | 2.32 / 5.87 | 57.79 / 66.30 | 21.98 / 29.16 | 100.0% | 12.53% |
| adaptive_hybrid | 25.0% | 3.01 / 3.99 | 56.80 / 68.74 | 18.42 / 22.81 | 100.0% | 12.89% |
| bounded_residual_rl | 29.2% | 2.30 / 3.37 | 57.04 / 68.74 | 14.80 / 18.86 | 100.0% | 15.91% |

## Interpretation

The bounded residual policy does not meet the pre-declared 90% case-pass target.
It improves case pass rate over the adaptive classical baseline.
Training return is therefore not used as evidence of deployment readiness.
See `comparison.csv` for every physical metric and failure label.
