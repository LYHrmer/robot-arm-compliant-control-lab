# v0.5 torque-safe Residual RL blind result

The first reveal evaluated 48 cases. Each method used the same case and noise seed.
The frozen primary rule requires every policy to pass at least 44/48 cases.

| Method | Pass | Force P95 [N] | Raw peak P95 [N] | Tangent P95 [mm] | Contact worst | Saturation worst |
|---|---:|---:|---:|---:|---:|---:|
| fixed_hybrid | 17/48 | 2.01 | 58.62 | 22.66 | 100.0% | 19.69% |
| adaptive_hybrid | 23/48 | 2.21 | 58.99 | 18.89 | 100.0% | 20.71% |
| safe_adaptive_hybrid | 24/48 | 2.33 | 59.54 | 18.89 | 100.0% | 0.00% |
| torque_residual_run_00 | 22/48 | 2.00 | 59.54 | 16.04 | 100.0% | 0.00% |
| torque_residual_run_01 | 25/48 | 2.03 | 59.54 | 16.13 | 100.0% | 0.00% |
| torque_residual_run_02 | 26/48 | 2.10 | 59.54 | 15.98 | 100.0% | 0.00% |
| torque_residual_run_03 | 24/48 | 2.12 | 59.54 | 16.50 | 100.0% | 0.00% |
| torque_residual_run_04 | 25/48 | 1.98 | 59.54 | 16.43 | 100.0% | 0.00% |

Mean residual pass rate hierarchical-bootstrap 95% interval: 36.2% to 65.0%.
Frozen primary rule: FAIL.
This directory is the first reveal. Later tuning against these cases is validation work, not blind evaluation.
