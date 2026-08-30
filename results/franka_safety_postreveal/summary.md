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

Across 5 residual policies, peak force failed in 20–24 cases and tangential tracking failed in 4–5 cases. The combined saturation-failure count was 0. The next experiment instruments nominal approach and contact transition before changing the policy class.

Source: [`../franka_safety_blind/comparison.csv`](../franka_safety_blind/comparison.csv).
