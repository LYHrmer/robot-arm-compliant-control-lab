# Surface-task development study

Full development grid: 24 cases, 96 rows.
NEW task and sensor/noise definitions: neither the old 24 holdout cases nor the 48 blind cases.
This is public development validation, NOT a new holdout. No fitting or tuning occurs.
All arms share the task, physical case, seed and sensor bias/noise definitions.
The task yaw is supplied explicitly; controller frames use yaw, yaw−5°, yaw+5°, or world x.
True wall normals are restricted to physics/scoring. No pass/fail gate is defined here.
surface_exact assumes ideal frame calibration. An accurate fixture description supplies
the task plane; the study does not test finding an unknown surface.
Payload mass changes are known to the robot dynamics model. Nominal sensor gravity
compensation can be mismatched, but its residual lies along world z and is not consumed
by these horizontal normal-force projections. This is not an unknown-payload test.
All six wrench channels are recorded; only the normal force drives force control.
Moment channels are diagnostic. Measurement RMSE mixes sensor error, filtering and timing lag.

Force RMSE uses current raw true force after evaluation_start; peak uses the full trial.
Tangent RMSE is the Euclidean position-error norm projected onto the true wall plane.
Measurement RMSE compares the actual scalar controller feedback to current true force.
Contact ratio uses raw true force > 0.5 N after evaluation_start. Saturation uses all steps.
Seconds over 35 N = count(raw true force > 35 N) × timestep.

| Arm | Cases | Cases with raw contact | Median contact ratio [%] |
|---|---:|---:|---:|
| world_safe_adaptive | 24 | 24 | 57.1333 |
| surface_exact | 24 | 24 | 56.6667 |
| surface_minus5 | 24 | 24 | 56.8 |
| surface_plus5 | 24 | 24 | 56.8 |

## Absolute metrics across executed cases

| Arm | Median force RMSE [N] | Peak P95 [N] | Median tangent RMSE [mm] | Median contact [%] | Worst saturation [%] |
|---|---:|---:|---:|---:|---:|
| world_safe_adaptive | 11.0888 | 32.34 | 12.4673 | 57.1333 | 0 |
| surface_exact | 10.9164 | 33.9738 | 11.8107 | 56.6667 | 0 |
| surface_minus5 | 10.8991 | 32.6605 | 11.7763 | 56.8 | 0 |
| surface_plus5 | 10.9951 | 32.7654 | 12.0656 | 56.8 | 0 |

## Paired differences from world_safe_adaptive

Differences are arm − baseline. Missing-contact peak/time differences are NA;
contact counts and tracking costs must accompany any peak comparison.

| Arm | Metric | Paired n | Median difference |
|---|---|---:|---:|
| surface_exact | force_rmse_n | 24 | -0.0602035 |
| surface_exact | peak_force_n | 24 | 0 |
| surface_exact | tangent_rmse_mm | 24 | -0.656704 |
| surface_exact | contact_ratio_pct | 24 | -0.966667 |
| surface_exact | saturation_pct | 24 | 0 |
| surface_exact | orientation_rmse_deg | 24 | 0.00113723 |
| surface_exact | measurement_rmse_n | 24 | -0.00824044 |
| surface_exact | seconds_over_35_n | 24 | 0 |
| surface_exact | first_raw_contact_time_s | 24 | 0 |
| surface_minus5 | force_rmse_n | 24 | -0.0937767 |
| surface_minus5 | peak_force_n | 24 | 0.163 |
| surface_minus5 | tangent_rmse_mm | 24 | -0.691124 |
| surface_minus5 | contact_ratio_pct | 24 | -0.8 |
| surface_minus5 | saturation_pct | 24 | 0 |
| surface_minus5 | orientation_rmse_deg | 24 | 0.00162593 |
| surface_minus5 | measurement_rmse_n | 24 | -0.0506871 |
| surface_minus5 | seconds_over_35_n | 24 | 0 |
| surface_minus5 | first_raw_contact_time_s | 24 | 0 |
| surface_plus5 | force_rmse_n | 24 | 0.0263357 |
| surface_plus5 | peak_force_n | 24 | -0.0988639 |
| surface_plus5 | tangent_rmse_mm | 24 | -0.393223 |
| surface_plus5 | contact_ratio_pct | 24 | -0.866667 |
| surface_plus5 | saturation_pct | 24 | 0 |
| surface_plus5 | orientation_rmse_deg | 24 | -0.000543768 |
| surface_plus5 | measurement_rmse_n | 24 | 0.0650429 |
| surface_plus5 | seconds_over_35_n | 24 | 0 |
| surface_plus5 | first_raw_contact_time_s | 24 | 0 |

Full canonical traces retained: representative_case_16_world_safe_adaptive.npz, representative_case_16_surface_exact.npz, representative_case_16_surface_minus5.npz, representative_case_16_surface_plus5.npz.
Only representative case 16 retains full traces; other cases retain full parameters
and summary metrics, not full traces. comparison.csv includes every executed arm/case.
The representative case was fixed before execution: yaw +15°, wall time 0.005 s,
tool mass 0.10 kg, noise seed 11. Existing v0.5 archives and conclusions are unchanged.
