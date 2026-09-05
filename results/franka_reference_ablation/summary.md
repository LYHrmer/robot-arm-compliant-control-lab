# Public-development reference ablation

All 48 public cases.
These are already public v0.5 cases, NOT a new holdout. No fitting or tuning occurs.
The frozen v0.5 FAIL is unchanged. All original per-case gates remain active.
Gate values: `{"max_force_rmse_n": 2.0, "max_peak_force_n": 35.0, "max_saturation_pct": 1.0, "max_tangent_rmse_mm": 15.0, "min_case_pass_rate_pct": 90.0, "min_contact_ratio_pct": 95.0}`

Legacy event diagnostics are NA because its log fields are not physically aligned.
Split-step raw force is solved at time[k] with action[k]; filtered force feedback is causal
and lagged. The wall normal is diagnostic ground truth only; controllers use world-x.
Normal/tangent task metrics retain their historical definitions, including world-y/z error.
Empty contact windows are NA. Contact count and contact ratio must accompany peak gains.
Seconds above 35 N = count(raw force > 35 N) × timestep; no interpolation is used.

| Metric | legacy_replay | timing_only | consistent_reference | rate_limited_reference |
|---|---:|---:|---:|---:|
| Cases | 48 | 48 | 48 | 48 |
| Cases with raw contact | 48 | 48 | 48 | 48 |
| Gate passes | 24 | 23 | 24 | 23 |
| First contact median [s] | NA | 0.569 | 0.511 | 0.565 |
| Contact ratio median [%] | 100 | 100 | 100 | 100 |
| Global peak P95 [N] | 59.537 | 60.893 | 59.03 | 61.628 |
| Early peak P95 [N] | NA | 60.893 | 59.03 | 61.628 |
| Late peak P95 [N] | NA | 44.163 | 43.703 | 42.255 |
| Time over 35 N median [s] | NA | 0 | 0 | 0 |

## Paired differences from timing_only

Values are arm − timing_only, each case paired by index and noise seed. Peak/time
differences require raw contact in both arms; n is reported for each metric.
Legacy differences change engine timing, so they are not controller improvements.

| Arm | Metric | Paired n | Median difference |
|---|---|---:|---:|
| legacy_replay | first_raw_contact_time_s | 0 | NA |
| legacy_replay | early_raw_peak_n | 0 | NA |
| legacy_replay | late_raw_peak_n | 0 | NA |
| legacy_replay | seconds_over_35_n | 0 | NA |
| legacy_replay | peak_force_n | 48 | -0.14809 |
| legacy_replay | force_rmse_n | 48 | -0.0063763 |
| legacy_replay | tangent_rmse_mm | 48 | -0.010561 |
| legacy_replay | contact_ratio_pct | 48 | 0 |
| consistent_reference | first_raw_contact_time_s | 48 | -0.03 |
| consistent_reference | early_raw_peak_n | 48 | 0 |
| consistent_reference | late_raw_peak_n | 48 | -0.28881 |
| consistent_reference | seconds_over_35_n | 48 | 0 |
| consistent_reference | peak_force_n | 48 | -0.43763 |
| consistent_reference | force_rmse_n | 48 | -0.022868 |
| consistent_reference | tangent_rmse_mm | 48 | -0.048943 |
| consistent_reference | contact_ratio_pct | 48 | 0 |
| rate_limited_reference | first_raw_contact_time_s | 48 | 0 |
| rate_limited_reference | early_raw_peak_n | 48 | 0 |
| rate_limited_reference | late_raw_peak_n | 48 | 0.0045588 |
| rate_limited_reference | seconds_over_35_n | 48 | 0 |
| rate_limited_reference | peak_force_n | 48 | -0.0044709 |
| rate_limited_reference | force_rmse_n | 48 | 0.0088406 |
| rate_limited_reference | tangent_rmse_mm | 48 | 0.0029004 |
| rate_limited_reference | contact_ratio_pct | 48 | 0 |

Assess early-peak reductions alongside later peaks, contact delay/loss and
force/tangent tracking costs. The paired ablations isolate component changes;
one peak sample does not identify its physical cause. Deployment readiness is untested.
A final performance claim requires a frozen new protocol
and unseen scenarios. Controller wall-clock latency is descriptive, not deterministic.
