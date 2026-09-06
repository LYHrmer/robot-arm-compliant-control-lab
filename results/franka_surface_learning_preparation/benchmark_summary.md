# Surface learning preparation checks

Public development checks, NOT a new holdout. No training or parameter tuning.
Regular-run hard safety target: 100% normal horizon completion, no safety termination or missing logs.
Separate engineering targets per case: wiping contact >=99%, tangent RMSE <=10 mm, force RMSE <=2 N.
Failures are retained, including episodes without an observed wiping window.
All runs use the same 50 Hz / 500 Hz environment. Teacher: adaptive nominal + feedback-only friction action.
Metrics are independently reconstructed from full physics traces; no environment episode metrics are copied.

| Method | Runs | Hard completions | Engineering targets | Force median [N] | Tangent median [mm] | Contact min [%] | Peak max [N] |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero_adaptive | 24 | 24/24 | 4/24 | 0.186212 | 11.8388 | 100 | 17.2067 |
| zero_friction | 24 | 24/24 | 24/24 | 0.187809 | 3.46597 | 100 | 17.2023 |
| friction_teacher_50hz | 24 | 24/24 | 24/24 | 0.188945 | 3.60601 | 100 | 17.2067 |

Unavailable metrics are excluded only from the displayed aggregate, never from run/pass counts.
Throughput and decision wall time include simulation, not policy-inference latency or real-time guarantees.
Complete local physics NPZ and decision/event JSON accompany this report.

## Cases missing completion or engineering targets

- g000_seed11 / zero_adaptive: hard=True, engineering=False; engineering/trace target not met.
- g000_seed29 / zero_adaptive: hard=True, engineering=False; engineering/trace target not met.
- g001_seed11 / zero_adaptive: hard=True, engineering=False; engineering/trace target not met.
- g001_seed29 / zero_adaptive: hard=True, engineering=False; engineering/trace target not met.
- g002_seed11 / zero_adaptive: hard=True, engineering=False; engineering/trace target not met.
- g002_seed29 / zero_adaptive: hard=True, engineering=False; engineering/trace target not met.
- g003_seed11 / zero_adaptive: hard=True, engineering=False; engineering/trace target not met.
- g003_seed29 / zero_adaptive: hard=True, engineering=False; engineering/trace target not met.
- g004_seed11 / zero_adaptive: hard=True, engineering=False; engineering/trace target not met.
- g004_seed29 / zero_adaptive: hard=True, engineering=False; engineering/trace target not met.
- g005_seed11 / zero_adaptive: hard=True, engineering=False; engineering/trace target not met.
- g005_seed29 / zero_adaptive: hard=True, engineering=False; engineering/trace target not met.
- g006_seed11 / zero_adaptive: hard=True, engineering=False; engineering/trace target not met.
- g006_seed29 / zero_adaptive: hard=True, engineering=False; engineering/trace target not met.
- g007_seed11 / zero_adaptive: hard=True, engineering=False; engineering/trace target not met.
- g007_seed29 / zero_adaptive: hard=True, engineering=False; engineering/trace target not met.
- g009_seed11 / zero_adaptive: hard=True, engineering=False; engineering/trace target not met.
- g009_seed29 / zero_adaptive: hard=True, engineering=False; engineering/trace target not met.
- g010_seed11 / zero_adaptive: hard=True, engineering=False; engineering/trace target not met.
- g010_seed29 / zero_adaptive: hard=True, engineering=False; engineering/trace target not met.
