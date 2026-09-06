# Surface learning preparation checks

Public development checks, NOT a new holdout. No training or parameter tuning.
Regular-run hard safety target: 100% normal horizon completion, no safety termination or missing logs.
Separate engineering targets per case: wiping contact >=99%, tangent RMSE <=10 mm, force RMSE <=2 N.
Stress runs may terminate safely; terminations and all available raw evidence remain published.
All runs use the same 50 Hz / 500 Hz environment. Teacher: adaptive nominal + feedback-only friction action.
Metrics are independently reconstructed from full physics traces; no environment episode metrics are copied.

| Method | Runs | Hard completions | Engineering targets | Force median [N] | Tangent median [mm] | Contact min [%] | Peak max [N] |
|---|---:|---:|---:|---:|---:|---:|---:|
| plus_one | 3 | 3/3 | 0/3 | 0.685446 | 15.7345 | 100 | 15.6714 |
| minus_one | 3 | 3/3 | 0/3 | 0.629531 | 15.5433 | 100 | 12.6279 |
| alternating | 3 | 3/3 | 3/3 | 0.322591 | 1.69973 | 100 | 13.1124 |
| seeded_random | 3 | 3/3 | 3/3 | 0.866764 | 3.29523 | 100 | 14.478 |

Unavailable metrics are excluded only from the displayed aggregate, never from run/pass counts.
Throughput and decision wall time include simulation, not policy-inference latency or real-time guarantees.
Complete local physics NPZ and decision/event JSON accompany this report.

## Cases missing completion or engineering targets

- g000_seed11 / plus_one: hard=True, engineering=False; engineering/trace target not met.
- g000_seed11 / minus_one: hard=True, engineering=False; engineering/trace target not met.
- g001_seed11 / plus_one: hard=True, engineering=False; engineering/trace target not met.
- g001_seed11 / minus_one: hard=True, engineering=False; engineering/trace target not met.
- g002_seed11 / plus_one: hard=True, engineering=False; engineering/trace target not met.
- g002_seed11 / minus_one: hard=True, engineering=False; engineering/trace target not met.
