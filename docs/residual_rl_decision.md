# Residual RL decision for compliant contact control

## Decision

**Proceed to a v0.4 research experiment, but keep the impact-aware classical hybrid controller as
the nominal and fallback controller. Do not replace it with an end-to-end joint-torque policy.**

This is an evidence-based `GO TO EXPERIMENT`, not proof that RL is necessary or a feature chosen for
its name. The impact-aware fixed-gain hybrid controller passed only 6 of 24 deterministic randomized
holdout cases (25.0%) against limits declared before running the benchmark. Contact was never lost,
but the main failures were raw contact force and
tangential tracking under contact/dynamics mismatch:

| Metric | Holdout result | Gate |
|---|---:|---:|
| Case pass rate | 25.0% | >= 90% |
| Force RMSE P95 / worst | 2.32 / 5.87 N | <= 2.0 N per case |
| Contact ratio worst | 100% | >= 95% |
| Raw peak force P95 / worst | 57.74 / 65.75 N | <= 35 N |
| Tangential RMSE P95 / worst | 21.98 / 29.16 mm | <= 15 mm |
| Torque saturation worst | 12.53% | <= 1% |

The complete conditions and metrics are in
[`results/franka_stress/metrics.csv`](../results/franka_stress/metrics.csv). The randomization covers
wall compliance and friction, +/-6 degree surface-normal error, sensor noise and bias, 0--30 ms
measurement delay, and +/-15% dynamics-bias mismatch. Seed 29 fixes scenario generation; each CSV
row also records the simulation-noise seed used for that case.

## Why residual rather than end-to-end RL

Johannink et al. formulate Residual RL as an additive policy
`u = pi_classical + pi_residual`, using the conventional controller for known robot structure and
learning the difficult contact/friction remainder. Their real block-assembly experiment also shows
the intended use case: the hand controller works in the nominal geometry but degrades under object
misalignment. See [Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201).

For contact-rich sim-to-real transfer, Zhang et al. similarly retain Cartesian admittance control and
adapt residual compliance parameters online; their experiments target the contact-parameter gap
rather than relearning rigid-body control. See
[Online Admittance Residual Learning, CoRL 2023](https://proceedings.mlr.press/v229/zhang23e.html).

That decomposition matches this repository: kinematics, bias compensation, null-space posture,
force PI, limits, and fallback behavior are already explicit and testable. Learning them again would
increase training cost and weaken safety/debuggability.

## Proposed v0.4 design

Run the trained policy at 50--100 Hz above the 500 Hz hybrid loop:

\[
\mathbf{w}_{cmd}=\mathbf{w}_{hybrid}+
S\,\mathrm{clip}(\Delta\mathbf{w}_{RL}),
\qquad
\Delta\mathbf{w}_{RL}=
[\Delta F_n,\Delta F_y,\Delta F_z,0,0,0]^T.
\]

- Observation: force error and short history, raw/filtered force, tangential pose/velocity error,
  joint position/velocity, prior residual, and contact phase.
- Action: a bounded translational wrench residual; start with `+/-4 N` normal and `+/-6 N`
  tangential bounds. Orientation remains under the classical impedance loop.
- Algorithm: SAC or TD3 with zero-initialized final policy layer and domain randomization.
- Reward: force and tangential error, strong peak-force/saturation penalties, residual magnitude and
  residual-rate penalties. Report each physical term separately, not only episodic return.
- Safety envelope: total normal-wrench clamp, torque and torque-rate limits, low-pass/rate-limited
  residual, non-finite/deadline watchdog, and immediate zero-residual fallback.

## Evaluation contract

Training cases and the seed-29 holdout cases must be disjoint. Report mean and 95% confidence
intervals over at least five training seeds for:

1. fixed hybrid baseline;
2. gain-scheduled or adaptive-admittance classical baseline;
3. end-to-end RL baseline;
4. hybrid + residual RL;
5. residual policy with domain randomization removed;
6. zero-residual fallback.

The learned controller succeeds only if it raises the same pre-declared holdout pass rate from 25.0%
to at least 90%, introduces no new force/torque-limit violations, and retains nominal behavior when
the residual is disabled. Until then, it is an experiment rather than a claimed improvement.
