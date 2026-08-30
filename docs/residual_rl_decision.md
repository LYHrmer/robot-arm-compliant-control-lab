# Residual RL decision for compliant contact control

## v0.4 decision

Do not deploy the v0.4 checkpoint. Keep the adaptive controller as the nominal path and evaluate
the residual again only after torque-aware projection and multi-seed training are frozen.

The v0.3 result was an evidence-based `GO TO EXPERIMENT`, not proof that RL was necessary. The
implemented v0.4 comparison now gives fixed/adaptive/residual pass counts of 6/24, 6/24 and 7/24.
Residual RL reduces tangential P95 error to 14.80 mm, but raw peak P95 remains 57.04 N and worst
torque saturation rises to 15.91%. It misses the unchanged 90% gate by a wide margin.

The original fixed-gain trigger was:

| Metric | Holdout result | Gate |
|---|---:|---:|
| Case pass rate | 25.0% | >= 90% |
| Force RMSE P95 / worst | 2.32 / 5.87 N | <= 2.0 N per case |
| Contact ratio worst | 100% | >= 95% |
| Raw peak force P95 / worst | 57.79 / 66.30 N | <= 35 N |
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

## Implemented v0.4 design

The trained policy runs at 50 Hz above the 500 Hz adaptive hybrid loop:

\[
\mathbf{w}_{cmd}=\mathbf{w}_{hybrid}+
S\,\mathrm{clip}(\Delta\mathbf{w}_{RL}),
\qquad
\Delta\mathbf{w}_{RL}=
[\Delta F_n,\Delta F_y,\Delta F_z,0,0,0]^T.
\]

- Observation: normal force error, corrected force and filtered force rate; normal/tangential
  position and velocity errors; target force, contact blend and the prior three-axis action. There
  are 14 ordered fields and no joint state or raw contact force.
- Action: a bounded translational wrench residual; start with `+/-4 N` normal and `+/-6 N`
  tangential bounds. Orientation remains under the classical impedance loop.
- Algorithm: an inspectable linear `tanh` policy trained from zero with Augmented Random Search and
  domain-randomized rollouts. SAC/TD3 remain future nonlinear baselines, not silently substituted
  claims.
- Cost: normalized force RMSE, raw peak force, tangential RMSE, saturation, contact shortfall and
  residual RMS. Residual rate is hard-limited by the controller but is not a cost term.
- Safety envelope: total normal-wrench clamp, low-pass/rate-limited residual,
  non-finite/deadline watchdog, 100 ms stable-contact enable delay, force-overshoot guard, and
  immediate zero-residual fallback. Joint torque remains limited by the existing simulation adapter.

## v0.5 evaluation contract

Seed 29 is public validation data. v0.5 uses eight training cases, eight development cases and a new
48-case first reveal derived from a future public random beacon. Five ARS seeds are frozen before
the beacon is published. Fixed hybrid, adaptive hybrid, torque-safe adaptive and all five residual
policies then run on the same physical cases and noise seeds.

The primary rule is fixed before reveal: every residual policy must pass at least 44/48 cases under
the existing force, contact, peak, tangential and torque-saturation gates. The repository keeps all
five policy results; it does not replace this rule with the best seed or mean pass rate. SAC/TD3 and
end-to-end RL are later comparisons, not part of this safety-baseline gate.
