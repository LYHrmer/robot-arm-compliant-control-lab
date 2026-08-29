# v0.4 implementation plan: adaptive compliance + bounded Residual RL

> Reference mechanisms: Johannink et al., *Residual Reinforcement Learning for Robot Control*
> (2018); Zhang et al., *Online Admittance Residual Learning* (CoRL 2023); Mania et al.,
> *Simple random search provides a competitive approach to reinforcement learning* (2018).
>
> Target framework: `compliant_control_lab` Franka Cartesian controller seam.

This iteration reproduces the additive residual-control mechanism and the idea of adapting
compliance under contact mismatch. It does **not** claim an exact reproduction of the papers'
robot tasks, neural networks, SAC/TD3 training, or online SQP solver.

## 1. Mechanism

This method changes **the translational Cartesian wrench** at
`FrankaController.compute(state, target, dt)`, first by scheduling the classical contact gains and
then by adding a learned residual, in order to improve force/trajectory robustness without bypassing
the nominal controller or joint-torque safety limits.

- Entity: adaptive classical gains and a 3D translational wrench residual.
- Existing call site wrapped: `FrankaHybridController.compute`.
- Effect: behavior change, not acceleration.
- Bounds: residual `|delta F| <= [4, 6, 6] N`, policy update at 50 Hz, explicit residual-rate
  limits, finite-value watchdog, and a clamped total normal wrench. The residual remains exactly
  zero until the classical contact blend has been complete for 100 ms.

## 2. Core invariants

- **Zero residual is numerically identical to the selected nominal controller.**
  - Maintained by an additive wrapper and verified over a state sequence.
- **The learned policy cannot command orientation torque or joint torque directly.**
  - Maintained by a three-element translational action and the existing `J.T @ wrench` adapter.
- **Every applied residual satisfies magnitude and rate bounds.**
  - Maintained by `tanh`, hard clipping, a rate limiter, and non-finite fallback in one module.
- **Training scenarios never contain the frozen seed-29 24-case evaluation set.**
  - Maintained by separate scenario/simulation seeds and recorded manifests.
- **Fixed, adaptive, and residual controllers see identical holdout cases and simulation seeds.**
  - Maintained by one comparison runner that loops controllers inside each frozen case.

## 3. Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Classical baseline | Bias estimator + force-rate/contact-stiffness gain schedule around impact-aware hybrid | Uses only deployable feedback and remains explainable |
| Residual nominal | Adaptive classical baseline | Tests whether RL adds value beyond a strong non-learning alternative |
| Residual policy | Fixed-normalization linear `tanh` policy | Small, inspectable and fast enough for deterministic CI smoke tests |
| RL optimizer | Augmented Random Search over episodic physical cost | Dependency-free continuous-control policy search; full runs remain reproducible |
| Policy frequency | 50 Hz above the 500 Hz control loop | Matches the documented two-rate architecture |
| Final claim metric | Existing physical gate, not training return | Prevents reward shaping from redefining success |

## 4. Known trade-offs

| Problem | Impact | Mitigation |
|---|---|---|
| Linear ARS policy is less expressive than SAC/TD3 | May leave nonlinear mismatch uncompensated | Treat as the smallest working Residual RL baseline; keep policy interface replaceable |
| Seed-29 results were already inspected in v0.3 | It is frozen and comparable, but no longer truly blind | Never train on seed 29; label it “frozen holdout”, not “blind test” |
| Simulator exposes one scalar normal force | Cannot learn from full wrist wrench | Restrict claims and observation vector to available signals |
| Wrench bounds do not prove hardware safety | Simulated torque clamp is not a certified safety layer | Keep zero-residual fallback and explicitly defer real-robot deployment |

## 5. Implementation checklist

| Item | File(s) |
|---|---|
| Adaptive estimator and gain-scheduled controller | `src/compliant_control_lab/franka_adaptive.py` |
| Bounded policy, observation encoder and safety wrapper | `src/compliant_control_lab/residual_rl.py` |
| Deterministic training scenarios, ARS trainer, checkpoint I/O | `src/compliant_control_lab/franka_learning.py` |
| Shared fixed/adaptive/residual 24-case comparison | `src/compliant_control_lab/franka_learning.py` |
| CLI entry point | `pyproject.toml` |
| Adaptive, safety, zero-residual and training-split tests | `tests/test_franka_adaptive.py`, `tests/test_residual_rl.py`, `tests/test_franka_learning.py` |
| Frozen policy and training manifest | `results/franka_learning/` |
| Comparison CSV/summary and current-state visualizations | `results/franka_learning/` |
| Method, limitations and reproduction commands | `README.md`, `docs/tutorial/05_residual_rl.md` |
| CI smoke coverage | `.github/workflows/tests.yml` and pytest suite |

## 6. Validation plan

- Plot training cost by ARS iteration and fixed/adaptive/residual metrics across all 24 current cases.
- Record every controller × case metric, gate failure, seeds, learned weights and safety configuration.
- Verify zero-residual parity, per-step bounds/rate limits, non-finite fallback and deterministic
  checkpoint round-trip in unit tests.
- Run the full seed-29 comparison once the policy/configuration is frozen.
- Success means adaptive and residual results are reported honestly under the unchanged gates;
  neither method is declared superior merely because its training return improves.

## 7. Completion audit

- [x] Adaptive bias estimation, stiffness estimation and gain scheduling implemented.
- [x] Bounded residual policy, 50 Hz scheduling, stable-contact delay and watchdogs implemented.
- [x] Training and frozen holdout scenario sets separated and recorded in the manifest.
- [x] One frozen policy trained from zero and evaluated against both baselines on the same 24 cases.
- [x] Zero-residual equivalence, bounds, fallback, split and determinism covered by tests.
- [x] Per-case CSV, policy checkpoint, manifest, plots and an honest gate/failure analysis committed.
- [ ] Multi-seed confidence intervals and nonlinear SAC/TD3 comparison; deliberately deferred until
  the current peak-force and torque-headroom safety failures are addressed.
- [ ] Real-robot deployment; deliberately blocked by the unchanged physical acceptance gate.
