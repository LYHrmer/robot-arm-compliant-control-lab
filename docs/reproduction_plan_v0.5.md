# v0.5 pre-holdout protocol

v0.4 improved tangential tracking but made the worst torque saturation larger. Its seed-29
evaluation had also been read several times. v0.5 treats seed 29 as public validation data and
creates a separate first-reveal experiment.

## Controller change

`FrankaState` now carries an optional numeric actuation context: the 6x7 Jacobian, bias and
null-space torque, and the seven actuator limits. The controller interface remains
`compute(state, target, dt)`. No MuJoCo object crosses that seam.

The safe adaptive baseline limits the normal reference lead to 10 mm before contact. That value was
chosen on the public seed-29 set after checking 3, 4.5, 6, 8, 10, 12 and 16 mm. Leads below 10 mm
lost contact on tilted-wall cases. This tuning history is part of the record; seed 29 is not used in
the v0.5 headline result.

The safe adaptive controller scales its complete task wrench along the requested ray when needed to
keep the same 10% joint-torque reserve. Bias compensation and null-space posture torque are left
unchanged. This conservative step removes actuator clipping in the public validation run, although
it can also increase tracking error.

For a candidate residual force `r`, the actuation context gives

$$
\tau_b=J^T w_{nom}+\tau_{offset}, \qquad \Delta\tau=J_v^T r.
$$

The actuator interval is reduced by a 10% reserve. The safety module finds the largest
`alpha` in `[0, 1]` for which

$$
\tau_{min,safe}\leq\tau_b+\alpha\Delta\tau\leq\tau_{max,safe}.
$$

It applies `alpha r`. A missing context, non-finite value or nominal torque outside the reserved
interval clears the residual for that cycle. The policy cannot use its residual to repair an unsafe
nominal command.

Six observation fields record the available scale for `+x, -x, +y, -y, +z, -z` actions. Together
with the v0.4 fields, the policy input has 20 values. A version-2 checkpoint stores the full schema;
the loader still accepts the published 14-field v0.4 checkpoint.

## Training freeze

Five ARS runs use policy seeds `17, 23, 31, 43, 59`. They share eight training scenarios and eight
development-validation scenarios. Simulation-noise streams differ by run and change at each ARS
iteration. The development set selects the checkpoint. Neither training nor selection reads the
blind beacon.

`franka-safety-learning-lab prepare` writes the five policies, their SHA256 values, training curves
and `protocol.json`. The protocol also records the implementation commit, controller settings,
MuJoCo model hash, hard gate, verifier/lockfile hashes and a future drand Quicknet round. Before and
after training, the frozen Node.js verifier reads `latest` from both official relays and then fetches
those advertised rounds again with BLS verification. The relays may differ by one three-second round;
a larger gap fails closed. The target must be at least 201 rounds ahead of the freshest final
reference, which guarantees ten unpublished minutes even if the next round is about to appear. This
check does not use the host clock. Generated files are committed in one child commit of the
implementation commit and tagged `v0.5-preholdout` before the target round is published.

Evaluation accepts only the protocol, checksum, policies and curves whose bytes are present in that
tag. It also recomputes the model and safety-manifest hashes. A protocol copied to `/tmp`, a policy
edited after the tag, a one-policy contract or a changed `HEAD` is rejected before any blind case is
derived.

## First reveal

After the frozen drand round appears, `tools/verify_drand_beacon.mjs` asks both
`api.drand.sh` and `drand.cloudflare.com` for that exact round. The verifier uses the pinned
`drand-client` 1.4.2 package and Quicknet chain hash/public key to check the BLS signature. Python
requires both verified responses to contain identical round, signature and randomness, and also
checks `randomness = SHA256(signature)`. There is no unverified fallback and the evaluator never
uses `latest` to choose a blind seed. The only `latest` requests are the pre-reveal freshness
evidence stored in the frozen protocol.

The blind root is

$$
SHA256(protocol\_sha256 \; || \; beacon\_randomness).
$$

HMAC-SHA256 namespaces derive 48 scenario seeds, 48 simulation-noise seeds and a reporting seed.
One process evaluates the fixed hybrid controller, adaptive controller, safe adaptive controller
and all five frozen policies. A case always uses the same physical parameters and noise seed across
methods. Intermediate output reports progress only.

The gate is unchanged: force RMSE at most 2 N, contact ratio at least 95%, raw peak force at most
35 N, tangential RMSE at most 15 mm and torque saturation at most 1%. A policy needs 44/48 passing
cases. The primary result passes only if all five policies reach that count. The report keeps the
five raw results and a case-and-training-seed bootstrap interval; it does not select the best seed.

## Recorded first reveal

The frozen implementation is `f186a19`; the artifact commit and dereferenced
`v0.5-preholdout` tag are `fa50f4d`. The protocol SHA256 is
`781839a62da0725bbe1ba8e321812dea2271ac7c8e646cdc9cf76732e4f02395`. GitHub created the
freeze CI run at `2026-08-30T06:12:33Z`; the run completed successfully before Quicknet round
`31756275` at `2026-08-30T06:43:09Z`.

| Method | Pass | Force P95 [N] | Raw peak P95 [N] | Tangent P95 [mm] | Saturation worst |
|---|---:|---:|---:|---:|---:|
| fixed_hybrid | 17/48 | 2.01 | 58.62 | 22.66 | 19.69% |
| adaptive_hybrid | 23/48 | 2.21 | 58.99 | 18.89 | 20.71% |
| safe_adaptive_hybrid | 24/48 | 2.33 | 59.54 | 18.89 | 0.00% |
| torque_residual_run_00 | 22/48 | 2.00 | 59.54 | 16.04 | 0.00% |
| torque_residual_run_01 | 25/48 | 2.03 | 59.54 | 16.13 | 0.00% |
| torque_residual_run_02 | 26/48 | 2.10 | 59.54 | 15.98 | 0.00% |
| torque_residual_run_03 | 24/48 | 2.12 | 59.54 | 16.50 | 0.00% |
| torque_residual_run_04 | 25/48 | 1.98 | 59.54 | 16.43 | 0.00% |

An independent CSV audit found 384 rows, eight methods with 48 rows each, one shared scenario/noise
seed pair per case, matching policy hashes and matching recomputed gate labels. All deadline,
torque-context and projection fallback counts were zero. The five residual policies passed 22, 25,
26, 24 and 25 cases, so the frozen primary result is `FAIL`. The raw files are in
`results/franka_safety_blind`; their hashes are listed in `manifest.json`.

Once `reveal.json` exists, the 48 cases become validation data. Any later tuning must use another
precommitted beacon round. A failed run may retry the same round; changing the round requires a new
protocol and freeze tag.
