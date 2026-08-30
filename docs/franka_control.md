# Franka 7-DOF control notes

## 1. Task state and Jacobian

The end-effector task state is

$$
\mathbf{x}=\begin{bmatrix}\mathbf{p}\\\boldsymbol{\phi}\end{bmatrix},\qquad
\dot{\mathbf{x}}=J(\mathbf{q})\dot{\mathbf{q}},
$$

where $\mathbf{p}\in\mathbb{R}^3$, $\boldsymbol{\phi}$ is orientation, and
$J\in\mathbb{R}^{6\times7}$ is MuJoCo's geometric site Jacobian. The world-frame orientation
error used by the controller is

$$
\mathbf{e}_R=\frac{1}{2}\sum_{i=1}^{3}
\left(\mathbf{R}_{:,i}\times\mathbf{R}_{d,:,i}\right).
$$

This representation is smooth around the small orientation errors used in the wall-wiping task.

## 2. Torque mapping and bias compensation

Each controller returns a 6D Cartesian wrench $\mathbf{w}=[\mathbf{f},\boldsymbol{\mu}]^T$.
The commanded arm torque is

$$
\boldsymbol{\tau}=
J^T\mathbf{w}+\mathbf{h}(\mathbf{q},\dot{\mathbf{q}})
+N\boldsymbol{\tau}_0,
$$

where $\mathbf{h}$ is MuJoCo's gravity/Coriolis bias and $\boldsymbol{\tau}_0$ is a joint-space
posture controller. With damping $\lambda$, the torque-space null-space projector is

$$
N=I-J^T\left(JJ^T+\lambda^2I\right)^{-1}J.
$$

The final torque is clipped to the Panda limits: 87 Nm for joints 1–4 and 12 Nm for joints 5–7.

v0.5 exposes the numeric terms needed to check that limit before the controller returns its wrench:

$$
\tau(w)=J^Tw+\tau_{offset},\qquad
\tau_{offset}=\mathbf h+N\tau_0.
$$

`FrankaActuationContext` contains copies of `J`, `tau_offset` and the lower/upper limits. It
contains no MuJoCo object, so the controller seam stays simulator-independent. The torque safety
module reserves 10% of each actuator interval and scales a requested wrench along its original ray.
It never changes bias compensation or null-space posture torque.

## 3. 6D Cartesian impedance

The impedance wrench is

$$
\mathbf{f}=K_p(\mathbf{p}_d-\mathbf{p})+D_p(\dot{\mathbf{p}}_d-\dot{\mathbf{p}}),
$$

$$
\boldsymbol{\mu}=K_R\mathbf{e}_R+D_R(\boldsymbol{\omega}_d-\boldsymbol{\omega}).
$$

Normal stiffness converts virtual wall penetration into force. It limits impact but does not
independently guarantee the requested 12 N contact force.

## 4. Normal admittance

Let $\mathbf{n}=[1,0,0]^T$ point from the robot into the wall. The force error drives a scalar
reference along the normal:

$$
M_a\ddot{x}_r+D_a\dot{x}_r+K_a(x_r-x_d)=F_d-F_m.
$$

The tangential reference is preserved with
$P_t=I-\mathbf{n}\mathbf{n}^T$. A 6D impedance inner loop tracks the combined normal/tangential
reference. Reference displacement is clamped to avoid unbounded motion after contact loss.

## 5. Hybrid force-position control

The translational command explicitly separates the force and motion subspaces:

$$
\mathbf{f}=\mathbf{n}
\left(F_d+K_f e_f+K_i\int e_fdt-D_f\mathbf{n}^T\dot{\mathbf{p}}\right)
+P_t\left(K_t\mathbf{e}_p+D_t\mathbf{e}_v\right).
$$

The force integral is updated only in confirmed contact and is clamped. This prevents wind-up during
approach or temporary contact loss. Orientation remains under impedance control.

The implementation also separates the approach and force-regulation phases. Before contact, a
bounded normal position controller approaches the surface:

$$
F_a=\mathrm{clip}\left(K_a\,\mathbf{n}^T(\mathbf{p}_d-\mathbf{p})+
D_a\,\mathbf{n}^T(\dot{\mathbf{p}}_d-\dot{\mathbf{p}}),0,F_{a,max}\right).
$$

Contact must remain above 3 N for 20 ms; then a blend factor transitions from the approach command
to the force PI command over 150 ms. A separate release threshold and 50 ms debounce prevent noisy
mode chatter. This reduced nominal full-trial peak force from 33.83 N to 26.72 N while preserving
100% contact and approximately 1 N steady-state force RMSE.

## 6. Adaptive and learned residual layers

v0.4 keeps the same Cartesian controller seam and adds two wrappers. `adaptive_hybrid` estimates
pre-contact force bias, local force rate and contact stiffness, then schedules the force/tangential
gains. `bounded_residual_rl` adds a learned three-axis translational wrench only after the classical
contact transition is stable, with component/rate limits and zero-residual fallback. The estimator,
ARS update equations and frozen comparison are derived in
[Adaptive compliance and bounded Residual RL](adaptive_residual_rl.md).

v0.5 adds `safe_adaptive_hybrid`, which limits pre-contact normal reference lead and projects the
complete adaptive wrench into the reserved torque interval. The learned controller then projects its
three-axis residual against the remaining interval. The 20 policy inputs are the v0.4 14 fields plus
the feasible scales for positive and negative x/y/z residuals. Derivation and the freeze protocol
are in [v0.5 pre-holdout protocol](reproduction_plan_v0.5.md), together with the failed 48-case first
reveal and its raw-result path.

## 7. Measurement and evaluation

MuJoCo provides raw contact impulses at 500 Hz. A 20 ms first-order low-pass filter approximates
force-sensor bandwidth. Scenario noise, signed bias and delay are applied after filtering; retaining
the sign is necessary for the pre-contact bias estimator to observe negative offsets.

- `force_rmse_n`: filtered force versus the 12 N target after the 1.5 s approach window.
- `peak_force_n`: unfiltered raw solver force over the complete trial, including first contact, so
  neither filtering nor the steady-state evaluation window can hide impact.
- `tangent_rmse_mm`: Euclidean y-z tracking error.
- `orientation_rmse_deg`: norm of the small-angle orientation error.
- `saturation_pct`: fraction of all trial steps where any arm torque was clipped.
- `controller_p95_us`: controller computation only; MuJoCo stepping and rendering are excluded.

The final gains deliberately trade a few millimetres of tangential error for zero torque
saturation in the 20 ms delay scenario. This change reduced the initial delayed-controller
saturation from over 40% to 0%.

## 8. Model provenance

The Panda geometry and inertial parameters come from Google DeepMind's MuJoCo Menagerie at commit
`da76818e269b82289eba39808e2fb91d679d6994` under Apache-2.0. The local
`panda_torque.xml` derivative replaces position actuators with torque motors and adds a rigid
spherical tool and end-effector site. See the vendored `UPSTREAM.md` and `LICENSE`.
