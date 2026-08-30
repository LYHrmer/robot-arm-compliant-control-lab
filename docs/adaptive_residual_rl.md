# Adaptive compliance and bounded Residual RL

This document explains the v0.4 implementation from estimator equations to the frozen 24-case
comparison. The implementation is a mechanism-level study, not an exact reproduction of the robot,
network, task, or optimizer in any one paper.

## 1. Control stack and seam

All three evaluated methods implement the same simulator-independent interface:

```python
controller.reset(initial_state)
wrench_6d = controller.compute(state, target, dt)
```

The MuJoCo adapter remains unchanged after that seam:

$$
\tau = J(q)^T w + \alpha h(q,\dot q) + N(q)\tau_{posture},
$$

followed by the same actuator torque clamp. Only the Cartesian controller changes; robot model,
trajectory, measurement delay, random seed and safety metrics are shared.

```text
measured Cartesian state
        |
        v
impact-aware hybrid -------- fixed_hybrid
        |
        +-- bias/stiffness/rate adaptation -------- adaptive_hybrid
                                                    |
normalized observation --> linear tanh policy --> bounded safety filter
                                                    |
                                                    +-- bounded_residual_rl
```

## 2. Adaptive classical baseline

Implementation: `src/compliant_control_lab/franka_adaptive.py`.

### 2.1 Pre-contact force-bias estimation

The simulated force sensor now preserves signed noise and bias. Clipping a measurement to zero
before estimation would make a negative bias unobservable. While the desired force is below 2 N and
contact has not been confirmed, the controller updates

$$
\hat b_{k+1}=\hat b_k+
\frac{\Delta t}{T_b+\Delta t}(F_{m,k}-\hat b_k),
\qquad T_b=80\text{ ms},
$$

then uses

$$
F_k=\max(0,F_{m,k}-\hat b_k).
$$

The estimate is limited to +/-3 N. This is an engineering estimator, not a proof that all sensor
drift is identifiable.

### 2.2 Force-rate and contact-stiffness estimates

The filtered force rate is

$$
\dot{\hat F}_{k+1}=\dot{\hat F}_k+
\frac{\Delta t}{T_{\dot F}+\Delta t}
\left(\frac{F_k-F_{k-1}}{\Delta t}-\dot{\hat F}_k\right),
$$

with derivative samples clipped before filtering. During confirmed loading, a local secant estimate

$$
K_{e,k}=\left|\frac{F_k-F_{k-1}}{x_{n,k}-x_{n,k-1}}\right|
$$

is clipped to `[500, 30000] N/m` and low-pass filtered. Samples with tiny displacement or opposing
force/displacement signs are rejected.

### 2.3 Gain schedule

The force PI scale combines stiffness and force-rate terms:

$$
s_K=\sqrt{\frac{K_{ref}}{\max(\hat K_e,1)}},
\qquad
s_{\dot F}=\frac{1}{1+|\dot{\hat F}|/250},
$$

$$
s=\operatorname{clip}(s_Ks_{\dot F},0.65,1.10).
$$

The normal proportional and integral gains are multiplied by `s`; normal damping is divided by
`sqrt(s)`. Tangential stiffness rises by at most 25% as tangential error approaches 20 mm, with
damping scaled by the square root of the same factor. The position-to-force transition is extended
to 500 ms for this adaptive baseline.

An independent impact guard reduces the normal command when corrected force exceeds the target by
6 N or when positive force rate exceeds 150 N/s. The guarded command remains in `[-2, 25] N`.

## 3. Bounded Residual RL

Implementation: `src/compliant_control_lab/residual_rl.py`.

The method follows the additive mechanism from
[Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201):

$$
w_{cmd}=w_{adaptive}+[\Delta F_x,\Delta F_y,\Delta F_z,0,0,0]^T.
$$

It also follows the structural motivation of
[Online Admittance Residual Learning](https://proceedings.mlr.press/v229/zhang23e.html): retain an
explicit compliance controller and adapt only the part affected by contact mismatch. It does not
reproduce that paper's online SQP formulation.

### 3.1 Observation

The policy uses 14 normalized values, all available at the controller seam:

| Feature | Scale |
|---|---:|
| Force error | 12 N |
| Corrected normal force | 20 N |
| Filtered force rate | 250 N/s |
| Normal position error | 20 mm |
| Normal velocity error | 0.10 m/s |
| y/z position error | 50 mm |
| y/z velocity error | 0.10 m/s |
| Target force | 12 N |
| Classical contact blend | `[0, 1]` |
| Previous three normalized actions | `[−1, 1]` |

Each feature is finally clipped to `[-3, 3]`. No wall friction, wall angle, true contact force or
other simulator-only parameter enters the policy.

### 3.2 Policy and action bounds

The first working policy is deliberately inspectable:

$$
a_k=\tanh(Wo_k+b), \qquad W\in\mathbb R^{3\times14}.
$$

The normalized action is hard-clipped and mapped to

$$
|\Delta F|\le[4,6,6]\text{ N}.
$$

This small linear policy is a baseline for the residual mechanism, not evidence that a linear policy
is sufficient for general contact-rich manipulation.

### 3.3 Safety filter order

The policy runs at 50 Hz above the 500 Hz classical loop. Each action passes through one safety
module in this order:

1. non-finite/exception/deadline watchdog; failure immediately sets residual to zero;
2. contact gate: residual is exactly zero until classical force blend has remained complete for
   100 ms; loss of contact immediately clears it;
3. `tanh` plus hard component clipping;
4. first-order filtering with 40 ms time constant;
5. rate limits `[40, 60, 60] N/s`;
6. force guard: positive normal residual is removed above target +3 N or +150 N/s force rate;
7. total normal wrench clamp to `[-4, 25] N`;
8. the existing Jacobian mapping and joint-torque limits.

A zero policy is tested to match the adaptive nominal wrench to `1e-12` over a state sequence.

## 4. ARS training

Implementation: `src/compliant_control_lab/franka_learning.py`.

The parameters are optimized with Augmented Random Search, following the static linear-policy idea
in [Simple random search provides a competitive approach to reinforcement learning](https://arxiv.org/abs/1803.07055).
For sampled directions `delta_i`, each iteration evaluates

$$
R_i^+=-C(\theta+\nu\delta_i), \qquad
R_i^-=-C(\theta-\nu\delta_i),
$$

keeps the top directions and updates

$$
\theta\leftarrow\theta+
\frac{\eta}{N_{top}\sigma_R}
\sum_i(R_i^+-R_i^-)\delta_i.
$$

The dimensionless physical rollout cost is

$$
C=
\left(\frac{e_F}{2}\right)^2+
0.75\left(\frac{F_{peak}}{35}\right)^2+
\left(\frac{e_t}{15}\right)^2+
2\left(\frac{s_{sat}}{1}\right)^2+
4\left(\frac{[95-r_c]_+}{5}\right)^2+
0.05\left(\frac{\Delta F_{rms}}{5}\right)^2.
$$

Final claims still use the original hard gates, not this weighted training cost.

The frozen run uses six training scenarios from randomization seed 101, simulation seeds 1001–1006,
policy seed 17, eight iterations, six directions and three top directions. The seed-29 24-case set
never enters training. Because its v0.3 results were already inspected, it is correctly labelled a
frozen public holdout, not a blind test.

## 5. Same-case result

Frozen artifacts: `results/franka_learning/`.

| Method | Case pass | Force P95 [N] | Peak P95 [N] | Tangent P95 [mm] | Saturation worst |
|---|---:|---:|---:|---:|---:|
| Fixed hybrid | 6/24 | 2.32 | 57.79 | 21.98 | 12.53% |
| Adaptive hybrid | 6/24 | 3.01 | 56.80 | 18.42 | 12.89% |
| Bounded Residual RL | 7/24 | 2.30 | 57.04 | 14.80 | 15.91% |

The residual lowers tangential P95 from 18.42 mm to 14.80 mm and adds one passing case. It does not
clear the full gate: raw impact force fails in 17 cases, force RMSE fails in three, and worst torque
saturation rises from 12.89% to 15.91%. The resulting 7/24 pass count is below the required 22/24.
This checkpoint remains a simulation ablation and is marked unsafe for hardware use.

v0.5 added a pre-contact reference governor, torque-headroom observations and wrench-to-joint-torque
projection. Its first reveal removed actuator saturation for the safe adaptive baseline and all five
residual policies, but the policies passed only 22–26/48 cases against a 44/48 requirement. Raw
impact P95 remained 59.54 N. The next experiment targets the nominal contact transient before testing
a larger policy class. See [`reproduction_plan_v0.5.md`](reproduction_plan_v0.5.md).

## 6. Reproduction commands

Train from zero and run all 72 holdout trials:

```bash
franka-learning-lab --output results/franka_learning
```

Re-evaluate the frozen checkpoint without retraining:

```bash
franka-learning-lab \
  --policy results/franka_learning/policy.json \
  --output results/franka_learning_eval
```

Fast smoke run:

```bash
franka-learning-lab \
  --iterations 1 --directions 2 --top-directions 1 \
  --training-cases 2 --training-duration 0.2 \
  --holdout-cases 2 --holdout-duration 0.2 \
  --output /tmp/franka-learning-smoke
```

The checkpoint contains the weights, observation schema, training config and safety envelope. The
manifest separately records every randomized training case, evaluation seed and hard gate.
