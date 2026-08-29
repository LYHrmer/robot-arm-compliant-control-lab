# 2-DOF baseline control notes

## 1. Plant and task coordinates

The arm has two revolute joints and link lengths \(l_1=0.45\,\mathrm{m}\) and
\(l_2=0.35\,\mathrm{m}\). Its end-effector position is

\[
\mathbf{x}(\mathbf{q}) =
\begin{bmatrix}
l_1\cos q_1 + l_2\cos(q_1+q_2) \\
l_1\sin q_1 + l_2\sin(q_1+q_2)
\end{bmatrix}.
\]

Differentiation gives the analytic Jacobian \(\dot{\mathbf{x}}=J(\mathbf{q})\dot{\mathbf{q}}\).
Each Cartesian controller returns a wrench \(\mathbf{w}=[F_x,F_y]^T\), which is mapped to
joint torque with

\[
\boldsymbol{\tau}=J(\mathbf{q})^T\mathbf{w}-D_q\dot{\mathbf{q}}.
\]

The wall normal is the x axis. Positive \(F_x\) pushes the fingertip into the wall; y is the
tangential motion axis.

## 2. Position baseline

The baseline is deliberately stiff:

\[
\mathbf{w}=K_p(\mathbf{x}_d-\mathbf{x})+K_d(\dot{\mathbf{x}}_d-\dot{\mathbf{x}}).
\]

It tracks free-space motion well but converts small geometric penetration or environment-model
error into large contact force.

## 3. Cartesian impedance

The impedance controller uses the same algebraic form with a much lower normal stiffness. It
therefore imposes a desired spring-damper relationship between displacement and contact wrench.
It limits impact but does **not** guarantee convergence to an independently specified force.

## 4. Admittance control

The normal force error drives a virtual mechanical system:

\[
M_a\ddot{x}_r + D_a\dot{x}_r + K_a(x_r-x_d)=F_d-F_m.
\]

The integrated \(x_r\) becomes the x reference of a Cartesian position inner loop. If measured
force is too low, the reference moves toward the wall. If it is too high, the reference retreats.
This structure is useful when the physical robot exposes a reliable position/velocity loop rather
than direct joint torque control.

## 5. Hybrid force-position control

Let the force selection matrix be \(S_f=\operatorname{diag}(1,0)\) and the complementary position
selection matrix be \(S_p=I-S_f\). The command is

\[
\mathbf{w}=
S_f\left(\mathbf{F}_d+K_f\mathbf{e}_f+K_i\int\mathbf{e}_fdt-D_f\dot{\mathbf{x}}\right)
+S_p\left(K_p\mathbf{e}_x+K_d\mathbf{e}_{\dot{x}}\right).
\]

In code the two diagonal selections reduce to an x force command and a y position command. The
force integral is contact-aware and clamped. It is frozen below 0.5 N to prevent wind-up during
free-space approach or temporary contact loss.

## 6. Measurement model

MuJoCo produces an impulsive contact wrench at every 2 ms simulation step. A first-order low-pass
filter with 20 ms time constant approximates force-sensor bandwidth:

\[
F_k^{f}=F_{k-1}^{f}+\alpha(F_k-F_{k-1}^{f}),\qquad
\alpha=\frac{\Delta t}{\tau_f+\Delta t}.
\]

Noise and delay are applied after this physical sensor filter. Force RMSE is computed from the
filtered signal, while peak force is computed from the unfiltered solver output.

## 7. Scope

- The robot is planar and uses ideal torque actuation.
- Link flexibility, motor current loops, encoder quantization and communication jitter are omitted.
- Contact parameters are phenomenological MuJoCo parameters rather than identified hardware values.
- The 7-DOF Franka extension adds orientation error, 6D wrench mapping, gravity compensation,
  null-space posture control and torque-limit monitoring. See `docs/franka_control.md`.

This planar model remains as a small analytic reference implementation and regression test.
