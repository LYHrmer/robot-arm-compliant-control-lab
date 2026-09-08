# C++17 / Eigen control core

## Purpose

The C++ module holds the simulator-independent control math. A caller supplies the public input
structures listed below. The module computes a Cartesian wrench and can project an additive wrench
into the remaining joint-torque envelope.

The surface implementation now also carries the state of the selected Python controller:
bias estimation, force-rate filtering, gain scheduling, contact transition, approach governor,
optional tangential compensation, and full-wrench torque projection. `online` is the bounded
error-driven compensation described in [online_compensation.md](online_compensation.md).
It is not the learned Residual RL policy.

```text
MuJoCo or robot hardware
        |
        v
CartesianState + CartesianTarget
        |
        v
surface frame -> adaptive hybrid -> tangential compensation
        |
        v
reserved-envelope projection -> world-frame 6D wrench
        |
        v
J(q)^T w + torque offset
```

The public interface contains fixed-size Eigen values only:

| Interface | Responsibility |
|---|---|
| `CartesianState` / `CartesianTarget` | Controller inputs |
| `WrenchController` | `reset()` and one `compute()` update |
| Three controller classes | Cartesian impedance, normal admittance and hybrid force-position control |
| `orientation_error()` / null-space projector | Rotation error and damped `6x7` kinematics |
| `FrankaActuationContext` | Current Jacobian, torque offset and asymmetric joint limits |
| Torque-safety functions | Full-wrench projection, residual-force projection and directional headroom |
| `SurfaceAdaptiveController` | Stateful surface-coordinate controller, timestamp checks and output status |
| `TangentialMode` | No compensation, integral, fixed friction, or online equivalent load |
| `SurfaceControlResult` | Wrench, parameter/contact telemetry, projection status and feasibility flag |

Configuration is validated in controller constructors. The original fixed-controller updates
are `noexcept`; the surface entry point additionally checks input validity. Numeric calculations
use fixed-size Eigen matrices and do not perform file I/O or blocking work.
`FrankaActuationContext` requires all four model fields at construction; no default robot model
is supplied. The probe constructs it only after reading a complete input case.
Projection errors return a typed status and zero additive wrench. Reversed limits raise
`std::invalid_argument` before the result can reach an actuator.

The projection core uses limits supplied by its caller. A timestamp watchdog is included for
mock input tests. A hardware adapter still owns the actual clock, packet freshness, robot model
and torque-rate limits. An offline trace with timestamps is not a hardware communication test.

## Stateful update order

Each accepted surface sample follows the Python sequence:

1. Rotate kinematics, targets and both Jacobian blocks into the fixed surface frame.
2. Govern the approach target using the previous cycle's force estimates.
3. Update bias/force-rate/stiffness estimates, schedule gains, and advance hybrid contact state.
4. Add tangential compensation using its current state, then project the complete wrench.
5. Commit the next integral/coefficient state only after an unchanged projection.
6. Rotate the wrench back to world coordinates and return telemetry.

The caller supplies a `FrankaActuationContext` containing the current Jacobian, torque offset
and joint limits. With no context, the numerical wrench can still be inspected, but torque
feasibility is not verified and adaptive compensation does not advance.

`fallback` and `feasible` have different meanings. A rejected input yields zero wrench; this
does not repair an offset torque that already exceeds the allowed envelope. Callers must check
the status fields rather than interpreting a zero wrench as a safe physical command.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
pytest tests/test_cpp_parity.py tests/test_cpp_surface_loop.py
./build/compliant_control_torque_benchmark
./build/compliant_control_surface_benchmark
```

`compliant_control_core_tests` checks controller behavior and configuration failure modes. The
`compliant_control_probe` executable evaluates all controllers at deterministic states, including
hybrid approach and contact-confirm/transition sequences. The Python parity test compares every
output component against `franka_control.py` with a `1e-12` tolerance.

The same probe checks torque projection and six-direction headroom on 160 deterministic randomized
7-DOF cases. Handwritten boundary cases are kept in
[`test_torque_safety.cpp`](../cpp/tests/test_torque_safety.cpp).

The benchmark runs each operation 100,000 times and reports latency percentiles plus the 2 ms
overrun ratio. One Release-mode observation is below; CI does not enforce these numbers.

| Host / compiler | Full wrench p99 | Residual p99 | Six-direction headroom p99 | Samples over 2 ms |
|---|---:|---:|---:|---:|
| i7-14650HX / GCC 11.4 | 0.188 us | 0.117 us | 0.495 us | 0 / 300,000 |

This measurement does not establish hard real-time behavior.

The separate `compliant_control_surface_benchmark` measures 20,000 complete numeric surface
updates, including state changes, coordinate conversion and projection. It reports p50/p99/max
and the count above 2 ms. It excludes sensor acquisition, model/Jacobian computation, network
communication and OS scheduling at the intended control frequency. Its synthetic input stream
does not integrate robot dynamics.

The new [surface parity tests](../tests/test_cpp_surface_loop.py) compare stateful sequences in
all four compensation modes. They also exercise invalid timestamps and infeasible torque
offsets. A separate [recorded-input verifier](../tools/verify_cpp_surface_replay.py) accepts full
simulation NPZ traces and checks both Python and C++ wrench/torque outputs. Recorded-input
replay is stronger than a single-state example, but still does not run a C++ hardware loop.

## Recorded surface-loop result

The [four-trace report](../results/franka_online_cpp_replay/report.json) uses the predeclared
friction step-down and stop/hold/reverse trials (seed 11, fixed friction and online modes).
Each trace has 6,000 samples: 24,000 complete controller updates in total.
All four replays match the saved Python commands. Maximum component errors are
`4.45e-15` in wrench and `3.56e-15` in joint torque, below the `1e-8` acceptance tolerance.
The two online traces contain 3,055 and 2,163 actual coefficient updates; their coefficient,
readiness, contact-blend and projection-scale telemetry match exactly.

The same report records a 20,000-update numeric benchmark: p50 `0.338 us`, p99 `0.358 us`,
maximum `3.498 us`, and no samples above 2 ms on this host. These are computation timings,
not a measured 500 Hz hardware loop. Replay supplies fresh synthetic timestamps; watchdog
fault handling is tested separately and is not evidence about real sensor transport.

After building the targets, reproduce the check into a new directory:

```bash
python -m tools.verify_cpp_surface_replay \
  --trace \
  results/franka_online_compensation_errors/traces/friction_step_down__seed_11__friction__full.npz \
  results/franka_online_compensation_errors/traces/friction_step_down__seed_11__online__full.npz \
  results/franka_online_compensation_errors/traces/stop_hold_reverse__seed_11__friction__full.npz \
  results/franka_online_compensation_errors/traces/stop_hold_reverse__seed_11__online__full.npz \
  --probe build/compliant_control_surface_probe --output /tmp/online-cpp-replay
```

The output contains per-step errors, source/input/binary hashes and runtime identity.
Numerical agreement verifies the port, not the suitability of the underlying controller:
the [reverse-ramp orientation failures](online_compensation.md#实测结果与没有通过的部分)
remain failures in both languages.

## ROS 2 integration boundary

A real Franka torque-controller plugin additionally needs the model/state signals from a hardware
interface. This repository has no ROS 2/Franka adapter or hardware validation. A future adapter
should be thin:

1. claim seven effort command interfaces and the Franka model/state interfaces;
2. convert hardware values to `CartesianState`;
3. call the selected controller; the surface entry point also needs trustworthy sample and clock timestamps;
4. map the wrench with `J^T`, add bias/null-space torque, then enforce torque/rate limits;
5. define and validate a robot-specific stop/hold response on stale input, non-finite values,
   infeasible offsets or deadline misses. Zero wrench alone is not such a response.

Keeping those hardware details outside the core makes the controller equations testable without
pretending that a simulator-only interface is ready for a physical robot.
