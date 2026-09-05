# C++17 / Eigen control core

## Purpose

The C++ module holds the simulator-independent control math. A caller supplies the public input
structures listed below. The module computes a Cartesian wrench and can project an additive wrench
into the remaining joint-torque envelope.

```text
MuJoCo or robot hardware
        |
        v
CartesianState + CartesianTarget
        |
        v
compliant_control_core  --->  6D Cartesian wrench
        |
        v
reserved-envelope projection
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

Configuration is validated in controller constructors. The 500 Hz update methods are `noexcept`
and use fixed-size Eigen matrices. They avoid blocking work and explicit allocation.
`FrankaActuationContext` requires all four model fields at construction; no default robot model
is supplied. The probe constructs it only after reading a complete input case.
Projection errors return a typed status and zero additive wrench. Reversed limits raise
`std::invalid_argument` before the result can reach an actuator.

The projection core uses limits supplied by its caller. A hardware adapter still owns data
validation and time-dependent safety handling such as torque-rate limits or watchdogs.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
pytest tests/test_cpp_parity.py
./build/compliant_control_torque_benchmark
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

## ROS 2 integration boundary

A real Franka torque-controller plugin additionally needs the model/state signals from a hardware
interface. This repository has no ROS 2/Franka adapter or hardware validation. A future adapter
should be thin:

1. claim seven effort command interfaces and the Franka model/state interfaces;
2. convert hardware values to `CartesianState`;
3. call `WrenchController::compute()`;
4. map the wrench with `J^T`, add bias/null-space torque, then enforce torque/rate limits;
5. zero the residual and enter a safe hold on stale input, non-finite values, or deadline misses.

Keeping those hardware details outside the core makes the controller equations testable without
pretending that a simulator-only interface is ready for a physical robot.
