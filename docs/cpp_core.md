# C++17 / Eigen controller core

## Purpose

The C++ module is the simulator-independent control seam. MuJoCo, Python experiments, and a future
Franka ROS 2 hardware adapter provide state and target data; the module owns the compliant-control
state machines and returns a Cartesian wrench.

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
J(q)^T w + bias + null-space torque
```

The public interface contains fixed-size Eigen values only:

- `CartesianState`: pose, twist, and measured normal force.
- `CartesianTarget`: desired pose, twist, and normal force.
- `WrenchController`: `reset()` plus one `compute()` update.
- Three implementations: Cartesian impedance, normal admittance, and hybrid force-position control.
- `orientation_error()` and the damped `6x7` Jacobian null-space projector.

Configuration is validated in controller constructors. The 500 Hz update methods are `noexcept`,
use fixed-size Eigen matrices, and contain no logging, locks, dynamic containers, or explicit heap
allocation. Hardware torque limits and watchdog behavior deliberately remain the responsibility of
the robot adapter because they depend on the hardware interface.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
pytest tests/test_cpp_parity.py
```

`compliant_control_core_tests` checks controller behavior and configuration failure modes. The
`compliant_control_probe` executable evaluates all controllers at deterministic states, including
hybrid approach and contact-confirm/transition sequences. The Python parity test compares every
output component against `franka_control.py` with a `1e-12` tolerance.
This prevents the deployable implementation and the research reference from silently diverging.

## ROS 2 integration boundary

A real Franka torque-controller plugin additionally needs a hardware model interface that supplies
the measured end-effector state, `6x7` Jacobian, dynamics bias, and a calibrated wrench signal. The
current machine has ROS 2 Humble and `ros2_control`, but not
`franka_semantic_components`/`franka_hardware`; therefore this repository does not claim a tested
Franka hardware plugin yet. The future adapter should be thin:

1. claim seven effort command interfaces and the Franka model/state interfaces;
2. convert hardware values to `CartesianState`;
3. call `WrenchController::compute()`;
4. map the wrench with `J^T`, add bias/null-space torque, then enforce torque/rate limits;
5. zero the residual and enter a safe hold on stale input, non-finite values, or deadline misses.

Keeping those hardware details outside the core makes the controller equations testable without
pretending that a simulator-only interface is ready for a physical robot.
