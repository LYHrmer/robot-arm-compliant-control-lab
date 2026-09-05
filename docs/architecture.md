# 系统架构

## RL 改了哪里

Residual policy 不接管机器人。它只在经典控制器给出的 nominal wrench 之后添加一个三维、
有界的平移力修正；禁用策略或推理异常时，该修正回到零。

```mermaid
flowchart LR
  INPUT["state + target"] --> NOMINAL["classical nominal<br/>500 Hz"]
  NOMINAL --> NPROJ["nominal wrench projection<br/>under torque limits"]
  NPROJ --> RESIDUAL["bounded residual<br/>50 Hz policy / 500 Hz guard"]
  RESIDUAL --> ROBOT["sum, Jᵀ mapping<br/>and robot"]
```

经典 impedance、admittance 和 fixed hybrid 使用同一个 `FrankaController` 接口，但不经过
自适应层或 residual policy。下面两张图再分别展开实时控制和实验证据流。

## 实时控制

```mermaid
flowchart LR
  ROBOT["MuJoCo Panda"] --> STATE["FrankaState<br/>pose, twist, force, J, limits"]
  TARGET["FrankaTarget<br/>pose, twist, 12 N"] --> NOMINAL["torque-safe adaptive<br/>contact state + gain schedule"]
  STATE --> NOMINAL
  NOMINAL --> NPROJ["nominal 6D wrench<br/>10% torque reserve"]
  STATE --> OBS["20-D observation<br/>including torque headroom"]
  OBS --> POLICY["3-D residual policy<br/>update at 50 Hz"]
  POLICY --> GUARD["contact gate, bounds, filter,<br/>rate and residual projection"]
  NPROJ --> GUARD
  STATE --> GUARD
  GUARD --> MAP["wrench sum + Jᵀ mapping<br/>bias + null-space torque"]
  MAP --> ROBOT
```

策略每 20 ms 更新一次动作。安全 wrapper 在每个 2 ms 周期继续滤波、限速并重算 torque
projection。两次 policy update 之间的 Jacobian 变化仍会进入检查。MuJoCo 的 actuator
clipping 只保留为监测项；控制器应在到达该处以前落入预留力矩区间。

## 训练、冻结与揭盲

```mermaid
flowchart LR
  TRAIN["8 train cases<br/>5 seeds"] --> DEV["8 development cases<br/>select checkpoints"]
  DEV --> ARTIFACT["5 policies + curves<br/>SHA256"]
  ARTIFACT --> FREEZE["protocol + commit<br/>preholdout tag"]
  FREEZE --> BEACON["future drand round<br/>two-relay verification"]
  BEACON --> CASES["48 scenario seeds<br/>48 noise seeds"]
  CASES --> EVAL["same cases<br/>3 baselines + 5 policies"]
  EVAL --> RESULT["CSV + reveal<br/>manifest + summary"]
```

训练集和 development set 在冻结前可见；first-reveal cases 由冻结后才发布的随机信标派生。
揭盲后，这 48 个 cases 只能算公开验证数据。具体选择理由见
[未来信标 ADR](adr/0002-future-beacon-first-reveal.md) 和
[nominal/residual 分级投影 ADR](adr/0003-project-nominal-and-residual-separately.md)。

## 模块接口与边界（seam）

| 接缝 | 接口 | 当前实现 |
|---|---|---|
| Cartesian controller | `FrankaController.reset()` 与 `compute(state, target, dt) -> wrench_6d` | [`franka_control.py`](../src/compliant_control_lab/franka_control.py)；MuJoCo 对象不会穿过此接缝 |
| Actuation context | `FrankaActuationContext` 中的 `J`、torque offset 和上下限 | [`franka_simulation.py`](../src/compliant_control_lab/franka_simulation.py) 提供数据；[Python](../src/compliant_control_lab/franka_torque_safety.py) 与 [C++](../cpp/src/torque_safety.cpp) 实现投影 |
| Residual policy | `ResidualPolicy.action(observation) -> normalized_action` | [`residual_rl.py`](../src/compliant_control_lab/residual_rl.py) 保存 20-D schema，并把策略异常收敛到零 residual |
| Event telemetry | `FrankaTrialResult` 数组与可选 controller snapshot | [`contact_event_analysis.py`](../src/compliant_control_lab/contact_event_analysis.py) 先对齐冻结指标，再提取公开 case 的峰值上下文 |
| 实验证据 | `prepare` 产出冻结协议，`evaluate` 只接受与 tag 字节一致的产物 | [`franka_safety_learning.py`](../src/compliant_control_lab/franka_safety_learning.py) 和冻结的 [protocol.json](../results/franka_safety_preholdout/protocol.json) |

前两个接缝把控制公式与仿真适配器分开。策略只能添加三维 Cartesian force，无法直接写
joint torque。训练产物通过第三个接缝进入控制周期。first-reveal set 完成初次评测后可用于
post-reveal analysis，但不会回流到已经冻结的 checkpoint。
