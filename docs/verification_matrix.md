# 验证矩阵

这张表用于追踪仓库中的核心主张。实现符号说明代码在哪里，自动测试检查数值或协议不变量，
实验产物记录完整 rollout 的结果。三者承担的证据角色不同。

## 经典控制

| 主张 | 实现 | 自动测试 | 实验产物 |
|---|---|---|---|
| 2-DOF FK、IK 与 Jacobian 解析式一致 | [`kinematics.py`](../src/compliant_control_lab/kinematics.py) | [finite difference 与 IK round trip](../tests/test_kinematics.py) | [metrics](../results/metrics.csv)、[GIF](../results/hybrid_demo.gif) |
| Franka 控制器统一输出 6D Cartesian wrench | [`franka_control.py`](../src/compliant_control_lab/franka_control.py) | [controller tests](../tests/test_franka_control.py)、[simulation tests](../tests/test_franka_simulation.py) | [Franka metrics](../results/franka/metrics.csv)、[plot](../results/franka/nominal.png) |
| Python 与 C++ 的三个 fixed controller 逐分量一致 | [C++ interface](../cpp/include/compliant_control_lab/franka_control.hpp)、[Python](../src/compliant_control_lab/franka_control.py) | [parity](../tests/test_cpp_parity.py)、[native tests](../cpp/tests/test_franka_control.cpp) | [CI](../.github/workflows/tests.yml) |

## 自适应与学习控制

| 主张 | 实现 | 自动测试 | 实验产物 |
|---|---|---|---|
| 自适应基线由反馈估计 bias、force rate 与接触刚度，再调度增益 | [`FrankaAdaptiveHybridController`](../src/compliant_control_lab/franka_adaptive.py) | [adaptive tests](../tests/test_franka_adaptive.py) | [v0.4 CSV](../results/franka_learning/comparison.csv)、[summary](../results/franka_learning/summary.md) |
| Nominal 与 residual 分别按关节力矩余量投影；异常时 fail closed | [torque projection](../src/compliant_control_lab/franka_torque_safety.py)、[adaptive nominal](../src/compliant_control_lab/franka_adaptive.py) | [torque-safety tests](../tests/test_franka_torque_safety.py) | [v0.5 CSV](../results/franka_safety_blind/comparison.csv) |
| Policy 以 50 Hz 更新三维 residual；安全 wrapper 以 500 Hz 运行 | [`residual_rl.py`](../src/compliant_control_lab/residual_rl.py) | [residual tests](../tests/test_residual_rl.py) | [五个冻结 policy](../results/franka_safety_preholdout/)、[result](../results/franka_safety_blind/summary.md) |

## 实验完整性

| 主张 | 实现 | 自动测试 | 冻结证据 |
|---|---|---|---|
| Train、development 与 first reveal 分离；同一 case 的方法共用 seed | [training](../src/compliant_control_lab/franka_learning.py)、[first reveal](../src/compliant_control_lab/franka_safety_learning.py) | [learning tests](../tests/test_franka_learning.py)、[protocol tests](../tests/test_franka_safety_learning.py) | [protocol](../results/franka_safety_preholdout/protocol.json)、[reveal](../results/franka_safety_blind/reveal.json)、[CSV](../results/franka_safety_blind/comparison.csv) |
| v0.5 绑定 commit、tag、policy hash 和未来 beacon | [`franka_safety_learning.py`](../src/compliant_control_lab/franka_safety_learning.py) | [protocol tests](../tests/test_franka_safety_learning.py)、[beacon verifier](../tools/verify_drand_beacon.mjs) | [protocol hash](../results/franka_safety_preholdout/protocol.sha256)、[manifest](../results/franka_safety_blind/manifest.json) |
| 离线 audit 重算 blind root、seed、gate 与摘要，且不覆盖 first reveal | [`published_results_audit.py`](../src/compliant_control_lab/published_results_audit.py) | [audit tamper tests](../tests/test_published_results_audit.py) | [reveal](../results/franka_safety_blind/reveal.json)、[384-row CSV](../results/franka_safety_blind/comparison.csv) |
| 预注册门槛为每个 residual 44/48；实际 22–26/48，结论为 `FAIL` | [gate](../src/compliant_control_lab/franka_stress.py)、[reporting](../src/compliant_control_lab/franka_safety_learning.py) | [gate tests](../tests/test_franka_stress.py) | [summary](../results/franka_safety_blind/summary.md)、[决策](residual_rl_decision.md) |
| Post-reveal 图只解释已公开 case | [`post_reveal_analysis.py`](../src/compliant_control_lab/post_reveal_analysis.py) | [analysis tests](../tests/test_post_reveal_analysis.py) | [figure](../results/franka_safety_postreveal/failure_analysis.png)、[summary](../results/franka_safety_postreveal/summary.md) |

## C++ parity 的范围

C++17/Eigen parity 当前只覆盖以下三类固定经典控制器：

- `CartesianImpedanceController` ↔ `FrankaImpedanceController`
- `CartesianAdmittanceController` ↔ `FrankaAdmittanceController`
- `HybridForcePositionController` ↔ `FrankaHybridController`

自适应增益调度、reference governor、torque projection、Residual RL、ARS 训练和 drand 揭盲
流程仍是 Python 实现。`1e-12` parity 结论不能外推到这些模块，也不能当作 ROS 2 或 Franka
真机插件已经完成的证据。

## 读表时的限制

单元测试证明局部不变量，例如零 residual 等价、投影后不越界、schema 错序会拒绝加载。
性能结论来自冻结 CSV。当前仿真仍使用理想 torque interface；0% saturation 只说明给定模型、
场景和关节力矩区间内没有触发 actuator clipping，不代表通过硬件安全认证。
