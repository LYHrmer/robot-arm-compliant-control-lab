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
| 有状态法向参考满足离散速度、加速度约束；实测超速请求制动 | [`FrankaRateLimitedAdaptiveController`](../src/compliant_control_lab/franka_reference.py) | [长序列与重置测试](../tests/test_franka_reference.py) | [公开开发对照](../results/franka_reference_ablation/summary.md)；约束对象为参考，非实际末端速度 |
| Nominal 与 residual 分别按关节力矩余量投影；异常时 fail closed | [Python projection](../src/compliant_control_lab/franka_torque_safety.py)、[C++ projection](../cpp/src/torque_safety.cpp)、[adaptive nominal](../src/compliant_control_lab/franka_adaptive.py) | [Python safety tests](../tests/test_franka_torque_safety.py)、[C++ edge cases](../cpp/tests/test_torque_safety.cpp)、[160-case parity](../tests/test_cpp_parity.py) | [v0.5 Python rollout](../results/franka_safety_blind/comparison.csv)；C++ port 属于揭盲后工程验证 |
| Policy 以 50 Hz 更新三维 residual；安全 wrapper 以 500 Hz 运行 | [`residual_rl.py`](../src/compliant_control_lab/residual_rl.py) | [residual tests](../tests/test_residual_rl.py) | [五个冻结 policy](../results/franka_safety_preholdout/)、[result](../results/franka_safety_blind/summary.md) |

## 表面任务开发路径

| 主张 | 实现 | 自动测试 | 实验产物 |
|---|---|---|---|
| 固定表面坐标中计算增益；同步旋转 Jacobian 后关节力矩不变 | [`surface_control.py`](../src/compliant_control_lab/surface_control.py) | [旋转协变与 identity 等价](../tests/test_surface_control.py) | [24-case 开发对照](../results/franka_surface_development/summary.md) |
| 工具端六轴传感使用 sensordata，仅补偿名义重力；法向投影进入闭环 | [`surface_sensing.py`](../src/compliant_control_lab/surface_sensing.py) | [外力符号与惯性残留](../tests/test_surface_sensing.py)、[因果采样与真值隔离](../tests/test_surface_simulation.py) | [建模限定](surface_frame_and_sensing.md) |
| 完整控制输入可重放 wrench 和未裁剪 joint torque，不积分新动力学 | [`surface_replay.py`](../src/compliant_control_lab/surface_replay.py) | [序列回放与格式拒绝](../tests/test_surface_replay.py)、[归档复核](../tests/test_surface_published_results.py) | [四份代表轨迹与 replay_checks](../results/franka_surface_development/manifest.json) |

## 实验完整性

| 主张 | 实现 | 自动测试 | 冻结证据 |
|---|---|---|---|
| Train、development 与 first reveal 分离；同一 case 的方法共用 seed | [training](../src/compliant_control_lab/franka_learning.py)、[first reveal](../src/compliant_control_lab/franka_safety_learning.py) | [learning tests](../tests/test_franka_learning.py)、[protocol tests](../tests/test_franka_safety_learning.py) | [protocol](../results/franka_safety_preholdout/protocol.json)、[reveal](../results/franka_safety_blind/reveal.json)、[CSV](../results/franka_safety_blind/comparison.csv) |
| v0.5 绑定 commit、tag、policy hash 和未来 beacon | [`franka_safety_learning.py`](../src/compliant_control_lab/franka_safety_learning.py) | [protocol tests](../tests/test_franka_safety_learning.py)、[beacon verifier](../tools/verify_drand_beacon.mjs) | [protocol hash](../results/franka_safety_preholdout/protocol.sha256)、[manifest](../results/franka_safety_blind/manifest.json) |
| 离线 audit 重算 blind root、seed、gate 与摘要，且不覆盖 first reveal | [`published_results_audit.py`](../src/compliant_control_lab/published_results_audit.py) | [audit tamper tests](../tests/test_published_results_audit.py) | [reveal](../results/franka_safety_blind/reveal.json)、[384-row CSV](../results/franka_safety_blind/comparison.csv) |
| 预注册门槛为每个 residual 44/48；实际 22–26/48，结论为 `FAIL` | [gate](../src/compliant_control_lab/franka_stress.py)、[reporting](../src/compliant_control_lab/franka_safety_learning.py) | [gate tests](../tests/test_franka_stress.py) | [summary](../results/franka_safety_blind/summary.md)、[决策](residual_rl_decision.md) |
| Residual paired effect 与 leave-one-gate-out 只解释已公开 case | [`post_reveal_analysis.py`](../src/compliant_control_lab/post_reveal_analysis.py) | [row-order、pair 完整性与固定数值](../tests/test_post_reveal_analysis.py) | [figure](../results/franka_safety_postreveal/failure_analysis.png)、[summary](../results/franka_safety_postreveal/summary.md) |
| Event replay 先逐 case 对齐 7 个冻结指标，再提取峰值上下文 | [`contact_event_analysis.py`](../src/compliant_control_lab/contact_event_analysis.py)、[telemetry adapter](../src/compliant_control_lab/franka_simulation.py) | [event/phase/replay tests](../tests/test_contact_event_analysis.py)、[non-interference tests](../tests/test_franka_simulation.py) | [48-case event report](../results/franka_safety_postreveal/contact_events/summary.md)、[CSV](../results/franka_safety_postreveal/contact_events/safe_adaptive_contact_events.csv) |
| 分步模式使用当前运动学与上一周期求解力，记录观测来源时间 | [`franka_simulation.py`](../src/compliant_control_lab/franka_simulation.py) | [独立前向计算核对](../tests/test_franka_timing.py)、[力缓存与延迟测试](../tests/test_franka_force_cache.py) | [采样约定](reference_governor_v0.6.md#每个周期的数据来自哪里) |
| 四组开发消融共用场景与噪声 seed，先验证旧模式，输出不覆盖冻结档案 | [`reference_ablation.py`](../src/compliant_control_lab/reference_ablation.py) | [配对、时序和目录保护测试](../tests/test_reference_ablation.py) | [参数与源码哈希](../results/franka_reference_ablation/manifest.json)、[CSV](../results/franka_reference_ablation/comparison.csv) |
| 一条 smoke 命令同时检查冻结档案与主仿真路径 | [`smoke.py`](../src/compliant_control_lab/smoke.py) | [smoke semantics](../tests/test_smoke.py)、[CI](../.github/workflows/tests.yml) | 输出同时显示 archive PASS、frozen decision FAIL 与 simulation PASS |

## C++ parity 的范围

C++17/Eigen parity 覆盖三类固定经典控制器：

- `CartesianImpedanceController` ↔ `FrankaImpedanceController`
- `CartesianAdmittanceController` ↔ `FrankaAdmittanceController`
- `HybridForcePositionController` ↔ `FrankaHybridController`

三个 torque-safety API 另用 160 个固定随机 7-DOF case 对照 Python。自适应增益调度与
reference governor 尚未移植。Policy 及其实验流水线仍是 Python 实现。Parity 结论不代表
ROS 2 或 Franka 真机插件已经完成。

## 读表时的限制

单元测试证明局部不变量，例如零 residual 等价、投影后不越界、schema 错序会拒绝加载。
性能结论来自冻结 CSV。当前仿真仍使用理想 torque interface；0% saturation 只说明给定模型、
场景和关节力矩区间内没有触发 actuator clipping，不代表通过硬件安全认证。
