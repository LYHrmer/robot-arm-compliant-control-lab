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
| 切向积分/前馈使用测量状态，在同次 nominal 投影之前相加；条件积分有界且丢失接触清零 | [`tangential_compensation.py`](../src/compliant_control_lab/tangential_compensation.py)、[`franka_adaptive.py`](../src/compliant_control_lab/franka_adaptive.py) | [范数与换向](../tests/test_tangential_feedforward.py)、[积分时序与冻结](../tests/test_tangential_safety.py)、[真值隔离与回放](../tests/test_tangential_replay.py) | [固定模型 72 行主对照与独立 9 行失配](../results/franka_tangential_development/summary.md)；不证明任意摩擦鲁棒性 |
| 在线补偿使用运动误差调整等效系数，受参数/请求变化率和投影接受条件约束 | [`tangential_compensation.py`](../src/compliant_control_lab/tangential_compensation.py) | [更新顺序、冻结与参数边界](../tests/test_online_tangential_compensation.py)、[回放与真值隔离](../tests/test_tangential_replay.py) | [原有 24-case 的 96 条对照](../results/franka_online_compensation_regression/comparison.csv)；不等同于摩擦辨识 |
| 动态摩擦、停顿换向与测量误差使用预定时序，保留阶段失败 | [`online_compensation_experiment.py`](../src/compliant_control_lab/online_compensation_experiment.py)、[只读审计](../tools/audit_online_compensation_errors.py) | [时序与真值隔离](../tests/test_online_compensation_experiment.py)、[重封篡改与非法重置拒绝](../tests/test_online_compensation_error_audit.py) | [80 次运行](../results/franka_online_compensation_errors/comparison.csv)、[168 条阶段结果](../results/franka_online_compensation_errors/phase_metrics.csv)；在线阶段通过 40/42，不是新 holdout |
| 常量旋转刚度乘 2、阻尼乘 sqrt(2)，四种方法使用同一预设 | [`surface_control.py`](../src/compliant_control_lab/surface_control.py)、[配置与推导](rotation_gain_comparison.md) | [增益隔离](../tests/test_surface_control.py)、[Python 回放](../tests/test_surface_replay.py)、[元数据篡改拒绝](../tests/test_online_compensation_error_audit.py) | [相同 80 次新对照](../results/franka_rotation_gain_comparison/comparison.csv)；在线阶段 42/42，切向代价保留，未改默认值 |
| 所选 C++ 表面控制链与保存的 Python 命令逐周期一致 | [`surface_control.cpp`](../cpp/src/surface_control.cpp)、[完整输入回放](../tools/verify_cpp_surface_replay.py) | [状态、换向与超时](../tests/test_cpp_surface_loop.py)、[解析及拒绝路径](../tests/test_cpp_surface_replay_verifier.py) | [四份轨迹、24,000 步报告](../results/franka_online_cpp_replay/report.json)；不涵盖真实机器人通信或实时性 |
| 新姿态预设能在 C++ 完整输入回放中重建 | [`surface_controller_probe.cpp`](../cpp/tools/surface_controller_probe.cpp)、[参数转发](../tools/verify_cpp_surface_replay.py) | [标量校验、默认兼容和 CLI 测试](../tests/test_cpp_surface_replay_verifier.py) | [新增益四份轨迹、24,000 步报告](../results/franka_rotation_gain_cpp_replay/report.json)；耗时附录仍对应独立默认参数 benchmark |
| 新姿态预设在原表面方向／已知质量网格做同方法配对，保留跟踪代价 | [192 次生成器](../tools/rotation_gain_regression.py)、[独立审计](../tools/audit_rotation_gain_regression.py) | [原网格、配对与边界](../tests/test_rotation_gain_regression.py)、[重封篡改和 full/compact 一致性](../tests/test_rotation_gain_audit.py) | [全部对照](../results/franka_rotation_gain_public24/)、[27,000 步回放](../results/franka_rotation_gain_public24_cpp/report.json)；[不改默认值的理由](rotation_gain_public24.md) |

## 实验完整性

[测得负载调度](load_budget.md)先经[逐周期控制测试](../tests/test_load_aware_compensation.py)，
再使用[测量输入白名单](../tests/test_load_budget_inputs.py)和
[数值重构检查](../tests/test_load_budget_validation.py)验证预算更新与补偿请求。
[协议/发布测试](../tests/test_load_budget_study.py)覆盖同工况配对、先 A 后 B 的执行条件及不完整输出隔离。
[归档](../results/franka_load_budget_pilot/comparison.json)含 18 条候选及 2 个复现控制组；
不把纯数组重构当作重新积分物理，也不声称已逐拍重算精简轨迹中未保存的测量状态更新。

[组合误差诊断](combined_residual_diagnosis.md)保留 12 次单因素移除运行。
[指标测试](../tests/test_combined_residual_diagnostics.py)核对真实切平面分解与滚动窗口，
[协议测试](../tests/test_combined_residual_ablation.py)限制每项移除只改一个输入，
[独立回放](../tests/test_combined_residual_observer.py)检查系数和命令时序，
[重封篡改测试](../tests/test_combined_residual_audit.py)检查观测标志、配对差值及物理边界。
这是输入反事实分析，不更新原配对方法的通过数。

[预算转移](budget_transfer.md)使用[生成与离线审计入口](../tools/budget_transfer.py)和
[独立轨迹校验](../tools/budget_transfer_validation.py)。[协议测试](../tests/test_budget_transfer.py)
检查固定矩阵与参考行身份；[归档](../results/franka_budget_transfer/)包含 120 行评价，其中
60 次新运行、60 行固定参考。两档增益的动态预算筛查均为 6/6，增益兼容为 12/12；public24 兼容仅 23/48，
最差切向速度 RMS 增量约 0.357 mm/s，超过 0.2 mm/s 限值，整体筛查 `FAIL`。
默认保持 6 N、增益 1，本轮没有新增 8 N 的 C++ 控制链回放。

[速度代价分解](velocity_cost.md)只读 96 条已有轨迹，保留全部 48 对和 288 个窗口。
[分析器测试](../tests/test_velocity_cost_analysis.py)检查正交分解及窗口权重；
[派生归档测试](../tests/test_velocity_cost_study.py)检查父归档身份和重封篡改。
[结果](../results/franka_velocity_cost/)没有新增仿真，也没有更改原来的兼容性判定。

[跨方向动态回归](cross_surface_regression.md)新增 36 次运行、144 个阶段及 90 个同身份增益配对。
[配置与边界测试](../tests/test_cross_surface_regression.py)、[配对测试](../tests/test_cross_surface_pairs.py)
和[重封篡改审计测试](../tests/test_cross_surface_audit.py)分别核对几何转换、配对身份、旧结果复现、
原始输入与紧凑指标、固定力矩限值及完整轨迹一致性。既有源代码与实验归档均未修改。

新表面任务的[学习前准备](surface_learning.md)与旧 v0.5 学习实验使用不同接口和数据身份：

| 检查范围 | 实现 | 可执行证据 |
|---|---|---|
| 批量与交互仿真使用同一循环；代表完整轨迹零残差一致 | [`surface_simulation.py`](../src/compliant_control_lab/surface_simulation.py)、[`surface_env.py`](../src/compliant_control_lab/surface_env.py) | [旧归档逐字段回归](../tests/test_surface_stepper.py)、[Gym/重置/零残差](../tests/test_surface_env.py) |
| 最后积分状态的速度/穿透不能被正常 truncation 隐藏 | [`SurfaceSimulator.evaluator_kinematics`](../src/compliant_control_lab/surface_simulation.py) | [端点故障注入](../tests/test_surface_endpoint.py)；不是自然触发概率或连续时间安全证明 |
| 同频教师标签、失败保留、按组切分、bootstrap/折扣时钟 | [`surface_dataset.py`](../src/compliant_control_lab/surface_dataset.py)、[`surface_splits.py`](../src/compliant_control_lab/surface_splits.py)、[`surface_transitions.py`](../src/compliant_control_lab/surface_transitions.py) | [重封哈希后的语义篡改](../tests/test_surface_dataset.py)、[分组测试](../tests/test_surface_splits.py)、[数组加载与 mask](../tests/test_surface_transitions.py) |
| 保存动作可重新积分，而不仅是重算标签或哈希 | [`surface_dataset_replay.py`](../src/compliant_control_lab/surface_dataset_replay.py) | [独立 fresh-env 重放](../tests/test_surface_dataset_replay.py) |
| 候选策略绑定新 schema/名义控制器/分组；冻结值由外部固定 | [`surface_policy_artifact.py`](../src/compliant_control_lab/surface_policy_artifact.py) | [旧策略拒绝、回调异常与重封篡改](../tests/test_surface_policy_artifact.py) |
| 有限 JSON MLP 可导出；冻结权重、runner 与运行版本后再进入评价 | [`surface_mlp_actor.py`](../tools/surface_mlp_actor.py)、[`evaluate_surface_candidate.py`](../tools/evaluate_surface_candidate.py) | [数值推理与格式拒绝](../tests/test_surface_mlp_actor.py)、[冻结选择范围与独立物理门](../tests/test_surface_candidate_evaluation.py) |
| BC 按完整验证动作 MSE 选 epoch；PPO 正确处理 GAE 边界，并保留零残差检查点 | [`train_surface_bc.py`](../tools/train_surface_bc.py)、[`train_surface_ppo.py`](../tools/train_surface_ppo.py) | [BC 拟合与非零导出](../tests/test_surface_bc.py)、[GAE、KL 与失败记录](../tests/test_surface_ppo.py)；训练测试需要可选 PyTorch |
| 模型选择不使用开发测试成绩，公开副本保留全部种子与失败成绩 | [`surface_learning_pilot.py`](../tools/surface_learning_pilot.py)、[`publish_surface_learning_pilot.py`](../tools/publish_surface_learning_pilot.py) | [完整验证集与配对拒绝](../tests/test_surface_learning_pilot.py)、[公开副本审计](../tests/test_surface_pilot_publication.py)、[首轮训练结果](../results/franka_surface_learning_pilot/) |
| BC 输入掩码同时作用于拟合与导出，配置选择覆盖全部种子 | [`train_surface_bc_transfer.py`](../tools/train_surface_bc_transfer.py)、[`surface_bc_transfer.py`](../tools/surface_bc_transfer.py) | [非零导出与输入不变性](../tests/test_surface_bc_transfer.py)、[缺失种子与来源检查](../tests/test_surface_bc_transfer_protocol.py)、[公开归档复核](../tests/test_surface_bc_transfer_publication.py) |
| PPO 只继承 BC 隐藏层，统一比较最终检查点，失败指标不剔除 | [`train_surface_ppo_transfer.py`](../tools/train_surface_ppo_transfer.py)、[`surface_ppo_transfer.py`](../tools/surface_ppo_transfer.py) | [复制层与零输出检查](../tests/test_surface_ppo_transfer.py)、[预算及开发集门控](../tests/test_surface_ppo_transfer_protocol.py)、[缺失指标否决](../tests/test_surface_ppo_transfer_failed_metrics.py)、[公开对照重算](../tests/test_surface_ppo_transfer_publication.py) |

表面任务的[接触模型修复](wiping_contact_diagnosis.md)由
[`surface_simulation.py`](../src/compliant_control_lab/surface_simulation.py) 显式选择；
[稳定性与真实几何回归](../tests/test_surface_contact_stability.py)检查接触、摩擦及默认旧路径，
[步长测试](../tests/test_contact_step_refinement.py)检查固定 500 Hz 的采样契约。
[192 行配对报告](../results/franka_surface_contact_fix/summary.md)由
[归档测试](../tests/test_surface_contact_published_results.py)核对。这些证据只适用于仿真模型，
不证明材料辨识或控制算法优越性。

后续[切向补偿实验](tangential_tracking.md)固定上述平滑模型，先逐项复现准确法向的旧 24 行，
再比较两种经典补偿。[实验契约测试](../tests/test_tangential_experiment.py)检查行数、配对、
目录与来源保护，[归档测试](../tests/test_tangential_published_results.py)重算哈希、代表轨迹指标
和失败判定。前馈达到本轮切向减半目标，积分未达到；两者的 force/姿态代价均保留。

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

补偿预算的[独立对照](compensation_budget.md)另外使用
[生成器](../tools/compensation_budget_study.py)、[固定筛选规则](../tools/compensation_budget_screen.py)
和[预算感知校验](../tools/compensation_budget_validation.py)。
[构造与协议测试](../tests/test_compensation_budget_study.py)确认只改变幅值上限，
[筛选测试](../tests/test_compensation_budget_screen.py)拒绝用平均值隐藏单组代价，
[篡改测试](../tests/test_compensation_budget_audit.py)独立复核
[12 次归档](../results/franka_compensation_budget/)。这轮没有新增 8 N 的 C++ 完整控制链回放，
不能把下面的原有 parity 结果自动推广到新配置。

C++17/Eigen parity 覆盖三类固定经典控制器：

- `CartesianImpedanceController` ↔ `FrankaImpedanceController`
- `CartesianAdmittanceController` ↔ `FrankaAdmittanceController`
- `HybridForcePositionController` ↔ `FrankaHybridController`

三个 torque-safety API 另用 160 个固定随机 7-DOF case 对照 Python。
新增的[表面控制器](../cpp/src/surface_control.cpp)包含自适应增益调度、接近 governor、
切向补偿与完整 wrench 投影，[长序列测试](../tests/test_cpp_surface_loop.py)核对有状态输出。
这里移植的是 `FrankaSafeAdaptiveController` 的 governor，不包括另一条
`FrankaRateLimitedAdaptiveController` 的速度/加速度参考实验。
Policy 及其训练流水线仍是 Python 实现。Parity 结论不代表 ROS 2 或 Franka 真机插件已完成。

## 读表时的限制

单元测试证明局部不变量，例如零 residual 等价、投影后不越界、schema 错序会拒绝加载。
性能结论来自冻结 CSV。当前仿真仍使用理想 torque interface；0% saturation 只说明给定模型、
场景和关节力矩区间内没有触发 actuator clipping，不代表通过硬件安全认证。
