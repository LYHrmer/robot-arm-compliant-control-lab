# 文档入口

这里按阅读目的组织现有材料。首页负责交代项目结论；教程解释推导过程；算法参考记录实现
细节；实验文档保存冻结条件和结果。遇到同一个数字在多处出现时，以 `results/` 中的 CSV
和 summary 为准。

## 3 分钟浏览

适合第一次打开仓库，或面试前快速回忆项目主线。

1. 看[项目首页](../README.md)，了解 500 Hz Franka 擦拭任务、控制器范围和当前局限。
   面试官可直接走[五分钟验收路径](recruiter_walkthrough.md)。
2. 安装后运行 `franka-smoke`，确认冻结档案和主仿真入口都能读取。
3. 看[系统架构](architecture.md)的五节点概览，再按需展开实时控制或实验冻结流程。
4. 打开[v0.5 first-reveal 摘要](../results/franka_safety_blind/summary.md)，确认预注册结论是
   `FAIL`，再读[为什么暂不部署](residual_rl_decision.md)。
5. 需要核实某项说法时，到[验证矩阵](verification_matrix.md)找源码、测试和实验产物。

这条路线应能回答四件事：项目解决什么问题，算法加在什么位置，结果是否通过门槛，哪些
能力还没有做到真机。

## 系统学习

入口是[教程目录](tutorial/README.md)。六章从 2-DOF 解析模型走到 Franka、实验方法和
Residual RL，每章都给出公式、源码入口和练习。

| 顺序 | 教程 | 读完以后应能做什么 |
|---:|---|---|
| 1 | [2-DOF 运动学](tutorial/01_2dof_kinematics.md) | 手推 FK、IK、Jacobian，并做有限差分检查 |
| 2 | [柔顺控制](tutorial/02_compliant_control.md) | 区分阻抗、导纳和法向控力/切向控位 |
| 3 | [Franka 数值解算](tutorial/03_franka_numerics.md) | 解释 6D wrench、`J.T @ w`、bias 和 null space |
| 4 | [实验与验证](tutorial/04_experiments_and_validation.md) | 区分稳态指标、完整轨迹安全指标和数据身份 |
| 5 | [Residual RL](tutorial/05_residual_rl.md) | 说明 observation、动作、安全包络和 ARS 训练 |
| 6 | [练习与面试](tutorial/06_exercises_and_interview.md) | 独立排错，并用证据讲清项目取舍 |

教程读到某个公式仍不清楚时，再查对应参考页：

| 范围 | 算法参考 |
|---|---|
| 2-DOF 控制公式 | [control_theory.md](control_theory.md) |
| Franka 6D 控制与指标 | [franka_control.md](franka_control.md) |
| C++17/Eigen 控制核心 | [cpp_core.md](cpp_core.md) |
| 自适应增益与 v0.4 residual | [adaptive_residual_rl.md](adaptive_residual_rl.md) |
| v0.5 torque projection | [torque_safe_residual_v0.5.md](torque_safe_residual_v0.5.md) |
| 接触峰值事件诊断 | [contact_event_diagnosis.md](contact_event_diagnosis.md) |
| 接近轨迹、参考限速与因果采样 | [reference_governor_v0.6.md](reference_governor_v0.6.md) |
| 表面坐标、工具端六维 F/T 与标定误差 | [surface_frame_and_sensing.md](surface_frame_and_sensing.md) |
| 擦拭微分离、接触柔度与步长检查 | [wiping_contact_diagnosis.md](wiping_contact_diagnosis.md) |
| 接触稳定后的切向误差、积分与摩擦前馈 | [tangential_tracking.md](tangential_tracking.md) |
| 在线等效负载补偿、停止更新条件和原有 24-case 回归 | [online_compensation.md](online_compensation.md) |
| 换向姿态代价、旋转增益推导与同配置配对 | [rotation_gain_comparison.md](rotation_gain_comparison.md) |
| 新姿态增益的表面方向／已知质量回归与力矩余量 | [rotation_gain_public24.md](rotation_gain_public24.md) |
| 换向／组合误差能否转移到其他表面方向 | [cross_surface_regression.md](cross_surface_regression.md) |
| 新表面任务的 RL / IL 环境、数据与评价准备 | [surface_learning.md](surface_learning.md) |
| 小规模行为克隆、残差 PPO 与闭环选型 | [surface_learning_pilot.md](surface_learning_pilot.md) |
| BC 离线拟合良好，闭环为什么偏：历史输入消融 | [bc_closed_loop_transfer.md](bc_closed_loop_transfer.md) |
| BC 表示能否帮助 PPO：隐藏层迁移与固定预算对照 | [bc_to_residual_rl.md](bc_to_residual_rl.md) |

## 复现实验

先按[教程环境说明](tutorial/README.md#环境与第一轮复现)安装 Python 依赖并运行测试。不同
实验的数据身份不能混用：

| 实验 | 说明与命令 | 已保存产物 |
|---|---|---|
| 2-DOF 与 Franka 标称场景 | [首页标称演示](../README.md#标称演示) | [`results/`](../results/)、[`results/franka/`](../results/franka/) |
| 固定增益随机压力测试 | [可信实验教程](tutorial/04_experiments_and_validation.md#9-复现实验) | [`results/franka_stress/`](../results/franka_stress/) |
| v0.4 自适应与 residual 同 case 对比 | [v0.4 实现记录](reproduction_plan_v0.4.md)、[复现命令](adaptive_residual_rl.md#6-reproduction-commands) | [`results/franka_learning/`](../results/franka_learning/) |
| v0.5 五 seed 冻结与 first reveal | [v0.5 协议](reproduction_plan_v0.5.md) | [`results/franka_safety_preholdout/`](../results/franka_safety_preholdout/)、[`results/franka_safety_blind/`](../results/franka_safety_blind/) |
| v0.5 揭盲后 paired/event 诊断 | [实验总账](experiments/README.md#揭盲后诊断) | [`results/franka_safety_postreveal/`](../results/franka_safety_postreveal/) |
| v0.6 开发：时序与接近参考四组对照 | [实现与复现命令](reference_governor_v0.6.md) | [`results/franka_reference_ablation/`](../results/franka_reference_ablation/) |
| 新表面任务：24-case 法向标定开发对照 | [传感器、指标与复现命令](surface_frame_and_sensing.md) | [`results/franka_surface_development/`](../results/franka_surface_development/) |
| 表面任务接触模型修复：24 × 4 × 2 配对 | [单因素排查与模型代价](wiping_contact_diagnosis.md) | [`results/franka_surface_contact_fix/`](../results/franka_surface_contact_fix/)、[诊断统计](../results/franka_surface_contact_diagnostics/) |
| 固定平滑模型的切向补偿：24 × 3，加独立长时 9 次 | [公式、时序和失配边界](tangential_tracking.md) | [81 次实验](../results/franka_tangential_development/)、[单因素与静态保持](../results/franka_tangential_diagnostics/diagnosis.json) |
| 在线补偿：原有 24 × 4 配对回归 | [更新律与共同 6 N 上限](online_compensation.md) | [96 次实验](../results/franka_online_compensation_regression/) |
| 12 秒误差对照：10 工况 × 2 噪声种子 × 4 方法 | [阶段结果与剩余问题](online_compensation.md#实测结果与没有通过的部分) | [80 次实验](../results/franka_online_compensation_errors/)、[对照图](../results/franka_online_compensation_figures/dynamic_errors.png) |
| 姿态阻抗常量预设：相同 80 次误差对照，所有方法统一增益 | [增益与跟踪代价](rotation_gain_comparison.md) | [新对照](../results/franka_rotation_gain_comparison/)、[新增益 C++ 回放](../results/franka_rotation_gain_cpp_replay/) |
| 姿态预设推广检查：原 24-case × 4 方法 × 2 增益 | [采用决定与力矩余量](rotation_gain_public24.md) | [192 次运行](../results/franka_rotation_gain_public24/)、[27,000 步 C++ 回放](../results/franka_rotation_gain_public24_cpp/) |
| 小规模跨方向动态回归：3 yaw × 2 工况 × 3 方法 × 2 增益，seed 11 | [阶段失效、残余误差与采用决定](cross_surface_regression.md) | [36 次运行与 144 个阶段](../results/franka_cross_surface_dynamic/) |
| C++ 完整数值控制链回放 | [输入、状态与硬件边界](cpp_core.md) | [四份完整轨迹核验](../results/franka_online_cpp_replay/) |
| 表面任务学习前准备：24 × 3、12 次压力测试与同频教师数据 | [训练接口、标签与复现](surface_learning.md) | [数值索引](../results/franka_surface_learning_preparation/)、[三回合示例](../results/franka_surface_learning_examples/)、[分层修正记录](../results/franka_surface_learning_partition/) |
| 小规模 BC 与 bounded Residual PPO：各三个训练种子 | [损失、更新与选择规则](surface_learning_pilot.md) | [完整种子对照](../results/franka_surface_learning_pilot/) |
| BC 输入消融：同网络、预算和三个种子 | [输入先验与导出掩码](bc_closed_loop_transfer.md) | [完整对照与代表轨迹](../results/franka_surface_bc_transfer/) |
| BC 隐藏层迁移到 PPO：固定第 32 回合，三个种子 | [动作语义、初始化和配对规则](bc_to_residual_rl.md) | [训练记录与最终策略对照](../results/franka_surface_ppo_transfer/) |

v0.5 的 round `31756275` 已经完成 first reveal。仓库中的 48 cases 随后转为 public
validation set（公开验证集）。复核时读取现有 `protocol.json`、`reveal.json`、
`comparison.csv` 和 `manifest.json`。围绕这些 cases 的诊断属于 post-reveal analysis；后续
调参若要形成新的未见数据主张，需要冻结新协议并选用尚未发布的 drand round。

已有结果可以用首页的[离线 audit 命令](../README.md#验证代码)复核。失败原因的派生图和计数
在 [post-reveal summary](../results/franka_safety_postreveal/summary.md)。事件重放在生成诊断前
还会逐 case 对齐 7 个冻结指标，输出见 [event summary](../results/franka_safety_postreveal/contact_events/summary.md)
和[诊断教程](contact_event_diagnosis.md)；两类分析都不写入 first-reveal 目录。

## 设计决策

ADR 记录跨模块、会影响后续实现的选择，并解释为什么边界放在这里。

- [ADR-0001：控制器边界放在 Cartesian wrench](adr/0001-cartesian-wrench-controller-seam.md)
- [ADR-0002：用未来公开信标生成 first-reveal cases](adr/0002-future-beacon-first-reveal.md)
- [ADR-0003：Nominal 与 residual 分级投影](adr/0003-project-nominal-and-residual-separately.md)

## 文档类型怎么区分

- 教程回答“为什么这样算”，允许按章节逐步推导。
- 算法参考回答“代码具体怎样实现”，适合结合源码查参数和状态机。
- 实验记录回答“何时冻结、用了哪些数据、结果是什么”，不会替代算法说明。
- ADR 回答“为什么采用这个模块边界或实验约束”，并列出后果和可复核证据。
- `results/` 保存数值事实。决策页只解释这些事实如何影响下一步。
