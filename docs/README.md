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
