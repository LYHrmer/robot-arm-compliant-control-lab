# 文档入口

先按任务选择入口。操作命令集中在复现与部署页，原始数值保存在对应实验归档；本页负责导航。

| 目的 | 首选文档 | 接下来能做什么 |
|---|---|---|
| 第一次了解项目 | [首页](../README.md)、[项目状态](project_status.md) | 识别可运行能力、验收范围和当前失败 |
| 跑通 BC/PPO 模型 | [部署与一键验收](learning_deployment.md) | 取得离线发行目录，在独立环境检查完整闭环 |
| 从干净环境重做实验 | [学习实验复现指南](learning_reproduction.md)、[固定环境](reproducible_environment.md) | 采集示教、训练、选模、打包并核对报告 |
| 学习原理和排错 | [教程目录](tutorial/README.md)、[BC 反馈实验](tutorial/labs/04_bc_feedback.md) | 从公式走到代码，并解释失败而不只看平均分 |
| 修改或扩展项目 | [项目结构](project_structure.md)、[系统架构](architecture.md) | 找到实现、调用关系、验证入口与冻结约束 |
| 核对某个历史结论 | [实验总账](experiments/README.md)、[验证矩阵](verification_matrix.md) | 从主张追到协议、源码、测试和原始数据 |

## 3 分钟浏览

看[首页演示](../README.md#当前负载调度演示)与[当前结果](../README.md#当前结果速览)，
再看[项目状态](project_status.md)的采用范围。需要演示完整走查时使用
[招聘方入口](recruiter_walkthrough.md)。快速工程检查是 `franka-smoke`；
学习策略的逐 case 门槛检查是[部署验收](learning_deployment.md)，两者回答不同问题。

## 系统学习

[教程 01–07](tutorial/README.md)依次讲 2-DOF 运动学、柔顺控制、Franka 数值解算、
实验方法、Residual RL、故障练习和[负载调度逐拍回放](tutorial/07_measured_budget_replay.md)。
已经学过机器人学，可从第 03 章开始。

| 动手练习 | 要核对的关系 |
|---|---|
| [误差到关节力矩](tutorial/labs/01_wrench_to_torque.md) | 位置／速度误差 → wrench → joint torque |
| [预算下降与缺包](tutorial/labs/02_budget_drop.md) | 当前上限、限速输出、下一拍系数各自何时生效 |
| [换向恢复排错](tutorial/labs/03_reversal_recovery.md) | 状态约束、局部改善和扩大回归的区别 |
| [BC 反馈实验](tutorial/labs/04_bc_feedback.md) | 教师输入与学生闭环为何不同，输入屏蔽如何部署 |

### 当前表面控制主线

按[表面坐标与 F/T](surface_frame_and_sensing.md) → [接触模型诊断](wiping_contact_diagnosis.md)
→ [积分与摩擦前馈](tangential_tracking.md) → [在线补偿](online_compensation.md)
→ [测得负载调度](load_budget.md) → [测量鲁棒性](measured_budget_robustness.md)阅读。
公式移植到 C++ 后用[同输入回放](cpp_core.md)验证，已有
[168,000 拍报告](../results/franka_measured_budget_cpp_replay/)含一份重复演示。

学习路线沿[表面学习任务](surface_learning.md) → [BC/PPO 小规模试验](surface_learning_pilot.md)
→ [BC 输入消融](bc_closed_loop_transfer.md) → [BC 到 PPO 迁移](bc_to_residual_rl.md)展开。
历史实验说明方法取舍；当前可执行训练参数与发行状态分别以
[复现指南](learning_reproduction.md)和[部署记录](learning_deployment.md)为准。

## 复现实验

| 复核层级 | 入口 | 输出意味着什么 |
|---|---|---|
| 环境与最短仿真 | [首页安装](../README.md#安装后快速复核) | 依赖可用，冻结归档可读，2 s nominal 可运行 |
| BC/PPO 全链路 | [学习实验复现指南](learning_reproduction.md) | 新计划、示教、模型、验证集选模及开发集验收 |
| 已有模型独立安装 | [部署与验收](learning_deployment.md) | 安装的源码、资源、模型身份一致，闭环指标是否达标 |
| 代码与 C++ 对齐 | [首页验证](../README.md#验证代码)、[C++ 文档](cpp_core.md) | 算法行为和同输入数值一致性 |
| 历史结果完整性 | [只读复核](velocity_evidence_audit.md)、[实验总账](experiments/README.md) | 保存的文件、指标和判定可追溯，不代表重跑实验 |

最近四项预算／速度实验可只读检查：

```bash
python -m tools.audit_velocity_evidence
```

历史正式实验使用各自协议：[v0.4](reproduction_plan_v0.4.md)、
[v0.5 首次揭盲](reproduction_plan_v0.5.md)、[v0.6 接近参考对照](reference_governor_v0.6.md)。
v0.5 已揭盲的 48 cases 是公开验证数据；复核时读取既有 protocol、reveal 和 manifest，
不重新制造一次“首次揭盲”。其 `FAIL` 与[揭盲后诊断](experiments/README.md#揭盲后诊断)均保留。

## 算法参考与历史诊断

本表按问题定位，不重复列各轮数值。完整实验顺序、产物目录和采用决定见
[实验总账](experiments/README.md)。

| 问题 | 参考与证据 |
|---|---|
| 控制公式与安全投影 | [2-DOF](control_theory.md)、[Franka](franka_control.md)、[自适应与旧 residual](adaptive_residual_rl.md)、[v0.5 torque projection](torque_safe_residual_v0.5.md) |
| 接触峰值与接近时序 | [事件诊断](contact_event_diagnosis.md)、[参考限速](reference_governor_v0.6.md)、[是否采用旧 residual](residual_rl_decision.md) |
| 姿态改善为何损失跟踪 | [增益对照](rotation_gain_comparison.md)、[原 public24 回归](rotation_gain_public24.md)、[跨方向回归](cross_surface_regression.md) |
| 力矩未饱和为何仍有残差 | [组合误差诊断](combined_residual_diagnosis.md)、[补偿预算](compensation_budget.md)、[预算转移](budget_transfer.md) |
| 速度代价集中在哪里 | [窗口分解](velocity_cost.md)、[内部限幅时序](onset_observer.md)、[速度误差时间系数](velocity_time.md) |
| 换向候选为什么没有推广 | [局部恢复](reversal_recovery.md)、[扩大回归](reversal_recovery_transfer.md)、[静止回退 3/4](stationary_recovery_pilot.md)、[可逆回退 32/36](reversible_recovery.md) |
| 学习输入、动作与选模 | [49 维任务](surface_learning.md)、[BC/PPO](surface_learning_pilot.md)、[BC 闭环输入](bc_closed_loop_transfer.md)、[PPO 初始化](bc_to_residual_rl.md) |

## 设计决策

[项目结构](project_structure.md)说明目录职责和修改落点；[系统架构](architecture.md)给出
控制数据流；[术语表](../CONTEXT.md)统一物理量、指标与数据身份。跨模块决策保存在：

- [ADR-0001：控制器接口采用 Cartesian wrench](adr/0001-cartesian-wrench-controller-seam.md)
- [ADR-0002：未来公开信标生成 first-reveal cases](adr/0002-future-beacon-first-reveal.md)
- [ADR-0003：nominal 与 residual 分级投影](adr/0003-project-nominal-and-residual-separately.md)
- [来源追踪整理](provenance_refactor_design.md)：静态依赖追踪与历史来源迁移的适用范围。

## 文档类型怎么区分

教程解释公式和思考过程；复现指南给出操作顺序和预期；算法参考解释实现；实验文档解释
冻结条件与结果；ADR 解释长期设计取舍。`results/` 保存提交过的历史数值事实，
`docs/evidence/` 保存交付摘要，`dist/` 保存本机生成的大型训练与发行产物。
三者的生命周期和来源关系见[项目结构](project_structure.md#结果放在哪里)。
