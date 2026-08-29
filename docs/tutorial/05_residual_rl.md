# 05｜Residual RL：何时值得做，怎样不破坏经典控制器

本章是设计规范，不宣称仓库当前已经训练出 RL improvement。完整 go/no-go 证据见
[`residual_rl_decision.md`](../residual_rl_decision.md)。

## 1. 为什么不是端到端 torque RL

当前系统已经有可靠的：

- kinematics/Jacobian；
- Cartesian impedance、force PI 和 contact state machine；
- gravity/Coriolis bias compensation；
- null-space posture；
- torque limits 和 zero-residual fallback 接口。

端到端 RL 若直接输出 7 维 torque，会重新学习这些已知结构，增大样本量、调试和安全难度。
Residual RL 保留 nominal controller，只学习未建模 remainder：

\[
\mathbf u=\pi_{classical}(\mathbf s)+\pi_\theta(\mathbf s).
\]

这一结构来自 Johannink et al. 的
[Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201)。他们的
目标也是让经典控制解决已知几何/机器人动力学，让 residual 处理接触、摩擦和对象变化。

CoRL 2023 的
[Online Admittance Residual Learning](https://proceedings.mlr.press/v229/zhang23e.html)
进一步说明：接触 sim-to-real gap 很大时，保留 admittance 结构并适应 compliance parameters
通常比直接丢弃控制结构更合理。

## 2. 本项目为什么进入“值得实验”阶段

impact-aware fixed hybrid 在 nominal 中表现稳定，但固定 holdout 只通过 25% case。主要失败
是擦拭阶段的 raw force peak、tangential tracking 和少量 torque saturation；这些误差与
摩擦、表面方向和 dynamics bias 共同变化，不容易用一组固定 gain 同时解决。

这只说明存在 learning/adaptation 问题，不证明 RL 优于：

- gain scheduling；
- disturbance observer；
- online stiffness/friction identification；
- adaptive admittance；
- MPC 或 passivity-based adaptation。

因此 v0.4 必须加入至少一个 adaptive classical baseline。

## 3. 推荐动作空间：有界 Cartesian residual

先只修正平移 wrench，姿态仍由 classical impedance 控制：

\[
\Delta\mathbf w=
[\Delta F_n,\Delta F_y,\Delta F_z,0,0,0]^T,
\]

\[
\mathbf w_{cmd}=\mathbf w_{hybrid}+S\,\mathrm{clip}(\Delta\mathbf w).
\]

初始建议 bounds：

- normal residual：+/-4 N；
- 每个 tangential residual：+/-6 N；
- residual rate 另行限幅并低通。

为什么不先学习 stiffness matrix？学习 gain 更容易保持物理可解释性，但正定性、阻尼关系和
快速变化都要额外约束。最小版本先用 bounded wrench residual，更容易做零残差回退和消融。

## 4. Observation

策略频率可为 50--100 Hz，下面的 classical loop 仍为 500 Hz。建议 observation 只使用真机
可获得量，避免 simulation privileged information：

```text
force error + short force history
raw/filtered force (training中注意真机是否真能得到raw equivalent)
tangential position and velocity error
joint position and velocity
contact phase / force blend
previous residual action
target force and target tangential velocity
```

所有量用训练分布固定的 scale 归一化；部署时不能在线改变 normalization statistics。

## 5. Reward 不是最终指标

可从以下物理项构造 dense reward：

\[
r_t=-w_f e_F^2-w_p\|\mathbf e_t\|^2
-w_{peak}\,[F_{raw}-F_{safe}]_+^2
-w_{sat}c_{sat}
-w_a\|\Delta\mathbf w\|^2
-w_{rate}\|\Delta\mathbf w_t-\Delta\mathbf w_{t-1}\|^2.
\]

训练曲线中的 return 不能替代 N、mm、Nm 等物理指标。最终仍使用第 04 章预先定义的 gate。
每个 reward term 应单独记录，防止一个大权重掩盖其他行为。

## 6. SAC 还是 TD3

- TD3 与原始 Residual RL 工作一致，deterministic actor 便于部署；
- SAC 的 entropy exploration 常在连续控制中更稳健；
- 算法选择的重要性低于 action bounds、domain randomization、reward、safety layer 和多 seed
  统计。

至少使用 5 个 training seeds，报告均值和 95% confidence interval。只挑一条最好曲线没有
统计意义。

## 7. Domain randomization 和数据拆分

```text
train seeds       policy optimization
validation seeds  reward/gain/model selection
blind holdout      final claim, evaluate once
```

随机化变量沿用压力测试，但 train range 与真实可能范围必须有物理依据。过宽会让策略保守，
过窄会 overfit simulator。seed 29 已在开发中多次查看，严格论文流程中应把它视为 validation，
再生成新的 blind holdout。

## 8. Safety envelope

RL 输出进入机器人前至少经过：

1. action clamp；
2. residual low-pass 和 rate limit；
3. total normal wrench clamp；
4. joint torque 和 torque-rate limit；
5. raw force / joint limit / non-finite termination；
6. inference deadline watchdog；
7. 异常时 residual 置零，由 classical controller safe hold/retreat。

“训练时惩罚大力”不是 hard safety guarantee。真机训练还需要独立安全监控和人工急停。

## 9. 必须做的对比和消融

| 方法 | 回答的问题 |
|---|---|
| impact-aware fixed hybrid | 强 nominal baseline 到什么水平？ |
| adaptive admittance / gain scheduling | 不用 RL 能否解决？ |
| end-to-end RL | classical prior 是否真的提高 sample efficiency？ |
| hybrid + residual RL | proposed method 是否达到 gate？ |
| no domain randomization | 泛化提升来自哪里？ |
| zero residual | policy failure 时 baseline 是否仍可用？ |

成功条件不是“return 更高”，而是 blind holdout pass rate >= 90%，且不新增 force/torque
violation。否则应诚实报告失败并分析原因。

## 10. 推荐实施顺序

1. 把 MuJoCo trial 包成 Gymnasium environment，但复用现有 controller/simulation，不复制公式；
2. 先实现 action=0 的环境回归测试，指标必须等于 classical baseline；
3. 加 action bounds/safety filter 的单元测试；
4. 小规模 train/validation smoke run；
5. 多 seed 正式训练；
6. 冻结 policy，再运行 blind holdout；
7. 导出 ONNX/TorchScript 前后做 inference parity；
8. 最后才考虑 ROS 2/真机部署。
