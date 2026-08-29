# 05｜Residual RL：何时值得做，怎样不破坏经典控制器

本章同时覆盖设计规范和 v0.4 的实际结果。仓库已经实现自适应经典基线、带安全包络的
Residual RL、ARS 训练器和同一 24-case 比较；结果表明策略学到了切向补偿，但尚未达到
部署门槛。完整公式与结果见
[`adaptive_residual_rl.md`](../adaptive_residual_rl.md)。

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

因此 v0.4 加入了 bias/contact-stiffness/force-rate estimator 和 gain scheduling 组成的
`adaptive_hybrid`。它把切向 P95 从 21.98 mm 降到 18.42 mm，但仍只有 6/24 case 通过，
说明“不用 RL 的强基线”本身也必须保留失败证据。

## 3. 推荐动作空间：有界 Cartesian residual

先只修正平移 wrench，姿态仍由 classical impedance 控制：

\[
\Delta\mathbf w=
[\Delta F_n,\Delta F_y,\Delta F_z,0,0,0]^T,
\]

\[
\mathbf w_{cmd}=\mathbf w_{hybrid}+S\,\mathrm{clip}(\Delta\mathbf w).
\]

v0.4 冻结 bounds：

- normal residual：+/-4 N；
- 每个 tangential residual：+/-6 N；
- residual rate 另行限幅并低通。

为什么不先学习 stiffness matrix？学习 gain 更容易保持物理可解释性，但正定性、阻尼关系和
快速变化都要额外约束。当前版本先用 bounded wrench residual，并在经典 contact blend 完成
100 ms 后才启用；丢失接触、非有限输出或超时会立即归零。

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

当前仓库先采用 ARS 训练 `3×14` 线性 `tanh` policy。它不是 SAC/TD3 的替代结论，而是一个
无需深度学习依赖、权重可直接检查、能快速验证 residual seam 的最小 RL baseline。冻结训练
将物理 rollout cost 从 2.0221 降到 1.6744。

至少使用 5 个 training seeds，报告均值和 95% confidence interval。只挑一条最好曲线没有
统计意义。

## 7. Domain randomization 和数据拆分

```text
train seeds       policy optimization
validation seeds  reward/gain/model selection
blind holdout      final claim, evaluate once
```

随机化变量沿用压力测试，但 train range 与真实可能范围必须有物理依据。过宽会让策略保守，
过窄会 overfit simulator。seed 29 已在开发中多次查看，因此 v0.4 把它准确标为 frozen
public holdout，而不是 blind holdout。训练使用 seed 101 和独立 simulation seeds。

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

实际同 case 结果为 fixed `6/24`、adaptive `6/24`、bounded Residual RL `7/24`。Residual RL
把切向 P95 降到 14.80 mm，但 raw peak P95 仍为 57.04 N，最差力矩饱和升到 15.91%。因此
它是“机制实现成功、部署判定失败”，不能写成 RL 已经解决柔顺接触。

## 10. 推荐实施顺序

1. 已完成：action=0 与 adaptive baseline 的 `1e-12` 回归测试；
2. 已完成：action bounds、rate limit、contact gate、force guard 和 watchdog 单元测试；
3. 已完成：独立训练场景上的 ARS rollout 和冻结 checkpoint；
4. 已完成：同一 seed-29 24-case public holdout 对比；
5. 待完成：加入新的真正 blind holdout 和至少五个 training seeds；
6. 待完成：把 torque headroom 纳入安全投影，而不是只在事后统计饱和；
7. 待完成：非线性 SAC/TD3 policy 与线性 ARS 的公平消融；
8. 上述门槛通过后才考虑 ROS 2/真机部署。
