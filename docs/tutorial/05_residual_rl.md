# 05｜Residual RL：何时值得做，怎样不破坏经典控制器

本章同时覆盖设计规范和 v0.4 的实际结果。仓库已经实现自适应经典基线、带安全包络的
Residual RL、ARS 训练器和同一 24-case 比较；结果表明策略学到了切向补偿，但尚未达到
部署门槛。完整公式与结果见
[`adaptive_residual_rl.md`](../adaptive_residual_rl.md)。

## 1. 为什么不让 RL 直接输出 torque

当前系统已经有可靠的：

- kinematics/Jacobian；
- Cartesian impedance、force PI 和 contact state machine；
- gravity/Coriolis bias compensation；
- null-space posture；
- torque limits 和 zero-residual fallback 接口。

RL 若直接输出 7 维 torque，会重新学习这些已知结构，增大样本量、调试和安全难度。
Residual RL 保留 nominal controller，只学习未建模 remainder：

$$
\mathbf u=\pi_{classical}(\mathbf s)+\pi_\theta(\mathbf s).
$$

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

v0.4 加入了 bias/contact-stiffness/force-rate estimator 和 gain scheduling 组成的
`adaptive_hybrid`。它把切向 P95 从 21.98 mm 降到 18.42 mm，但仍只有 6/24 case 通过，
说明“不用 RL 的强基线”本身也必须保留失败证据。

## 3. 推荐动作空间：有界 Cartesian residual

先只修正平移 wrench，姿态仍由 classical impedance 控制：

$$
\Delta\mathbf w=
[\Delta F_n,\Delta F_y,\Delta F_z,0,0,0]^T,
$$

$$
\mathbf w_{cmd}=\mathbf w_{hybrid}+S\,\mathrm{clip}(\Delta\mathbf w).
$$

v0.4 冻结 bounds：

- normal residual：+/-4 N；
- 每个 tangential residual：+/-6 N；
- residual rate 另行限幅并低通。

为什么不先学习 stiffness matrix？学习 gain 更容易保持物理可解释性，但正定性、阻尼关系和
快速变化都要额外约束。当前版本先用 bounded wrench residual，并在经典 contact blend 完成
100 ms 后才启用；丢失接触、非有限输出或超时会立即归零。

## 4. Observation

v0.4 的策略频率是 50 Hz，经典控制器仍为 500 Hz。输入是固定顺序的 14 个量：

```text
法向力误差、校正后的法向力、滤波力变化率
法向位置误差、法向速度误差
y/z 位置误差、y/z 速度误差
目标法向力、接触 blend、上一次三维归一化动作
```

这里没有关节位置/速度、原始接触力、墙面摩擦或墙面法向等仿真特权量。每一项使用源码中
固定的物理尺度归一化，最后截到 `[-3, 3]`。v0.5 额外加入六个方向的归一化力矩余量，
输入随之变为 20 维；checkpoint 会保存字段名和顺序，错序时直接拒绝加载。

## 5. 当前训练 cost

ARS 最小化每个 rollout 的物理 cost：

$$
C=
\left(\frac{e_F}{2}\right)^2+
0.75\left(\frac{F_{peak}}{35}\right)^2+
\left(\frac{e_t}{15}\right)^2+
2\left(\frac{s_{sat}}{1}\right)^2+
4\left(\frac{[95-r_c]_+}{5}\right)^2+
0.05\left(\frac{\Delta F_{rms}}{5}\right)^2.
$$

`e_F`、`F_peak`、`e_t`、`s_sat`、`r_c` 分别对应力 RMSE、完整轨迹原始峰值、切向
RMSE、饱和率和接触率。当前 cost 没有 residual-rate 项；动作变化率由控制器硬限幅。
选 checkpoint 可以用开发集 cost，最终判定仍逐 case 检查 N、mm 和饱和率门槛。

## 6. 为什么先用 ARS

v0.4 只有 45 个策略参数：`3×14` 权重加 3 个 bias。ARS 不需要神经网络框架，正负扰动
可复用同一组仿真噪声，checkpoint 也能直接检查。单次公开训练把 rollout cost 从 2.0221
降到 1.6744，但一个 seed 不能回答稳定性问题。

v0.5 冻结了五个 ARS seed 和新的 48-case first-reveal set。五个策略全部未达到 44/48，
raw peak 是最大的失败来源。揭盲后的基线重放同时找到了入触初段和擦拭阶段的峰值，
见[事件诊断](../contact_event_diagnosis.md)。下一轮先分段检查 classical path，再比较
非线性策略的收益；稳定接触 100 ms 后才启用 residual 的规则不足以解释全部峰值。

## 7. Domain randomization 和数据拆分

```text
train seeds       policy optimization
validation seeds  reward/gain/model selection
blind holdout      final claim, evaluate once
```

随机化变量沿用压力测试，但 train range 与真实可能范围必须有物理依据。过宽会让策略保守，
过窄会 overfit simulator。seed 29 已在开发中多次查看，v0.4 把它准确标为 frozen
public holdout，而不是 blind holdout。训练使用 seed 101 和独立 simulation seeds。

## 8. Safety envelope

v0.4 的实际执行顺序是：

1. action clamp；
2. residual low-pass 和 rate limit；
3. total normal wrench clamp；
4. contact gate 和 force-overshoot guard；
5. non-finite、异常和 inference deadline 时 residual 立即归零；
6. 仿真适配器最后做 joint torque clamp。

v0.4 没有 torque-rate limiter、joint-limit termination 或自动 safe retreat，不能在文档中把
这些写成已经完成。v0.5 把 bias/null-space torque、Jacobian 和 actuator limits 放进数值
actuation context，名义 wrench 与 residual 都在映射到关节力矩后做 ray projection，并保留
10% 余量。它仍不是 Franka 真机安全认证；真机还需要碰撞阈值、状态机、watchdog 和急停。

## 9. 必须做的对比和消融

| 方法 | 回答的问题 |
|---|---|
| impact-aware fixed hybrid | 强 nominal baseline 到什么水平？ |
| adaptive admittance / gain scheduling | 不用 RL 能否解决？ |
| end-to-end RL | classical prior 是否真的提高 sample efficiency？ |
| hybrid + residual RL | proposed method 是否达到 gate？ |
| no domain randomization | 泛化提升来自哪里？ |
| zero residual | policy failure 时 baseline 是否仍可用？ |

实际同 case 结果为 fixed `6/24`、adaptive `6/24`、bounded Residual RL `7/24`。Residual RL
把切向 P95 降到 14.80 mm，但 raw peak P95 仍为 57.04 N，最差力矩饱和升到 15.91%。
它没有达到 22/24 的门槛，也不能用于真机。

v0.5 first reveal 中，torque-safe adaptive 的 saturation 是 0%，五个 torque-projected
residual 也都是 0%；但它们只通过 22、25、26、24、25/48，raw peak P95 均为 59.54 N。
这说明 joint-torque safety layer 按设计工作，任务级 gate 仍然失败。

## 10. 推荐实施顺序

1. 已完成：action=0 与 adaptive baseline 的 `1e-12` 回归测试；
2. 已完成：action bounds、rate limit、contact gate、force guard 和 watchdog 单元测试；
3. 已完成：独立训练场景上的 ARS rollout 和冻结 checkpoint；
4. 已完成：同一 seed-29 24-case public holdout 对比；
5. 已完成：五个 training seeds、开发集 checkpoint selection 和新的 48-case first reveal；
6. 已完成：torque-headroom observation、名义 wrench 投影和 residual 二次投影；
7. 待完成：记录 peak 时间/接触相位，降低名义接近与切换阶段的冲击；
8. 待完成：用新的未来 round 复测，再决定是否做 SAC/TD3 公平消融；
9. 上述门槛通过后才考虑 ROS 2/真机部署。
