# 06｜从会运行到能独立设计：练习、排错与面试表达

## 1. 分级练习

每项标注已有实现或待完成的工作，避免把仓库已经做过的实验当成下一步成果。
先做两份带答案的小实验：[误差到七轴力矩](labs/01_wrench_to_torque.md)和
[预算下降与缺包](labs/02_budget_drop.md)。它们分别核对静态公式与逐拍状态更新，运行时不改历史数据。
随后做[换向恢复故障分析](labs/03_reversal_recovery.md)，练习用原始状态排除错误假设，
并判断通过门槛的候选是否仍有代价。

### Level 1：运动学入门

1. 已有实现，可复现：运行[有限差分 Jacobian 与 IK 测试](../../tests/test_kinematics.py)，再把测试位置移近 $q_2=0,\pi$，观察逆解的敏感性。
2. 新增练习：画 2-DOF 工作空间，标出同一位置的 elbow-up/elbow-down 两支 IK。
3. 推导练习：证明 $\det J=l_1l_2\sin q_2$，解释两个奇异位置。
4. 已有答案，可核对：用虚功推导 $\tau=J^T w$，完成[小实验一](labs/01_wrench_to_torque.md)中的手算和功率核对。

验收：测试可重复，能解释 rad/m/N/Nm 单位，不只贴代码输出。

### Level 2：控制器实现

1. 已有实现，可复现：阅读[2-DOF 控制器](../../src/compliant_control_lab/controllers.py)，先独立写出 impedance，再对照代码与单位。
2. 新增练习：在单独脚本中比较显式 Euler 与半隐式 Euler 的 admittance 更新，不替换默认控制器。
3. 新增练习：给 hybrid 构造持续饱和后解除约束的输入序列，检查积分器是否及时退出饱和。
4. 新增练习：只改 contact transition time，同时记录完整轨迹峰值与接触比例；再看过渡段跟踪，不能只凭平均力误差判断。

验收：每项改动有假设、单变量实验和回归测试。

### Level 3：7-DOF 数值与工程

1. 新增练习：扫描 null-space damping，画 `||J N||`，解释阻尼引入的泄漏。
2. 已有实现，可复现：从 [C++ 控制链](../cpp_core.md)检查 Eigen 矩阵维度，运行逐分量 parity，再回放带测量包的完整轨迹。
3. 已有实现，可复现：先手算不对称力矩区间的 wrench ray projection，再对照[数值测试](../../tests/test_franka_torque_safety.py)与 C++ 输出。
4. 已有答案，可核对：完成[小实验二](labs/02_budget_drop.md)，解释预算回到 6 N 后为什么还有几拍请求高于 6 N。
5. 尚未实现的扩展：增加关节 torque-rate limit，区分 clipping 与 rate saturation，并检查引入约束后的闭环代价。
6. 尚未实现的扩展：用 Pinocchio 交叉验证 Jacobian 和 bias；这不在当前 C++ parity 证据内。

验收：CTest、pytest、Ruff、parity 和 MuJoCo closed-loop 全部通过。

### Level 4：研究扩展

1. 已有实现，可复现：比较自适应基线、解析前馈与在线补偿，先解释[原公开 24-case 回归](../online_compensation.md)的配对结果。
2. 已有实现，可复现：阅读[49 维学习任务](../surface_learning.md)，运行零残差，再核对 BC 教师与 PPO 名义控制器的区别。
3. 已有历史记录，可复核：对照[旧 v0.5 首次揭盲](../../results/franka_safety_blind/summary.md)和当前按物理任务组划分的 16/4/4 个公开 case，说明为什么后者不能叫新盲测，也不是训练 rollout 的计数。
4. 已有小规模实验，可复现：检查[换向恢复候选](../reversal_recovery.md)的四对比较，保留“高负载下短暂停顿后恢复”的代价。下一轮算法工作仍限定为换向恢复的更广回归，不改默认值、不扩展 RL。
5. 尚未实现的扩展：补齐包含 end-to-end 与 no-randomization 的统一消融。已有零残差对照不能替代这套完整比较，也不因增加训练规模就有必要优先做它。

验收：运行前固定配对工况和采用门槛，报告所有失败。两个噪声种子不算两个独立物理场景。
BC/PPO 保持研究支线，本轮不扩大训练。

## 2. 常见故障定位

| 现象 | 优先检查 | 原因示例 |
|---|---|---|
| 一接触就产生巨大峰值 | full-trial raw force、接近速度、状态切换 | 自由空间直接 force control、blend 太短 |
| force RMSE 很低但轨迹漂移 | tangential RMSE、bias scale、friction | 法向目标达成，切向扰动未补偿 |
| 只有 noisy-delay 饱和 | phase lag、integrator、derivative gain | 延迟导致过补偿和 wind-up |
| posture controller 扰动末端 | `||J N||`、damping、posture gain | 阻尼 projector 非严格 null space |
| Python 正常、C++ 异常 | state reset、单位、矩阵布局、parity sequence | rotation row/column convention 不一致 |
| RL return 上升但峰值恶化 | reward term 与 physical metrics | reward scale 掩盖 safety cost |
| 峰值只归因于首次接触 | peak 距首次接触时间、contact phase、motion phase | 擦拭过程仍可能产生更晚的全局峰值 |
| 降载换向后，最后两秒正常但恢复段误差较大 | 分别查看 7–8 s、8–10 s、10–12 s；对照旧系数、运动确认和补偿向量 | 停顿时参数保留、重新确认运动、输出限速都可能影响恢复；需逐项验证 |
| 仿真好、真机振荡 | delay、sensor bandwidth、contact stiffness | sim-to-real contact gap、未做 rate limit |

最后一行是迁移时的排查建议，不代表本仓库做过真机振荡实验。
排错顺序应从 measurement/units -> math primitive -> controller output -> torque mapping -> contact
dynamics。一次同时改十个 gain 会失去定位依据。

## 3. 高频面试问题

### 阻抗和导纳怎样区分？

阻抗把 motion error 映射为 wrench，适合可直接控制 torque/wrench 的系统；导纳把 measured
force 映射为 motion reference，常放在稳定的位置内环外面。两者都能产生柔顺行为，但因果
方向、内部状态和适用执行器接口不同。

### 为什么 hybrid control 要用 selection projector？

同一 Cartesian 方向不能独立同时强制精确位置和精确接触力，否则约束可能冲突。用
$S_f=nn^T$ 在法向控力，用 $S_p=I-S_f$ 在切向控位置，明确分离任务子空间。

### 为什么 torque 使用 $J^Tw$？

由虚功/功率对偶得到。Jacobian 把 joint velocity 映射为 task velocity，它的转置把 task
wrench 映射为 generalized force；不要求 J 为方阵或可逆。

### 阻尼伪逆解决什么问题？

接近奇异位形时小奇异值会放大噪声和命令。阻尼最小二乘将增益限制为
$\sigma/(\sigma^2+\lambda^2)$，提高数值稳定性，但引入 task/null-space 泄漏。

### 如何证明 controller 是实时可部署的？

目前只能证明数值移植和本机计算计时，不能证明真机实时可部署。
C++ 已覆盖完整表面控制计算，不只固定控制器和投影：包含接触状态、在线补偿与可选测得负载调度。
当前回放核对 28 条输入轨迹、168,000 周期的状态、wrench 和七轴力矩，其中一条是重复演示。
它没有重新积分动力学，也未覆盖真实硬件通信或操作系统调度。
独立计算 benchmark 的范围和原始数据见 [C++ 说明](../cpp_core.md)。

### 为什么考虑 Residual RL？

随机 contact/model mismatch holdout 暴露了固定控制器的剩余误差。Residual policy 只补偿
这部分误差，经典控制器仍负责接触状态与 fallback；评测时和 adaptive classical baseline
使用同一批 case。

当前 49 维表面任务还要求与解析摩擦前馈强基线比较。BC/PPO 尚未跨种子一致超过该基线，
所以不能把“强化学习优于传统控制”写成项目结论。在线补偿的 24/24 改善是解析更新律的结果。

### 为什么当前 Residual RL 不能部署？

先看当前学习任务的仿真门槛，再谈硬件接口；两者都不能被训练回报代替。
以下数字属于历史 20 维任务，不是当前 49 维 BC/PPO：
v0.4 只通过 7/24，且最差 torque saturation 为 15.91%。v0.5 加了关节力矩投影后，五个
策略的 saturation 都是 0%，但首次 48-case 揭盲只通过 22、25、26、24、25 个 case，
没有一个达到 44/48。raw peak P95 仍是 59.54 N。

同 case 配对显示，residual 在 34–35/48 个 case 改善切向误差，但 force RMSE 和 raw peak
的中位数都略有增加。事件重放又找到两批 peak failure：7 个在首次接触后 0.432 s 内，
11 个在 1.222 s 以后。安全层完成了 actuator clipping 限制，接触性能仍未过 gate。

## 4. 30 秒项目介绍

> 我做的是 Franka 七轴接触擦拭仿真。自适应力位混合控制负责接触，在线切向补偿处理跟踪误差。
> 在原公开 24 个配对 case 中，切向 RMSE 中位数从 1.885 降到 1.359 mm，每个 case 都有改善。
> 完整表面控制计算已移植到 C++，用保存的输入逐拍核对。可选负载调度在测量低估和换向恢复上
> 仍有局限，所以默认保留 6 N；当前没有真机部署结论。

被追问时再打开[当前架构](../architecture.md)。C++ 回放是 28 条轨迹、168,000 周期，
包含一份重复演示；[测量回归](../measured_budget_robustness.md)是 9 场景 × 3 方法共 27 次仿真，
其中 8 个组合误差采用检查通过 7 个，幅值 ×0.8 为 `FAIL`。不要把两个分母混在一起。

## 5. 简历表述示例

中文：

> 构建 Franka Panda 7-DOF MuJoCo 柔顺接触控制平台，实现 6D 阻抗、导纳与力位混合控制、
> 接触状态机及在线切向补偿；在原公开 24-case 配对回归中全部改善切向跟踪，RMSE 中位数
> 从 1.885 降至 1.359 mm。将完整表面控制计算移植为 C++17/Eigen 核心，核对 28 条已记录轨迹
> 共 168,000 周期的状态、wrench 与七轴力矩，最大分量误差小于 3.56e-15。

English:

> Built a MuJoCo compliant-contact lab for the 7-DoF Franka Panda, adding online tangential
> compensation to adaptive hybrid control. Reduced median tangential RMSE from 1.885 to 1.359 mm
> against fixed feedforward in a public 24-case paired regression, with improvement in every case.
> Ported the stateful surface-control computation to C++17/Eigen and checked state, wrench and
> seven-joint torque across 28 recorded traces (168,000 updates, including one duplicate demo),
> with maximum component error below 3.56e-15.

这段适合控制算法岗位；若版面有限，保留在线补偿与配对比较即可。C++ 的数字是回放计算，
不能写成 168,000 周期真机闭环。可选测得负载预算的 27 次仿真只通过 7/8 个组合误差采用检查，
幅值 ×0.8 仍失败；它没有成为默认配置。

投递学习控制方向时，可另写“完成 BC 与 bounded Residual PPO 小规模仿真及强基线对照，
保留未跨种子一致超越解析前馈的结果”。历史 v0.5 首次揭盲仍为 `FAIL`，不能用当前解析方法的
改善替换它，更不能概括成 RL 优势或真机部署能力。

## 6. 自测清单

- [ ] 能从 FK 推到 Jacobian，并解释 singular values；
- [ ] 能从虚功推导 $J^Tw$；
- [ ] 能从连续 admittance 写出离散 update；
- [ ] 能说明 anti-windup、hysteresis 和 smooth transition 的必要性；
- [ ] 能解释 orientation error 的 frame 与适用范围；
- [ ] 能手工检查 6x7、6x6、7x7 矩阵维度；
- [ ] 能解释 `solve`/LDLT 比 `inverse` 更合理；
- [ ] 能区分 tracking、safety、latency 三类指标；
- [ ] 能设计 train/validation/blind holdout；
- [ ] 能说明仿真结果为什么不等于真机安全保证；
- [ ] 能给 Python/C++/ROS 2 划分清晰接口；
- [ ] 能把入触峰值和擦拭阶段峰值分开定义指标；
- [ ] 能保留失败结果并提出可证伪的下一步实验。
