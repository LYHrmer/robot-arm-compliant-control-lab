# 06｜从会运行到能独立设计：练习、排错与面试表达

## 1. 分级练习

### Level 1：运动学入门

1. 写有限差分 Jacobian test，并避开 (q_2=0,\pi)；
2. 画 2-DOF 工作空间与 elbow-up/elbow-down 两支 IK；
3. 证明 (det J=l_1l_2\sin q_2)；
4. 用虚功推导 (	au=J^Tw)。

验收：测试可重复，能解释 rad/m/N/Nm 单位，不只贴代码输出。

### Level 2：控制器实现

1. 从零实现 2-DOF impedance；
2. 用半隐式 Euler 实现 admittance，并比较显式 Euler；
3. 给 hybrid 增加 anti-windup 单元测试；
4. 改 contact transition time，只通过 full-trial peak 和 contact ratio 判断结果。

验收：每项改动有假设、单变量实验和回归测试。

### Level 3：7-DOF 数值与工程

1. 扫描 null-space damping 并画 `||J N||`；
2. 检查每个 Eigen fixed-size 矩阵的维度；
3. 给 C++ controller 增加新参数，同时保持 Python/C++ parity；
4. 增加 torque-rate limit，并区分 clipping 与 rate saturation；
5. 用 Pinocchio 交叉验证 Jacobian 和 bias。

验收：CTest、pytest、Ruff、parity 和 MuJoCo closed-loop 全部通过。

### Level 4：研究扩展

1. 做 adaptive admittance baseline；
2. 实现 bounded Residual RL Gymnasium environment；
3. train/validation/blind-test 三分离；
4. 完成 fixed/adaptive/end-to-end/residual/no-randomization/zero-residual 消融；
5. 报告多 seed confidence interval 和所有 safety violations。

验收：预注册 gate 不因结果修改，失败 case 留在仓库。

## 2. 常见故障定位

| 现象 | 优先检查 | 原因示例 |
|---|---|---|
| 一接触就产生巨大峰值 | full-trial raw force、接近速度、状态切换 | 自由空间直接 force control、blend 太短 |
| force RMSE 很低但轨迹漂移 | tangential RMSE、bias scale、friction | 法向目标达成，切向扰动未补偿 |
| 只有 noisy-delay 饱和 | phase lag、integrator、derivative gain | 延迟导致过补偿和 wind-up |
| posture controller 扰动末端 | `||J N||`、damping、posture gain | 阻尼 projector 非严格 null space |
| Python 正常、C++ 异常 | state reset、单位、矩阵布局、parity sequence | rotation row/column convention 不一致 |
| RL return 上升但峰值恶化 | reward term 与 physical metrics | reward scale 掩盖 safety cost |
| 仿真好、真机振荡 | delay、sensor bandwidth、contact stiffness | sim-to-real contact gap、未做 rate limit |

排错顺序应从 measurement/units -> math primitive -> controller output -> torque mapping -> contact
dynamics，而不是一上来同时改十个 gain。

## 3. 高频面试问题

### 阻抗和导纳的本质区别是什么？

阻抗把 motion error 映射为 wrench，适合可直接控制 torque/wrench 的系统；导纳把 measured
force 映射为 motion reference，常放在稳定的位置内环外面。两者都能产生柔顺行为，但因果
方向、内部状态和适用执行器接口不同。

### 为什么 hybrid control 要用 selection projector？

同一 Cartesian 方向不能独立同时强制精确位置和精确接触力，否则约束可能冲突。用
(S_f=nn^T) 在法向控力，用 (S_p=I-S_f) 在切向控位置，明确分离任务子空间。

### 为什么 torque 是 (J^Tw)，不是 (J^{-1}w)？

由虚功/功率对偶得到。Jacobian 把 joint velocity 映射为 task velocity，它的转置把 task
wrench 映射为 generalized force；不要求 J 为方阵或可逆。

### 阻尼伪逆解决什么问题？

接近奇异位形时小奇异值会放大噪声和命令。阻尼最小二乘将增益限制为
(\sigma/(\sigma^2+\lambda^2))，提高数值稳定性，但引入 task/null-space 泄漏。

### 如何证明 controller 是实时可部署的？

先区分“算法快”和“整个系统实时”。本项目 C++ update 使用 fixed-size Eigen、无日志/锁/显式
动态容器，并有数值 parity；但真正 hard real-time 还要验证 scheduler、memory allocation、
hardware interface、deadline、watchdog、torque-rate limit。当前仓库不声称已完成真机认证。

### 为什么考虑 Residual RL？

不是 nominal 做不动，而是 impact-aware fixed controller 在随机 contact/friction/model
mismatch holdout 只通过 25%。Residual policy 只补偿这一 remainder，经典 controller 保留
稳定结构和 fallback；同时必须和 adaptive classical baseline 公平比较。

## 4. 30 秒项目介绍

> 我做了一个 Franka 7-DOF 柔顺接触控制实验平台，用 MuJoCo 比较 6D 阻抗、导纳和力位
> 混合控制。控制器通过 Jacobian transpose 映射到关节力矩，包含 bias compensation、阻尼
> null-space posture、接触状态机和安全指标。我还写了 C++17/Eigen 等价核心，用逐分量
> parity test 防止 Python 与部署实现漂移。随机 contact/model mismatch holdout 暴露了固定
> gain 的失败，因此下一步是 bounded Residual RL 与 adaptive admittance 的公平对比。

这段介绍包含问题、方法、工程验证、真实失败和下一步，而不是堆技术名词。

## 5. 简历表述示例

中文：

> 构建 Franka Panda 7-DOF MuJoCo 柔顺接触控制平台，实现 6D 阻抗、导纳与力位混合控制、
> Jacobian-transpose torque mapping、bias/null-space compensation 和接触状态机；开发
> C++17/Eigen 控制核心及 Python/C++ 1e-12 数值一致性测试，并以 24 个随机未见工况量化
> force tracking、raw impact、trajectory error 与 saturation。

English:

> Built a reproducible MuJoCo contact-control lab for the 7-DoF Franka Panda, implementing 6D
> impedance, admittance, hybrid force-position control, Jacobian-transpose torque mapping,
> bias/null-space compensation, and contact-phase transitions. Developed a C++17/Eigen controller
> core with component-wise Python parity tests and evaluated robustness on 24 randomized holdout
> contact/model-mismatch cases using physical tracking and safety metrics.

只写已经完成的内容。Residual RL 在训练与 holdout 改进完成前，应写成“designed/evaluating”，
不能写成“improved”。

## 6. 真正的“精通”检查表

- [ ] 能从 FK 推到 Jacobian，并解释 singular values；
- [ ] 能从虚功推导 (J^Tw)；
- [ ] 能从连续 admittance 写出离散 update；
- [ ] 能说明 anti-windup、hysteresis 和 smooth transition 的必要性；
- [ ] 能解释 orientation error 的 frame 与适用范围；
- [ ] 能手工检查 6x7、6x6、7x7 矩阵维度；
- [ ] 能解释 `solve`/LDLT 比 `inverse` 更合理；
- [ ] 能区分 tracking、safety、latency 三类指标；
- [ ] 能设计 train/validation/blind holdout；
- [ ] 能说明仿真结果为什么不等于真机安全保证；
- [ ] 能给 Python/C++/ROS 2 划分清晰接口；
- [ ] 能保留失败结果并提出可证伪的下一步实验。
