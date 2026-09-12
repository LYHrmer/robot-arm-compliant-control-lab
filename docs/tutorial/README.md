# 从 2-DOF 到 Franka 7-DOF：柔顺接触控制学习路线

每章先给控制目标和连续公式，再落到离散实现。源码入口和验证方式放在同一章，方便边读边跑。
读完一章，应该能回答下面的问题：

1. 控制目标和物理量是什么；
2. 连续公式如何离散并稳定地求解；
3. 公式对应哪段 Python/C++ 源码；
4. 用什么实验和指标证明实现没有“看起来能跑、其实算错”。

## 建议顺序

| 阶段 | 主题 | 学完应能回答 | 对应章节 |
|---|---|---|---|
| 0 | 环境与复现 | 如何一条命令复现图表和测试？ | 本页 |
| 1 | 2-DOF 解析运动学 | FK、IK、Jacobian 为什么这样写？ | [01](01_2dof_kinematics.md) |
| 2 | 柔顺控制 | 阻抗、导纳、力位混合到底差在哪？ | [02](02_compliant_control.md) |
| 3 | Franka 数值解算 | 6D wrench 如何变成 7 维 torque？阻尼伪逆怎么解？ | [03](03_franka_numerics.md) |
| 4 | 实验方法 | 怎样定义指标、压力测试和可复现实验？ | [04](04_experiments_and_validation.md) |
| 5 | Residual RL | 什么时候该加 RL，动作/奖励/安全层如何设计？ | [05](05_residual_rl.md) |
| 6 | 进阶练习与面试 | 如何从“会运行”进阶到“能解释、能扩展”？ | [06](06_exercises_and_interview.md) |
| 7 | 负载调度实战 | 预算下降、缺包和在线系数怎样逐拍核对？ | [07](07_measured_budget_replay.md) |

如果刚接触机器人控制，按 01 -> 02 -> 04 的 2-DOF 部分学习；如果已有机器人学基础，
可从 03 开始；如果目标是复现实验或准备面试，至少完整阅读 03、04 和 06。

## 读完 03 之后的实战路线

六章讲通用方法，下面用当前表面接触任务把公式、误差和实验连起来。经典控制是主线，BC 与
PPO 是在强基线之后的延伸，不需要先训练网络才能学习这个项目。

1. **先分清接触问题和跟踪问题。** 读[擦拭微分离诊断](../wiping_contact_diagnosis.md)，
   在新的输出目录比较同一 case 的原接触模型与平滑模型。检查接触率、分离持续时间、
   真实法向力和姿态代价，不能只看动作是否平滑。
   对照[仿真实现](../../src/compliant_control_lab/surface_simulation.py)与
   [接触测试](../../tests/test_surface_contact_stability.py)。

2. **把控制量放进正确坐标系。** 做[表面坐标与 F/T 练习](../surface_frame_and_sensing.md)：
   旋转墙面后，检查切向命令有没有漏入真实法向，并解释工具重力补偿与传感器偏置的区别。
   对照[传感实现](../../src/compliant_control_lab/surface_sensing.py)和
   [坐标测试](../../tests/test_surface_sensing.py)。

3. **建立能解释的经典基线。** 读[积分与固定摩擦前馈](../tangential_tracking.md)，在相同输入下
   比较不补偿、积分和前馈。回答它们各自改善什么误差、限幅后还剩什么误差。
   对照[补偿实现](../../src/compliant_control_lab/tangential_compensation.py)和
   [前馈测试](../../tests/test_tangential_feedforward.py)。

4. **跟踪一个完整的在线更新周期。** 按[在线补偿](../online_compensation.md)手算本周期输出和
   下一周期系数，解释换向、失去接触或力矩投影时为什么冻结更新。再从阶段 CSV 中找出没有
   通过的 case，不能用全程平均值覆盖它。
   对照[在线更新测试](../../tests/test_online_tangential_compensation.py)与
   [安全时序测试](../../tests/test_tangential_safety.py)。

5. **用单因素诊断验证解释。** 读[残差诊断](../combined_residual_diagnosis.md)和
   [跨方向回归](../cross_surface_regression.md)，区分补偿幅值、变化速率、力矩投影和执行器
   裁剪。自己重算沿／横轨迹误差，解释为什么“输出长度小于 6 N”不代表内部没有限幅。
   对照[分析模块](../../tools/combined_residual_diagnostics.py)和
   [观测回放测试](../../tests/test_combined_residual_observer.py)。
   再读[6 N／8 N 预算对照](../compensation_budget.md)，检查更低的窗口 RMSE 是否等于
   所有瞬时误差都通过，并区分增加控制预算与改进算法。

6. **先看 BC 的闭环，不只看训练损失。** 复核[BC 输入消融](../bc_closed_loop_transfer.md)
   的完整 49 维输入与屏蔽历史残差方案。解释教师标签、历史动作反馈和闭环分布变化，
   并核对是否真的超过解析前馈。
   对照[策略实现](../../src/compliant_control_lab/surface_policy.py)、
   [BC 测试](../../tests/test_surface_bc.py)与[实验归档](../../results/franka_surface_bc_transfer/)。

7. **最后再判断 Residual PPO 是否值得加入。** 读[小规模 BC/PPO](../surface_learning_pilot.md)
   和[隐藏层迁移](../bc_to_residual_rl.md)，写清名义控制器、残差动作、训练预算和选模规则。
   区分验证集选择的 checkpoint 与固定第 32 回合的对照，保留每个种子的结果。
   对照[PPO 测试](../../tests/test_surface_ppo.py)与
   [学习归档](../../results/franka_surface_learning_pilot/)。

完成第 5 步后，做[负载调度与完整回放练习](07_measured_budget_replay.md)，区分目标预算与
全局硬上限。先验证缺包和换向的状态更新，再进入学习方法的对照。

完成第 4 步后，就可以并行学习 [C++ 控制核心](../cpp_core.md)，用相同记录输入核对状态与
命令（[完整控制链测试](../../tests/test_cpp_surface_loop.py)）；不必等到学习算法完成。

注意数据身份：在线补偿的原有 24-case 回归为 4.5 秒；学习任务是 12 秒，24 个公开 case 按物理
任务组分为 16 train / 4 validation / 4 development test。三个训练种子重复同四个开发 case，
不是 12 个独立场景。冻结 v0.5 的 48-case 首次揭盲是另一轮实验，结论仍为 `FAIL`。

## 环境与第一轮复现

安装步骤见[首页安装说明](../../README.md#安装后快速复核)，这里不重复。装好以后先跑三种不同
性质的检查，它们回答的问题不一样：

```bash
franka-smoke   # 冒烟：代码能跑通，且冻结归档仍是 frozen_decision=FAIL
pytest         # 测试：数学、控制器、时序和协议的代码行为
```

- `franka-smoke` 只做最短可运行性检查：校验 384 行冻结归档，再跑 2 秒 nominal 仿真。它不重跑
  完整对照或训练，`smoke: PASS` 不改变冻结结论。
- `pytest` 检查代码行为，其中部分测试也会重算归档指标；通过测试不代表重新执行了整套实验。
- `*-audit` 命令是第三类检查：只离线核对已发布产物是否被改动、指标能否重算，不训练也不重新
  积分动力学。命令列表在[招聘方走查](../recruiter_walkthrough.md#3-跑最短检查约-1-分钟)。

入门示例的输出一律写到仓库外的新目录，避免覆盖 `results/` 下已发布的归档
（`results/metrics.md`、`results/franka/` 等都是提交进仓库的结果）。**重复运行时请换一个新的
目录名**，本页命令不会替你清理旧目录：

```bash
compliant-control-lab --output /tmp/compliant-control-planar-01 --gif
franka-control-lab --output /tmp/compliant-control-franka-01 --gif
```

然后编译 C++ 参考实现：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
pytest tests/test_cpp_parity.py
```

一轮完整复现应得到：

- Python 测试全部通过；
- CTest 通过；
- `tests/test_cpp_parity.py` 没有 skip，且 Python/C++ wrench 逐项一致；
- `/tmp/compliant-control-franka-01/metrics.md` 中 hybrid nominal 力 RMSE 约 1 N、接触率 100%、
  无力矩饱和，与已发布的 [`results/franka/metrics.md`](../../results/franka/metrics.md) 一致。

上面两条示例命令只跑默认标称场景，不会生成压力测试摘要。固定增益控制器在未见工况中的失败
保存在 [`results/franka_stress/summary.md`](../../results/franka_stress/summary.md)，由单独的
`franka-stress-lab` 生成，复现方式见[实验与验证](04_experiments_and_validation.md#9-复现实验)；
同样记住把它的 `--output` 指向仓库外的新目录。

## 源码导航

```text
2-DOF analytic lab
├── src/compliant_control_lab/kinematics.py
├── src/compliant_control_lab/controllers.py
└── src/compliant_control_lab/simulation.py

Franka 7-DOF lab
├── src/compliant_control_lab/franka_control.py
├── src/compliant_control_lab/franka_adaptive.py
├── src/compliant_control_lab/franka_torque_safety.py
├── src/compliant_control_lab/residual_rl.py
├── src/compliant_control_lab/franka_learning.py
├── src/compliant_control_lab/franka_safety_learning.py
├── src/compliant_control_lab/franka_simulation.py
├── src/compliant_control_lab/franka_stress.py
└── cpp/                         # C++17/Eigen 等价实现

Surface wiping task
├── src/compliant_control_lab/surface_control.py
├── src/compliant_control_lab/surface_sensing.py
├── src/compliant_control_lab/tangential_compensation.py
├── src/compliant_control_lab/surface_env.py
├── src/compliant_control_lab/surface_dataset.py
└── src/compliant_control_lab/surface_replay.py
```

核心原则是：控制器只接收状态/目标并输出 Cartesian wrench；MuJoCo、未来的 ROS 2
适配器和绘图代码不参与控制公式。这个边界让公式、仿真和部署代码可以分别验证。

## 学习完成标准

不要只以“GIF 能动”为完成。至少应能独立完成以下检查：

- 手推 2R 平面臂 FK、IK 和 Jacobian，并用有限差分验证 Jacobian；
- 解释阻抗与导纳的因果方向，以及 hybrid 中法向/切向选择矩阵；
- 解释 `solve(A, B)` 或 Eigen `LDLT.solve()` 相对显式矩阵逆的数值优势；
- 说明 `J^T w`、bias compensation 和 null-space posture torque 各自负责什么；
- 区分 filtered-force tracking RMSE 与 full-trial raw peak force；
- 说明在线补偿的更新时机、换向冻结条件和共同 6 N 上限，并指出它没有通过的阶段检查；
- 给出 Residual RL 的 nominal controller、残差动作、安全限制和零残差回退；
- 说明为什么当前 BC 与 PPO 都没有取代解析摩擦前馈；
- 改动 Python 公式后，同步修改 C++ 并让 parity test 继续通过。
