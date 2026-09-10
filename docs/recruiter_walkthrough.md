# 招聘方 5 分钟走查

这一页只回答三个证据问题：当前控制做到了什么并付出什么代价、学习方法有没有超过解析基线、
失败有没有被藏起来。每个数字都能在仓库的公开产物里找到；四个 development-test case 已经公开，
不是盲测。安装步骤不在这里重复，见[首页安装](../README.md#安装后快速复核)。

## 1. 任务与范围（约 30 秒）

Franka Panda 在 MuJoCo 中沿表面擦拭，同时维持 12 N 法向接触力。经典控制器运行在 500 Hz，
学习策略每 20 ms 输出一次三轴请求；请求还要经过接触门控、滤波、变化率限制和关节力矩包络投影。
没有真机部署，也没有 ROS 2 / Franka hardware adapter。

三类结果的数据身份不同，不能互相覆盖：在线补偿用**原有 24 case、4.5 秒**回归；BC/PPO 用
**12 秒任务的 4 个开发测试 case**；`FAIL` 的首次揭盲属于**冻结 v0.5 的 48 case**。

## 2. 三个证据问题（约 3 分钟）

### 问题一：当前控制结果好在哪，代价是什么

[在线负载补偿](online_compensation.md)按测量到的运动误差在线调整等效切向负载系数，保留固定
前馈同样的 6 N 补偿上限。原有 24-case、4.5 秒回归中，切向 RMSE 中位数从 1.885 mm 降到
**1.359 mm**，24 个配对 case 全部改善（[96 次运行](../results/franka_online_compensation_regression/)）。

代价写在同一页：另一组 12 秒、80 次误差对照里，在线方法降低了全部 20 个配对运行的全程切向
误差，但 42 条阶段检查只通过 40 条，两条失败都是反向加速时的姿态误差增量超过 0.1°。
[逐阶段结果](../results/franka_online_compensation_errors/phase_metrics.csv)没有删掉这两条。
把全部方法统一改成 2 倍旋转刚度、 $\sqrt{2}$ 倍旋转阻尼后，在线方法阶段通过
**42/42**（[增益对照](rotation_gain_comparison.md)），[原 24-case 的 192 次回归](rotation_gain_public24.md)
也显示姿态 RMSE 中位数由 0.464° 降到 0.248°；但切向 RMSE 中位数从 1.359 升到 1.409 mm，切向
速度误差在全部 case 中增加，所以这只是姿态优先任务的可选预设，默认配置不替换。

剩余误差也有进一步诊断：[跨方向回归](cross_surface_regression.md)显示组合误差后段仍有约
3.25–3.33 mm 切向 RMSE，[单因素诊断](combined_residual_diagnosis.md)进一步指出该段
95.6%–98.0% 的采样触发切向补偿幅值限制，主要表现为沿轨迹滞后，属于输入反事实分析，不是
算法改进。后续[6 N／8 N 预算对照](compensation_budget.md)在六组相同输入配对中将后段
误差从 3.26–4.20 mm 降到 1.22–1.47 mm，姿态和控力代价均通过预定检查。它增加了补偿
预算。后续[跨工况和两档姿态增益检查](budget_transfer.md)中，动态预算筛查均通过 6/6，
但原 public24 仅 23/48 兼容，默认上限仍为 6 N。
[八次内部观测复现](onset_observer.md)进一步检查了速度代价：评估开头的位置滞后驱动系数增长，
速度项的窗口均值则起抵消作用。[时间系数加倍](velocity_time.md)的四次新对照降低了速度代价，
但 8 N 追赶略慢，高增益仍未过原门槛，默认参数保持不变。

### 问题二：学习方法有没有超过解析基线

24 个 case 按物理任务组分为 16 train / 4 validation / 4 development test。三个训练种子重复运行
的是同四个开发 case，所以表中的 `4/4` 不能累加成 12 个独立场景。

| 方法（12 秒任务） | 每个种子的跟踪通过 | 开发测试平均切向 RMSE [mm] |
|---|---:|---:|
| 解析摩擦前馈 | 4/4 | 2.403 |
| BC，屏蔽上一步残差，seed 11/29/47 | 4/4 | 2.818 / 2.825 / 2.781 |
| Bounded PPO，验证集选 checkpoint，seed 11/29/47 | 4/4 | 2.467 / 2.470 / 2.381 |

三行使用相同开发 case 和相同门槛，但名义控制器不同：BC 模仿摩擦补偿，PPO 在已带摩擦前馈的
控制器上学残差，网络动作不能直接互换。结论是两者都没有相对 2.403 mm 一致改善，项目继续保留
经典控制器作为首选。[BC 输入消融](bc_closed_loop_transfer.md)记录了首轮闭环偏移怎么修好、
仍未超过解析前馈；原始表格与选择记录在 [BC 归档](../results/franka_surface_bc_transfer/)和
[BC/PPO 首轮归档](../results/franka_surface_learning_pilot/)。

还有一组容易混淆的 PPO 结果：它固定比较第 32 回合的 fresh PPO 与 BC 隐藏层初始化，三个验证
配对差值为 `+0.039、+0.063、−0.092 mm`，只有一个种子改善，冻结规则选择 `fresh_ep32`。这组
比较不包含首轮试验中 seed 47 的第 16 回合选模成绩，见[迁移对照](bc_to_residual_rl.md)与
[迁移归档](../results/franka_surface_ppo_transfer/)。

顺着公式找代码：

| 问题 | 公式和协议 | 源码 |
|---|---|---|
| 补偿系数如何更新，什么时候冻结 | [在线负载补偿](online_compensation.md) | [`tangential_compensation.py`](../src/compliant_control_lab/tangential_compensation.py) |
| 怎样验证姿态调参，避免不公平比较 | [同增益对照与代价](rotation_gain_comparison.md) | [`surface_control.py`](../src/compliant_control_lab/surface_control.py) |
| 状态、动作和数据怎么定义 | [表面学习任务](surface_learning.md) | [`surface_env.py`](../src/compliant_control_lab/surface_env.py)、[`surface_dataset.py`](../src/compliant_control_lab/surface_dataset.py) |
| BC 学什么，PPO 的 clipped loss 与变时长 GAE | [BC 输入消融](bc_closed_loop_transfer.md)、[小规模 BC/PPO](surface_learning_pilot.md) | [`train_surface_bc.py`](../tools/train_surface_bc.py)、[`train_surface_ppo.py`](../tools/train_surface_ppo.py) |
| 有界残差怎样进入力矩安全层 | [torque-safe residual](torque_safe_residual_v0.5.md) | [`residual_rl.py`](../src/compliant_control_lab/residual_rl.py)、[`franka_torque_safety.py`](../src/compliant_control_lab/franka_torque_safety.py) |
| C++ 控制链怎么接输入、处理过期数据 | [C++ 控制核心](cpp_core.md) | [`surface_control.cpp`](../cpp/src/surface_control.cpp) |

### 问题三：失败有没有被藏起来

冻结 v0.5 的首次揭盲要求五个 residual 策略分别通过 44/48，实际是 fixed hybrid 17/48、adaptive
hybrid 23/48、torque-safe adaptive 24/48、五个 torque residual 22–26/48，主结果 `FAIL`。
Torque projection 把最差 actuator saturation 从 20.71% 降到 0.00%，但完整轨迹的 raw-force
peak P95 仍是 59.54 N，超过 35 N gate。原始 384 行数据在
[comparison.csv](../results/franka_safety_blind/comparison.csv)，冻结条件和哈希在
[v0.5 protocol](reproduction_plan_v0.5.md)。软件版本现在是 `0.5.1`，不改变这轮实验的 v0.5 身份。

仓库保留这个结论，不部署策略：揭盲后的配对分析、gate 敏感性与
[接触峰值诊断](contact_event_diagnosis.md)只用已公开数据解释失败来源，不算新一轮 blind
evaluation，也不回写冻结结论。同样保留的还有在线补偿那两条姿态阶段失败、积分方法在 12 秒
实验中的两条阶段失败，以及组合误差后段约 3.3 mm 的切向残差。

## 3. 跑最短检查（约 1 分钟）

按[首页安装](../README.md#安装后快速复核)装好后运行 `franka-smoke`，预期同时出现：

```text
archive: PASS (384 rows, frozen_decision=FAIL)
simulation: PASS (safe_adaptive_hybrid/nominal, steps=1000, ...)
smoke: PASS
```

`simulation` 只跑 2 秒 nominal 仿真；`archive: PASS` 说明旧归档通过完整性检查，括号里的
`frozen_decision=FAIL` 才是实验结论。这条命令不训练 BC/PPO，也不重跑 48-case 揭盲。

误差实验与三个学习归档各有离线复核入口：

```bash
python -m tools.audit_online_compensation_errors --audit results/franka_online_compensation_errors
python -m tools.audit_online_compensation_errors --audit results/franka_rotation_gain_comparison
python -m tools.publish_surface_learning_pilot --audit results/franka_surface_learning_pilot
python -m tools.publish_surface_bc_transfer --audit results/franka_surface_bc_transfer
python -m tools.publish_surface_ppo_transfer --audit results/franka_surface_ppo_transfer
```

它们校验分发文件、重算选择或代表轨迹指标，不重新训练，也不重新积分全部动力学。
本页提到的预算转移、速度代价分解、内部观测和时间系数对照，还需运行：

```bash
python -m tools.audit_velocity_evidence
```

四项都应返回 `audit_status: PASS`，但预算转移仍为实验 `FAIL`，时间系数对照仍为
`do_not_expand`。命令不会新跑仿真，运行时间取决于读取和重算归档的速度；不计入上面的
最短冒烟检查。字段含义、覆盖范围和超时设置见[复核说明](velocity_evidence_audit.md)。

想直接看动作和指标，可打开[12 秒擦拭视频](../results/franka_tangential_demo/demo.mp4)；它来自
保存的仿真日志，不是真机录像。

## 4. 范围有没有说过头（约 30 秒）

当前证据限于 MuJoCo 和公开开发域。命令有界不等于接触力有界；4/4 通过不证明未知表面泛化。
公开学习归档只附少量代表性完整轨迹，其余原始轨迹留在本地源产物中。

[C++ 表面控制核心](cpp_core.md)已在四份完整仿真轨迹、24,000 个周期上逐步核验，
[最大 wrench 分量误差小于 4.45e-15](../results/franka_online_cpp_replay/report.json)，新增益的
[另四份轨迹](../results/franka_rotation_gain_cpp_replay/report.json)同样通过。这验证的是数值
移植，不包括传感器驱动、机器人模型计算，也不等同于真机实时性测试。仓库没有硬件 safety 或
passivity 证明。逐条主张对应的实现、测试与产物见[验证矩阵](verification_matrix.md)，版本顺序
见[实验记录](experiments/README.md)。
