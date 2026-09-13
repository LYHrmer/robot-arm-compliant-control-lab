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

目前可展示的改进是[按测得负载开放 6–8 N 补偿预算](load_budget.md)。原 public24 的两档
增益检查 48/48 通过，动态配对 12/12、增益交互 6/6 通过；组合误差后段切向 RMSE 从
3.255–3.333 mm 降到 1.465–1.581 mm。普通工况预算始终保持 6 N，与旧 6 N 的原有轨迹字段
完全相同，所以这里验证的是回退兼容性，不能改写固定 8 N 原先的 23/48。
先读[完整回归说明](load_budget.md#完整回归与推广决定)；逐项数字保存在
[42 条新增＋18 条复用候选](../results/franka_measured_budget_full/comparison.json)中。

[当前演示](../results/franka_measured_budget_demo/demo.mp4)只选一个 12 秒组合误差 case，
动作与误差、力和预算曲线同步。它增加了辅助切向力测量输入，默认控制器仍用固定 6 N。
下面说明这一步之前的对照与取舍，避免只展示最终较好的配置。

后续[27 次测量误差对照](measured_budget_robustness.md)完整回放了 162,000 拍。
组合误差采用检查 7/8 通过，辅助力幅值低估 20% 时未达门槛；下降负载后的换向也有位置代价。
这些结果保留在同一份公开归档里，项目不把它描述成对传感器误差普遍稳健的控制器。

<details>
<summary>展开：经典补偿怎样走到测得负载预算</summary>

[经典切向补偿](tangential_tracking.md)先建立积分与摩擦前馈基线。[在线负载补偿](online_compensation.md)
再按运动误差调整等效负载系数；原有 24-case 回归的切向 RMSE 中位数由 1.885 mm 降到
1.359 mm。另一组 12 秒实验保留了两条反向加速姿态失败，只通过 40/42 阶段检查。

[统一提高旋转阻抗](rotation_gain_comparison.md)后，在线阶段变为 42/42，但
[原 24-case 回归](rotation_gain_public24.md)的切向 RMSE 中位数由 1.359 mm 升到 1.409 mm，
速度误差也全部增加。[组合误差诊断](combined_residual_diagnosis.md)随后定位到
补偿幅值限制；固定 8 N 虽改善高摩擦段，[跨工况检查](budget_transfer.md)仍只有 23/48，整体
`FAIL`。内部观测与[时间系数对照](velocity_time.md)没有改变这个判定。测得负载预算沿用这些
失败约束，普通工况回到 6 N，高负载时才开放额外预算。完整顺序保留在
[实验总账](experiments/README.md)。

</details>

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

BC 隐藏层初始化 PPO 的结果也没有三个种子一致改善，冻结规则仍选择 `fresh_ep32`。详情见
[迁移对照](bc_to_residual_rl.md)；完整检查点与原始评估仍可从[迁移归档](../results/franka_surface_ppo_transfer/)读取。

顺着公式找代码：

| 问题 | 公式和协议 | 源码 |
|---|---|---|
| 补偿系数如何更新，什么时候冻结 | [在线负载补偿](online_compensation.md) | [`tangential_compensation.py`](../src/compliant_control_lab/tangential_compensation.py) |
| 预算下降和缺包怎样逐拍验证 | [带练习的状态回放](tutorial/07_measured_budget_replay.md) | [`measured_budget_validation.py`](../tools/measured_budget_validation.py) |
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

完成首页安装后，可先跑两条不生成结果文件的数值实验，分别检查 wrench 到 torque 的映射和预算下降时的限速输出：

```bash
python -m tools.tutorials.wrench_to_torque
python -m tools.tutorials.budget_drop
```

逐步答案见[实验一：误差到关节力矩](tutorial/labs/01_wrench_to_torque.md)和
[实验二：预算下降与缺包](tutorial/labs/02_budget_drop.md)；前置章节见
[教程实验目录](tutorial/README.md#带答案的数值实验)。

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

测得负载调度的原 60 条候选归档另用：

```bash
python -m tools.measured_budget_study audit results/franka_measured_budget_full
python -m tools.measured_budget_robustness audit results/franka_measured_budget_robustness
```

原归档是精简轨迹，完整的测得位置/速度系数回放仍标记为未检查。新增完整轨迹的验收与
缺包测试见[回放练习](tutorial/07_measured_budget_replay.md)。27 次鲁棒性运行是后一份独立归档，
不会把旧精简轨迹改写成完整证据。

`audit_velocity_evidence` 汇总的四项都应返回 `audit_status: PASS`，但预算转移仍为实验
`FAIL`，时间系数对照仍为 `do_not_expand`。命令不会新跑仿真，运行时间取决于读取和重算
归档的速度；不计入上面的最短冒烟检查。字段含义、覆盖范围和超时设置见
[复核说明](velocity_evidence_audit.md)。

想直接看动作和指标，可打开[12 秒擦拭视频](../results/franka_tangential_demo/demo.mp4)；它来自
保存的仿真日志，不是真机录像。

## 4. 范围有没有说过头（约 30 秒）

当前证据限于 MuJoCo 和公开开发域。命令有界不等于接触力有界；4/4 通过不证明未知表面泛化。
公开学习归档只附少量代表性完整轨迹，其余原始轨迹留在本地源产物中。

[C++ 表面控制核心](cpp_core.md)新增了带测量包的完整回放：
[27 次配对仿真＋1 份重复演示](../results/franka_measured_budget_cpp_replay/report.json)，
共 168,000 个周期，状态和命令逐步对齐，最大分量误差小于 `3.56e-15`。
原[固定预算四份报告](../results/franka_online_cpp_replay/report.json)和
[姿态新增益四份报告](../results/franka_rotation_gain_cpp_replay/report.json)仍单独保留。
这些验证的是数值移植，不包括传感器驱动、机器人模型计算，也不等同于真机实时性测试。仓库没有硬件 safety 或
passivity 证明。逐条主张对应的实现、测试与产物见[验证矩阵](verification_matrix.md)，版本顺序
见[实验记录](experiments/README.md)。
