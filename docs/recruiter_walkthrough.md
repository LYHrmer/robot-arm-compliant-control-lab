# 招聘方 5 分钟走查

这条路线先看当前误差补偿、学习对照与 C++ 核验，再核对旧 v0.5 的失败归档。所有数字都能在仓库的
公开产物中找到；四个 development-test case 已经公开，不是盲测。

最新的[在线负载补偿](online_compensation.md)在原有 24-case、4.5 秒回归中，将切向 RMSE
中位数从 1.885 mm 降至 1.359 mm，24 个配对 case 均改善。下方 BC/PPO 表格使用另一份
12 秒学习任务，不能直接与这个中位数比较。

另有 80 次误差对照：在线方法降低了全部 20 个配对运行的全程切向误差，
但 42 条阶段检查有 2 条未通过，均发生在反向加速时的姿态误差增量。
[逐阶段结果](../results/franka_online_compensation_errors/phase_metrics.csv)没有删掉这两条。
后续[同增益配对](rotation_gain_comparison.md)给全部方法统一调整常量姿态阻抗，
在线阶段通过 42/42；自身平均切向 RMSE 增加约 0.016 mm，默认配置暂不替换。
C++ 在四份完整仿真轨迹、24,000 个周期上逐步对齐，
[最大 wrench 误差小于 4.45e-15](../results/franka_online_cpp_replay/report.json)。
新增益的[另四份完整轨迹](../results/franka_rotation_gain_cpp_replay/report.json)也通过回放。

## 1. 先看现在做的任务（约 1 分钟）

Franka Panda 在 MuJoCo 中沿表面擦拭，同时维持 12 N 法向接触力。经典控制器运行在
500 Hz，学习策略每 20 ms 输出一次三轴请求。请求还要经过接触门控、滤波、变化率限制和
关节力矩包络投影。

24 个 case 按物理任务组分为 16 train / 4 validation / 4 development test。三个训练种子
重复运行同四个开发 case，所以表中的 `4/4` 不能累加成 12 个独立场景。

| 方法 | 每个种子的跟踪通过 | 开发测试平均切向 RMSE [mm] |
|---|---:|---:|
| 解析摩擦前馈 | 4/4 | 2.403 |
| BC，屏蔽上一步残差，seed 11/29/47 | 4/4 | 2.818 / 2.825 / 2.781 |
| Bounded PPO，验证集选 checkpoint，seed 11/29/47 | 4/4 | 2.467 / 2.470 / 2.381 |

三行使用相同开发 case 和评价门槛。BC 模仿摩擦补偿，PPO 在已经带摩擦前馈的名义控制器上
学习残差，所以二者的网络动作不能直接互换。

BC 的输入消融改善了首轮明显的闭环偏移，但仍没有超过解析前馈。PPO 的三个结果也没有
相对 2.403 mm 基线一致改善，项目因此继续保留经典控制器作为首选。原始表格和选择记录在
[BC 归档](../results/franka_surface_bc_transfer/)与
[BC/PPO 首轮归档](../results/franka_surface_learning_pilot/)。

还有一组容易混淆的 PPO 结果。它固定比较第 32 回合的 fresh PPO 和 BC-hidden-layer
初始化，三个验证配对差值为 `+0.039、+0.063、−0.092 mm`。只有一个种子改善，冻结规则
选择 `fresh_ep32`。这组结果没有使用首轮试验中 seed 47 的第 16 回合选模成绩。记录见
[迁移归档](../results/franka_surface_ppo_transfer/)。

## 2. 顺着公式找到代码（约 1 分钟）

| 问题 | 公式和协议 | 源码 |
|---|---|---|
| 补偿系数如何更新，什么时候冻结 | [在线负载补偿](online_compensation.md) | [`tangential_compensation.py`](../src/compliant_control_lab/tangential_compensation.py) |
| 怎样验证姿态调参，避免不公平比较 | [同增益对照与代价](rotation_gain_comparison.md) | [`surface_control.py`](../src/compliant_control_lab/surface_control.py) |
| C++ 控制链怎么接输入、处理过期数据 | [C++ 控制核心](cpp_core.md) | [`surface_control.cpp`](../cpp/src/surface_control.cpp) |
| 状态、动作和数据怎么定义 | [表面学习任务](surface_learning.md) | [`surface_env.py`](../src/compliant_control_lab/surface_env.py)、[`surface_dataset.py`](../src/compliant_control_lab/surface_dataset.py) |
| BC 学什么，为什么要屏蔽历史残差 | [BC 输入消融](bc_closed_loop_transfer.md) | [`surface_policy.py`](../src/compliant_control_lab/surface_policy.py)、[`train_surface_bc.py`](../tools/train_surface_bc.py) |
| PPO 的 clipped loss 和变时长 GAE | [小规模 BC/PPO](surface_learning_pilot.md) | [`train_surface_ppo.py`](../tools/train_surface_ppo.py) |
| 有界残差怎样进入力矩安全层 | [torque-safe residual](torque_safe_residual_v0.5.md) | [`residual_rl.py`](../src/compliant_control_lab/residual_rl.py)、[`franka_torque_safety.py`](../src/compliant_control_lab/franka_torque_safety.py) |

想直接看动作和指标，可打开[12 秒擦拭视频](../results/franka_tangential_demo/demo.mp4)。它来自
保存的仿真日志，不是真机录像。

## 3. 跑最短工程检查（约 1 分钟）

需要 Python 3.10+，不需要 ROS 2 或 Franka 硬件接口。安装后运行：

```bash
git clone https://github.com/LYHrmer/robot-arm-compliant-control-lab.git
cd robot-arm-compliant-control-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
franka-smoke
```

预期输出同时出现：

```text
archive: PASS (384 rows, frozen_decision=FAIL)
simulation: PASS (safe_adaptive_hybrid/nominal, steps=1000, ...)
smoke: PASS
```

`simulation` 只跑 2 秒 nominal 仿真。`archive: PASS` 表示旧归档通过完整性检查，括号里的
`frozen_decision=FAIL` 才是实验结论。这条命令不训练 BC/PPO，也不重跑 48-case 揭盲。

误差实验及三个学习归档可以分别做离线复核：

```bash
python -m tools.audit_online_compensation_errors --audit results/franka_online_compensation_errors
python -m tools.audit_online_compensation_errors --audit results/franka_rotation_gain_comparison
python -m tools.publish_surface_learning_pilot --audit results/franka_surface_learning_pilot
python -m tools.publish_surface_bc_transfer --audit results/franka_surface_bc_transfer
python -m tools.publish_surface_ppo_transfer --audit results/franka_surface_ppo_transfer
```

这些命令校验分发文件、重算选择或代表轨迹指标，不重新训练，也不重新积分全部动力学。

## 4. 核对没有藏掉的失败（约 1 分钟）

冻结 v0.5 的首次揭盲要求五个 residual 策略分别通过 44/48。实际只有 22–26/48，结论为
`FAIL`。Torque projection 把最差 actuator saturation 降到 0%，但完整轨迹 raw-force
peak P95 仍为 59.54 N，超过 35 N gate。

| v0.5 方法 | 通过数 | Raw peak P95 [N] | Saturation worst |
|---|---:|---:|---:|
| Fixed hybrid | 17/48 | 58.62 | 19.69% |
| Adaptive hybrid | 23/48 | 58.99 | 20.71% |
| Torque-safe adaptive | 24/48 | 59.54 | 0.00% |
| Torque residual，5 seeds | 22–26/48 | 59.54 | 0.00% |

原始 384 行数据在 [comparison.csv](../results/franka_safety_blind/comparison.csv)，冻结条件和
哈希在 [v0.5 protocol](reproduction_plan_v0.5.md)。软件版本目前是 `0.5.1`，不改变这轮
实验的 v0.5 身份。

## 5. 判断范围有没有说过头（约 1 分钟）

当前证据限于 MuJoCo 和公开开发域。命令有界不等于接触力有界；4/4 通过不证明未知表面
泛化。公开学习归档只附少量代表性完整轨迹，其余原始轨迹留在本地源产物中。

仓库没有 ROS 2/Franka hardware adapter，没有硬件 safety 或 passivity 证明。
在线负载补偿的默认配置在反向加速时仍未通过姿态阶段门槛。可选的新增益在相同 80 次
误差对照中通过，但尚未覆盖原 24-case 几何网格；积分方法也仍有两条阶段失败。
[C++ 表面控制器](cpp_core.md)已通过记录输入的完整序列核验；状态更新、补偿与力矩投影
属于数值控制核心，不包括传感器驱动和机器人模型计算，也不等同于真机实时性测试。
逐条主张对应的实现、测试与产物见[验证矩阵](verification_matrix.md)，版本顺序见
[实验记录](experiments/README.md)。
