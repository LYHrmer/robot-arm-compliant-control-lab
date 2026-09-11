# Robot Arm Compliant Control Lab

[![tests](https://github.com/LYHrmer/robot-arm-compliant-control-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/LYHrmer/robot-arm-compliant-control-lab/actions/workflows/tests.yml)

Franka Panda 7-DOF 在 MuJoCo 中沿表面擦拭，同时跟踪 12 N 法向接触力。经典控制器运行在
500 Hz；独立的 BC/PPO 实验用 50 Hz 策略叠加三轴有界请求。当前 torque-safe 表面控制链包含接触处理和
6D wrench 到 7 关节力矩的包络投影。

范围先说清楚：“有界”只描述命令接口，不保证真实接触力有界；当前没有真机部署，仓库也没有
ROS 2 / Franka hardware adapter。冻结 v0.5 的 48-case 首次揭盲结论仍然是 `FAIL`，原始归档
没有重算、没有改名，见下面的[结果与决定](#结果与决定)。

## 安装后快速复核

需要 Python 3.10+。MuJoCo 仿真不要求 ROS 2 或 Franka hardware interface。

```bash
git clone https://github.com/LYHrmer/robot-arm-compliant-control-lab.git
cd robot-arm-compliant-control-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

franka-smoke
```

`franka-smoke` 先校验 384 行冻结归档，再运行 2 s torque-safe adaptive nominal 仿真。正常
输出会同时保留“实验失败”和“工程检查通过”两个状态：

```text
archive: PASS (384 rows, frozen_decision=FAIL)
simulation: PASS (safe_adaptive_hybrid/nominal, steps=1000, ...)
smoke: PASS
```

`smoke: PASS` 不会把冻结结论改成通过，也不等于重跑 48 cases。CI 使用同一个入口。

## 结果与决定

下表列出主要结果，每行链接到对应协议与公开产物。三种实验的数据身份不同，不要互相
覆盖：**原有 24-case、4.5 秒公开回归**用于在线补偿；**12 秒擦拭任务的 4 个开发测试 case**
用于 BC/PPO；**48-case 首次揭盲**属于冻结的 v0.5，只能整轮引用。

| 证据 | 结果与取舍 |
|---|---|
| [在线补偿：24 case、4.5 s](docs/online_compensation.md)，[96 次运行](results/franka_online_compensation_regression/) | 相对固定前馈，切向 RMSE 中位数 **1.885 → 1.359 mm**，24/24 配对改善，旧三方法指标精确复现。 |
| [动态误差：80 次运行](docs/online_compensation.md#实测结果与没有通过的部分) | 原增益阶段检查 **40/42**；[可选姿态增益](docs/rotation_gain_comparison.md)达到 42/42，但[192 次回归](docs/rotation_gain_public24.md)显示切向位置与速度有代价，不替换默认增益。 |
| [跨方向 36 次回归](docs/cross_surface_regression.md)与[12 次残差诊断](docs/combined_residual_diagnosis.md) | 组合误差后段仍约 **3.3 mm**；95.6%–98.0% 的采样触发内部补偿幅值限制。移除输入因素不是算法提升。 |
| [补偿预算初筛](docs/compensation_budget.md)，12 次运行；[120 行转移检查](docs/budget_transfer.md) | 原高摩擦 RMSE **3.26–4.20 → 1.22–1.47 mm**；后续两档增益的预算筛查均为 6/6，但 public24 兼容仅 **23/48**，整体 `FAIL`。默认保持 6 N、增益 1。 |
| [测得负载调度预算初筛](docs/load_budget.md)，18 条候选＋2 个复现控制组 | 组合误差 8–12 s RMSE **3.25–3.33 → 1.47–1.58 mm**，6/6 通过；12 条普通工况验收指标与旧 6 N 相同。新增切向力测量输入，尚未补齐 48 组兼容检查，不改默认值。 |
| [BC 输入消融](docs/bc_closed_loop_transfer.md)与[bounded PPO](docs/surface_learning_pilot.md)：12 s、4 个开发 case | 解析前馈均值 **2.403 mm**；三个种子的 BC 为 2.781–2.825 mm，PPO 为 2.381–2.470 mm，均无跨种子一致优势。 |
| [冻结 v0.5：48 case](results/franka_safety_blind/summary.md) | 五个 residual 策略通过 **22–26/48**，未达到各 44/48；主结果 **FAIL**，保留[全部 384 行数据](results/franka_safety_blind/comparison.csv)，不部署策略。 |
| [C++ 完整控制链回放](results/franka_online_cpp_replay/report.json) | 4 份轨迹、24,000 周期，最大 wrench 分量误差 < `4.45e-15`；[新增益另 4 份](results/franka_rotation_gain_cpp_replay/report.json)也通过。这是数值移植证据，不是真机实时性保证。 |

关于 BC/PPO 那一行：三种方法用同一批开发 case 和同一套门槛，但名义控制器不同——BC 模仿摩擦
补偿，PPO 在已有摩擦前馈之上学残差，网络动作不能直接互换；PPO 另有一组固定第 32 回合的迁移
比较（验证差值 `+0.039、+0.063、−0.092 mm`，冻结规则保留 `fresh_ep32`），不要和验证集选模的
结果混在一起。24 个公开 case 按物理任务组固定为 16 train / 4 validation / 4 development test，
三个种子重复的是同四个 case，不能算成 12 个独立场景或新的盲测。

关于 v0.5 那一行：`Force-tracking error` 在进入稳定任务阶段后由滤波力反馈计算，`Raw peak` 取
完整轨迹的未滤波最大接触力，所以约 2 N 稳态误差与约 60 N 瞬时峰值并不矛盾。Residual 策略确实
改善了切向跟踪（同 case 配对后 34–35/48 个场景降低 tangent RMSE），torque projection 也把最差
actuator saturation 降到 0%，但完整轨迹峰值仍未过 gate。揭盲后的配对效应、gate 敏感性和接触
事件重放见[post-reveal 摘要](results/franka_safety_postreveal/summary.md)与
[接触峰值诊断](docs/contact_event_diagnosis.md)，它们不构成新的 blind evaluation，也不回写冻结
结论；冻结条件与哈希在[v0.5 protocol](docs/reproduction_plan_v0.5.md)。

学习策略只在仿真公开开发域评价。命令限幅、4/4 通过和 100% 接触率都不是硬件安全、未知表面泛化
或收敛证明。公开归档含全部候选与训练统计，但每轮只分发少量代表性完整轨迹。

## 算法与源码入口

| 想看什么 | 说明 | 实现 |
|---|---|---|
| 误差驱动的在线补偿 | 更新顺序、换向冻结、共同 6 N 上限 | [在线补偿](docs/online_compensation.md)、[`tangential_compensation.py`](src/compliant_control_lab/tangential_compensation.py) |
| 仅高负载时开放额外预算 | 测量符号、下一拍生效、当前上限 anti-windup | [负载调度初筛](docs/load_budget.md)、[`load_aware_compensation.py`](tools/load_aware_compensation.py) |
| 换向时的姿态代价 | 旋转刚度与阻尼同步缩放、同增益配对比较 | [姿态增益对照](docs/rotation_gain_comparison.md)、[`surface_control.py`](src/compliant_control_lab/surface_control.py) |
| 摩擦强基线与 BC 教师 | 有界摩擦前馈公式、输入消融与闭环偏移 | [切向补偿](docs/tangential_tracking.md)、[BC 对照](docs/bc_closed_loop_transfer.md)、[`surface_policy.py`](src/compliant_control_lab/surface_policy.py) |
| 擦拭任务与 49 维观测 | 50 Hz / 500 Hz 时序、数据分组、教师标签 | [学习任务](docs/surface_learning.md)、[`surface_env.py`](src/compliant_control_lab/surface_env.py)、[`surface_dataset.py`](src/compliant_control_lab/surface_dataset.py) |
| Bounded PPO 与残差安全边界 | clipped objective、变时长 GAE、接触门控、joint-torque headroom | [学习试验](docs/surface_learning_pilot.md)、[`residual_rl.py`](src/compliant_control_lab/residual_rl.py)、[`franka_torque_safety.py`](src/compliant_control_lab/franka_torque_safety.py) |
| Python 到 C++ 的控制链 | 表面坐标、自适应状态、接触过渡、力矩投影、输入超时 | [C++ 接口与范围](docs/cpp_core.md)、[`surface_control.cpp`](cpp/src/surface_control.cpp) |
| 500 Hz 接触状态机与经典控制器 | 阻抗／导纳／力位混合、bias 与刚度估计、参考限速 | [`franka_control.py`](src/compliant_control_lab/franka_control.py)、[`franka_adaptive.py`](src/compliant_control_lab/franka_adaptive.py)、[`franka_reference.py`](src/compliant_control_lab/franka_reference.py) |

逐条主张对应的实现、测试与产物集中在[验证矩阵](docs/verification_matrix.md)；从 2-DOF 解析
模型到当前任务的版本顺序、假设与被否定的结论集中在[实验记录](docs/experiments/README.md)；
模块边界的取舍见 [ADR](docs/adr/0001-cartesian-wrench-controller-seam.md)，数据流见
[architecture](docs/architecture.md)。

## 标称演示

![Franka hybrid force-position control](results/franka/hybrid_demo.gif)

这个 GIF 只展示 fixed hybrid 的 nominal 动作，v0.5 residual 结果见上表。想自己跑一遍，把输出
写到仓库外的新目录，不要覆盖已发布归档（重复运行请换一个新目录）：

```bash
franka-control-lab --output /tmp/compliant-control-franka-quick --gif
compliant-control-lab --output /tmp/compliant-control-planar-quick --gif
```

无显示器的 Linux 环境可加 `MUJOCO_GL=egl`。[12 秒擦拭视频](results/franka_tangential_demo/demo.mp4)
同步显示七轴日志、原始法向力和切向误差，它是已有仿真日志的可视化，不是真机演示。

## 学习与导航

| 顺序 | 动手或阅读 | 核对什么 |
|---:|---|---|
| 1 | 安装后运行 `franka-smoke` | 代码能运行，且冻结 v0.5 归档仍是 `FAIL` |
| 2 | 打开[招聘方走查](docs/recruiter_walkthrough.md) | 当前结果、对照是否公平、范围限制 |
| 3 | 读[切向补偿](docs/tangential_tracking.md)与[在线更新](docs/online_compensation.md) | 从误差来源到固定前馈，再到有界参数更新 |
| 4 | 按在线补偿页复核阶段指标 | 不只看平均误差，也看换向、测量误差和失败项 |
| 5 | 构建 [C++ 控制核心](docs/cpp_core.md)并跑 parity | 同一输入序列下状态与命令能否逐步对齐 |
| 6 | 读[表面学习任务](docs/surface_learning.md)，audit 学习归档 | 49 维观测、16/4/4 分组，BC/PPO 是否超过强基线 |

系统学习柔顺控制从[教程目录](docs/tutorial/README.md)开始，由 2-DOF 推到 Franka、在线补偿与
Residual RL；面试练习见[练习、故障定位和项目表达](docs/tutorial/06_exercises_and_interview.md)。
完整导航和术语见 [docs/README.md](docs/README.md) 与 [CONTEXT.md](CONTEXT.md)。

## 验证代码

三种检查回答不同问题，不能互相替代：`pytest` / `ctest` 检查代码行为，`franka-smoke` 是最短的
可运行性冒烟检查，`*-audit` 只离线核对已发布归档是否被改动。

```bash
pytest
ruff check src tests tools

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
pytest tests/test_cpp_parity.py
```

离线审核已发布的 v0.5 产物，不联网也不重跑仿真：

```bash
franka-published-results-audit \
  --protocol results/franka_safety_preholdout/protocol.json \
  --result results/franka_safety_blind
```

它核对固定 manifest 与 `COMPLETE` 标记、文件 hash、两路 beacon 归档、blind root 与 HMAC seed
推导、384 行 case-method 网格、policy 身份、gate 标签和 summary 通过数；不重新执行 BLS 验签或
仿真。`audit PASS` 只表示归档未变且内部推导自洽，冻结结论仍是 `FAIL`。在线补偿与三个学习归档
有各自的 audit 入口，命令列在[招聘方走查](docs/recruiter_walkthrough.md#3-跑最短检查约-1-分钟)。

近期预算与速度误差的四项实验可用一个命令只读复核，不运行新仿真：

```bash
python -m tools.audit_velocity_evidence
```

它覆盖预算转移、速度代价分解、内部观测和时间系数对照，逐项保留实验 `FAIL`／`do_not_expand`。
输出字段与失败处理见[复核说明](docs/velocity_evidence_audit.md)。它只覆盖这四项，不包含全仓库所有实验。

正式训练、公开验证和新一轮 first-reveal 命令放在[实验复现文档](docs/reproduction_plan_v0.5.md)，
以免把一次正式实验误当成快速示例。

## 当前限制

- 仿真使用理想力矩接口，没有电流环、编码器量化和真实通信抖动。
- 接触参数直接取自 MuJoCo，未做真机辨识。
- 零空间投影采用阻尼运动学形式，尚未实现 dynamically consistent operational-space control。
- C++ 只包含选定表面控制器的状态更新与 torque projection；加速度受限参考实验、policy 和训练
  尚未移植，传感器采集、机器人模型计算和真实通信仍由外部接口负责。
- torque projection 没有提供 torque-rate、碰撞阈值或硬件安全认证。
- 默认在线补偿在 12 秒对照的反向加速段仍未通过姿态阶段门槛；组合误差后段仍有约 3.3 mm 切向
  残差。[预算转移](docs/budget_transfer.md)在原 public24 仅通过 23/48，25 组因切向速度代价失败。
  [窗口分解](docs/velocity_cost.md)显示平均净速度 MSE 代价主要集中在 1.5–2.0 s，最后 1 s 的两个窗口速度 MSE 在全部配对中均降低。
  [八次内部观测复现](docs/onset_observer.md)精确匹配旧轨迹：该窗口的位置滞后驱动系数增长，速度项平均起抵消作用。
  限幅先改变本周期请求，随后阻断用于下一周期的系数正增量。
  [速度误差权重单因素对照](docs/velocity_time.md)降低了两档增益的速度代价，但高增益仍未过门槛，8 N 追赶略慢，暂不推广。
  8 N 只保留为高摩擦实验配置；默认保持 6 N、增益 1，也没有新增 8 N 的 C++ 完整控制链回放。
- 当前机器没有 Franka hardware/model interface，仓库不声称完成 ros2_control 真机插件。

## 模型与许可证

Franka 模型来自 Google DeepMind
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda)，
固定到上游 commit `da76818e269b82289eba39808e2fb91d679d6994`。模型资源使用 Apache-2.0，许可证和
修改说明保存在
[assets/franka_emika_panda](src/compliant_control_lab/assets/franka_emika_panda/UPSTREAM.md)。
本项目其余代码使用 MIT License，见 [LICENSE](LICENSE)。
