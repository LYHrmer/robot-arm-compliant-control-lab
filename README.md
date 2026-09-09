# Robot Arm Compliant Control Lab

[![tests](https://github.com/LYHrmer/robot-arm-compliant-control-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/LYHrmer/robot-arm-compliant-control-lab/actions/workflows/tests.yml)

Franka Panda 7-DOF 在 MuJoCo 中沿表面擦拭，同时跟踪 12 N 法向接触力。
经典控制器运行在 500 Hz；独立的 BC/PPO 实验使用 50 Hz 策略叠加三轴有界请求。
控制请求经过接触处理和 6D wrench 到 7 关节力矩的包络投影。
“有界”指命令接口，不保证真实接触力有界。当前没有真机部署。

## 当前重点：在线补偿与误差验证

在固定摩擦前馈上，根据测量到的运动误差在线调整等效切向负载系数，保留同一 6 N
补偿上限。原有 24-case、4.5 秒公开回归中，切向 RMSE 中位数从固定前馈的
**1.885 mm 降到 1.359 mm**；全部 24 个配对 case 都有改善，并通过本轮验收检查。
旧三种方法的 72 条指标逐项复现，最大差值为 0。

[算法与状态更新](docs/online_compensation.md)解释了换向、接触丢失和力矩约束下何时停止
学习；[完整结果](results/franka_online_compensation_regression/)保留所有配对差值。
另做了 10 种误差工况 × 2 个预定噪声种子 × 4 种方法的 12 秒对照。
在线方法在全部 20 个配对运行中降低全程切向误差，但阶段检查只通过 **40/42**：
两个反向加速段的姿态误差增量超过 0.1° 门槛。高摩擦变化后也未恢复到 3 mm 阈值。
[全部 80 次结果](results/franka_online_compensation_errors/)与
[阶段失败说明](docs/online_compensation.md#实测结果与没有通过的部分)均保留，没有重新调门槛。

后续给四种方法统一使用两倍旋转刚度、 $\sqrt{2}$ 倍旋转阻尼，重跑相同的 80 次对照。
在线方法阶段通过 **42/42**，反向加速的姿态增量降至约 0.087°。
代价是在线方法的平均切向 RMSE 从 2.559 增至 2.575 mm；积分方法仍有两条阶段失败。
[增益对照与推导](docs/rotation_gain_comparison.md)说明了被否定的假设和完整配对结果。
随后完成[原 24-case 的双增益回归](docs/rotation_gain_public24.md)，共 192 次运行。
在线姿态 RMSE 中位数从 0.464° 降到 0.248°，切向位置误差则从 1.359 增到 1.409 mm，
速度误差在全部 case 中增加。因此保留为姿态优先任务的可选预设，默认及 BC/PPO 基线不变。

再将换向／组合误差扩展到三个表面方向，完成[36 次单种子回归](docs/cross_surface_regression.md)。
新增方向没有额外工程阶段失效；原 +15° 的两条姿态失败保留。组合误差后段仍有约
3.25–3.33 mm 切向 RMSE，新增益也仍有跟踪代价，因此本轮未修改控制算法。

所选 C++ 表面控制链已完成四份完整仿真轨迹的逐步核验，共 24,000 个周期，
最大 wrench 分量误差小于 `4.45e-15`。见 [C++ 回放报告](results/franka_online_cpp_replay/report.json)；
[新增益的另四份轨迹](results/franka_rotation_gain_cpp_replay/report.json)也通过 24,000 步核验。
这验证的是数值移植，不是真机部署或实时性保证。

## BC 与 Residual RL 的独立对照

24 个公开 case 按物理任务组固定为 16 train / 4 validation / 4 development test。
种子 `11 / 29 / 47` 重复的是同四个开发 case，不能算作 12 个独立场景或新的盲测。

| 方法 | 开发测试跟踪通过 | 平均切向 RMSE [mm] | 结论 |
|---|---:|---:|---|
| 解析摩擦前馈 | 4/4 | 2.403 | 这轮学习对照的强基线 |
| BC，屏蔽上一步残差，seed 11/29/47 | 各 4/4 | 2.818 / 2.825 / 2.781 | 这批 case 均通过，仍不及解析前馈 |
| Bounded PPO，验证集选 checkpoint，seed 11/29/47 | 各 4/4 | 2.467 / 2.470 / 2.381 | 没有跨种子一致改善 |

这三行使用相同开发 case 和评价门槛，但 BC 与 PPO 的名义控制器不同：BC 模仿摩擦补偿，
PPO 则在已有摩擦前馈上学习残差，网络动作不能直接互换。
这轮是 12 秒任务，不能与上方 4.5 秒回归的中位数直接比较。

BC 使用 `49 → 32 → 32 → 3` 网络模仿摩擦教师。首轮 full-49 输入虽然离线 MSE 只有
0.0012–0.0016，开发测试却只通过 `0/4、4/4、2/4`。固定训练预算后，仅屏蔽上一步施加的
三轴残差，三个种子都通过 4/4；[输入消融](docs/bc_closed_loop_transfer.md)和
[完整归档](results/franka_surface_bc_transfer/)保留了未成功的原模型及两种屏蔽方案。

PPO 在已经带摩擦前馈的强基线上学习 bounded residual。上表使用首轮试验按验证集选择的
checkpoint：seed 11/29 选第 32 回合，seed 47 选第 16 回合。另一个预先固定的对照只比较
第 32 回合，并把 BC 隐藏层迁移给 PPO、重新置零动作层。迁移相对 fresh PPO 的验证差值为
`+0.039、+0.063、−0.092 mm`，没有达到三个种子一致改善的条件，因此保留 `fresh_ep32`。
不要把这组固定终点比较与上表的验证集选模结果混在一起。细节见
[PPO 公式与首轮训练](docs/surface_learning_pilot.md)、[迁移对照](docs/bc_to_residual_rl.md)和
[迁移归档](results/franka_surface_ppo_transfer/)。

这些策略只在仿真公开开发域中评价。命令限幅、4/4 通过和 100% 接触率都不是硬件安全、
未知表面泛化或收敛证明。公开归档含全部候选与训练统计，但每轮只分发少量代表性完整轨迹；
其余原始轨迹留在本地源产物中。上方在线负载补偿则另保留全部 80 次可重算指标的紧凑
轨迹及 4 份完整输入轨迹，不与这些学习归档混为同一实验。

## 任务、公式与源码入口

| 想看什么 | 说明 | 实现 |
|---|---|---|
| 误差驱动的在线补偿 | 当前力与下一步系数的更新顺序、换向冻结、共同 6 N 上限 | [在线补偿](docs/online_compensation.md)、[`tangential_compensation.py`](src/compliant_control_lab/tangential_compensation.py) |
| 换向时的姿态代价 | 旋转刚度与阻尼的同步缩放、同增益配对比较 | [姿态增益对照](docs/rotation_gain_comparison.md)、[`surface_control.py`](src/compliant_control_lab/surface_control.py) |
| Python 到 C++ 的控制链 | 表面坐标、自适应状态、接触过渡、力矩投影和输入超时 | [C++ 接口与范围](docs/cpp_core.md)、[`surface_control.cpp`](cpp/src/surface_control.cpp) |
| 擦拭任务与 49 维观测 | 50 Hz/500 Hz 时序、数据分组、教师标签 | [学习任务](docs/surface_learning.md)、[`surface_env.py`](src/compliant_control_lab/surface_env.py)、[`surface_dataset.py`](src/compliant_control_lab/surface_dataset.py) |
| 摩擦强基线与 BC 教师 | 有界摩擦前馈公式、输入消融和闭环偏移 | [切向补偿](docs/tangential_tracking.md)、[BC 对照](docs/bc_closed_loop_transfer.md)、[`surface_policy.py`](src/compliant_control_lab/surface_policy.py) |
| Bounded PPO | clipped objective、变时长 GAE、checkpoint 规则 | [学习试验](docs/surface_learning_pilot.md)、[`train_surface_ppo.py`](tools/train_surface_ppo.py) |
| 残差安全边界 | 接触门控、slew/filter、joint-torque headroom | [`residual_rl.py`](src/compliant_control_lab/residual_rl.py)、[`franka_torque_safety.py`](src/compliant_control_lab/franka_torque_safety.py) |

[12 秒擦拭视频](results/franka_tangential_demo/demo.mp4)同步显示七轴日志、原始法向力和切向
误差。它是已有仿真日志的可视化，不是真机演示，也没有重新积分动力学。

早期开发从接触微分离排查、平滑接触模型、切向积分和固定摩擦前馈一路推进到当前任务。
实验次序及各版本证据见[实验记录](docs/experiments/README.md)。冻结 v0.5 的 48-case first
reveal 仍为 `FAIL`，原始归档没有重算或改名。

## v0.5 首次揭盲结果

机器人使用相同的 48 个仿真参数场景和相同的逐 case 噪声 seed。预注册规则要求五个 residual
策略分别达到 44/48，不能挑最好 seed 或改用平均通过率。

| 方法 | 通过数 | Force-tracking error P95 [N] | Raw peak P95 [N] | Tangent P95 [mm] | Saturation worst |
|---|---:|---:|---:|---:|---:|
| Fixed hybrid | 17/48 | 2.01 | 58.62 | 22.66 | 19.69% |
| Adaptive hybrid | 23/48 | 2.21 | 58.99 | 18.89 | 20.71% |
| Torque-safe adaptive | 24/48 | 2.33 | 59.54 | 18.89 | 0.00% |
| Torque residual（5 runs） | 22–26/48 | 1.98–2.12 | 59.54 | 15.98–16.50 | 0.00% |

`Force-tracking error` 在进入稳定任务阶段后由滤波力反馈计算；`Raw peak` 取完整轨迹中的未滤波
最大接触力，包含首次碰撞。表中的约 2 N 稳态误差和约 60 N 瞬时峰值并不矛盾。

主结果是 `FAIL`。Residual policy 改善了切向跟踪，torque projection 也完成了限幅职责；
完整轨迹的 raw-force peak 仍然超过 35 N gate。仓库保留这一结果，不部署策略，也不把训练
回报当作安全证据。

同 case 配对后，五个 residual 分别在 34–35/48 个场景中降低 tangent RMSE，中位降幅为
1.47–2.09 mm；force RMSE 中位增加 0.05–0.11 N，raw peak 中位增加 0.36–0.59 N。
即使在揭盲后省略 peak-force gate，五个策略也只有 40–41/48，仍低于 44/48。这些数字
属于公开数据的描述性分析，不计作新的 blind evidence。

原始 384 行数据在 [comparison.csv](results/franka_safety_blind/comparison.csv)，生成摘要在
[summary.md](results/franka_safety_blind/summary.md)。冻结、信标和结果哈希见
[v0.5 protocol](docs/reproduction_plan_v0.5.md)。全部版本的假设与结论集中在
[experiment record](docs/experiments/README.md)。

![v0.5 post-reveal gate diagnosis](results/franka_safety_postreveal/failure_analysis.png)

上图只使用揭盲后已经公开的数据。它解释失败来源，不构成一轮新的 blind evaluation；
生成方法和逐项计数见 [post-reveal summary](results/franka_safety_postreveal/summary.md)。

![Safe-adaptive contact peak timing](results/franka_safety_postreveal/contact_events/contact_peak_timing.png)

第二张图重放 torque-safe adaptive 的 48 个公开 case。生成器先核对冻结指标，再按峰值距首次
raw contact 的时间作图；颜色表示运动阶段，形状表示 controller contact phase。事件定义和
两簇峰值的数值见 [contact-event diagnosis](docs/contact_event_diagnosis.md)。

## 核心实现入口

| 问题 | 代码 | 怎么检查 |
|---|---|---|
| 500 Hz 接触状态机与经典柔顺控制 | [`franka_control.py`](src/compliant_control_lab/franka_control.py)、[`franka_control.cpp`](cpp/src/franka_control.cpp) | [controller tests](tests/test_franka_control.py)、[fixed-controller parity](tests/test_cpp_parity.py) |
| 在线 bias/刚度估计与 gain scheduling | [`franka_adaptive.py`](src/compliant_control_lab/franka_adaptive.py) | [adaptive tests](tests/test_franka_adaptive.py)、[48-case event replay](results/franka_safety_postreveal/contact_events/summary.md) |
| 有状态接近参考与因果力反馈 | [`franka_reference.py`](src/compliant_control_lab/franka_reference.py)、[split-step simulation](src/compliant_control_lab/franka_simulation.py) | [参考约束](tests/test_franka_reference.py)、[采样契约](tests/test_franka_timing.py)、[四组实验](docs/reference_governor_v0.6.md) |
| 表面坐标、工具端 F/T 与完整输入回放 | [`surface_control.py`](src/compliant_control_lab/surface_control.py)、[`surface_sensing.py`](src/compliant_control_lab/surface_sensing.py)、[`surface_replay.py`](src/compliant_control_lab/surface_replay.py) | [坐标与传感器教程](docs/surface_frame_and_sensing.md)、[因果采样测试](tests/test_surface_simulation.py) |
| 切向负载偏差、有界积分、固定前馈与在线补偿 | [`tangential_compensation.py`](src/compliant_control_lab/tangential_compensation.py) | [固定补偿诊断](docs/tangential_tracking.md)、[在线更新与回归](docs/online_compensation.md)、[安全时序](tests/test_tangential_safety.py) |
| 新表面任务的交互环境、端点安全与学习数据 | [`surface_env.py`](src/compliant_control_lab/surface_env.py)、[`surface_dataset.py`](src/compliant_control_lab/surface_dataset.py)、[`surface_transitions.py`](src/compliant_control_lab/surface_transitions.py) | [学习前准备](docs/surface_learning.md)、[末步回归](tests/test_surface_endpoint.py)、[数据因果重放](tests/test_surface_dataset_replay.py) |
| 6D wrench 到 7 关节力矩包络投影 | [Python](src/compliant_control_lab/franka_torque_safety.py)、[C++17](cpp/src/torque_safety.cpp) | [native edge cases](cpp/tests/test_torque_safety.cpp)、[160-case randomized parity](tests/test_cpp_parity.py) |
| 50 Hz bounded residual 与 500 Hz safety wrapper | [`residual_rl.py`](src/compliant_control_lab/residual_rl.py) | [residual tests](tests/test_residual_rl.py)、[paired effect](results/franka_safety_postreveal/summary.md) |
| 五 seed 冻结、first reveal 与离线复核 | [`franka_safety_learning.py`](src/compliant_control_lab/franka_safety_learning.py)、[`published_results_audit.py`](src/compliant_control_lab/published_results_audit.py) | [protocol tests](tests/test_franka_safety_learning.py)、[tamper tests](tests/test_published_results_audit.py) |

## 实现范围

控制路径从 2-DOF 解析模型开始，随后进入 Franka 6D Cartesian wrench 控制：

- impedance、admittance 和 hybrid force-position control；
- bias/force-rate/contact-stiffness estimator 与 gain scheduling；
- contact confirmation、force transition、anti-windup 和 reference governor；
- nominal wrench 与 residual wrench 的两级 joint-torque projection；
- 50 Hz、三轴有界 residual policy，外层仍由 500 Hz 经典控制器运行。

实验路径使用独立 train/development 数据、五个训练 seed、冻结 tag、SHA256 manifest，以及
未来 drand Quicknet round 生成首次揭盲场景。架构和数据流见
[architecture](docs/architecture.md)。

### Python / C++ 范围

| 能力 | Python | C++17/Eigen | 公开实验 |
|---|---:|---:|---:|
| Impedance / admittance / fixed hybrid | 是 | 是，逐分量 parity | nominal、v0.3、v0.4、v0.5 |
| Adaptive gain scheduling / surface controller | 是 | 是，有状态序列 parity | Python 仿真；C++ 记录输入核验 |
| 在线切向负载补偿 | 是 | 是，含更新冻结与重置 | 原有 24-case 回归；动态误差对照另列 |
| Torque projection / residual headroom | 是 | 是，160-case parity | v0.5 rollout 使用 Python；C++ port 为揭盲后工程验证 |
| Bounded Residual RL | 是 | 否 | v0.4、v0.5、独立表面学习试验 |
| ROS 2 / Franka hardware adapter | 否 | 否 | 无 |

C++ 数值接口使用固定尺寸状态、目标、actuation context 和 Cartesian wrench，并返回状态标志。
完整的主张、测试和结果对应关系
见 [verification matrix](docs/verification_matrix.md)。

## 新手最短路线

| 顺序 | 动手或阅读 | 核对什么 |
|---:|---|---|
| 1 | 安装后运行 `franka-smoke` | 代码能运行，旧 v0.5 归档仍是 `FAIL` |
| 2 | 打开[招聘方走查](docs/recruiter_walkthrough.md) | 当前结果、对照是否公平、范围限制 |
| 3 | 阅读[切向补偿](docs/tangential_tracking.md)和[在线更新](docs/online_compensation.md) | 从误差来源到固定前馈，再到有界参数更新 |
| 4 | 按在线补偿页复核实验和阶段指标 | 不只看平均误差，也检查换向、测量误差和失败项 |
| 5 | 构建 [C++ 控制核心](docs/cpp_core.md)，运行 parity | 同一输入序列下，状态与命令能否逐步对齐 |
| 6 | 再读[表面学习任务](docs/surface_learning.md)，运行学习归档的 audit | 49 维观测、16/4/4 分组，以及 BC/PPO 是否超过强基线 |

系统学习柔顺控制可从[教程目录](docs/tutorial/README.md)开始，由 2-DOF 推到 Franka 与
Residual RL。面试练习见[练习、故障定位和项目表达](docs/tutorial/06_exercises_and_interview.md)。

完整导航和术语说明分别在 [docs/README.md](docs/README.md) 与
[CONTEXT.md](CONTEXT.md)。

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

该命令先校验 384 行冻结归档，再运行 2 s torque-safe adaptive nominal 仿真。正常输出会同时保留
实验失败和工程检查通过这两个状态：

```text
archive: PASS (384 rows, frozen_decision=FAIL)
simulation: PASS (safe_adaptive_hybrid/nominal, steps=1000, ...)
smoke: PASS
```

`smoke: PASS` 不会把冻结结论改成通过，也不等于重跑 48 cases。CI 使用同一个入口。

## 标称演示

```bash
franka-control-lab --output results/franka-quick --gif
```

![Franka hybrid force-position control](results/franka/hybrid_demo.gif)

这个 GIF 只展示 fixed hybrid 的 nominal 动作；v0.5 residual 结果见前面的冻结表和诊断图。

无显示器的 Linux 环境可指定 EGL：

```bash
MUJOCO_GL=egl franka-control-lab --output results/franka-quick --gif
```

2-DOF 教学基线：

```bash
compliant-control-lab --output results/planar-quick --gif
```

## 验证代码

Python 测试和静态检查：

```bash
pytest
ruff check src tests
```

C++17/Eigen 核心及 Python/C++ 数值一致性：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
pytest tests/test_cpp_parity.py
./build/compliant_control_torque_benchmark
```

离线审核仓库中已经发布的 v0.5 产物，不联网，也不重新运行仿真：

```bash
franka-published-results-audit \
  --protocol results/franka_safety_preholdout/protocol.json \
  --result results/franka_safety_blind
```

该命令先核对固定的 v0.5 manifest 与 `COMPLETE` 标记，再检查文件 hash、两路 beacon 归档、
blind root 与 HMAC seed 推导、384 行 case-method 网格、policy 身份、gate 标签和 summary
通过数。它不联网，也不重新执行 BLS 验签或仿真；BLS 校验记录已包含在固定 manifest 覆盖的
`reveal.json` 中。`audit PASS` 表示发布归档未变且内部推导自洽；冻结的实验结论仍然是 `FAIL`。

完整训练、公开验证和新一轮 first-reveal 命令放在
[实验复现文档](docs/reproduction_plan_v0.5.md)，避免把一次正式实验误当成快速示例运行。

## 项目结构

```text
src/compliant_control_lab/
├── franka_control.py          # fixed 6D Cartesian controllers and state interface
├── franka_adaptive.py         # estimators, gain scheduling and torque-safe adaptive nominal
├── franka_torque_safety.py    # wrench-to-joint-torque projection
├── residual_rl.py             # observation, policy and residual safety rules
├── franka_learning.py         # ARS training and 24-case public validation
├── franka_safety_learning.py  # five-seed freeze and 48-case first reveal
├── franka_simulation.py       # MuJoCo adapter, contact task and metrics
├── contact_event_analysis.py  # verified event replay of public v0.5 cases
├── post_reveal_analysis.py    # paired effects and gate sensitivity
├── smoke.py                   # archive + short simulation check
└── controllers.py             # 2-DOF teaching baseline
cpp/
├── include/                   # public fixed-size Eigen interface
├── src/                       # fixed controllers and torque-envelope projection
└── tests/                     # native behavior tests and parity probe
docs/                          # navigation, tutorials, method notes and experiment records
results/                       # committed CSV, figures, policies and integrity manifests
tests/                         # math, controller, simulation, safety and protocol tests
```

## 当前限制

- 仿真使用理想力矩接口，没有电流环、编码器量化和真实通信抖动。
- 接触参数直接取自 MuJoCo，未做真机辨识。
- 零空间投影采用阻尼运动学形式，尚未实现 dynamically consistent operational-space control。
- C++ 已包含选定表面控制器的状态更新与 torque projection。另一条加速度受限参考实验、
  policy 和训练尚未移植；传感器采集、机器人模型计算和真实通信仍由外部接口负责。
- torque projection 没有提供 torque-rate、碰撞阈值或硬件安全认证。
- 当前机器没有 Franka hardware/model interface，仓库不声称完成 ros2_control 真机插件。

## v0.5 揭盲后的事件诊断

事件重放先逐 case 核对 7 个冻结指标，再读取峰值时刻的 controller 与 actuation telemetry。
48/48 case 的最大指标误差为 0。18 个 peak-gate
failure 中，7 个峰值位于首次 raw contact 后 0.432 s 内；另外 11 个在 1.222 s 以后达到
峰值。独立的 motion-phase 统计为 pre-wiping 6 / wiping 12。

当时据此把问题拆成入触与擦拭两个 cohort，并把公开 48 cases 只用于生成后续假设。
随后完成的 reference、接触模型、切向补偿和表面学习实验都列在
[实验记录](docs/experiments/README.md)中；它们不回写 v0.5 的冻结结论。

现有 failure breakdown 与 event replay 可用下面的命令重新生成：

```bash
franka-post-reveal-analysis
franka-contact-event-analysis
```

## 模型与许可证

Franka 模型来自 Google DeepMind
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda)，
固定到上游 commit `da76818e269b82289eba39808e2fb91d679d6994`。模型资源使用 Apache-2.0，
许可证和修改说明保存在
[assets/franka_emika_panda](src/compliant_control_lab/assets/franka_emika_panda/UPSTREAM.md)。
本项目其余代码使用 MIT License。
