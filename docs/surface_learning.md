# 从稳定擦拭到学习控制

这条开发线先准备训练接口和数据，不把“能运行训练脚本”当作算法有效的证据。
旧 v0.5 的首次揭盲结论仍是 `FAIL`；这里使用新的表面任务、平滑接触模型和公开开发域，
不是旧策略重跑，也不是新一轮盲测。

先读[切向控制实验](tangential_tracking.md)。如果一个解析摩擦前馈已经能消除大部分误差，
学习算法就必须与它比较，而不能只挑较弱的无补偿基线。

## 两个不同的学习问题

| 路线 | 名义控制器 | 学习输出 | 用来回答什么 |
|---|---|---|---|
| Bounded Residual RL | 自适应导纳 + 有界摩擦前馈 | 表面坐标下三轴残余力 | 在已有强基线上还能改善多少？ |
| 摩擦教师模仿学习 | 自适应导纳，不加名义摩擦前馈 | 50 Hz 教师的三轴归一化动作 | 能否从测量输入学到教师规则并在闭环保持效果？ |

两个路线不能混用 checkpoint。把摩擦教师输出叠在已有摩擦前馈上，会重复补偿；
把 RL 的零动作结果当作 IL 基线，也改变了被比较的控制器。
教师是可解析的工程规则，不是人类示教、最优控制解或未知环境的专家。
其主要价值是检查模仿学习流程和闭环误差，不是证明神经网络优于解析公式。

## 本轮实测结果

固定代码后运行 24 个 case × 三种方法，每回合 12 s。以下是公开准备域的描述性统计，
不是新 holdout；每个物理任务组的两个 seed 也不是两个独立物理场景。

| 方法 | 安全完成 | 跟踪目标通过 | 法向力 RMSE 中位数 | 切向 RMSE 中位数 |
|---|---:|---:|---:|---:|
| 自适应导纳，无摩擦补偿 | 24/24 | 4/24 | 0.186 N | 11.839 mm |
| 有界摩擦前馈强基线 | 24/24 | 24/24 | 0.188 N | 3.466 mm |
| 自适应导纳 + 50 Hz 摩擦教师 | 24/24 | 24/24 | 0.189 N | 3.606 mm |

跟踪目标在运行前固定为：擦拭接触 ≥99%、切向 RMSE ≤10 mm、法向力 RMSE ≤2 N。
三种方法都保持 100% 擦拭接触、0% actuator clipping；教师的最差切向 RMSE 为 7.337 mm。
教师略逊于 500 Hz 强基线，不能用旧前馈成绩替代同频教师成绩。
这组结果支持开始受限仿真的训练试验，不证明神经网络必要或更优。

另有三标称锚点 × 四种合法极端动作的 12 次压力测试，均完整结束。
常值正/负边界动作使切向误差中位数升到约 15.5–15.7 mm，未过跟踪目标；
这也说明“动作合法、安全层未终止”不等于“控制效果好”。原始力/穿透/速度越界的终止路径
另由故障注入测试覆盖，不能把注入测试说成自然出现的物理事件。

[完整数值索引](../results/franka_surface_learning_preparation/README.md)包含 72 行对照、
12 行压力测试、分组统计和来源哈希；大体积全集留在本地生成。
[公开示例](../results/franka_surface_learning_examples/)只包含 `g000_seed11`、`g005_seed11`、
`g007_seed11` 三回合：一个标称 train、一个联合扰动 development_test、一个联合扰动
validation。三份重新积分的 18,000 步与保存数据逐字段一致；另外，数据采集器与 benchmark
教师的全部 24 回合共享轨迹字段也已逐字段核对。

先快速复核随仓库提供的数据，不必重跑全集：

```bash
python -m compliant_control_lab.surface_dataset --audit results/franka_surface_learning_examples
python -m compliant_control_lab.surface_dataset_replay --dataset results/franka_surface_learning_examples
```

## 先跑一个零残差回合

在仓库根目录安装 `pip install -e ".[dev]"` 后：

```python
import numpy as np
from compliant_control_lab.surface_env import SurfaceLearningEnv
from compliant_control_lab.surface_simulation import yaw_frame

env = SurfaceLearningEnv(yaw_frame(0))  # 默认 smooth contact、friction nominal
observation, info = env.reset(seed=11)
episode_return = 0.0
while True:
    observation, reward, terminated, truncated, info = env.step(np.zeros(3))
    episode_return += reward
    if terminated or truncated:
        break
print(episode_return, info["termination_reasons"])
env.close()
```

这只是接口例子。完整准备域使用后文的 case 文件，不能用一个 nominal 回合代替域内测试。
环境没有在线渲染；已有擦拭视频是日志可视化。

## 一次动作怎样进入机器人

```text
500 Hz：采样/延迟输入 → 名义控制器更新一次 → 生成观测
50 Hz ：策略读取观测 → 输出三轴动作，保持 10 个物理步
500 Hz：接触门控 → 力保护 → 滤波/限速 → 力矩包络投影 → MuJoCo 积分
        每步保存实际输入、动作各阶段、原始力与终止原因
```

实现只有一个动力学循环：[SurfaceSimulator](../src/compliant_control_lab/surface_simulation.py)。
批量经典实验和 [Gymnasium 环境](../src/compliant_control_lab/surface_env.py) 都调用它。
`sample()` 重复读取不应重复抽噪声或推进滤波器；`prepare()` 和 `apply()` 成对调用，
避免为构造观测额外更新一次名义控制器。

传感器力经过滤波和延迟，是当前控制器已经拿到的输入。当前动作积分后得到的 raw wrench
只能供后续观测或评价使用，不能倒填到当前策略输入。关节编码器与延迟 F/T 的采样年龄分别
记录。默认模型仍向名义控制器提供已知动力学上下文，这不是纯黑箱、无模型控制。

### 49 维观测

Schema 为 `surface_env_v1`，全部为 `float64`。物理量除以固定尺度后截断到 `[-3, 3]`；
没有从 validation 或 development_test 拟合均值方差。精确顺序、尺度和单位由
[`OBSERVATION_NAMES/SCALES/UNITS`](../src/compliant_control_lab/surface_env.py) 给出。

| 索引（从 0 起） | 内容 | 主要尺度 |
|---|---|---|
| 0–2 | bias 修正后的法向力、力误差、变化率 | 20 N、12 N、250 N/s |
| 3–12 | 局部位置/速度误差、目标速度、目标力 | 法向 20 mm、切向 50 mm、0.1 m/s、12 N |
| 13–22 | 接触过渡、上一实际残差、六方向力矩余量 | 1、(4, 6, 6) N、1 |
| 23–32 | 姿态/角速度误差、接触确认时间、当前名义力 | 0.2 rad、1 rad/s、0.1 s、25 N |
| 33–46 | 七轴位置和速度编码器 | π rad、2 rad/s |
| 47–48 | 运动学输入与力输入的年龄 | 0.1 s |

真实摩擦系数、场景 ID、理想接触力、真实穿透量不进入 actor。
评价真值保留在 `info` 和 `result()`；训练代码不得把整个 `info` 拼进观测。
单帧观测没有包含全部滤波和接触内部状态，因此不要把它宣称为严格的全状态 Markov 表示。
以后使用历史窗口或循环网络时，必须在 episode 边界清空状态。

### 三轴动作与安全边界

策略动作 `a ∈ [-1,1]^3` 对应控制器表面坐标 `(normal, tangent1, tangent2)`，默认请求力
`diag(4, 6, 6) a` N。接触未确认时残差不启用；确认后还有 0.1 s 等待。
普通变化率上限为 `(40,60,60)` N/s，低通时间常数 0.04 s，关节力矩投影保留 10% 余量。

记录必须区分：策略归一化动作、门控后的请求力、滤波后的力、最终施加的局部残差。
模仿标签是第一项，不是投影后的力。紧急清零和投影可以超过普通限速，优先满足保护条件。

环境严格拒绝形状不符、NaN/Inf 和超出 Box 的动作，清零残差后终止该回合。
低层控制器另有截断逻辑，但环境不以截断掩盖策略的接口错误。
以上约束的是命令；它们不保证实际接触力、运动速度、被动性或真机安全。

## 回报和回合结束

每个 2 ms 物理步积累存活回报，扣除六项截断到 `[0,1]` 的代价：
法向力、切向误差、失去接触、姿态、动作幅度、保护介入。
权重分别为 `0.2, 0.2, 0.2, 0.1, 0.1, 0.2`；前三种连续误差的参考尺度为
12 N、20 mm、0.2 rad。正常回合的未折扣总回报在 `[0, H]` 内，`H` 为回合秒数。

每个物理步检查原始法向力 >35 N、穿透 >2 mm、末端速度 >0.3 m/s、关节力矩裁剪，
以及评价阶段连续 0.1 s 失去接触。异常数值、控制器/积分失败也会终止。
触发回合扣除 `H+1`，所以失败回合的**未折扣**总回报至多为 −1。
这不是任意折扣因子下的最优策略证明，更不是“训练回报高就安全”。

到达时间上限返回 `truncated=True`；失败返回 `terminated=True`。
时间截断的最后观测是最后动作后的真实 next observation，可以用于 bootstrap。
失败一般不 bootstrap；若最后观测无法构造，环境给出最后一个有限观测占位，
同时设置 `terminal_observation_valid=False`。数据集将这种无效 next observation
存为带 mask 的 NaN，不能当作训练样本的正常下一状态。
这是 [Gymnasium 时间上限约定](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
在本项目中的具体处理。

终止是在仿真中观察到越界后停下，不能撤销已发生的峰值。部分失败甚至无法确认物理积分
是否完成，因此日志区分 `attempted`、`recorded` 和 `unconfirmed`，不虚构未执行证明。
速度和几何量同时检查积分前 `x[k]` 与刷新后的 `x[k+1]`，包括最终的 `x[N]`；
后者单独保存在 `endpoint_*` 字段中，不改变旧日志时刻。无法构造端点时用 `endpoint_valid`
标记未知并终止，不能把到达时间上限判为成功。这里没有额外求解 `x[N]` 的下一周期接触力。

## 数据：先分组，再切窗口

准备域默认 12 个物理/传感/任务组，每组两个噪声 seed，共 24 个 12 s 回合。
它包含三个标称锚点与九组联合扰动：墙面 yaw ±15°、法向标定误差 ±5°、摩擦系数
0.25–0.65、工具质量 0.10–0.13 kg，并改变力偏置、噪声、延迟、目标力和轨迹。
完整边界在 [case 生成器](../src/compliant_control_lab/surface_readiness_cases.py)，不是对每个
边界组合的穷举。墙面位置与任务平面仍已知；没有移动墙面、未知平面搜索或非 Coulomb 摩擦。

分组 ID 包含完整物理、传感及任务参数。仅噪声 seed、方法、回合长度或法向标定不同的
变体留在同组，不能通过改 seed 制造看似独立的测试数据。
按组分为 `train / validation / development_test`（默认 8/2/2 组、16/4/4 回合），之后才能切时间窗口。
三个集合都是公开开发数据；最后一项的名字不赋予它盲测身份。

初始 seed `20260906` 恰把两个标称组放到 development_test，覆盖不合理。
训练前增加了纯配置分层条件：三个标称锚点分别进入三个 split，从原 seed 起取第一个满足
条件的值，得到 `20260908`。validation 和 development_test 各有一个标称组、一个联合扰动组。
这次整理发生在经典实验之后，全部数据仍是公开开发数据。它只改 split 标签，不重新仿真，
不改观测、动作或指标；[原始清单与全部候选 seed](../results/franka_surface_learning_partition/)
均保留。文中的 v1/v2 指分组版本，不是软件版本或一轮新的首次揭盲。

教师使用已测法向力和当前切向目标速度，在 adaptive nominal 上生成平滑有界摩擦补偿。
它以真实 50 Hz 频率决策，通过同一滤波和安全层执行。不能将旧的 500 Hz 轨迹简单抽帧，
也不能用两个不同回合的 wrench 相减来制造同状态残差标签。

```bash
python -m compliant_control_lab.surface_readiness_cases --output dist/surface-preparation-cases-v1.json
python -m compliant_control_lab.surface_dataset \
  --cases dist/surface-preparation-cases-v1.json --output dist/surface-preparation-dataset-v1
python -m compliant_control_lab.surface_readiness_benchmark \
  --cases dist/surface-preparation-cases-v1.json --output dist/surface-preparation-benchmark-v1
python -m compliant_control_lab.surface_readiness_benchmark \
  --cases dist/surface-preparation-cases-v1.json --output dist/surface-preparation-stress-v1 --stress
python tools/repartition_surface_preparation.py \
  --benchmark dist/surface-preparation-benchmark-v1 --stress dist/surface-preparation-stress-v1 \
  --dataset dist/surface-preparation-dataset-v1 --output dist/surface-preparation-v2
python -m compliant_control_lab.surface_dataset --audit dist/surface-preparation-v2/dataset
```

输出必须是新路径。采集先在临时目录完整保存，再原子发布；配置错误会中止发布，真实安全
失败回合则保留。审计独立重算教师公式、回报、时钟、动作保持、输入/日志对应关系和分组；
因果重放用保存动作重新积分，对比观测和完整轨迹，检查这些数据是否来自同一次因果执行。
精确重放以相同软件和物理引擎版本为目标，不承诺跨版本逐位一致。
相同版本在不同 CPU 或数值库上也可能出现浮点末位差异。CI 对历史轨迹的浮点重算采用
`1e-10` 绝对容差，仍精确检查哈希、时间轴和离散状态；同进程的一致性测试继续逐位比较。
因果重放工具会如实报告是否完全一致，不据此改写归档，也不调整接触力或穿透的安全门槛。
本轮 Python 3.10.12、NumPy 2.2.6、MuJoCo 3.12.0、Gymnasium 1.3.0；详细版本在清单中。
保留初始全集和分层副本约需 2.6 GiB。测得的仿真吞吐中位数约为 0.83 模拟秒/墙钟秒，
运行时存在并行采集任务；这不是单独测得的策略推理延迟，也不是实时性承诺。

完整数据适合本地生成。公开的小样本若标注为 `example_subset`，会保留父数据的完整分组
计划和哈希，但只代表实际附带的 episodes，不能作为完整训练语料的替身。

### 读成训练数组

```python
from compliant_control_lab.surface_transitions import load_transition_batch

batch = load_transition_batch("dist/surface-preparation-v2/dataset", "train",
                              gamma_per_policy_step=0.99)
x, y = batch["observations"], batch["actions"]  # 模仿学习的输入和标签
```

加载器先审计原始数据，再选指定 split，不会删除失败回合或拼入评价真值。
无效 terminal next observation 只在返回数组中置零，磁盘原始 NaN 不改；
`bootstrap_mask = ~terminated & terminal_observation_valid`。
`discounts` 已包含这个 mask，实际指数为 `physics_substeps / 10`，因此不足 10 步的末块
也按真实时长折扣。`0.99` 在这里是每个 50 Hz 学习步的折扣，不是每个 2 ms 物理步。
以后拼历史窗口时，使用 `episode_id / episode_start`，不能跨回合拼接。
示例子集默认禁止当作完整训练语料加载；只有显式 `allow_examples=True` 才能用来走通接口。

## 保存候选策略与冻结评价

策略文件绑定观测顺序/尺度、动作阶段、名义控制器、任务域和分组计划。
文件评价器支持受限的前馈 MLP：49 维输入、1–3 个隐藏层（每层至多 128 单元）、
三维 tanh 输出，隐藏激活可选 tanh 或 ReLU。每层权重的形状是 `输出维 × 输入维`，
bias 是输出维向量。训练后只需导出这些数组，不必引入框架专用 checkpoint 加载器。

下面用固定 seed 构造小幅非零网络，检查导出和评价接口；它没有经过训练。

```python
import json
from pathlib import Path
import numpy as np
from compliant_control_lab.surface_policy_artifact import (
    policy_contract, load_policy_artifact,
    freeze_evaluation_protocol, verify_evaluation_protocol, ActorEvaluator,
)
from tools.surface_mlp_actor import save_mlp_candidate, actor_from_artifact

cases = json.loads(Path("dist/surface-preparation-v2/dataset/manifest.json").read_text())["cases"]
for case in cases:
    case["nominal_kind"] = "friction"  # RL 用强基线；不改变物理组身份
contract = policy_contract(cases, purpose="rl_residual", nominal_kind="friction")
candidate = "dist/mlp-interface-fixture.json"
rng = np.random.default_rng(7)
widths = [49, 16, 8, 3]
layers = [{"weights": rng.normal(0, 0.03, (out_dim, in_dim)),
           "bias": rng.normal(0, 0.01, out_dim)}
          for in_dim, out_dim in zip(widths[:-1], widths[1:])]
save_mlp_candidate(candidate, layers, contract=contract, activation="tanh")
protocol = "dist/mlp-interface-protocol.json"
limits = {"peak_force_n": ("max", "<=", 35),
          "force_rmse_n": ("max", "<=", 2),
          "tangent_rmse_mm": ("max", "<=", 10),
          "contact_ratio_pct": ("min", ">=", 99),
          "max_speed_m_s": ("max", "<=", 0.3)}
frozen_sha = freeze_evaluation_protocol(protocol, candidate,
    expected_contract=contract, cases=cases,
    metrics=list(limits),
    thresholds={name: dict(zip(("aggregation", "operator", "value"), rule))
                for name, rule in limits.items()})
verify_evaluation_protocol(protocol, candidate, expected_protocol_sha256=frozen_sha,
                           expected_contract=contract, cases=cases)
artifact = load_policy_artifact(candidate, expected_contract=contract)
callback, runner = actor_from_artifact(artifact)
actor = ActorEvaluator(artifact, actor=callback)
```

在仓库根目录运行。MLP 权重、runner/evaluator 源码哈希、包源码及 Python/NumPy/MuJoCo/
Gymnasium 版本共同写入候选文件，再由协议冻结候选文件哈希。加载时逐项核对，
环境重置前就拒绝不匹配的运行程序。参数只允许有限数字，禁止 pickle 和候选代码导入。
上述示例可换成训练后的同格式权重；RNN、额外 normalizer 或任意网络结构尚不支持。

务必在评价前把 `frozen_sha` 保存在独立记录中；之后不能读取一个可能已被改过的文件，
重新算 SHA 并把它称为原冻结值。哈希用于发现改动，不证明作者身份或首次揭盲。

`ActorEvaluator.step(env, observation)` 只把独立的 49 维数组交给回调，检查名义控制器
和残差配置。回调异常、非法动作或超时会走名义控制回退并终止。
超时是同步回调返回后的计时检查，不会抢占卡死的推理，也不构成真机 watchdog。
任意 Python 闭包仍可能访问外部对象；这是接口隔离，不是恶意代码沙箱。

执行冻结的选择范围可用 [`evaluate_surface_candidate.py`](../tools/evaluate_surface_candidate.py)
中的 `evaluate_candidate(candidate, protocol, cases, frozen_sha, output, purpose=...,
nominal_kind=...)`。它在创建环境前校验身份，只跑协议所选 cases，独立重算物理安全门，
保存失败和全部可用日志，再按冻结门槛汇总。没有观测到擦拭窗口、缺指标或任何失败，
都不能因回报较高而变成 PASS。文件加载器通过上面的受限 runner 执行 MLP；旧的
`linear_tanh` 单层接口夹具仍可使用。

```python
from tools.evaluate_surface_candidate import evaluate_candidate

evaluate_candidate(candidate, protocol, cases, frozen_sha,
                   "dist/mlp-interface-evaluation",
                   purpose="rl_residual", nominal_kind="friction")
```

本轮已实际执行这个非零 `49→16→8→3` 测试网络：四个 development_test 回合共
24,000 个物理步，均完成且没有 actor 失败。它只验证网络文件到闭环评价的接入；
权重来自 seed 7 的随机初始化，不能作为学习结果。线性零残差夹具也完成了相同范围的评价。

## 真正开始训练时的顺序

先固定数据和控制器配置，做一个小批次拟合检查，确认标签、动作尺度和 loss 实现正确。
再只用 train 训练，使用 validation 选超参数；同时运行闭环回合，不能只报离线动作 MSE。
候选策略和评价规则冻结后，再评价 development_test。若据此继续修改方法，它继续是开发
数据，不能重新命名为未见数据。需要新的泛化主张时，另外安排尚未使用的冻结测试协议。

IL 的首要对照是同频教师；RL 的首要对照是零残差强基线。两条路线都报告安全失败、
接触比例、力/切向误差、干预率、实际模拟步数和推理耗时。所有训练 seed 分别报告，
不只展示最好 checkpoint。

本轮不提供新的已训练策略，也不增加 ROS、Franka 硬件适配或在线学习。
准备域结果支持先做小规模模仿学习，再尝试 bounded Residual RL。解析强基线已经通过
全部跟踪目标，RL 即使没有额外收益，也应保留并报告这个结果。
