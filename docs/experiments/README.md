# 实验总账（Experiment Record）

这里索引已经发布的实验。叙述文档中的数字是阅读摘要；最终以链接的 CSV 和冻结产物为准。

| 版本 | 数据身份 | 结果 | 决策 |
|---|---|---|---|
| Initial | 命名标称场景 | Hybrid force RMSE 0.95 N，saturation 0% | 增加随机参数失配测试 |
| v0.3 | 当时的首次运行；现在是公开验证集 | Fixed hybrid 通过 6/24 | 可以开展 bounded residual 实验；尚未证明 RL 更好 |
| v0.4 | seed-29 公开验证集 | Fixed 6/24，adaptive 6/24，residual 7/24 | 保留自适应 nominal；不部署 checkpoint |
| v0.5 | 48-case first reveal；揭盲后转为公开验证集 | 五个 residual 为 22–26/48，均未达到 44/48；torque-safe adaptive 与五个 residual 的 saturation 均为 0% | 不部署；分别检查早期入触和较晚擦拭峰值 |
| v0.6 开发对照 | 同一批 48 个公开场景，四组共 192 次仿真 | 同分步时序下原始参考 23/48、解析速度 24/48、限速参考 23/48 | 限速参考保留为实验项；尚未完成低冲击控制 |
| Surface development v1 | 新任务与六轴传感定义；24 个公开开发 case，四组共 96 次 | 准确法向的切向误差配对中位差 −0.657 mm，接触比例 −0.967 个百分点；四组接触比例中位数约 57% | 保留坐标修正与传感接口；先解决间歇接触，未加入 RL |
| Surface contact-model repair v1 | 同一 24 × 4 开发网格，旧/平滑模型共 192 次 | 新模型 96 组全部接触率 100%、饱和率 0%；准确法向 raw RMSE 中位数 0.181 N | 保留显式模型选项；软接触压入增加，不声称控制算法或真机改进 |
| Tangential compensation v1 | 固定平滑模型，24 × 3 主网格；12 s × 3 摩擦 × 3 方法另存 9 行 | 切向 RMSE 中位数：基线 11.800、积分 8.093、前馈 1.885 mm；81 次接触率均 100%、饱和率均 0% | 前馈达到主网格减半目标，积分未达到；保留摩擦失配和姿态代价，不加 RL |
| Surface learning pilot v1 | 24 个公开 case 按组分为 16/4/4；BC 与 PPO 各三个训练种子 | BC 开发跟踪 0/4、4/4、2/4；PPO 均 4/4，但相对强基线切向均值差为 +0.064、+0.068、−0.022 mm | BC 尚未可靠迁移到闭环；PPO 没有一致收益，保留解析前馈 |
| Surface BC input transfer v1 | 沿用 16/4/4；两个输入掩码各三个种子，48 个评测回合 | 两种掩码均全部通过；验证集选中的历史残差屏蔽方案开发误差为 2.818、2.825、2.781 mm | 原来的 49 输入跟踪失效明显改善；未超过解析前馈，不增加新盲测主张 |

## 版本锚点与证据

- **Initial**：revision `f5755e5`；[metrics](../../results/franka/metrics.md)。
- **v0.3**：revision `b126ef3`；[summary](../../results/franka_stress/summary.md)、
  [CSV](../../results/franka_stress/metrics.csv)。
- **v0.4**：revision `9eda43f`；[summary](../../results/franka_learning/summary.md)、
  [manifest](../../results/franka_learning/manifest.json)。
- **v0.5**：implementation `f186a19`，tag `v0.5-preholdout`，result `54ab506`；
  [summary](../../results/franka_safety_blind/summary.md)、
  [protocol](../../results/franka_safety_preholdout/protocol.json)、
  [reveal](../../results/franka_safety_blind/reveal.json)、
  [manifest](../../results/franka_safety_blind/manifest.json)。

## 证据生命周期（Evidence lifecycle）

`summary.md` 是生成的阅读页；CSV 保留每个 case 和失败标签。protocol、reveal 与 manifest
用于确认运行的是哪版实现，以及结果字节是否发生变化。

v0.5 揭盲数据现在可以用于诊断。任何针对这 48 个 cases 调过的控制器，都必须换用新冻结
协议和未来 beacon，才能再次提出 first-reveal 主张。现有 freeze 和 result 目录保持不变。

## 揭盲后诊断

同 case 配对显示，五个 residual 各自在 34–35/48 个 case 降低切向 RMSE，中位降幅为
1.47–2.09 mm。它们的 force RMSE 和 raw peak 中位数都略有增加。省略 peak-force gate 的
post-hoc 计数为 40–41/48，仍低于冻结阈值 44/48。

Safe-adaptive event replay 在分析前重新核对了 48 个 case 的 7 项冻结指标，观测到的最大绝对
误差为 0。18 个 peak-gate failure 里，7 个峰值距首次 raw contact 不超过 0.432 s；11 个
至少晚 1.222 s，且 12 个发生在擦拭阶段。这批数据继续保持 public validation 身份。

方法细节见 [v0.5 protocol](../reproduction_plan_v0.5.md)；是否部署和下一条假设见
[Residual RL decision](../residual_rl_decision.md)。独立的
[post-reveal report](../../results/franka_safety_postreveal/summary.md) 记录 paired effect 和 gate
sensitivity。Event replay 由
[`contact_event_analysis.py`](../../src/compliant_control_lab/contact_event_analysis.py) 生成；
[event summary](../../results/franka_safety_postreveal/contact_events/summary.md) 保存重放核对和
峰值阶段。两类诊断都不改动冻结证据。

## 接近参考开发对照

四组实验先验证旧模式仍能复现 48 个场景的 7 项确定性指标，最大绝对误差为 0。随后固定
分步采样时序，分别比较原始参考、解析速度与有状态限速参考；三组全轨迹 raw peak P95
为 60.89、59.03、61.63 N。限速参考的通过数没有增加，继续保持可选实验实现。

这次没有训练或评测新的 residual policy，也没有进行新一轮首次揭盲。
实现说明见[接近参考与采样时序](../reference_governor_v0.6.md)；
[CSV](../../results/franka_reference_ablation/comparison.csv) 与
[manifest](../../results/franka_reference_ablation/manifest.json) 记录数值和实际运行版本。

## 表面坐标与六轴测量

本轮单独给出名义任务平面，控制器分别使用世界 x、准确法向及 ±5° 标定偏差。
每个 case 的任务轨迹与随机数流相同。Force RMSE 使用未滤波接触力，切向误差使用真实
墙面上的欧氏长度；这两项定义均不同于旧实验。较低的峰值不能直接算作对旧任务的改善。

全体参数及四份预选 case 16 完整输入轨迹的重放结果见
[manifest](../../results/franka_surface_development/manifest.json)，数值见
[96-row CSV](../../results/franka_surface_development/comparison.csv) 和
[摘要](../../results/franka_surface_development/summary.md)。这轮没有定义通过门槛，也没有新揭盲。
六轴测量和质量误差的适用范围在[教程](../surface_frame_and_sensing.md)中说明。

后续接触模型修复先逐项复现旧 96 行指标，最大绝对差为 0，再运行同参数的平滑模型。
所有配对配置、原归档引用与新轨迹哈希见
[contact-repair manifest](../../results/franka_surface_contact_fix/manifest.json)。
[诊断记录](../wiping_contact_diagnosis.md)保留无效探针、固定控制周期的物理子步对照和压入量
代价。这轮仍是公开开发实验，没有新训练或新 holdout。

## 固定模型后的切向补偿

先用无噪声 3 s 单因素实验区分摩擦负载与动态滞后，保留零摩擦、双刚度和半速探针。
两个补偿器参数在主网格运行前固定，均在原 nominal wrench 投影之前相加，不改变接触模型。
旧准确法向基线 24 行既有指标完全复现，最大绝对差为 0。

前馈名义摩擦始终为 0.45；独立的 12 s 诊断同时改变工具和墙面的实际摩擦输入。
实际 0.25/0.45/0.65 时，前馈切向 RMSE 为 3.875/1.705/6.484 mm；没有把这九行混入
主网格，也没有重新冻结 holdout。积分未达到主网格误差减半目标，结果仍完整保存。

证据：[算法与学习页](../tangential_tracking.md)、[主 CSV](../../results/franka_tangential_development/comparison.csv)、
[独立长时 CSV](../../results/franka_tangential_development/long_comparison.csv)、
[实际参数与来源哈希](../../results/franka_tangential_development/manifest.json)、
[七次诊断统计](../../results/franka_tangential_diagnostics/diagnosis.json)。

## 小规模学习试验

先完成 BC 的三个种子，再在带摩擦前馈的名义控制器上训练独立 PPO。后者没有使用 BC
权重。训练前固定数据和所有预算；BC 以验证动作 MSE 选 epoch，PPO 仅用验证任务比较
第 0、16、32 回合检查点，全部种子选定后再执行该路线的开发测试。

PPO 共执行 96 个训练回合、576,000 个物理步，没有安全终止；这只包含每种子 8 次
采样更新，不能当作收敛结果。完整验证/开发评价另执行 72 回合。
BC 的 NumPy 导出重新计算离线 MSE，与训练记录差异小于 `1e-18`，但学生实际访问状态
上的教师动作误差明显增大。这个诊断不改变已选权重，也不构成分布偏移原因的消融证明。

[全部种子结果](../../results/franka_surface_learning_pilot/README.md)保存数值、检查点选择与
两份可重算物理指标的代表轨迹。[学习指南](../surface_learning_pilot.md)给出 BC/PPO 公式和复现命令。

### BC 输入消融

首轮拟合完成后，分别屏蔽上一步残差和全部教师非必要输入，保留相同网络与训练预算。
两种配置的三个种子均通过验证和开发测试。配置选择只使用完整验证集，开发测试之前
固定全部六个候选；学生状态上的教师动作误差只做诊断，不回填训练数据。

[全部对照](../../results/franka_surface_bc_transfer/README.md)包含原始 49 输入成绩和两份
完整代表轨迹；[输入索引、公式和复现](../bc_closed_loop_transfer.md)解释了掩码怎样合入
第一层权重。48 个新回合重复使用既有公开 case，不是 48 个独立场景。
公开的四个开发测试 case 只有两个物理任务组；不能把重复训练种子计为新的独立场景。

### BC 隐藏层初始化的 PPO

BC 完成后另行冻结这轮对照。每个种子复制对应 BC 的两个隐藏层，动作层重新置零，
价值网络保持原种子的全新初始化。两组都在带摩擦前馈的控制器上学习残差，不能把
自适应导纳上的 BC 动作直接拿来作为同一残差。

三个种子各训练 32 回合，只比较最终检查点；历史 fresh PPO 也统一使用第 32 回合，
不用先前验证选出的不同回合混合比较。全部验证成绩和候选哈希固定后才运行开发测试。
验证物理或跟踪门槛失败时，保留否决记录，不靠筛除失败回合改善平均数。

[初始化与复现说明](../bc_to_residual_rl.md)解释了动作语义和共享随机数流的限制。
[公开对照](../../results/franka_surface_ppo_transfer/README.md)保存训练曲线、所有迁移检查点、
历史最终检查点及原始评估报告。仍是公开任务域上的方法开发，不是新盲测。

实际完成 96 个新训练回合、576,000 个物理步，没有失败训练回合。验证集切向差值
（迁移减去 fresh）为 `+0.039、+0.063、−0.092 mm`，不满足三个种子一致改善的规则，
因此保留 `fresh_ep32`。之后两组共 24 个开发回合全部通过原门槛，但仍没有一致收益。
