# 实验总账（Experiment Record）

这里索引已经发布的实验。叙述文档中的数字是阅读摘要；最终以链接的 CSV 和冻结产物为准。

| 版本 | 数据身份 | 结果 | 决策 |
|---|---|---|---|
| Initial | 命名标称场景 | Hybrid force RMSE 0.95 N，saturation 0% | 增加随机参数失配测试 |
| v0.3 | 当时的首次运行；现在是公开验证集 | Fixed hybrid 通过 6/24 | 可以开展 bounded residual 实验；尚未证明 RL 更好 |
| v0.4 | seed-29 公开验证集 | Fixed 6/24，adaptive 6/24，residual 7/24 | 保留自适应 nominal；不部署 checkpoint |
| v0.5 | 48-case first reveal；揭盲后转为公开验证集 | 五个 residual 为 22–26/48，均未达到 44/48；torque-safe adaptive 与五个 residual 的 saturation 均为 0% | 不部署；分别检查早期入触和较晚擦拭峰值 |
| v0.6 开发对照 | 同一批 48 个公开场景，四组共 192 次仿真 | 同分步时序下原始参考 23/48、解析速度 24/48、限速参考 23/48 | 限速参考保留为实验项；尚未完成低冲击控制 |

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
