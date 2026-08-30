# 实验总账（Experiment Record）

这里索引已经发布的实验。叙述文档中的数字是阅读摘要；最终以链接的 CSV 和冻结产物为准。

| 版本 | 数据身份 | 结果 | 决策 |
|---|---|---|---|
| Initial | 命名标称场景 | Hybrid force RMSE 0.95 N，saturation 0% | 增加随机参数失配测试 |
| v0.3 | 当时的首次运行；现在是公开验证集 | Fixed hybrid 通过 6/24 | 可以开展 bounded residual 实验；尚未证明 RL 更好 |
| v0.4 | seed-29 公开验证集 | Fixed 6/24，adaptive 6/24，residual 7/24 | 保留自适应 nominal；不部署 checkpoint |
| v0.5 | 48-case first reveal；揭盲后转为公开验证集 | 五个 residual 为 22–26/48，均未达到 44/48；torque-safe adaptive 与五个 residual 的 saturation 均为 0% | 不部署；检查 nominal approach 和 contact transition |

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

方法细节见 [v0.5 protocol](../reproduction_plan_v0.5.md)；是否部署和下一条假设见
[Residual RL decision](../residual_rl_decision.md)。独立的
[post-reveal report](../../results/franka_safety_postreveal/summary.md) 只统计失败原因，不改动
冻结证据。
