# v0.5 揭盲后接触事件诊断

这份诊断只重放已经公开的 48 个 v0.5 cases。控制器是 torque-safe adaptive baseline，参数和
seed 都来自冻结协议。它用来定位失败，不增加新的 holdout 证据，也不改变 first-reveal 的
`FAIL`。

## 先证明重放的是同一批轨迹

生成器按下面的顺序工作：

1. 用离线 audit 检查 protocol、manifest、reveal 和冻结 CSV；
2. 从 protocol hash 与已归档 beacon 重新派生 scenario/noise seed；
3. 重放全部 48 个 torque-safe adaptive cases；
4. 逐 case 对照 7 个确定性指标，容差为 `1e-12`；
5. 对齐通过后才提取事件并写报告。

本次重放使用 Python 3.10.12、NumPy 2.2.6、MuJoCo 3.12.0，7 项最大绝对误差都是 0。
报告写入器要求每项绝对误差不超过 `1e-12`；更换数值依赖后，需要重新通过这项检查。
`controller_p95_us` 依赖墙钟调度，没有进入确定性对照。

日志在每次 `mj_step` 前写入。此时 contact、site position 和 Jacobian 仍来自上一次 forward
计算，关节速度已经经过前一步积分；表中的 wrench 则是本周期新计算、尚待下一步施加的
指令。这些量组成同一控制周期的记录，不能解释为严格同步的物理瞬时，也不能用当前 wrench
认定已记录峰值的成因。诊断保留了冻结仿真的采样顺序，时间细节约有一个 2 ms 步长的限制。

## 事件怎样定义

| 字段 | 定义 |
|---|---|
| first raw contact | 第一个 $F_{raw}>0$ 的采样 |
| controller confirmation | `contact_blend` 第一次大于 0；这是 controller state 的推断，不等同于第一次物理接触 |
| blended contact | 当前采样仍有 raw contact，且 `contact_blend >= 0.99` |
| wiping | 冻结 target generator 使用 $t>1.20\,s$；500 Hz 下第一个 wiping sample 是 1.202 s |
| global peak | 完整 4.5 s rollout 中的 $\max_t F_{raw}(t)$ |
| torque headroom | $\min_i\{\tau_i-l_i,\ u_i-\tau_i\}$，使用同周期未裁剪 torque |

Contact phase 和 motion phase 分开保存。一个峰值可以同时是 `blended_contact` 与 `wiping`，
不会因为两个状态共存而被塞进含糊的“接触阶段”。

## 48-case 结果

![Safe-adaptive contact peak timing](../results/franka_safety_postreveal/contact_events/contact_peak_timing.png)

| 范围 | Cases | Delay median [P25, P75] | Early <=0.5 s | Late >=1.0 s | Pre-wiping / wiping |
|---|---:|---:|---:|---:|---:|
| 全部峰值 | 48 | 1.976 [1.308, 2.306] s | 9 | 39 | 8 / 40 |
| 35 N gate failures | 18 | 1.774 [0.312, 1.983] s | 7 | 11 | 6 / 12 |

18 个失败峰值的 contact phase 是 raw 2、transition 5、blended 11。按相对首次接触的时间
排序后，中间没有 0.5–1.0 s 的样本：7 个落在 0–0.432 s，另外 11 个落在
1.222–2.484 s。这里更像两个待分别验证的问题。

## 峰值时刻的控制上下文

| 峰值所在周期的量 | Early <=0.5 s，n=7 | Late >=1.0 s，n=11 |
|---|---:|---:|
| 实际 world-x 速度 median [range]，m/s | 0.131 [0.045, 0.197] | 0.0053 [0.0015, 0.0080] |
| 目标 world-x 速度 max abs，m/s | 0.000 | 0.000 |
| 指令 world-x 力 median [range]，N | 11.30 [5.93, 14.38] | 11.62 [9.35, 14.82] |
| Torque headroom min / median，Nm | 5.08 / 6.57 | 7.51 / 7.94 |
| Projection scale min | 1.000 | 1.000 |

表中 x 是世界坐标系的控制器接近轴。场景包含墙面偏航，x 分量与真实墙面法向投影需要
区分；这里没有计算速度或指令力在倾斜墙面法向上的投影。CSV 里的原始接触力仍是 MuJoCo
接触坐标系的法向力。表格描述峰值采样的状态，下面两项解释属于待验证假设。

Early cohort 的实际 x 速度明显更高，而 target x velocity 是 0。当前 reference governor
限制的是 target x velocity；接近动作主要由位置误差和 approach stiffness 产生，所以
`max_approach_velocity` 没有直接限制实际接近速度。下一次可比较带状态速度约束的 reference
ramp 或能量限制，同时检查是否丢失接触。

Late cohort 已进入低 x 速度的擦拭过程。峰值处 projection scale 全为 1，最小 torque
headroom 仍有 7.51 Nm。这里应检查擦拭轨迹的法向耦合和 force-loop dynamics，不能继续把
问题只写成 approach impact。

## 复现

```bash
franka-contact-event-analysis
```

命令会覆盖
[`results/franka_safety_postreveal/contact_events/`](../results/franka_safety_postreveal/contact_events/)
中的派生文件，不会写入冻结的 `results/franka_safety_blind/`。完整事件表在
[`safe_adaptive_contact_events.csv`](../results/franka_safety_postreveal/contact_events/safe_adaptive_contact_events.csv)，
两类峰值上下文在
[`peak_context.csv`](../results/franka_safety_postreveal/contact_events/peak_context.csv)，
生成摘要在
[`summary.md`](../results/franka_safety_postreveal/contact_events/summary.md)。实现和输入校验见
[`contact_event_analysis.py`](../src/compliant_control_lab/contact_event_analysis.py)。

这些 48 cases 已经参与诊断。v0.6 若要给出新的性能结论，需要先冻结分开的 entry/in-contact
peak 指标，再使用未来 beacon 生成未见场景。
