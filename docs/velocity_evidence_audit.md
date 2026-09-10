# 复核预算与速度误差这条证据链

在仓库根目录运行：

```bash
python -m tools.audit_velocity_evidence
```

这条命令依次检查四份已发布归档：预算转移、速度代价分解、内部观测、速度误差时间系数对照。
它先核对每份 manifest 的固定 SHA-256，再调用对应的原审计工具。不训练、不重新积分动力学，
也不修改归档。依赖沿用[首页安装](../README.md#安装后快速复核)，仍需安装 MuJoCo；
原审计工具会导入它，但这次没有新仿真。

## 输出应该怎样读

顶层 `audit_status` 只回答四份数据能否按原口径复核。当前应为 `PASS`，
`simulations_executed` 应为 `0`。每个条目单独保留 `experiment_decision`：

| 归档 | 复核状态 | 实验结论或检查内容 |
|---|---|---|
| budget_transfer | PASS | 实验筛查 FAIL，原 public24 兼容 23/48 |
| velocity_cost | PASS | 48 组预算配对、288 个窗口；没有新增采用判据 |
| onset_observer | PASS | 8 条父轨迹精确复现、18,000 周期内部更新校验 |
| velocity_time | PASS | do_not_expand，默认时间系数仍为 0.05 s |

两个描述性分析条目的 `experiment_decision` 为 `null`，表示没有新增实验采用决策。
每个条目中的 `archived_summary` 原样保留该审计工具的输出。
其中 `new_simulations: 8` 或 `4` 记录的是当时生成归档的次数，不是这条复核命令新跑的次数。
`audit_status: PASS` 与 `experiment_decision: FAIL` 可以同时出现。
两项采用决策也按固定归档核对：预算转移须为 `FAIL`，时间系数对照须为 `do_not_expand`；
子审计若返回相反结论，聚合器会报错，不会把它当成一次性能改进。

## 复核失败时

任一归档缺失、哈希不符、子命令报错、超时或返回不合格 JSON，顶层状态都会是 `FAIL`，
进程退出码为 `1`。对应条目给出 `error`，其余检查继续执行，不把失败项当作跳过成功。
参数不合法时退出码为 `2`。全部审计通过才返回 `0`，即使实验筛查结论本来就是 `FAIL`。

单项审计默认最多等待 300 秒。较慢的机器可以调整每项等待上限：

```bash
python -m tools.audit_velocity_evidence --timeout-seconds 600
```

固定 manifest 哈希约束的是仓库当前发布的这四份数据。原审计工具还会检查 `COMPLETE`、
文件清单和源码哈希，并从轨迹重算指标；只给改过的数据重新生成一份 manifest 不能通过。
如果修改了控制器源码，先检查它是否仍匹配归档，不要删除这些校验来消除报错。

## 范围

本命令只覆盖上述四项，不代表全仓库所有实验都已审计，也不重新做历史 v0.5 的 BLS 验签。
v0.5 保留原 `franka-published-results-audit` 入口；学习实验等其余检查见
[招聘方走查](recruiter_walkthrough.md#3-跑最短检查约-1-分钟)。
数值结论分别见[预算转移](budget_transfer.md)、[窗口分解](velocity_cost.md)、
[内部观测](onset_observer.md)和[时间系数对照](velocity_time.md)。
