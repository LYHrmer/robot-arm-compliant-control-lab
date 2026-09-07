# BC 隐藏层初始化的 PPO 对照

该归档比较固定第 32 回合的历史 fresh PPO 与 BC 隐藏层初始化 PPO。
validation/development_test 都是已公开开发域，不是新盲测；结果不能推出真机表现。
终态：`complete`。只比较第 32 回合，没有从 0/16/32 中挑选检查点。

验证允许进入开发集：`True`；
三种子一致改善：`False`；
按冻结规则保留：`fresh_ep32`。

| seed | fresh 验证切向 RMSE [mm] | transfer 验证切向 RMSE [mm] | 配对差值 [mm] |
|---:|---:|---:|---:|
| 11 | 4.530 | 4.569 | 0.039 |
| 29 | 4.553 | 4.617 | 0.063 |
| 47 | 4.579 | 4.487 | -0.092 |

训练 manifest 保存完整曲线；公开副本仅保留每个 transfer 种子的 0/16/32 权重。
成功完成时附 fresh/transfer 各一份首种子、首开发 case 的完整物理 NPZ；
其他训练和评价原始轨迹保留在本地。因此 `all_raw_traces_distributed=false`。

实验语义和后续步骤见 [BC 到残差 RL](../../docs/bc_to_residual_rl.md)。

```bash
python -m tools.publish_surface_ppo_transfer --audit results/franka_surface_ppo_transfer
```
