# 表面行为克隆输入消融

该归档比较历史 full49 控制与两个预先固定的输入掩码。所有结果来自同一公开 16/4/4 分组、
同一训练预算和种子 11/29/47；development_test 已经是公开开发集，不是新盲测。
这是输入组干预比较，但不能据此断言某个单独通道是唯一原因，也不能推出真机泛化能力。

| 输入 | seed | 集合 | 全部门槛 | 安全完成 | 跟踪通过 | 平均切向 RMSE [mm] | 平均力 RMSE [N] |
|---|---:|---|---|---:|---:|---:|---:|
| full49 | 11 | validation | FAIL | 4/4 | 0/4 | 14.577 | 0.143 |
| full49 | 29 | validation | FAIL | 4/4 | 2/4 | 8.477 | 0.155 |
| full49 | 47 | validation | FAIL | 4/4 | 0/4 | 14.723 | 0.146 |
| drop_previous_residual | 11 | validation | PASS | 4/4 | 4/4 | 5.072 | 0.151 |
| drop_previous_residual | 29 | validation | PASS | 4/4 | 4/4 | 5.083 | 0.147 |
| drop_previous_residual | 47 | validation | PASS | 4/4 | 4/4 | 5.002 | 0.149 |
| teacher_inputs | 11 | validation | PASS | 4/4 | 4/4 | 5.435 | 0.148 |
| teacher_inputs | 29 | validation | PASS | 4/4 | 4/4 | 5.799 | 0.148 |
| teacher_inputs | 47 | validation | PASS | 4/4 | 4/4 | 5.788 | 0.146 |
| full49 | 11 | development_test | FAIL | 4/4 | 0/4 | 14.787 | 0.204 |
| full49 | 29 | development_test | PASS | 4/4 | 4/4 | 3.801 | 0.212 |
| full49 | 47 | development_test | FAIL | 4/4 | 2/4 | 8.721 | 0.198 |
| drop_previous_residual | 11 | development_test | PASS | 4/4 | 4/4 | 2.818 | 0.210 |
| drop_previous_residual | 29 | development_test | PASS | 4/4 | 4/4 | 2.825 | 0.206 |
| drop_previous_residual | 47 | development_test | PASS | 4/4 | 4/4 | 2.781 | 0.207 |
| teacher_inputs | 11 | development_test | PASS | 4/4 | 4/4 | 3.683 | 0.205 |
| teacher_inputs | 29 | development_test | PASS | 4/4 | 4/4 | 3.991 | 0.202 |
| teacher_inputs | 47 | development_test | PASS | 4/4 | 4/4 | 3.732 | 0.201 |

验证集预先选择：`selected_arm=drop_previous_residual`，`eligible=True`；完整均值和排序见 [selection.json](selection.json)。该选择不使用 development_test。

每个掩码和种子的权重、完整 epoch 曲线、训练报告、12 份闭环评价报告及其协议均已复制。
归档仅附每个掩码各一份代表性完整物理 NPZ：首个种子和按字典序首个开发 case。
其他物理轨迹和逐决策事件只保留在本地源产物中；source_manifest/source_COMPLETE 只是哈希身份记录。
根 manifest.json 仅绑定实际分发文件。所有候选仅限仿真，不能直接部署到真机。

```bash
python -m tools.publish_surface_bc_transfer --audit results/franka_surface_bc_transfer
```
