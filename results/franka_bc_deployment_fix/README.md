# BC 输入依赖修复与离线部署证据

固定当前 24-case 数据分组、教师标签、adaptive nominal 和全部门槛，只将 BC 的
`previous_residual` 三个观测通道（14、15、16）在训练时屏蔽，导出时合入第一层权重。
seed 11，100 epochs，按离线验证动作 MSE 选择 epoch 96；随后四个闭环验证 case 全部通过，
才固定模型进行四个开发测试。最差开发切向 RMSE 由旧 full49 的 11.494 mm 降为
2.162 mm。源码与无 Torch 安装版各 16 回合全部正常完成，BC/PPO 均达标，命令退出 0。

| 文件 | 内容 |
|---|---|
| [evidence.json](evidence.json) | 本轮完整门槛、旧 BC 对照、训练选择与部署核验 |
| [bundle/](bundle/) | BC、PPO、各自 nominal、全部 cases、冻结协议及封存 manifest |
| [study_plan.json](study_plan.json)、[eligibility.json](eligibility.json) | 训练前计划、开发测试前的验证通过记录与候选 SHA |
| [training_report.json](training_report.json)、[epoch_losses.csv](epoch_losses.csv) | 训练／验证损失、epoch 96 选择、导出一致性检查 |
| [validation_report.json](validation_report.json)、[development_report.json](development_report.json) | 4 + 4 个完整任务的逐例指标与门槛 |
| [validation_protocol.json](validation_protocol.json) | 绑定公开 BC 候选和 4 个验证 case 的冻结闭环评估协议 |
| [source_acceptance.json](source_acceptance.json)、[installed_acceptance.json](installed_acceptance.json) | 源码及安装版的 16 回合配对验收 |
| [installation_audit.json](installation_audit.json) | 35 项独立安装核验，含来源、依赖与无 Torch 检查 |
| [diagnosis_before.json](diagnosis_before.json)、[diagnosis_after_validation.json](diagnosis_after_validation.json) | 动作历史依赖诊断及屏蔽后的独立输入扰动检查 |

训练使用 commit `097e460bd55fe7145678fb23f85181a815d1b818` 中已有的输入消融 trainer，
部署源码冻结于 `b698e84dd02ece4063ef46812b318d9260195924`。后者只接入该训练路径，
没有修改 teacher、控制器、物理环境或旧归档。PPO 模型与上一版逐字节相同。

bundle manifest 的外部 SHA256：
`1048a6280d3231ac661a3f0b0ad86b68a208eb18f82c785a227fc52218de6961`。
BC 模型 SHA256：`8f1a3e0b3b2263cfadb0a8fda40e398db9e0d9c19bb4e67ab07a21e4be9e1569`。

本目录提供小型模型与完整数值报告。大体积原始轨迹保存在本机
`dist/bc-feedback-fix-20261009/{validation,development}` 及两份部署验收目录，
原始文件 SHA 保存在对应 manifest；它们不随 Git 上传。
可按[复现指南](../../docs/learning_reproduction.md#复核公开-bc-的四个闭环验证任务)重跑公开
BC 的 4 个闭环验证任务，再运行 bundle 的 16 个部署开发测试回合，得到新的轨迹与报告。
这些重跑产物不保证与原始归档逐字节一致。常规 `prepare` 只包含 16 个部署开发测试回合，
不会另行执行候选的 4 个闭环 validation 回合；实际研究先验证、后固定候选的资格流程
由本目录的 `study_plan.json`、`eligibility.json` 另行记录。底层评估 CLI 退出 `0` 也不等于
门槛通过，须检查报告的 `acceptance_met`、`all_episodes_succeeded` 和 `all_metrics_available`。
这是公开任务域中的单训练种子工程验证，不构成新 holdout 或跨种子优越性结论。
