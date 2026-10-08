# 实验四：BC 的离线误差更低，闭环为什么反而不达标

预计学习 30–45 分钟。先读[表面残差策略](../05_residual_rl.md)中的观测、动作与安全层，
再完成本页四个练习。前三个练习和第四个练习的读数命令只读取 Git 中的 JSON；
每条通常在数秒内完成，不训练、不启动 MuJoCo、不生成结果文件，也不依赖本机的大型 `dist/`。
所有命令都从**仓库根目录**运行。练习三需要 NumPy，其余读数只用 Python 标准库。

本页使用 [2026-10-09 BC 修复证据](../../../results/franka_bc_deployment_fix/evidence.json)。
同四个公开 development case 上，原 BC 最差切向 RMSE 为 **11.493746 mm**，
重新训练的 BC 为 **2.162149 mm**，原门槛是 10 mm。
这次实验解释一个已知失败并复核修复；`new_holdout=false`，不能把这些 case 称为新的盲测。

## 先把教师、学生和控制器分开

这里的 BC（behavior cloning，行为克隆）学习的是**残差动作**。每次策略调用输入 49 维
归一化观测，输出 3 维归一化动作，分别对应表面法向和两个切向；动作范围为 `[-1, 1]`。
它们经比例换算、滤波、变化率限制与安全投影后叠加到 adaptive nominal 控制器。
策略每 20 ms 更新一次，底层控制／仿真步长为 2 ms。

教师是 [`friction_teacher_action`](../../../src/compliant_control_lab/surface_policy.py)：
利用修正法向力和目标切向速度生成有界摩擦补偿，再除以动作尺度得到标签。
教师不读取仿真隐藏的真实摩擦系数；本实验也没有更改示教标签。
学生是 `49 → 32 → 32 → 3` 的 tanh MLP，训练目标是在教师轨迹的观测上减小动作 MSE：

$$
L(\theta)=\frac{1}{3N}\sum_{i=1}^{N}\|\pi_\theta(o_i)-a_i^{\mathrm{teacher}}\|^2.
$$

这个式子衡量归一化动作拟合误差，既不是毫米，也不等于接触任务的闭环误差。
部署后，学生的动作会改变下一时刻的运动、力和观测；此时访问的状态可能偏离教师示教。
原模型的索引 14、15、16 分别是上一时刻实际施加的法向、切向 1、切向 2 残差力的归一化值。
教师轨迹里，这些历史量与下一动作高度相关；学生可以依赖它们形成预测捷径。
学生自己运行时，历史量来自自身动作经过控制处理后的结果，错误可能沿反馈继续传播。

因此，“训练集和验证集上的动作 MSE 很低”并没有证明闭环能跟上轨迹。
本次修复用训练时输入消融检验这一解释：将三个历史残差观测置零后重新训练，
不改变教师、网络宽度、动作接口或验收门槛。只在旧权重推理时突然删掉输入不能替代这个实验。

## 练习一：谁可以参与训练，谁可以选择 checkpoint

先预测：24 个回合是否等于 24 个独立物理工况？检查训练计划与实际选模记录：

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path('results/franka_bc_deployment_fix')
plan = json.loads((root / 'study_plan.json').read_text())
training = json.loads((root / 'training_report.json').read_text())
groups = {}
for split in ('train', 'validation', 'development_test'):
    cases = [case for case in plan['cases'] if case['split'] == split]
    groups[split] = {case['group_id'] for case in cases}
    print(split, 'episodes=', len(cases), 'physical_groups=', len(groups[split]))
print('groups_disjoint=', all(
    not groups[a] & groups[b]
    for a, b in (('train', 'validation'), ('train', 'development_test'),
                 ('validation', 'development_test'))))
print('training=', plan['training'])
print('selection=', training['selection'])
print('dev_loaded_for_training_or_selection=',
      training['data_use']['development_test_loaded_for_training_or_selection'])
PY
```

<details>
<summary>参考答案：按物理组隔离，按验证动作 MSE 选 epoch</summary>

应看到 16 train／4 validation／4 development-test 回合，分别属于 8／2／2 个物理组，
`groups_disjoint=True`。每个物理组各用 seed 11、29 采集一次；这些噪声重复不增加独立工况数。
网络训练种子另为 11，训练预算为 100 epochs，batch size 256，学习率 0.001。

只用 train 更新权重；每个 epoch 在 validation 的示教观测上计算动作 MSE，
取最小有限值，完全相同则保留更早的 epoch。本轮选中 **epoch 96**，
validation action MSE 为 **0.009877272832683642**。
`dev_loaded_for_training_or_selection=False`。

选定权重后，另跑 4 个 validation 闭环回合；全部原门槛通过，才固定候选并运行 development。
这个资格检查记录在 [`eligibility.json`](../../../results/franka_bc_deployment_fix/eligibility.json)。
development 用于报告最终表现，不能再拿它挑 epoch 或悄悄修改阈值。
此前已经看过的 development 失败仍属于公开诊断资料；本轮没有新留出集。

</details>

## 练习二：动作 MSE 和闭环 RMSE 会给出同一个排序吗

原 full49 BC 的 validation action MSE 为 **0.002936716749148043**（选中 epoch 68）。
它比新模型低。先判断哪一个闭环切向误差更小，再读取同一组 development case：

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path('results/franka_bc_deployment_fix')
evidence = json.loads((root / 'evidence.json').read_text())
training = json.loads((root / 'training_report.json').read_text())
old = {row['case_id']: row for row in evidence['old_bc_report']['runs']}
new = {row['case_id']: row for row in evidence['development']['runs']}
assert old.keys() == new.keys()
print('old validation action MSE:',
      evidence['old_bc_training_selection']['selected_validation_action_mse'])
print('new validation action MSE:',
      training['selection']['selected_validation_action_mse'])
for case in sorted(old):
    print(case, f"{old[case]['tangent_rmse_mm']:.6f} -> "
                f"{new[case]['tangent_rmse_mm']:.6f} mm")
print('new validation closed-loop worst:',
      evidence['validation']['gates']['tangent_rmse_mm']['actual'], 'mm')
PY
```

<details>
<summary>参考答案：同组闭环改善，离线 MSE 却增加</summary>

| development case | 原 BC 切向 RMSE (mm) | 新 BC 切向 RMSE (mm) |
|---|---:|---:|
| `g000_seed11` | 11.367344 | 2.158166 |
| `g000_seed29` | 11.363376 | 2.161156 |
| `g001_seed11` | 11.493746 | 2.157123 |
| `g001_seed29` | 11.489305 | 2.162149 |

新模型的离线 validation action MSE 增到约 0.009877，但这四个 development 闭环回合均明显改善。
移除历史残差后，模型不能再用同一捷径拟合教师轨迹，需要更多依赖当前测量和目标信息。
这支持该输入消融在当前任务上的作用，也说明用离线损失替代闭环检查会漏掉问题。
它不证明屏蔽历史动作对所有模仿学习任务都更好。

新模型 validation 闭环最差值是 **8.099535 mm**。validation 使用 `g003/g007`，
development 使用 `g000/g001`，所以不能把 8.099535 → 2.162149 说成同一工况的改进。
可配对的前后比较是上表中的相同 case。

</details>

## 练习三：删掉三个输入，为什么仍然接收 49 维观测

设掩码 $m\in\{0,1\}^{49}$，只有 $m_{14},m_{15},m_{16}=0$。训练时输入为
$o\odot m$；第一层计算 $W_1(o\odot m)+b_1$。导出时使用

$$
W'_1=W_1\operatorname{diag}(m),\qquad W'_1o+b_1=W_1(o\odot m)+b_1.
$$

所以部署第一层仍为 `32 × 49`，只是对应的三列为零。运行时无需增加额外 mask 参数，
也无需把环境接口改成 46 维。对照
[`_mask_observations` 与 `_fold_input_mask`](../../../tools/train_surface_bc_transfer.py)。

在已安装 NumPy 的 Python 环境中执行下列数值核对。这里只按 JSON 权重计算网络；
正式部署还须使用 `franka-learning-deploy verify` 核对完整接口、运行版本和外部哈希。

```bash
python3 - <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

path = Path('results/franka_bc_deployment_fix/bundle/bc.json')
model = json.loads(path.read_text())['body']['payload']
layers = model['layers']
first = np.asarray(layers[0]['weights'], dtype=np.float64)
assert first.shape == (32, 49)
assert np.count_nonzero(first[:, 14:17]) == 0

def predict(observations):
    values = observations.copy()
    for layer in layers:
        weight = np.asarray(layer['weights'], dtype=np.float64)
        bias = np.asarray(layer['bias'], dtype=np.float64)
        values = np.tanh(values @ weight.T + bias)
    return values

rng = np.random.default_rng(11)
observations = rng.uniform(-3, 3, (256, 49))
changed = observations.copy()
changed[:, 14:17] = rng.uniform(-3, 3, (256, 3))
actions = predict(observations)
print('first_layer_shape=', first.shape)
print('masked_column_nonzeros=', np.count_nonzero(first[:, 14:17]))
print('max_action_change=', np.max(np.abs(actions - predict(changed))))
print('actions_shape=', actions.shape)
print('finite_and_bounded=', bool(np.isfinite(actions).all() and (np.abs(actions) <= 1).all()))
print('torch_imported=', 'torch' in sys.modules)
PY
```

<details>
<summary>参考答案：96 个权重严格为零，扰动不改变输出</summary>

应得到 `(32, 49)`、`0`、`0.0`、`(256, 3)`、`True`、`False`。
三列共 32 × 3 = 96 个权重严格为零，因此改变这些输入不改变网络动作。
这个练习只证明导出模型对被屏蔽输入不敏感，以及这段推理不导入 Torch；随机观测没有物理
轨迹含义，也不能单靠动作有限有界判断接触稳定。

训练报告还保存真实的 9,600 train + 2,400 validation 观测上的导出对照：
`diagnostics.folded_export.torch_masked_vs_numpy_raw_max_abs` 为约 **7.75e-16**，
低于 **2e-14** 容差；被屏蔽维度的极值扰动导致的最大动作变化为 **0**。
这样，训练时的 masked Torch 网络与部署时接收原始 49 维观测的 NumPy 网络得到同一动作。

BC 在 adaptive nominal 上学习教师补偿，PPO 在 friction nominal 上学习残差；
输入输出维数相同不表示契约相同。不能把 BC 权重直接交给 PPO 的评估协议。

</details>

## 练习四：切向达标，是否就能宣布验收通过

先列出其余六项原门槛，再核对报告。接触率取所有回合的最小值，其余指标取最大值：

```bash
python3 - <<'PY'
import json
from pathlib import Path

evidence = json.loads(Path('results/franka_bc_deployment_fix/evidence.json').read_text())
for name in ('validation', 'development'):
    report = evidence[name]
    print(name, 'episodes=', len(report['runs']),
          'complete=', report['all_episodes_succeeded'],
          'metrics_available=', report['all_metrics_available'],
          'acceptance_met=', report['acceptance_met'])
    assert len(report['gates']) == 7
    for metric, gate in sorted(report['gates'].items()):
        rule = gate['rule']
        print(metric, f"{gate['actual']:.6f}", rule['operator'], rule['value'],
              'PASS' if gate['passed'] else 'FAIL')
PY
```

<details>
<summary>参考答案：七项均通过，工程状态还要单独核对</summary>

两个报告都应为 4 个回合，完成、指标齐全与 `acceptance_met` 均为 `True`。
本轮 development 的原门槛和结果如下：

| 指标 | 原门槛 | development 最差值 |
|---|---:|---:|
| 接触率 `contact_ratio_pct` | ≥ 99% | 100% |
| 切向 RMSE `tangent_rmse_mm` | ≤ 10 mm | 2.162149 mm |
| 力 RMSE `force_rmse_n` | ≤ 2 N | 0.164872 N |
| 穿透深度 `max_penetration_mm` | ≤ 2 mm | 0.850948 mm |
| 末端速度 `max_speed_m_s` | ≤ 0.3 m/s | 0.076700 m/s |
| 力峰值 `peak_force_n` | ≤ 35 N | 12.648972 N |
| 力矩饱和比例 `saturation_pct` | ≤ 0% | 0% |

完整部署报告中的 `engineering_status` 检查回合完成与本机推理耗时；
`policy_acceptance` 检查策略的绝对物理门槛；`paired_comparisons` 报告相对各自名义控制器的差值。
三类证据回答不同问题。绝对门槛通过不等于优于解析控制，也不代表新工况或真机验证完成。
动作 MSE、JSON 完整性和推理耗时都不能替代这七项闭环门槛。

</details>

需要实际重跑时，按[部署说明](../../learning_deployment.md)取得对应发行目录，
在仓库根目录执行 `./scripts/accept_learning_release.sh`。这一步会建立或检查独立运行环境，
用 NumPy 推理运行 BC、PPO 及各自名义控制器，合计 **16 个 12 秒仿真回合**；
它不重新训练，但会写入新的结果目录。墙钟耗时取决于主机，建议预留 5–15 分钟，
这只是操作时间预算，不是本页实测性能。加 `--dry-run` 只校验，不运行新仿真。
完整复现训练、模型导出与部署的路径见[复现指南](../../learning_reproduction.md)。

最后写一段自己的实验结论，至少包含：输入消融、按验证集选中的 epoch、同 case 的前后结果、
其余六项门槛，以及 `new_holdout=false`。如果只写“BC 达标”而省略这些条件，别人无法判断
你修复了哪个问题，也无法重复验证。
