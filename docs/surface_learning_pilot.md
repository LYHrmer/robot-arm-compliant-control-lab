# 小规模行为克隆，再做有界残差 PPO

这次训练沿用[表面任务的接口和教师数据](surface_learning.md)，先检查网络能否学到解析教师，
再检查学习残差能否改善已有的摩擦前馈。实际数值、所有训练种子与检查点选择记录见
[训练结果](../results/franka_surface_learning_pilot/README.md)。旧 v0.5 的 `FAIL` 保持不变。

首轮结果是：BC 离线拟合良好，闭环跟踪对初始化敏感；PPO 在验证集得到微小改善，
开发测试却没有一致收益。三个 PPO 候选相对强基线的平均切向误差差值为
`+0.064、+0.068、−0.022 mm`。目前没有充分理由用这批模型替换解析摩擦前馈。

## 两个网络，各自补什么

BC 的名义控制器是自适应导纳，网络学习摩擦教师的请求动作。PPO 的名义控制器已经带有
摩擦前馈，因此从零均值残差开始。两者都输出三轴动作，但不能直接交换权重。

网络是 `49 → 32 → 32 → 3`，两个隐藏层和输出层均为 `tanh`。输入只包含已有的测量观测，
按固定物理尺度归一化；不额外拟合 normalizer，也不输入真实摩擦系数或评价真值。
三轴归一化请求对应 `(4, 6, 6)` N 的力边界，随后经过已有的门控、滤波与力矩投影。
命令有界不保证接触力有界；仍需检查原始力和实际运动。

## 行为克隆的目标

教师通过已测法向力与切向目标速度计算摩擦补偿。数据中的标签是进入安全层前的
归一化请求 `a_teacher`，不能替换成最终施加的力。

对训练批次的 N 个样本，优化目标为：

```text
a_pred = tanh(W3 tanh(W2 tanh(W1 observation + b1) + b2) + b3)
L_BC   = sum((a_pred[i,j] - a_teacher[i,j])²) / (3N)
```

这里拟合的是控制规则，不是直接拟合末端轨迹。某个观测上的动作误差很小，也可能在
连续反馈中累积成偏差，所以离线 MSE 之后还有闭环仿真。

完整语料按物理任务组划分：训练 16 回合、9,600 步；验证 4 回合、2,400 步。
另 4 回合用于开发测试，不参与优化和模型选择。动作序列没有随机拆散后跨集合分配。
加载器会审计完整语料的完整性，但只将 train/validation 的数组交给 BC。

每个种子先用独立初始化的网络拟合固定 64 个训练样本，运行 300 次更新。
这是实现检查，不能用来选架构或训练预算。正式训练用 Adam，学习率 `0.001`、
batch 256，完整运行 100 epoch。每轮记录训练与验证误差，取完整验证集 MSE 最低的
epoch；数值完全相同时保留更早的 epoch。

种子固定为 `11 / 29 / 47`。报告同时列出零动作与训练标签均值预测器的 MSE，
并拆分法向和两条切向轴，避免零法向标签掩盖切向拟合问题。
实现见 [`train_surface_bc.py`](../tools/train_surface_bc.py)。

### 这次 BC 的离线成绩为什么不够

三个种子的离线验证 MSE 都很低，但在四个验证任务中，没有一个种子通过全部跟踪门槛。
将保存的候选文件重新加载到 NumPy，离线 MSE 与训练记录的差异小于 `1e-18`，
因此这组现象不能归咎于导出后 loss 对不上。

在学生已经执行过的验证轨迹上，用原教师公式重新计算标签，得到下面的动作误差：

| 训练 seed | 教师数据上的 MSE | 学生实际访问状态上的 MSE |
|---:|---:|---:|
| 11 | 0.001564 | 0.361475 |
| 29 | 0.001180 | 0.090741 |
| 47 | 0.001570 | 0.344482 |

两列都计算三轴归一化请求的平均平方误差，但观测分布不同。这与 BC 的闭环分布偏移问题
相符；目前还没有通过输入消融或重新采样，证明是哪一项观测依赖导致偏差。
这项复核只读取已完成的 validation 轨迹，不改变选中的 epoch。

开发测试的跟踪通过数分别为 `0/4、4/4、2/4`，12 回合均安全完成。不能只展示 seed 29，
也不能把安全完成等同于成功模仿教师。后续若继续改善 BC，优先检查学生状态覆盖和
不必要的观测依赖，再决定是否增加网络规模。

## PPO 怎样更新残差

PPO 使用独立的 actor 和 value 网络。actor 最后一层权重和偏置均初始化为零，
所以第 0 个检查点的确定性动作严格为零。训练时在 `tanh` 之前加入固定标准差
`σ = 0.05` 的高斯探索；评价时只执行均值的 `tanh`。

```text
z = μθ(observation) + σ ε,   ε ~ N(0, I)
a = tanh(z)
log πθ(a|o) = sum(log Normal(zj; μθ,j(o), σ) - log(1 - tanh(zj)²))
```

采样时保留 `z` 和旧 log probability，更新时不对接近边界的动作做 `atanh`。
代码用等价的 softplus 表达计算 Jacobian，减轻浮点溢出问题。
PPO 的比值和裁剪目标为：

```text
ratio = exp(log π_new - log π_old)
L_actor = -mean(min(ratio * A, clip(ratio, 0.8, 1.2) * A))
L_value = 0.5 * mean((V(observation) - return_target)²)
L_total = L_actor + L_value
```

优势在每次更新的采样批次内标准化，环境回报不做标准化，也不另加一次失败惩罚。
固定探索方差，没有额外熵奖励。PPO 的 clipped objective 来自
[原论文](https://arxiv.org/abs/1707.06347)；本项目采用上述小网络与固定预算，
不宣称复现论文中的性能。

### `terminated` 和 `truncated` 不能合成同一个 mask

一次学习动作通常执行 10 个物理步，也可能提前终止。设本次实际执行步数为 `n`：

```text
duration_fraction = n / 10
discount = 0.99 ** duration_fraction
trace_decay = 0.95 ** duration_fraction
bootstrap = not terminated and terminal_observation_valid
continue_trace = bootstrap and not truncated
δt = rt + discount * bootstrap * V(next_observation) - V(observation)
At = δt + discount * trace_decay * continue_trace * A[t+1]
return_target = At + V(observation)
```

时间上限截断仍可用最后的有效状态估值，但优势递推不能穿过 reset 接到下一回合。
安全终止不 bootstrap。无效下一观测先按 mask 处理，不能依靠 `0 × NaN` 清除 NaN。
这是 [GAE](https://arxiv.org/abs/1506.02438) 在本环境的变时长与回合边界处理。

每个种子尝试 32 个训练回合。16 个训练 case 每轮打乱，遍历两轮；每次 reset 另取
可追溯的随机种子。每 4 回合更新一次，因此总共只有 8 次采样更新。
每次最多 4 轮 minibatch 更新，batch 128、Adam 学习率 `0.0003`，梯度范数上限 `0.5`，
KL 停止阈值 `0.02`。失败回合也占预算并保留记录，不补跑来凑成功次数。
KL 在一次梯度更新后计算，超过阈值就停止本批剩余更新，不回滚刚完成的那一步。
所以它是提前停止条件，日志中的近似 KL 可能超过 `0.02`，不构成硬约束。

每个正常 12 s 回合有 600 个学习步，完整预算每种子最多 19,200 个学习步。
这只够做小规模试验；网络没有改善时，不能据此断言残差 RL 无效。
实现见 [`train_surface_ppo.py`](../tools/train_surface_ppo.py)。

## 先选模型，再打开开发测试结果

BC 先按离线验证 MSE 选 epoch，然后运行完整闭环验证；闭环成绩不再用于挑另一个 epoch。
PPO 保留第 `0 / 16 / 32` 回合检查点，每个都运行相同的四个验证 case。
四回合都安全完成且满足跟踪门槛，才有资格参加选择。

跟踪门槛保持为接触比例 ≥99%、切向 RMSE ≤10 mm、法向力 RMSE ≤2 N。
物理门槛包含原始力 ≤35 N、穿透 ≤2 mm、速度 ≤0.3 m/s、零执行器裁剪，
擦拭连续失联严格少于 0.1 s。缺指标或缺回合均不通过。

合格检查点按平均切向 RMSE 排序，再比较平均法向力 RMSE，最后取更早的检查点。
若没有合格项，记录“无合格策略”，选择零残差用于对照，但不把它标为通过。
第 32 回合的已训练检查点始终保留验证成绩，即使最终选择的是零残差。

每条路线的三个种子全部选定后，保存选择文件及候选哈希，才运行开发测试。
对照来自相同 case、相同实际仿真种子的历史强基线和同频教师；所有种子分别报告。
开发测试仍是公开域中的 4 个 case，不能把这 4 个或重复的训练种子说成新的 24-case 盲测。

## 本地复现

在仓库根目录安装可选训练依赖：

```bash
pip install -e ".[dev,learning]"
```

按[准备数据的命令](surface_learning.md#数据先分组再切窗口)生成 v2 全集。
随仓库附带的三个示例回合不够训练，训练器会拒绝它们。
本次使用 CPU 和 float64，每次训练限一个 PyTorch 计算线程。
软件版本随产物记录；换物理引擎或数值库后，应重新生成相匹配的数据和基线。

```bash
python -m tools.surface_learning_pilot freeze \
  --dataset dist/surface-preparation-v2/dataset \
  --benchmark dist/surface-preparation-v2/benchmark \
  --output dist/surface-learning-pilot-v1
```

保存命令输出的 SHA，以下两步中的 `FROZEN_PLAN_SHA` 替换成这个原始值。
不要在文件改变后重新计算一个值来冒充训练前的记录。

```bash
python -m tools.surface_learning_pilot bc \
  --dataset dist/surface-preparation-v2/dataset \
  --benchmark dist/surface-preparation-v2/benchmark \
  --output dist/surface-learning-pilot-v1 --expected-plan-sha256 FROZEN_PLAN_SHA
python -m tools.surface_learning_pilot ppo \
  --dataset dist/surface-preparation-v2/dataset \
  --benchmark dist/surface-preparation-v2/benchmark \
  --output dist/surface-learning-pilot-v1 --expected-plan-sha256 FROZEN_PLAN_SHA
```

PPO 阶段要求 BC 产物先完成。训练失败、已完成阶段和来源不匹配都会明确报错，
不会覆盖结果重跑开发测试。中途退出可复用已经完整保存且哈希匹配的子任务；
未完成的训练不会冒充一次完整运行。

训练产物保存 epoch/update 曲线与候选 JSON，PPO 另存每回合全部物理轨迹和采样动作。
闭环评价仍由已有的受限 NumPy runner 执行，不加载 pickle 或任意候选代码。
评价器只测推理调用返回后的耗时，没有实时抢占或真机 watchdog。
哈希能发现产物改动，不能证明独立预注册或首次揭盲。

公开结果含完整评测候选权重和训练统计，另附首个训练种子、首个开发 case 的 BC/PPO
完整物理轨迹各一份；其余原始 NPZ 与事件日志保留在本地全集中。
读取公开副本并重算两份代表轨迹的指标：

```bash
python -m tools.publish_surface_learning_pilot --audit results/franka_surface_learning_pilot
```

重训完成后可重新生成小体积公开副本，输出必须使用新目录：

```bash
python -m tools.publish_surface_learning_pilot \
  --pilot dist/surface-learning-pilot-v1 \
  --dataset dist/surface-preparation-v2/dataset \
  --benchmark dist/surface-preparation-v2/benchmark \
  --output dist/surface-learning-pilot-public \
  --expected-plan-sha256 FROZEN_PLAN_SHA
```
