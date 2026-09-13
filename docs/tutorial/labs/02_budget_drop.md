# 实验二：预算回到 6 N，为什么这一拍仍输出 6.171 N

预计 20–30 分钟。需要会算三维向量的长度，并了解“本拍输入、下一拍状态”的区别。
先读 [07 章的更新顺序](../07_measured_budget_replay.md#沿时间顺序看状态)；不需要训练网络。

本实验只读取已发布的 12 秒缺包轨迹，从重置状态重算 6,000 拍切向补偿与负载调度状态。不会重新积分
MuJoCo，也不会生成或修改 `results/` 文件。故障只作用于辅助负载通道，原法向力反馈仍在工作。

## 1. 运行并找到预算下降的那一拍

按[首页](../../../README.md#安装后快速复核)安装后，在仓库根目录执行：

```bash
python -m tools.tutorials.budget_drop
```

应得到下面的输出。它来自保存的测量输入，屏幕上保留 6–9 位小数：

```text
full-state replay: PASS (6000 cycles)
event: missing; rejected packets: 150
first rejection: k=4000, t=8.000 s
applied budget: 6.230884 -> 6.000000 N
load estimate: 5.982861 -> 0.000000 N
coefficient: 0.510661310 -> 0.510661310
previous_force_n: [ 0.000000  5.312579 -3.217125]
desired_force_n: [ 0.000000  5.122746 -3.086342]
force_n: [ 0.000000  5.279640 -3.194431]
vector change: 0.040000 N (limit 0.040000 N/step)
request norm: 6.210748 -> 6.170817 N
first norm <= 6 N: k=4005, t=8.010 s
packet recovery: t=8.300 s
```

`k` 从 0 开始，控制周期为 0.002 s。因此 `k=4000` 对应 8.000 s。
辅助包缺失期间没有把上一份测量再送给调度器：负载估计立即清零，预算回到 6 N。
在线等效系数保留下来；这一拍因请求仍受变化率限制，更新条件不满足，系数前后相同。
不能据此推断所有缺包周期都冻结系数。原运动误差反馈及其更新条件仍然有效。

## 2. 先手算，再展开答案

只看上面的三个向量，回答：

1. 从上一拍请求直接跳到期望请求，向量一共要变化多少 N？
2. 20 N/s 的变化率限制在 2 ms 内允许变化多少 N？
3. 限制后的请求向量及其长度是多少？为什么长度仍大于 6 N？

<details>
<summary>参考答案：限制的是向量变化量</summary>

所有向量都在控制器的表面坐标系中，第一个分量是法向；这里计算的补偿只有切向分量。
上一拍请求为 $f_{k-1}$，本拍期望请求为 $f_k^*$。

$$
\Delta f^*=f_k^*-f_{k-1}
\approx[0,-0.189833,0.130783]\ \mathrm N.
$$

它的长度约为 0.230523 N。单拍允许的最大变化为

$$
20\ \mathrm{N/s}\times0.002\ \mathrm{s}=0.04\ \mathrm N.
$$

因此只走向期望请求的一部分：

$$
f_k=f_{k-1}+\Delta f^*\frac{0.04}{\|\Delta f^*\|}
\approx[0,5.279640,-3.194431]\ \mathrm N,
\qquad \|f_k\|\approx6.170817\ \mathrm N.
$$

预算 6 N 约束生成期望请求时的幅值；输出还必须满足原有的变化率约束和全局 8 N 上限。
这条轨迹中有 5 拍输出高于刚下降的预算，8.010 s 首次回到 6 N 以下。这里没有发生接触丢失；
接触丢失的立即清零分支不属于连续变化率保证。

注意 `||f_k − f_{k-1}|| = 0.04 N`，但两个向量长度之差约为 0.039931 N。
运动方向略有变化时，不能用两个长度相减代替向量差的长度。

</details>

期望请求也可以往前追一层：把经偏置修正的法向力和旧系数相乘，再以本拍预算限制幅值。
本例中 $0.510661310\times12.208478387>6$，所以幅值取 6 N。
目标切向速度在控制器坐标中约为 $[0,0.05318107,-0.03204042]\ \mathrm{m/s}$，
使用 $d=v_t/\sqrt{\|v_t\|^2+0.005^2}$ 平滑方向后，即得到打印的 `desired_force_n`。
此拍接触混合权重为 1。该方向长度略小于 1，因此期望向量长度也略小于预算。

## 3. 只换成陈旧包，会发生什么

先预测：重复收到同一份旧包，是否应该在 8.000 s 立刻当作缺包？随后运行：

```bash
python -m tools.tutorials.budget_drop --event stale
```

<details>
<summary>参考答案：包龄未超限时仍可接受</summary>

重复旧包的时间戳为 7.996 s。到 8.016 s 时包龄恰好为 20 ms，仍可接受；
到 8.018 s 时包龄为 22 ms，首次超限。因此前 9 拍仍可接受，首次拒收在 `k=4009`，即 8.018 s，
共拒收 141 拍，8.300 s 恢复正常测量。首个拒收周期的请求范数为
`6.224654 → 6.184715 N`，向量变化仍为 0.04 N；`k=4014`、8.028 s 首次回到 6 N 以下。

拒收状态为 `stale`，与 `missing` 的原因不同，但调度器都不会复用拒收的力向量。
这些时间和数值属于本例，不能当成任意输入下的固定恢复时间。

</details>

## 4. 验收能发现被改错的系数吗

只在内存副本中把一拍的系数加上 `0.0001`：

```bash
python -m tools.tutorials.budget_drop --check-tamper
```

正常回放后会多一行 `in-memory coefficient +0.0001: REJECTED (expected)`。
这次拒绝是练习的预期结果，原 NPZ 文件没有变化。若只检查力是否小于 8 N，就发现不了
这个被单独改错的系数；[独立验收](../../../tools/measured_budget_validation.py)还会从测得运动
状态重算系数、运动确认和预算更新。

## 5. 用测试核对答案

```bash
pytest tests/test_tutorial_budget_drop.py -q
```

`test_published_fault_answers` 核对两类故障的时间、预算和变化率。
`test_first_missing_vector_matches_rounded_hand_calculation` 核对手算向量；
`test_documented_stdout_is_the_actual_command_output` 将本页输出与真实命令逐行比较。
实验源码在 [`budget_drop.py`](../../../tools/tutorials/budget_drop.py)。

输出中的 `full-state replay` 指切向补偿与负载调度状态的完整回放。
接触混合权重、接触标记、修正法向力及投影接受标记仍取自日志；本实验不会重算完整法向
控制器、力矩投影求解或关节力矩。要核对整个实验的协议、哈希和配对指标，使用
[测量鲁棒性页](../../measured_budget_robustness.md#复核与重跑)的完整 audit 命令。
算法采用检查仍为 7/8，默认保持固定 6 N；本次练习不增加新的仿真证据。

返回[教程目录](../README.md)，或回顾[实验一：误差到关节力矩](01_wrench_to_torque.md)。
