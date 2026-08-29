# 04｜怎样做可信的控制实验

控制项目的含金量不在于 GIF 是否顺滑，而在于：问题是否可复现、指标是否对应物理风险、
对比是否公平、失败是否保留。

对应源码与结果：

- [`franka_experiments.py`](../../src/compliant_control_lab/franka_experiments.py)
- [`franka_stress.py`](../../src/compliant_control_lab/franka_stress.py)
- [`nominal metrics`](../../results/franka/metrics.md)
- [`stress summary`](../../results/franka_stress/summary.md)

## 1. 控制变量和公平比较

三种控制器在同一场景中使用完全相同的：

- 初始关节状态；
- 500 Hz timestep；
- 目标接触力和 y-z 擦拭轨迹；
- 随机种子；
- 传感器滤波、噪声和延迟；
- actuator limits；
- MuJoCo 模型和接触参数。

每个 trial 都构造新 controller，防止 admittance reference 或 force integral 从上一回合泄漏。

## 2. 默认三场景

| 场景 | 改动 | 想检验什么 |
|---|---|---|
| `nominal` | 标称墙体、低噪声 | 基本 tracking 和稳定性 |
| `stiff_wall` | 更短 contact time constant | 硬接触下的 raw force peak |
| `noisy_delay` | 0.5 mm、0.6 N、20 ms | 相位滞后、噪声、wind-up 和 saturation |

只在 nominal 调通不能说明 robust。相反，只展示极端场景也不能说明正常工况性能。

## 3. 指标为什么分两个时间窗口

### 稳态/任务指标：1.5 s 之后

- `force_rmse_n`：filtered normal force 对目标力的 RMSE；
- `tangent_rmse_mm`：y-z 轨迹欧氏误差的 RMSE；
- `orientation_rmse_deg`：小角度姿态误差范数的 RMSE；
- `contact_ratio_pct`：filtered force 大于 0.5 N 的比例；
- `torque_rms_nm`、`controller_p95_us`。

排除 approach 是为了不让“还没接触时目标力正在 ramp”污染稳定跟踪能力。

### 安全指标：完整 trial

- `peak_force_n`：raw MuJoCo contact force 的最大值，包括首次接触；
- `saturation_pct`：任意关节 torque 被裁剪的全程比例。

这两个指标不能使用稳态 mask。项目有专门的 regression test，构造一个只在 approach
发生 50 N 峰值和 torque saturation 的 trial，确保安全指标仍能看到它。

## 4. 不要把 filtered force 当安全峰值

20 ms 低通的目标是模拟传感器带宽并给 controller 提供较平滑反馈。它必然削弱高频冲击。
因此：

```text
filtered force -> control feedback / tracking RMSE
raw force      -> impact / safety peak
```

面试中如果只给出低通曲线的峰值，面试官通常会追问滤波前是否更高。

## 5. 随机 holdout 压力测试

固定 seed 29 生成 24 个未见工况，每一行还记录自己的 simulation-noise seed。范围为：

| 变量 | 范围 | 近似物理含义 |
|---|---:|---|
| wall time constant | 4--25 ms（log-uniform） | 接触柔度/求解器刚度 |
| sliding friction | 0.20--0.90 | 材料与表面变化 |
| wall yaw | -6--6 deg | 法向估计/装夹误差 |
| position noise | 0--0.8 mm | 估计噪声 |
| force noise | 0.1--0.8 N | F/T 反馈噪声 |
| force bias | -1.5--1.5 N | 零漂/标定误差 |
| measurement delay | 0--30 ms | 通信与滤波延迟 |
| bias compensation scale | 0.85--1.15 | 负载/动力学模型误差 |

## 6. 预先声明 go/no-go，而不是结果后改标准

在运行 holdout 前固定每个 case 的门槛：

- force RMSE <= 2 N；
- contact ratio >= 95%；
- raw peak force <= 35 N；
- tangential RMSE <= 15 mm；
- torque saturation <= 1%。

整体要求至少 90% case 同时满足全部门槛。当前 impact-aware hybrid 是 6/24（25%），
所以结论是“值得评估 Residual RL 或 adaptive classical control”，不是“RL 已经更好”。

### 为什么不能在 holdout 上继续调参

如果看完这 24 个 case 后反复调 gain，seed 29 就从 holdout 变成 training set。正确做法是：

1. 保留 seed 29 不再用于调参；
2. 创建独立 train randomization seeds；
3. 在 train/validation 上开发；
4. 最后只评估一次固定 holdout；
5. 若研究过程已多次查看 holdout，应声明它已成为 validation，并生成新的 blind test set。

## 7. 怎样读当前结果

默认 nominal 中，hybrid 约 1 N force RMSE、100% contact、0% saturation；这说明基本闭环
成立。随机 holdout 中接触率仍为 100%，但高摩擦、法向偏差和 bias mismatch 会推高
接触峰值、切向误差和个别 case 的 saturation。这说明下一步要解决的是接触耦合与适应，
而不是重新学习 FK 或 gravity compensation。

失败结果必须进入仓库。一个可信结论可以是“方法在这些条件下失败，并据此定义下一阶段
问题”，不必把每张表都调到全绿。

## 8. 测试金字塔

```text
fast math tests
  FK / IK / Jacobian / orientation error / projector
        |
controller unit tests
  direction, axis separation, clamp, state transition
        |
Python <-> C++ parity
  same state sequence, component-wise wrench equality
        |
short MuJoCo closed-loop tests
  finite state, contact, RMSE, saturation
        |
full benchmark + randomized holdout
  published CSV / Markdown / plots / GIF
```

单元测试不能证明整个接触闭环稳定，GIF 也不能证明公式正确；两者需要同时存在。

## 9. 复现实验

```bash
# 默认 3 controllers x 3 scenarios
franka-control-lab --output results/franka --gif

# 固定 holdout
franka-stress-lab \
  --output results/franka_stress \
  --cases 24 \
  --duration 4.5 \
  --seed 29

# 全部自动验证
pytest
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## 10. 实验记录模板

每次改 controller 至少记录：

```text
Hypothesis:
  Example: longer contact transition reduces full-trial peak without losing contact.

One change:
  force_transition_time: 0.10 -> 0.15 s

Primary metrics:
  raw peak force, contact ratio

Guardrail metrics:
  force RMSE, tangent RMSE, saturation

Result:
  report all scenarios, not only nominal

Decision:
  keep / revert / run another controlled experiment
```

这种记录方式比“调到感觉不错”更接近实际算法工程工作。
