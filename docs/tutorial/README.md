# 从 2-DOF 到 Franka 7-DOF：柔顺接触控制学习路线

这套教程不是把公式和代码并排放在一起就结束。每一章都回答四个问题：

1. 控制目标和物理量是什么；
2. 连续公式如何离散并稳定地求解；
3. 公式对应哪段 Python/C++ 源码；
4. 用什么实验和指标证明实现没有“看起来能跑、其实算错”。

## 建议顺序

| 阶段 | 主题 | 学完应能回答 | 对应章节 |
|---|---|---|---|
| 0 | 环境与复现 | 如何一条命令复现图表和测试？ | 本页 |
| 1 | 2-DOF 解析运动学 | FK、IK、Jacobian 为什么这样写？ | [01](01_2dof_kinematics.md) |
| 2 | 柔顺控制 | 阻抗、导纳、力位混合到底差在哪？ | [02](02_compliant_control.md) |
| 3 | Franka 数值解算 | 6D wrench 如何变成 7 维 torque？阻尼伪逆怎么解？ | [03](03_franka_numerics.md) |
| 4 | 实验方法 | 怎样定义指标、压力测试和可复现实验？ | [04](04_experiments_and_validation.md) |
| 5 | Residual RL | 什么时候该加 RL，动作/奖励/安全层如何设计？ | [05](05_residual_rl.md) |
| 6 | 进阶练习与面试 | 如何从“会运行”进阶到“能解释、能扩展”？ | [06](06_exercises_and_interview.md) |

如果刚接触机器人控制，按 01 -> 02 -> 04 的 2-DOF 部分学习；如果已有机器人学基础，
可从 03 开始；如果目标是复现实验或准备面试，至少完整阅读 03、04 和 06。

## 环境与第一轮复现

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest
compliant-control-lab --output results --gif
franka-control-lab --output results/franka --gif
```

然后编译 C++ 参考实现：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
pytest tests/test_cpp_parity.py
```

一轮完整复现应得到：

- Python 测试全部通过；
- CTest 通过；
- `tests/test_cpp_parity.py` 没有 skip，且 Python/C++ wrench 逐项一致；
- `results/franka/metrics.md` 中 hybrid nominal 力 RMSE 约 1 N、接触率 100%、无力矩饱和；
- `results/franka_stress/summary.md` 明确保留固定增益控制器在未见工况中的失败，而非只展示最好结果。

## 源码导航

```text
2-DOF analytic lab
├── src/compliant_control_lab/kinematics.py
├── src/compliant_control_lab/controllers.py
└── src/compliant_control_lab/simulation.py

Franka 7-DOF lab
├── src/compliant_control_lab/franka_control.py
├── src/compliant_control_lab/franka_adaptive.py
├── src/compliant_control_lab/franka_torque_safety.py
├── src/compliant_control_lab/residual_rl.py
├── src/compliant_control_lab/franka_learning.py
├── src/compliant_control_lab/franka_safety_learning.py
├── src/compliant_control_lab/franka_simulation.py
├── src/compliant_control_lab/franka_stress.py
└── cpp/                         # C++17/Eigen 等价实现
```

核心原则是：控制器只接收状态/目标并输出 Cartesian wrench；MuJoCo、未来的 ROS 2
适配器和绘图代码不参与控制公式。这个边界让公式、仿真和部署代码可以分别验证。

## 学习完成标准

不要只以“GIF 能动”为完成。至少应能独立完成以下检查：

- 手推 2R 平面臂 FK、IK 和 Jacobian，并用有限差分验证 Jacobian；
- 解释阻抗与导纳的因果方向，以及 hybrid 中法向/切向选择矩阵；
- 解释为什么使用 `solve(A, B)` 或 Eigen `LDLT.solve()`，而不是显式计算矩阵逆；
- 说明 `J^T w`、bias compensation 和 null-space posture torque 各自负责什么；
- 区分 filtered-force tracking RMSE 与 full-trial raw peak force；
- 给出 Residual RL 的 nominal controller、残差动作、安全限制和零残差回退；
- 改动 Python 公式后，同步修改 C++ 并让 parity test 继续通过。
