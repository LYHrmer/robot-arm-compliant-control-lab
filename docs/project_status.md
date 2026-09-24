# 项目状态与验收路径

主线已经可以不经训练直接运行；换向恢复候选仍未通过采用门槛。
数值以 `results/` 下的归档为准；各轮实验的完整顺序留在[实验总账](experiments/README.md)。

## 可运行主线

500 Hz 经典柔顺控制到在线补偿，再到 Python/C++ 同输入回放：

1. 阻抗／导纳／力位混合与接触状态机：[`franka_control.py`](../src/compliant_control_lab/franka_control.py)、[控制参考](franka_control.md)
2. 切向积分与摩擦前馈基线：[tangential_tracking.md](tangential_tracking.md)
3. 在线等效负载补偿：[online_compensation.md](online_compensation.md)，原 24-case 切向 RMSE 中位数 1.885 → 1.359 mm
4. 同输入的 C++ 回放：[cpp_core.md](cpp_core.md)，168,000 周期，最大分量误差 < `3.56e-15`；含一份重复演示，不计为新工况

默认配置为补偿上限固定 6 N、姿态增益 1、速度误差时间系数 0.05 s。6–8 N 测得负载调度是
已覆盖仿真工况的可选预设（[load_budget.md](load_budget.md)），默认入口不启用。
控制器只接收状态与目标并输出六维 Cartesian wrench，仿真、绘图和未来的 ROS 2 适配层不参与
控制公式（[ADR-0001](adr/0001-cartesian-wrench-controller-seam.md)）。主线不要求训练网络。

## 研究支线

- BC 与 bounded residual PPO：同 4 个开发测试 case 上解析前馈平均 2.403 mm，BC 2.781–2.825 mm，
  PPO 2.381–2.470 mm，两种学习方法均未跨种子一致优于前馈（[surface_learning_pilot.md](surface_learning_pilot.md)、
  [bc_closed_loop_transfer.md](bc_closed_loop_transfer.md)）。BC 模仿补偿，PPO 在已有前馈上学残差，
  两者名义控制器不同，网络动作不能直接互换。
- 换向恢复候选：[局部实验](reversal_recovery.md)、[跨方向回归](reversal_recovery_transfer.md)、
  [静止回退 pilot](stationary_recovery_pilot.md)。三轮都停在候选状态，默认控制器未改。

六轴机械臂在独立项目中开发，不并入本仓库，也不以它的实物条件替代七轴项目的验证。

## 不支持的能力

- 没有 Franka hardware/model interface，也没有 ros2_control 插件；运动演示均为仿真，没有真机验证。
- 没有硬件 safety 或 passivity 证明；torque projection 不提供 torque-rate、碰撞阈值或认证。
- 接触参数直接取自 MuJoCo，未做真机辨识；命令有界不保证真实接触力有界。

## 安装之后的验收路径

先按[首页安装](../README.md#安装后快速复核)完成固定环境安装，再留出约 30 分钟依次操作并
对照输出。安装时间不计，实际耗时取决于机器。这条路径含一次 2 秒 nominal 新仿真，其余步骤
检查环境或已有证据，不训练网络。命令都在仓库根目录执行；需要另存结果时使用仓库外的新目录，
不覆盖 `results/`。

| 步骤 | 命令 | 预期 | 不能证明什么 |
|---:|---|---|---|
| 1 | `python -m tools.ci.check_replay_kernel` | 打印 `numpy_version` 与 `loaded_blas`；每个 OpenBLAS 库报告 Haswell、1 线程，否则报错退出 | 只核对当前进程的数值内核；不保证其他 CPU 上逐位一致 |
| 2 | `franka-smoke` | `archive: PASS (384 rows, frozen_decision=FAIL)`、`simulation: PASS (safe_adaptive_hybrid/nominal, steps=1000, ...)`、`smoke: PASS` | 仿真只跑 2 秒 nominal；不重跑 48-case 揭盲，也不改冻结 `FAIL` |
| 3 | `python -m tools.tutorials.wrench_to_torque` | 打印一次阻抗 wrench 与七个关节力矩 | 合成 Jacobian 的单步计算，不含 bias、零空间、投影或动力学积分 |
| 4 | `python -m tools.tutorials.budget_drop` | 先 `full-state replay: PASS (6000 cycles)`，再列出预算下降、缺包与系数更新 | 重算已保存轨迹，不新跑仿真 |
| 5 | `MUJOCO_GL=disable python -m tools.stationary_recovery.study --audit results/franka_stationary_recovery_pilot` | `archive_integrity: PASS`、`comparison_status: FAIL`、`new_holdout: false` | 只核对归档完整性与配对重算；不积分新动力学，`FAIL` 仍是实验判定 |
| 6 | `pytest tests/test_documentation_links.py tests/test_tutorial_wrench_to_torque.py` | 两个目标通过 | 只检查本地链接和一个数值例子 |

完整 `pytest`、`ruff`、C++ 构建与 `ctest`、BC/PPO 学习测试都不在这条路径里，
命令见[首页验证代码](../README.md#验证代码)；CPU 学习依赖与禁止跳过的测试要求见
[固定环境说明](reproducible_environment.md)。

## 学习路线

从[教程目录](tutorial/README.md)按 01 → 07 读；已有机器人学基础可从
[03 Franka 数值解算](tutorial/03_franka_numerics.md)开始。三份带答案实验分别针对力矩映射、
预算下降和换向排错：[实验一](tutorial/labs/01_wrench_to_torque.md)、
[实验二](tutorial/labs/02_budget_drop.md)、[实验三](tutorial/labs/03_reversal_recovery.md)。
按目的检索文档用[文档入口](README.md)；把某条主张落到源码和测试用[验证矩阵](verification_matrix.md)。

## 当前失败与后续边界

最新已发布实验是[静止回退 pilot](stationary_recovery_pilot.md)，4 个配对通过 3 个，整体 `FAIL`。
高负载 7–8 s 窗口相对原 6–8 N 调度的额外代价：seed 11 位置 +0.095464 mm、速度 +0.458643 mm/s；
seed 29 位置 +0.106956 mm、速度 +0.545732 mm/s。门槛为 +0.1 mm 和 +0.5 mm/s，seed 29 两项都没过。
这些数字是相对原 6–8 N 调度的增量，不对应默认固定 6 N，也不是绝对 RMSE。两个种子相对旧回退
候选都改善，但没有通过全部门槛，两次噪声重复也不足以支持推广，
采用范围与默认值不变。[跨方向回归](reversal_recovery_transfer.md)的 34/36 保持原样。
这轮属于公开开发实验，不构成新的 holdout。

保持阶段的系数只降不升，是[控制器测试](../tests/test_stationary_recovery_controller.py)验证过的
结构现象：预算 6 N、初始系数 0.5，法向力 12 → 14 → 12 N 时系数为 0.5 → 0.4994 → 0.4994。
这个无动力学例子不证明它解释了 seed 29 的全部剩余代价。下一轮干预尚未执行。

其余保留的失败项：默认固定 6 N 在组合误差后段仍有约 3.3 mm 切向残差；可选负载调度在
辅助力幅值 ×0.8 时后段 RMSE 为 3.010 mm，
测量鲁棒性采用检查 7/8（[measured_budget_robustness.md](measured_budget_robustness.md)）；
在线补偿在 12 秒对照里有两条反向加速姿态阶段失败（[online_compensation.md](online_compensation.md)）。
固定 8 N 预算在原 public24 只通过 23/48，整体 `FAIL`（[budget_transfer.md](budget_transfer.md)）。
冻结 v0.5 的 48-case 首次揭盲也仍为 `FAIL`，仓库不部署这些 residual 策略。

## 读这些结果时的边界

`archive: PASS` 和各 `*-audit` 的 `PASS` 只表示归档未被改动、内部推导自洽或指标可重算；
实验判定由归档中的 `frozen_decision`、`comparison_status` 给出。精确复核旧摘要要求
[固定环境](reproducible_environment.md)：Linux x86_64、锁定依赖、单线程 Haswell OpenBLAS，
CPU 需支持 AVX2／FMA3。执行新实验前提交源码，输出写到新目录；失败归档保留，不覆盖重跑。

相关主张已列入[验证矩阵](verification_matrix.md)。静止回退小试的可执行证据是
[候选控制器](../tools/stationary_recovery/controller.py)、[协议与执行入口](../tools/stationary_recovery/study.py)、
[控制器测试](../tests/test_stationary_recovery_controller.py)、[协议测试](../tests/test_stationary_recovery_study.py)、
[审计测试](../tests/test_stationary_recovery_audit.py)与
[归档比较](../results/franka_stationary_recovery_pilot/comparison.json)。
