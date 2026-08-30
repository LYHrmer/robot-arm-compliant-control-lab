# ADR-0003：Nominal 与 residual 分级投影

## Status

Accepted。

## Context

如果只投影 nominal 与 residual 的最终和，策略可能用自己的动作抵消一个已经越界的 nominal
请求。这样既会削弱 zero-residual fallback，也很难判断饱和来自经典控制器还是学习修正。

## Decision

先把完整 nominal wrench 投影到预留关节力矩区间，再按剩余 headroom 投影 residual。策略只能
增加三维 Cartesian force，不能直接写 joint torque。

## Consequences

关闭策略时仍得到可独立运行的 torque-safe adaptive nominal，失败归因也更清楚。代价是
residual 的可用动作范围依赖当前 Jacobian、bias 和 nominal wrench，不能只靠固定 Cartesian
action bound 判断。

## Evidence

实现见 [`franka_torque_safety.py`](../../src/compliant_control_lab/franka_torque_safety.py)，边界与
fallback 测试见 [`test_franka_torque_safety.py`](../../tests/test_franka_torque_safety.py)，公开
结果见 [v0.5 comparison](../../results/franka_safety_blind/comparison.csv)。
