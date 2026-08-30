# ADR-0001：控制器边界放在 Cartesian wrench

## Status

Accepted。

## Context

柔顺控制公式需要同时服务 MuJoCo、C++ 数值核心和未来机器人适配器。如果控制器直接依赖
MuJoCo 对象、Jacobian 映射或硬件句柄，同一套公式会和每个 runtime 重复耦合。

## Decision

控制器接收任务状态和目标，只返回 6D Cartesian wrench。仿真或机器人适配器负责 Jacobian
映射、动力学补偿和 actuator limit。

## Consequences

Python 与 C++ 可以围绕同一固定尺寸接口做逐分量测试。代价是适配层必须正确提供 pose、
twist、wrench、Jacobian 与 bias；控制器接口本身不承诺 ROS 2 或真机接入已经完成。

## Evidence

接口见 [`franka_control.py`](../../src/compliant_control_lab/franka_control.py) 和
[`franka_control.hpp`](../../cpp/include/compliant_control_lab/franka_control.hpp)；数值一致性见
[`test_cpp_parity.py`](../../tests/test_cpp_parity.py)。
