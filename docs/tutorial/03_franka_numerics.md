# 03｜Franka 7-DOF 的 Jacobian、阻尼解算与 torque mapping

这一章解释仿真循环中每个数值量如何得到，以及为什么实现选择“解线性方程”而不是“求逆”。

对应源码：

- [`franka_simulation.py`](../../src/compliant_control_lab/franka_simulation.py)
- [`franka_control.py`](../../src/compliant_control_lab/franka_control.py)
- [`C++ core`](../../cpp/src/franka_control.cpp)

## 1. 状态向量和 6x7 geometric Jacobian

Franka 有 7 个 arm joints，末端任务是 6D：

$$
\dot{\mathbf x}=
\begin{bmatrix}\mathbf v\\\boldsymbol\omega\end{bmatrix}
=J(\mathbf q)\dot{\mathbf q},
\qquad J\in\mathbb R^{6\times7}.
$$

MuJoCo 的 `mj_jacSite` 分别返回 site position Jacobian 和 rotational Jacobian。代码将两者
堆叠，再只取前 7 个 arm DoF。随后直接用 `twist = J @ qvel[:7]` 得到末端线速度和角速度。

这里不对 Jacobian 做有限差分，是因为 MuJoCo 已根据模型树计算 analytic geometric
Jacobian；但项目仍用独立单元测试检查形状、有限性和 null-space 性质。

## 2. Cartesian wrench 到 joint torque

控制器返回

$$
\mathbf w=[\mathbf f,\boldsymbol\mu]^T\in\mathbb R^6.
$$

由虚功关系得到 task torque：

$$
\boldsymbol\tau_{task}=J^T\mathbf w.
$$

最终命令为

$$
\boldsymbol\tau=
J^T\mathbf w+\mathbf h(\mathbf q,\dot{\mathbf q})
+N\boldsymbol\tau_0.
$$

- `J.T @ wrench`：完成当前 Cartesian task；
- `qfrc_bias`：MuJoCo 给出的重力、科氏/离心 bias；
- `N @ posture_torque`：只使用任务剩余自由度保持舒适关节姿态。

压力测试中的 `bias_compensation_scale` 故意把精确 bias 乘以 0.85--1.15，模拟质量参数、
工具负载或动力学模型误差。

## 3. 阻尼 null-space projector 怎么解

项目使用 torque-space projector：

$$
N=I-J^T(JJ^T+\lambda^2I)^{-1}J.
$$

维度检查：

```text
J                 6 x 7
J J^T             6 x 6
(...)^-1 J        6 x 7
J^T (...)^-1 J    7 x 7
N                 7 x 7
```

### 为什么加 $\lambda^2I$

当 Jacobian 接近降秩， $JJ^T$ 的最小特征值接近零。直接求伪逆会极度放大噪声。
加入阻尼后，奇异方向的增益从 $1/\sigma$ 变为近似

$$
\frac{\sigma}{\sigma^2+\lambda^2},
$$

牺牲一点严格投影精度换取有界数值结果。

### 为什么不显式求逆

Python 实际求解的是

$$
(JJ^T+\lambda^2I)X=J,
$$

使用 `np.linalg.solve(A, J)`；C++ 使用 Eigen `A.ldlt().solve(J)`。两者都不构造
$A^{-1}$。解线性方程通常更快、更稳定，也让底层库选择适合的分解。

`LDLT` 适用于这里的对称正定/半正定加阻尼矩阵。若实现改成一般非对称矩阵，就不能不加
判断地继续用同一分解。

### 阻尼 projector 不是严格零

当 $\lambda>0$，一般只有 $JN\approx0$，而不是数学上的精确零。因此 posture gain 过大
仍可能扰动末端 task。测试使用很小阻尼检查极限性质；实际控制使用 0.03 保持条件良好。

## 4. Posture torque

关节姿态控制为

$$
\boldsymbol\tau_0=K_q(\mathbf q_{nom}-\mathbf q)-D_q\dot{\mathbf q}.
$$

它经 $N$ 投影后加入主任务。常见错误包括：

- 把 $N\tau_0$ 写成 $N^T\tau_0$ 而不确认 projector 定义；
- posture gain 过大，使阻尼 projector 的泄漏不可忽略；
- nominal pose 接近 joint limit；
- 忘记最终 torque limits。

更严格的 operational-space controller 会使用质量矩阵构造 dynamically consistent
inverse 和 projector；本项目明确把它列为进阶项，而不把运动学 projector 说成完整 WBC。

## 5. Torque clipping 和 saturation 指标

Panda 模型中的 actuator range 是 joints 1--4 为 +/-87 Nm，joints 5--7 为 +/-12 Nm。命令
按每个 actuator 的上下限裁剪：

$$
\tau_i\leftarrow\mathrm{clip}(\tau_i,\tau_{min,i},\tau_{max,i}).
$$

只看裁剪后的 torque 无法知道控制器原本要求了多少，因此代码先保留 `torque_unclipped`，
并记录任意关节是否被裁剪。`saturation_pct` 覆盖完整 trial，包括 approach。

真机还应加入 torque-rate limit：

$$
|\tau_{k}-\tau_{k-1}|\le \dot\tau_{max}\Delta t,
$$

以及通信 watchdog、非有限值检查和碰撞状态机。

## 6. MuJoCo 接触力怎么读取

每个仿真步遍历 `data.contact`，只接受 `{tool_geom_id, wall_geom_id}` 这一对 geom，避免把
其他自碰撞或环境碰撞算入工具法向力。`mj_contactForce` 返回 contact frame 下的 6D
wrench；第 0 分量是法向力。本项目累加所有工具-墙接触点的正法向分量。

要点：

- contact wrench 是求解器的原始瞬时输出，可能有尖峰；
- controller 使用 20 ms 一阶低通后的力；
- `peak_force_n` 使用完整 trial 的原始力；
- `force_rmse_n` 使用 approach 结束后的 filtered force；
- 不能用低通后的曲线证明“没有碰撞尖峰”。

一阶滤波为

$$
F_k^f=F_{k-1}^f+\alpha(F_k-F_{k-1}^f),
\qquad
\alpha=\frac{\Delta t}{\tau_f+\Delta t}.
$$

测量延迟用定长 `deque` 实现：先写入当前 measurement，控制器读取队首；10 steps 在
500 Hz 下等于 20 ms。

## 7. Python/C++ 数值一致性

C++ core 只暴露固定尺寸状态、目标和 wrench；配置阶段可检查非法参数，500 Hz `compute`
路径不做日志、锁或动态容器操作。`compliant_control_probe` 在同一组状态上运行：

- orientation error；
- impedance；
- admittance；
- 已接触 hybrid；
- approach hybrid；
- contact-confirm/blend 状态序列。

Python 测试逐分量以 `rtol=atol=1e-12` 对比，防止两套实现只在“趋势上差不多”。

## 8. 本章实验

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
pytest tests/test_cpp_parity.py -v
```

建议练习：

1. 扫描 $\lambda\in[10^{-6},10^{-1}]$，画出 `||J @ N||` 与 projector 条件性的取舍；
2. 故意把 `solve` 改为显式 `inv`，比较结果和运行时间，但不要提交退化版本；
3. 增大 posture gain，观察末端误差和 torque saturation；
4. 用 Pinocchio 计算同一姿态的 Jacobian/bias，注意 frame convention 后再交叉验证。
