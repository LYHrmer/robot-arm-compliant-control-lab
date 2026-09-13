# 实验一：从 10 mm 位置误差算到七个关节力矩

预计 20–30 分钟。先读 [02｜柔顺控制](../02_compliant_control.md) 的阻抗公式，再做这个单拍计算。
需要会向量逐元素相乘，以及矩阵转置乘法；不需要启动 MuJoCo。

本实验使用便于手算的 6×7 Jacobian，未取自 Franka 的某个姿态。所有量都表达在同一个
固定坐标系里，当前和目标姿态均为单位矩阵。算出的力矩仅用于核对公式，不能发送到实物。

## 运行

完成[仓库安装](../../../README.md#安装后快速复核)后，在仓库根目录运行：

```bash
python -m tools.tutorials.wrench_to_torque
```

命令只向终端打印答案，不创建实验目录。实现直接调用
[`FrankaImpedanceController.compute()`](../../../src/compliant_control_lab/franka_control.py)，
再用同文件的 `FrankaActuationContext.joint_torque()` 做映射；没有另写一套阻抗控制器。

## 题目：先算 wrench

误差采用“目标减当前”。本例目标速度为零，当前线速度为 `[0.02, -0.01, 0] m/s`，
所以速度误差的 x 分量为负。输入如下：

| 量 | 数值 | 单位 |
|---|---|---|
| 位置误差 $e_p$ | `[0.010, -0.020, 0.005]` | m |
| 线速度误差 $e_v$ | `[-0.020, 0.010, 0]` | m/s |
| 平移刚度 $K_p$ 的对角元素 | `[300, 200, 400]` | N/m |
| 平移阻尼 $D_p$ 的对角元素 | `[20, 30, 40]` | N·s/m |
| 姿态误差 $e_R$ | `[0, 0, 0]` | rad，小角度记法 |
| 角速度误差 $e_\omega$ | `[0.10, -0.20, 0.05]` | rad/s |
| 旋转刚度 $K_R$ 的对角元素 | `[20, 20, 20]` | N·m/rad |
| 旋转阻尼 $D_R$ 的对角元素 | `[2, 3, 4]` | N·m·s/rad |

当前角速度为 `[-0.10, 0.20, -0.05] rad/s`。虽然这一拍姿态已经重合，
角速度仍未归零，所以旋转阻尼仍会输出力矩。

用下面两个式子算出六维 wrench，再展开下方答案：

$$
f=K_p e_p+D_p e_v,\qquad m=K_R e_R+D_R e_\omega.
$$

<details>
<summary>参考答案：力和末端力矩</summary>

平移刚度项与阻尼项分别为：

```text
Kp * ep = [300×0.010, 200×(-0.020), 400×0.005] = [3, -4, 2] N
Dp * ev = [20×(-0.020), 30×0.010, 40×0]        = [-0.4, 0.3, 0] N
f       = [2.6, -3.7, 2.0] N
```

x 轴阻尼力为负，抵消了一部分正方向弹簧力。y 轴位置目标在负方向，算出的合力也为负。

由于 $e_R=0$，末端力矩只剩阻尼项：

```text
m = [2×0.10, 3×(-0.20), 4×0.05] = [0.2, -0.6, 0.2] N·m
```

按生产代码的顺序拼接：

```text
w = [fx, fy, fz, mx, my, mz] = [2.6, -3.7, 2.0, 0.2, -0.6, 0.2]
```

前三项单位为 N，后三项为 N·m，不能把整行都标成 N。

</details>

## 把 wrench 映射成 task torque

下式中的七个关节均按转动关节计。 $J$ 的前三行把 rad/s 映射为 m/s，系数单位为 m/rad；
后三行把 rad/s 映射为 rad/s。行顺序和 wrench 的力／力矩顺序必须对应。

```text
J = [ 0    0.2   0   -0.1   0    0    0.1 ]  vx
    [ 0.1  0     0.3  0    -0.2  0    0   ]  vy
    [ 0   -0.1   0    0.2   0    0.1  0   ]  vz
    [ 0    0     1    0     0    1    0   ]  wx
    [ 0    1     0    0     1    0    0   ]  wy
    [ 1    0     0    1     0    0    1   ]  wz
```

计算 $\tau_{task}=J^T w$。正力矩方向由每一列对应关节的正转方向定义，
不能仅凭末端 x 方向力为正，就判断所有关节力矩都应为正。

<details>
<summary>参考答案：逐个关节计算</summary>

每个关节取 $J$ 的一列与 $w$ 做内积：

```text
tau1 =  0.1×(-3.7) + 0.2                 = -0.17 N·m
tau2 =  0.2×2.6 - 0.1×2.0 - 0.6         = -0.28 N·m
tau3 =  0.3×(-3.7) + 0.2                 = -0.91 N·m
tau4 = -0.1×2.6 + 0.2×2.0 + 0.2         =  0.34 N·m
tau5 = -0.2×(-3.7) - 0.6                 =  0.14 N·m
tau6 =  0.1×2.0 + 0.2                    =  0.40 N·m
tau7 =  0.1×2.6 + 0.2                    =  0.46 N·m
```

终端的对应输出为：

```text
force [fx, fy, fz] (N): [2.600, -3.700, 2.000]
moment [mx, my, mz] (N m): [0.200, -0.600, 0.200]
task torque [tau1..tau7] (N m): [-0.170, -0.280, -0.910, 0.340, 0.140, 0.400, 0.460]
```

转置来自功率对应关系：若末端速度为 $v=J\dot q$，则
$w^T v=w^T J\dot q=(J^T w)^T\dot q=\tau_{task}^T\dot q$。
测试 `test_transpose_mapping_preserves_instantaneous_power` 检查这个等式。

</details>

## 只改一个量：x 位置误差变成 20 mm

其余输入保持不变，预测哪些关节力矩会变化，再运行：

```bash
python -m tools.tutorials.wrench_to_torque --x-error-mm 20
```

<details>
<summary>参考答案：增加的 3 N 去了哪里</summary>

位置误差只增加 $0.010$ m，新增弹簧力为 $300\times0.010=3$ N。
阻尼项没有变化，所以 x 方向合力为 `5.6 N`。各关节的增量由 $J$ 第一行决定：

```text
delta_tau = 3×[0, 0.2, 0, -0.1, 0, 0, 0.1]
          = [0, 0.6, 0, -0.3, 0, 0, 0.3] N·m
tau_new   = [-0.17, 0.32, -0.91, 0.04, 0.14, 0.40, 0.76] N·m
```

第 4 关节力矩反而减小，这是该列线速度映射的 x 分量为 `-0.1` 导致的。

</details>

## 检查与源码边界

装有开发依赖时运行：

```bash
python -m pytest tests/test_tutorial_wrench_to_torque.py -q
```

`test_default_wrench_and_task_torque_match_hand_calculation` 对照第一组手算答案；
`test_twenty_mm_variant_changes_only_x_force_and_its_joint_contributions` 对照变体。
数值比较允许浮点舍入误差，终端保留三位小数。

这里将 `joint_torque_offset` 设为零。结果没有重力／科氏力补偿，也没有零空间力矩。
代码中的力矩上下限只是构造 `FrankaActuationContext` 所需的占位值；
`joint_torque()` 自身不做安全投影，也不检查命令是否可执行。实际 Franka 仿真使用当前姿态
计算的 Jacobian，完整输出还要经过安全处理，见 [03｜Franka 数值计算](../03_franka_numerics.md)。

回到[教程目录](../README.md)。
