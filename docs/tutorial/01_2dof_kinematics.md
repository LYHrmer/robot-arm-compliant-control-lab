# 01｜2-DOF 解析运动学：从几何到 Jacobian

2-DOF 平面臂不是“玩具代码”。它把坐标、解析解、奇异性和虚功映射压缩到一个能手算、
能画图、能写单元测试的最小系统。Franka 章节使用的核心概念都能在这里先验证。

对应源码：

- [`kinematics.py`](../../src/compliant_control_lab/kinematics.py)
- [`simulation.py`](../../src/compliant_control_lab/simulation.py)
- [`test_kinematics.py`](../../tests/test_kinematics.py)

## 1. 坐标与正运动学

两根连杆长度为 (l_1,l_2)，关节角为 (q_1,q_2)。第二个关节角是相对第一根连杆的
转角，所以第二根连杆相对世界坐标的角度是 (q_1+q_2)：

\[
\mathbf{x}(\mathbf{q})=
\begin{bmatrix}
x\\y
\end{bmatrix}
=
\begin{bmatrix}
l_1\cos q_1+l_2\cos(q_1+q_2)\\
l_1\sin q_1+l_2\sin(q_1+q_2)
\end{bmatrix}.
\]

这就是 `forward_kinematics(q)`。输入单位必须是 rad，输出单位是 m。常见错误是把第二项
写成 `cos(q2)`，等价于误把第二根连杆当成相对世界坐标旋转。

## 2. 逆运动学：余弦定理和两支解

给定目标 ((x,y))，先由余弦定理求第二关节：

\[
c_2=\cos q_2=\frac{x^2+y^2-l_1^2-l_2^2}{2l_1l_2}.
\]

若 (c_2\notin[-1,1])，目标在工作空间之外，代码直接抛出 `ValueError`。数值计算中即使
理论上可达，浮点误差也可能得到 `1.0000000001`，所以求 `arccos` 前仍使用 `clip`。

肘下/肘上两支解分别来自

\[
q_2=\pm\arccos(c_2),
\]

再求

\[
q_1=\operatorname{atan2}(y,x)-
\operatorname{atan2}(l_2\sin q_2,l_1+l_2\cos q_2).
\]

验证 IK 的正确方式不是只看角度，而是检查
`forward_kinematics(inverse_kinematics(x))` 是否回到原位置。

## 3. Jacobian：速度的一阶线性化

对 FK 关于关节角求偏导：

\[
J(\mathbf q)=\frac{\partial\mathbf x}{\partial\mathbf q}=
\begin{bmatrix}
-l_1\sin q_1-l_2\sin(q_1+q_2) & -l_2\sin(q_1+q_2)\\
l_1\cos q_1+l_2\cos(q_1+q_2) & l_2\cos(q_1+q_2)
\end{bmatrix}.
\]

局部速度关系为

\[
\dot{\mathbf x}=J(\mathbf q)\dot{\mathbf q}.
\]

### 有限差分验证

任选单位方向 (mathbf v)，用很小的 (epsilon)：

\[
\frac{\mathbf x(\mathbf q+\epsilon\mathbf v)-\mathbf x(\mathbf q)}{\epsilon}
\approx J(\mathbf q)\mathbf v.
\]

若误差随 (epsilon) 从 (10^{-3}) 降到 (10^{-6}) 而减小，通常说明导数正确；若一直
很大，多半是符号或角度叠加写错；若 (epsilon) 极小后误差反而增大，是浮点消减误差。

## 4. 奇异性和条件数

这个 2x2 Jacobian 的行列式为

\[
\det J=l_1l_2\sin q_2.
\]

当 (q_2=0) 或 (pi) 时两根连杆共线，Jacobian 降秩。机械臂沿某个 Cartesian 方向的
速度/力能力丢失，直接求 `J^{-1}` 会放大噪声甚至失败。Franka 的 6x7 Jacobian 不使用
普通逆，正是同一个原因；第 03 章会使用阻尼最小二乘解。

可用奇异值分解理解接近奇异位形的程度：

\[
J=U\Sigma V^T,\qquad
\kappa(J)=\frac{\sigma_{max}}{\sigma_{min}}.
\]

最小奇异值越接近零，条件数越大，数值解越敏感。

## 5. 为什么 wrench 用 (J^T) 映射到关节力矩

由虚功守恒：

\[
\delta W=\boldsymbol\tau^T\delta\mathbf q
=\mathbf w^T\delta\mathbf x
=\mathbf w^TJ\delta\mathbf q.
\]

对任意 (delta\mathbf q) 成立，因此

\[
\boldsymbol\tau=J^T\mathbf w.
\]

注意这不是“把速度公式转置一下凑出来”，而是功率/虚功对偶关系。2-DOF 仿真中的
Cartesian controller 输出 ([F_x,F_y])，再由 `jacobian(q).T @ wrench` 得到两个关节力矩。

## 6. 本章实验

```bash
pytest tests/test_kinematics.py -v
compliant-control-lab --controllers position impedance --output results/kinematics-study
```

建议练习：

1. 增加一个有限差分 Jacobian 测试，随机采样 20 个非奇异位形；
2. 画出 (q_2\to0) 时最小奇异值和条件数；
3. 分别求肘上/肘下 IK，并用 FK 回代；
4. 解释为什么 (J^T\mathbf w) 不需要 Jacobian 可逆。

能完成这四项，再进入柔顺控制；否则在 7-DOF 中遇到数值异常会很难定位。
