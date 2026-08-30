# 02｜阻抗、导纳与力位混合控制

柔顺控制的核心不是“把增益调软”，而是明确机器人与环境之间希望呈现什么动力学关系，
以及力/位置分别由哪一层闭环调节。

对应源码：

- 2-DOF：[`controllers.py`](../../src/compliant_control_lab/controllers.py)
- Franka：[`franka_control.py`](../../src/compliant_control_lab/franka_control.py)
- C++：[`franka_control.cpp`](../../cpp/src/franka_control.cpp)

## 1. 先区分四种控制器

| 控制器 | 输入误差 | 直接输出 | 适合回答的问题 |
|---|---|---|---|
| Position PD | 位置/速度 | Cartesian wrench | 刚性轨迹跟踪能有多准？ |
| Impedance | 位置/速度 | Cartesian wrench | 偏离目标时，希望呈现多硬的弹簧阻尼？ |
| Admittance | 力 | 修正后的位置/速度参考 | 测到外力后，参考轨迹应该怎样退让？ |
| Hybrid force-position | 法向力 + 切向位置 | 分子空间 wrench | 哪些方向控力，哪些方向控位置？ |

Position PD 和 Cartesian impedance 在代码形式上可能完全相同，区别在设计目标和增益：
高刚度 PD 追求几何误差小；阻抗控制把刚度/阻尼理解为机器人对外呈现的机械关系。

## 2. Cartesian impedance

平移部分为

$$
\mathbf f=K_p(\mathbf p_d-\mathbf p)+D_p(\dot{\mathbf p}_d-\dot{\mathbf p}).
$$

单位检查：

- $K_p$：N/m；
- $D_p$：N·s/m；
- 位置误差：m；
- 输出：N。

旋转部分使用小角度误差

$$
\mathbf e_R=\frac12\sum_{i=1}^3
(\mathbf R_{:,i}\times\mathbf R_{d,:,i}),
$$

并输出

$$
\boldsymbol\mu=K_R\mathbf e_R+D_R(\boldsymbol\omega_d-\boldsymbol\omega).
$$

最终 wrench 排列固定为

$$
\mathbf w=[f_x,f_y,f_z,\mu_x,\mu_y,\mu_z]^T.
$$

这套姿态误差在本项目的小姿态偏差下连续、计算便宜；接近 180 度误差时不适合作为全局
姿态表示，工程中可改为带符号一致性处理的 unit-quaternion error。

### 阻抗为什么不能保证 12 N

稳态法向速度为零时，近似有 $F=K\Delta x$。环境位置、工具半径或接触刚度改变，
$\Delta x$ 就改变，所以同一位置目标不会自动维持固定接触力。阻抗是“位移到力的机械
关系”，不是独立的力跟踪闭环。

## 3. Admittance：把力误差变成参考运动

法向虚拟动力学为

$$
M_a\ddot x_r+D_a\dot x_r+K_a(x_r-x_d)=F_d-F_m.
$$

代码没有调用通用 ODE solver，而是在 500 Hz 下使用半隐式 Euler：

$$
a_k=\frac{F_d-F_m-D_av_k-K_a(x_k-x_d)}{M_a},
$$

$$
v_{k+1}=v_k+a_k\Delta t,
\qquad
x_{k+1}=x_k+v_{k+1}\Delta t.
$$

先更新速度再更新位置，比完全显式 Euler 对弹簧系统通常更稳定。积分状态必须在每次 trial
开始时 `reset()`，否则上一回合的参考偏移会泄漏到下一回合。

参考位移还被限制在 `target +/- max_normal_offset`。这个限制用于接触丢失时防止参考无限
向墙内漂移，但它不是稳定性证明；更严格的真机方案还需要工作空间、速度、加速度、能量
或 passivity 限制。

任意单位法向量 $\mathbf n$ 的切向投影为

$$
P_t=I-\mathbf n\mathbf n^T.
$$

组合后的参考是

$$
\mathbf p_r=P_t\mathbf p_d+\mathbf n x_r,
\qquad
\dot{\mathbf p}_r=P_t\dot{\mathbf p}_d+\mathbf n\dot x_r.
$$

内部 impedance loop 再跟踪 $(\mathbf p_r,\dot{\mathbf p}_r)$。

## 4. Hybrid：法向控力，切向控位置

令 $\mathbf n$ 为表面法向，法向选择矩阵和切向选择矩阵分别为

$$
S_f=\mathbf n\mathbf n^T,
\qquad
S_p=I-S_f=P_t.
$$

进入稳定接触后，法向命令为

$$
F_f=\mathrm{clip}\left(
F_d+K_f(F_d-F_m)+K_i\int(F_d-F_m)dt
-D_f\mathbf n^T\dot{\mathbf p},
0,F_{max}\right).
$$

切向位置命令为

$$
\mathbf f_t=K_tP_t(\mathbf p_d-\mathbf p)
+D_tP_t(\dot{\mathbf p}_d-\dot{\mathbf p}).
$$

于是平移 wrench 是

$$
\mathbf f=\mathbf nF_f+\mathbf f_t.
$$

### Anti-windup

如果机械臂还没碰到墙， $F_m=0$，直接积分会让积分项持续增大；真正接触时会瞬间输出很大
命令。本项目只在“已确认接触”状态更新积分，并将积分限制在
`[-integral_limit, integral_limit]`。

### 为什么还需要接触状态机

纯 force command 在自由空间会让机械臂持续加速，直到撞上环境。因此 Franka hybrid 有
四个阶段：

```text
free-space approach
    -> force above 3 N for 20 ms
confirmed contact
    -> 150 ms smooth blend
force regulation
    -> force below 1 N for 50 ms
back to approach
```

未接触时使用限幅 normal position PD：

$$
F_a=\mathrm{clip}\left(
K_a\mathbf n^T(\mathbf p_d-\mathbf p)
+D_a\mathbf n^T(\dot{\mathbf p}_d-\dot{\mathbf p}),0,F_{a,max}\right).
$$

设 blend factor $\beta\in[0,1]$，实际法向命令为

$$
F_n=(1-\beta)F_a+\beta F_f.
$$

确认时间抑制传感器噪声导致的误切换，release hysteresis 抑制模式抖动，平滑 blend 避免
控制律突变。这个状态机使 nominal full-trial peak 从 33.83 N 降到 26.72 N。

## 5. 离散实现中的四个坑

1. **忘记乘 `dt`**：积分增益会随控制频率改变，500 Hz 和 100 Hz 表现完全不同。
2. **法向未归一化**：`n @ velocity` 和 `n*n^T` 会带错误尺度。
3. **在不同坐标系相加**：世界系误差、工具系 wrench、contact-frame force 必须先明确变换。
4. **只限制最终 torque**：内部积分/参考仍可能 wind up，解除饱和后产生二次冲击。

## 6. 调参顺序

不要同时调整所有增益：

1. 关闭接触，先调切向 position/rotation impedance；
2. 用低 approach stiffness 和较高 damping 建立安全接触；
3. 只开法向比例和 damping，观察 force step response；
4. 小幅增加 $K_i$ 去掉稳定偏差，同时监控 wind-up；
5. 改变墙刚度、延迟和噪声，检查峰值与饱和；
6. 最后才提高轨迹速度或接触力。

临界阻尼公式 $D\approx2\zeta\sqrt{MK}$ 可作初值，但 Cartesian 有效质量随姿态变化，
接触求解器和采样延迟也会改变稳定边界，所以最终必须用闭环实验验证。

## 7. 本章实验

```bash
franka-control-lab --controllers impedance admittance hybrid \
  --output results/control-study
```

观察：

- impedance 的位置误差如何转换为接触力；
- admittance/hybrid 的 force RMSE 是否更低；
- stiff-wall 中 raw peak 是否上升；
- noisy-delay 中积分、相位滞后和 torque saturation 是否恶化。

练习时每次只改一个参数，并保留“假设 -> 参数 -> 指标 -> 结论”记录。
