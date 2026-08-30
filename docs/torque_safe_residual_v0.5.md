# v0.5｜从 Cartesian wrench 到关节力矩安全包络

v0.4 的 residual 能改善切向跟踪，但它只在 Cartesian 空间限幅。相同 residual 经
`J(q)^T` 映射后，可能集中到 Panda 末端几个 12 Nm 关节上。v0.5 先把这个映射放进
控制器可见的数值上下文，再训练策略。

## 一次 2 ms 控制周期

```text
q, dq ──> MuJoCo Jacobian / bias
              │
              ├─ J(6x7)
              ├─ torque_offset = bias + null-space posture
              └─ actuator lower / upper limits
                         │
                         v
                 FrankaActuationContext
                         │
target ──> reference governor ──> adaptive hybrid ──> nominal wrench projection
                                                        │
20-D observation ──> 50 Hz policy ──> bounded residual ─┤
                                                        v
                                              residual torque projection
                                                        │
                                                        v
                    tau = J^T (w_nom + delta_w) + torque_offset
```

`FrankaActuationContext` 只有 NumPy 数组，不含 `MjModel` 或 `MjData`。这样既能使用当前
位形的精确 Jacobian，又不把控制算法绑死在 MuJoCo 上。以后接 ROS 2 时，同一接口可由
Franka model handle 填充。

## ray projection 怎么算

设安全余量比例为 \(\rho=0.1\)。原 actuator interval \([l,u]\) 缩成

\[
l_s=l+\frac{\rho}{2}(u-l),\qquad
u_s=u-\frac{\rho}{2}(u-l).
\]

已知名义 wrench \(w_0\) 和待加入的 \(\Delta w\)：

\[
\tau_0=J^Tw_0+\tau_{offset},\qquad d=J^T\Delta w.
\]

安全模块不改变方向，只求最大的 \(\alpha\in[0,1]\)：

\[
l_s\leq\tau_0+\alpha d\leq u_s.
\]

对第 \(i\) 个关节，若 \(d_i>0\)，上界是
\((u_{s,i}-\tau_{0,i})/d_i\)；若 \(d_i<0\)，上界是
\((l_{s,i}-\tau_{0,i})/d_i\)。取全部有限上界与 1 的最小值，再用
`nextafter(alpha, 0)` 留出浮点误差。最后重新计算关节力矩；复核失败就返回零增量。

名义关节力矩若已经在安全区间外，residual 不能反向“修好”它。该周期 residual 直接清零，
状态记为 `nominal_outside`。缺少 context、NaN/Inf 或投影后仍越界也按同样原则处理。

实现位于
[`franka_torque_safety.py`](../src/compliant_control_lab/franka_torque_safety.py)。

## 为什么投影两次

第一层把 adaptive controller 的完整 6D wrench 视为从零开始的增量。bias compensation 和
null-space posture torque 保持不变，只缩放 task wrench。第二层以已经安全的名义 wrench
为基点，投影策略给出的三维 residual force。最终仍由仿真适配器用同一份 context 计算
`J.T @ wrench + offset`。

这不是二次 clipping。两层都保留 wrench 的方向；第二层只能使用第一层留下的余量。公开
seed-29 验证中，safe adaptive 的最差 saturation 从 12.89% 降到 0%，但 force 和 tangent
gate 仍有大量失败，所以这一改动只说明 actuator clipping 已被消除。

## 六个 torque-headroom 输入

策略动作上界为 `[4, 6, 6] N`。在每个 50 Hz policy tick，控制器分别尝试
`+x, -x, +y, -y, +z, -z` 六个满幅动作，并记录各自可行的 \(\alpha\)。例如 `+y` 字段为
0.25，表示当前位形最多安全施加 `+1.5 N` 的 y 向 residual。

这六个数附加在 v0.4 的 14 个反馈量后面。策略文件保存全部 20 个字段名；维度相同但字段
错序也会拒绝构造 controller。headroom 只需按 50 Hz 更新，实际 residual 每个 500 Hz
周期都重新投影，两次 policy update 之间的 Jacobian 变化不会绕过安全层。

## reference governor 与 adaptive nominal

`FrankaSafeAdaptiveController` 把接触前法向 reference lead 限在 10 mm。3、4.5、6、8 mm
在公开 seed-29 倾斜墙工况中会丢失接触；10、12、16 mm 不再改善通过数，最后冻结 10 mm。
这段调参记录只属于 public validation，不进入新的 first reveal。

接触力超过目标 3 N 或滤波力变化率超过 120 N/s、且 classical force blend 尚未完成时，
governor 不再给正向 lead/velocity。adaptive 层仍负责偏置、接触刚度和力变化率估计，以及
force/tangential gain scheduling。

## 五个训练 seed 怎样冻结

每次 ARS 训练使用 8 个固定 training cases；另有 8 个 development cases 选择 checkpoint。
五个 policy seed 是 `17, 23, 31, 43, 59`，simulation seed 分别从
`10001, 20001, 30001, 40001, 50001` 开始。每轮换一组仿真噪声，正负方向共用同一组噪声。

训练时关闭 5 ms 墙钟 deadline，避免多进程调度改变策略输出；seed、动作和 MuJoCo 轨迹
仍是确定的。正式评测重新启用 deadline，超时或策略异常会在当前周期把 residual 清零。
训练 CSV 记录每轮实际使用的 training/development simulation seed。

## first reveal 为什么不能换文件

`prepare` 完成训练后会再次检查 `HEAD`、tracked diff、模型 hash 和 safety manifest。五个
policy、CSV、PNG、协议和 checksum 随后作为 implementation commit 的单一子提交，tag 为
`v0.5-preholdout`。

协议预先指定一个尚未发布的 drand Quicknet round。训练前后，冻结的 Node.js helper 都会
并行读取两个官方 relay 的最新轮次，再按固定公钥逐个验签。两个 relay 最多允许相差一轮；
最终目标至少领先较新轮次 201 轮，机器时钟快慢不会改变这项判断。目标轮次到达后，helper
再从两个 relay 请求同一个 exact round。beacon randomness 与 protocol hash 一起派生 48 个
scenario seed 和 48 个 noise seed。

评测前会逐字节比较工作区文件和 tag 中的文件，并检查 tag 的父 commit。只复制一份新的
`protocol.json`、删除四个 policy 或在 reveal 前改安全参数都会失败。48 个 case 一旦生成，
它们就转为公开 validation 数据；后续实验要使用新的 protocol、tag 和未来 round。

## first reveal 读出了什么

round `31756275` 的 48-case 结果没有通过预注册门槛。safe adaptive 通过 24/48；五个
residual 分别通过 22、25、26、24、25/48，平均 24.4。规则要求每个策略都达到 44/48，
不能用最好 run 02 的 26/48 代替。

安全层和学习层要分开看：

- safe adaptive 与五个 residual 的最差 saturation 都是 0%，三类 fallback 总数也都是 0。
  residual 自身最多有 15.71% 的控制周期触发 torque projection，最小平均缩放系数为 0.843。
- residual 的切向 P95 为 15.98–16.50 mm，低于 safe adaptive 的 18.89 mm，但仍有 4–5
  个 case 超过 15 mm。
- 五个 residual 的 raw peak P95 都是 59.54 N，并有 20–24 个 case 超过 35 N。这一项是
  通过数上不去的主要原因。

residual 要等稳定接触 100 ms 才启用，而 safe adaptive 和五个 residual 的 peak P95 完全
相同。两条证据都指向接近速度、reference governor 或接触切换阶段；这是根据结果做的诊断，
还需要记录 peak 时间点才能定因。当前不急着把 ARS 换成 SAC/TD3。先修名义层的入触瞬态，
再用新的未来 round 做第二次盲测，信息量更大。

## 建议从哪些测试读起

- `tests/test_franka_torque_safety.py`：手算 projection、reserve、NaN/Inf、双层投影和真实
  MuJoCo 零饱和 rollout。
- `tests/test_residual_rl.py`：20 维 schema、50/500 Hz 更新、context 丢失与零 residual。
- `tests/test_franka_safety_learning.py`：五 seed 合同、tag 字节绑定、伪 beacon 和双 relay
  不一致的拒绝路径。

实验参数、命令和 reveal 规则见
[`reproduction_plan_v0.5.md`](reproduction_plan_v0.5.md)。
