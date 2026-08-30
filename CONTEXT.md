# 柔顺接触控制术语表

括号内是文档和代码采用的英文规范名。统一术语是为了避免把稳态误差、接触峰值或不同数据
身份混为一谈。

## 控制

**接触任务（contact task）**：末端沿表面切向轨迹运动，同时调节表面法向力。脱离具体演示
轨迹讨论控制器时，避免只称为 wiping task。

**法向方向（normal direction）**：垂直于接触表面的单位方向，力控制沿该方向执行。避免称为
force axis。

**切平面（tangential plane）**：与表面法向正交的二维平面，位置跟踪在该平面内执行。避免称为
position axes。

**名义控制器（nominal controller）**：加入学习修正以前，先计算任务 wrench 的经典控制器。
避免称为 base policy 或 RL baseline。

**固定混合控制器（fixed hybrid controller）**：使用固定增益和接触切换状态机的力位混合
nominal controller。避免称为 vanilla controller。

**自适应混合控制器（adaptive hybrid controller）**：估计接触状态并调度经典增益的 nominal
controller。避免称为 adaptive RL。

**力矩安全自适应控制器（torque-safe adaptive controller）**：把完整 nominal wrench 限制在
预留关节力矩区间内的 adaptive hybrid controller。避免缩写成 safe controller 或 safe
adaptive。

**有界残差策略（bounded residual policy）**：叠加在 nominal wrench 上的学习型三维平移力
修正；动作仍受经典控制器的安全规则约束。避免称为 RL controller 或 end-to-end policy。

## 实验证据

**公开验证集（public validation set）**：结果已经被查看，可用于诊断和调参的数据集。它不能
继续支持新的未见数据主张。避免称为 blind test 或 hidden test。

**首次揭盲集（first-reveal set）**：实现、策略和协议冻结后，才由不可预知随机量派生的数据。
完成第一次评测后，它就转为公开验证数据。揭盲后避免继续称为 blind set。

**揭盲后分析（post-reveal analysis）**：首次揭盲完成后的诊断。它可以解释结果，但不是一轮
新的 blind evidence。避免称为 holdout result。

**冻结协议（frozen protocol）**：首次揭盲以前固定的控制器、策略、数据派生和验收规则记录。
它不是普通 experiment notes。

**验收门槛（gate）**：由力跟踪、接触率、冲击、切向跟踪和 actuator saturation 共同组成的
逐 case 判定条件。避免用单一 score 或 reward 代指。

**主要结论（primary result）**：按揭盲前已经固定的规则得到的 PASS 或 FAIL。它不是 best seed
或平均结果。

## 指标

**稳态力跟踪误差（force tracking error）**：越过 approach 区间后，用滤波力反馈计算的任务期
误差。它不是 peak force。

**全轨迹未滤波峰值（raw peak force）**：完整 trial 中最大的未滤波接触力，包含首次接触。
它不是 force RMSE 或 filtered peak。

**执行器饱和率（actuator saturation）**：请求关节力矩超出 actuator limit 的控制周期比例。
它不是 torque projection rate。
