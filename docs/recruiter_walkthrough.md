# 招聘方 5 分钟走查

这页面向没有时间读完整仓库的招聘方或面试官。所有命令和数字都来自仓库里已经存在的入口
和已发布产物，本页不新增结论，只挑一条最短路径核对它们。

## 1. 准备环境

下面的五分钟走查从依赖安装完成后开始。首次下载时间取决于网络；已有仓库可直接在仓库
根目录创建环境。

```bash
git clone https://github.com/LYHrmer/robot-arm-compliant-control-lab.git
cd robot-arm-compliant-control-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

需要 Python 3.10+，不需要 ROS 2 或 Franka 硬件接口。

## 2. 跑一条命令看工程状态（约 1 分钟）

```bash
franka-smoke
```

预期输出同时包含实验失败和工程检查通过两种状态，二者不矛盾：

```text
archive: PASS (384 rows, frozen_decision=FAIL)
simulation: PASS (safe_adaptive_hybrid/nominal, steps=1000, ...)
smoke: PASS
```

`archive` 校验的是已冻结的 384 行 v0.5 结果文件没有被改动；`simulation` 只跑
2 秒 nominal 仿真确认代码能跑。`smoke: PASS` 的含义到此为止，v0.5 实验仍为 `FAIL`，
48 个 first-reveal cases 也没有在这条命令中重跑。

## 3. 看主结果：v0.5 是 FAIL（约 1 分钟）

| 方法 | 通过数 | Raw peak P95 [N] | Saturation worst |
|---|---:|---:|---:|
| Fixed hybrid | 17/48 | 58.62 | 19.69% |
| Adaptive hybrid | 23/48 | 58.99 | 20.71% |
| Torque-safe adaptive | 24/48 | 59.54 | 0.00% |
| Torque residual（5 个 seed） | 22–26/48 | 59.54 | 0.00% |

预注册门槛是每个 residual 策略单独达到 44/48。五个策略最好也只到 26/48，冻结结论是
`FAIL`，仓库没有部署这些策略。Torque-safe 方法的最差 actuator saturation 是 0%，
说明在这批场景内没有触发执行器裁剪；完整轨迹的 raw-force peak 仍超过 35 N gate。

原始数据：[comparison.csv](../results/franka_safety_blind/comparison.csv)、
[summary.md](../results/franka_safety_blind/summary.md)。完整背景见首页的
[v0.5 首次揭盲结果](../README.md#v05-首次揭盲结果)。

## 4. 区分软件版本与实验结论（约 1 分钟）

仓库的软件版本是 `0.5.1`（见 [pyproject.toml](../pyproject.toml)），记录协议冻结后的
工程修复，例如事件重放和离线 audit 工具。本页仍引用原来的 v0.5 first reveal，已发布
结果文件保持不变。实验的协议身份仍然是 `v0.5`，对应
[reproduction_plan_v0.5.md](reproduction_plan_v0.5.md) 里冻结的 commit、tag 和 beacon
round；这份文档不会因为软件小版本号继续增长而改变。

## 5. 想核实某个具体主张（剩下的时间）

- 逐条主张对应的实现、测试和产物：[verification_matrix.md](verification_matrix.md)。
- 版本演进和每个版本的证据锚点（commit、tag、结果哈希）：
  [experiments/README.md](experiments/README.md)。
- 离线复核已发布的 v0.5 归档，不联网也不重新仿真：

  ```bash
  franka-published-results-audit \
    --protocol results/franka_safety_preholdout/protocol.json \
    --result results/franka_safety_blind
  ```

  `audit PASS` 只表示归档文件字节没有被改动、内部推导自洽；不会把 `FAIL` 改成
  `PASS`。

- 跑测试和静态检查确认这份走查引用的命令确实可用：

  ```bash
  pytest
  ruff check src tests
  ```

## 继续阅读

深入算法推导看[教程目录](tutorial/README.md)；面试用的口头介绍和简历措辞已经写在
[练习与面试](tutorial/06_exercises_and_interview.md)；完整文档导航在
[docs/README.md](README.md)。这页只负责让招聘方在五分钟内验证“结果是否可信、失败有没有
被藏起来”，其余内容不重复展开。
