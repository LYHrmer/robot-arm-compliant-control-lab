# 固定实验环境与兼容性检查

首次复现七轴项目，优先使用本页的固定环境。六轴项目有自己的依赖，不共用这里的虚拟环境。只做控制与教学实验时，安装 `core` 即可；BC/PPO 需要额外的 CPU 学习环境。

## 固定了什么

目标是 Ubuntu 22.04、Linux x86_64。Python 的准确版本由 [`environment/python-version`](../environment/python-version) 指定。安装器检查 CPython 版本、架构和 glibc，拒绝在系统 Python 中安装。其他 Linux 发行版满足这些检查，也不代表经过本项目验证；Windows、macOS、ARM 请用首页的普通开发安装方式。

[`core.lock`](../environment/core.lock) 固定直接及传递依赖，并保存下载包的 SHA-256。NumPy、MuJoCo 与 Python 版本来自已发布的 [27 次负载调度回归](measured_budget_robustness.md)运行环境。锁文件包含构建所需的 setuptools 与 wheel，安装本项目时关闭构建隔离，避免另拉一套未锁定的构建依赖。

[`learning-cpu.lock`](../environment/learning-cpu.lock) 保留 core 的全部版本，再加入 PyTorch 及其依赖。Torch 指向官方 CPU wheel，不安装 CUDA、torchvision 或 torchaudio。两份 `.in` 是生成输入；运行安装只读取 `.lock`。测试会检查两份锁文件的共同依赖是否一致。

这个锁覆盖 Python 包。Eigen、CMake、编译器和 Node.js 的完整系统环境没有锁成容器镜像；C++ 的回放容差仍需单独检查。相同 Python 包也不保证不同 CPU 上逐位一致，不能用锁文件代替数值验证。[pip 的固定版本与哈希说明](https://pip.pypa.io/en/stable/topics/repeatable-installs/)解释了这两层约束。

## 在新环境中安装

先确认 `python3` 与 `environment/python-version` 一致，再在仓库根目录运行。下面使用新目录，不覆盖已有 `.venv` 或 `.local-deps`：

```bash
unset PYTHONPATH
python3 -m venv .venv-repro
source .venv-repro/bin/activate
export OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python tools/ci/install_locked.py --profile core --check-only
python tools/ci/install_locked.py --profile core
python -m tools.ci.check_replay_kernel
franka-smoke
python -m tools.tutorials.wrench_to_torque
python -m tools.tutorials.budget_drop --check-tamper
```

`--check-only` 只核对解释器与平台、打印将执行的命令，不安装包。正式安装使用 `--require-hashes --only-binary=:all:`，随后执行可编辑安装和 `pip check`。相同命令可以重跑；下载失败会返回非零退出码，不退回浮动依赖。安装器拒绝非空 `PYTHONPATH` 和使用 `--system-site-packages` 建立的环境，避免 ROS 或 `.local-deps` 中的旧包遮住新环境。`unset PYTHONPATH` 只影响当前终端，不修改 shell 配置文件。

`franka-smoke` 应以 `smoke: PASS` 完成。两份小实验的参考输出见[误差到关节力矩](tutorial/labs/01_wrench_to_torque.md)和[预算下降与缺包](tutorial/labs/02_budget_drop.md)。这些命令不重新生成历史归档。

## CPU 学习检查

单独建一个环境，避免把 Torch 装进只用于控制实验的环境：

```bash
unset PYTHONPATH
python3 -m venv .venv-learning-cpu
source .venv-learning-cpu/bin/activate
export OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python tools/ci/install_locked.py --profile learning-cpu
python -m tools.ci.check_replay_kernel
python -c 'import torch; assert torch.version.cuda is None; print(torch.__version__)'
mapfile -t learning_tests < environment/learning-tests.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest "${learning_tests[@]}" --junitxml=/tmp/franka-learning-junit.xml
python tools/ci/check_learning_junit.py /tmp/franka-learning-junit.xml
```

这里的 `mapfile` 需要 Bash。检查范围由 [`learning-tests.txt`](../environment/learning-tests.txt) 列出，覆盖目前所有包含可选 Torch 导入的测试模块。报告检查器要求每个模块都有测试结果，且没有任何 skipped、failure 或 error。模块缺失、缺包导致部分测试跳过，都不能给出成功结果。

这一步验证 BC/PPO 的实现与小规模训练路径，不意味着学习方法在跨种子评估中超过解析前馈。项目仍保留此前没有达到采用门槛的结果。

## CI 怎么分工

| 检查 | 环境 | 它回答的问题 |
|---|---|---|
| `locked-core` | 固定 Python 与 core 锁文件 | 同一套依赖能否安装，并跑通测试、C++ 与 smoke？ |
| `floating-compatibility` | Python 3.11、3.12；按 pyproject 版本范围安装 | 新的兼容依赖是否破坏已有功能？ |
| `learning-cpu` | 固定 Python 与 CPU 学习锁文件 | BC/PPO 测试是否真的执行，而非缺包后跳过？ |

core 和兼容性任务允许原有可选学习测试跳过；学习任务单独强制执行这些测试。每个任务有 25 分钟上限，只授予仓库读取权限。固定环境缓存绑定锁文件；日志记录准确 commit 和锁文件哈希。学习测试的 JUnit 报告保存 14 天，失败时也尝试上传。

浮动任务可能比固定任务先暴露依赖升级问题。遇到这种情况，不要直接覆盖锁文件：先复现失败，再判断是兼容性缺陷还是某项新依赖不适用。

### MuJoCo 3.14.0 的兼容边界

2026-09-23 的[浮动检查](https://github.com/LYHrmer/robot-arm-compliant-control-lab/actions/runs/35842201029)
安装了 MuJoCo 3.14.0，而[前一次通过的检查](https://github.com/LYHrmer/robot-arm-compliant-control-lab/actions/runs/35723669103)
使用 3.13.0。两个 Python 任务各自的解释器、NumPy 和运行器版本都没有变化。

随后在两个独立环境中固定 Python 3.10.12、NumPy 2.2.6 和其余 core 依赖，仅切换 MuJoCo，
并给 3.14 安装它新增的合法依赖 websockets 16.1.1。四项历史轨迹检查在 3.13 下全部通过，
在 3.14 下全部失败：关节位置最大差异约为 `1.56e-10`–`2.09e-10`，超过原来的 `1e-10`
数值复现容差。这不是控制性能或真机安全门槛，也不据此认定新引擎的物理结果错误。

浮动 CI 还报告旧换向归档的精确摘要不一致；这一项在上述本地 Python 3.10 环境中没有复现，
不能把两类差异未经检查归为同一个机制。3.13 下，本轮 72 条换向回归归档仍完整复算通过，
性能结论保持 34/36、整体 `FAIL`。

安装范围因此暂设为 `mujoco>=3.1,<3.14`。固定锁文件仍使用 3.12.0，没有修改旧轨迹、
审计容差或任何性能判据，也没有跳过兼容性测试。上限表示 3.14 及以后的版本尚未完成本项目
的复现验证，不代表已证明所有后续版本都失败。将来升级须另做环境对照，不能用更新归档或
放宽原测试来掩盖差异。

### 旧换向摘要的末位差异

同日的[失败任务](https://github.com/LYHrmer/robot-arm-compliant-control-lab/actions/runs/35850262605/job/107146005977)
已取出具体字段。Python 3.11.16、NumPy 2.4.6、MuJoCo 3.13.0 下，旧 8 条轨迹的
复算摘要有 16 处差异，全部位于 `runs[*].audit`：

| 审计字段 | 差异数量 | 最大绝对差 |
|---|---:|---:|
| 补偿力重构残差 | 6 | `8.881784197001252e-16` N |
| 调度重构残差 | 8 | `8.881784197001252e-16` N |
| 系数转移残差 | 2 | `1.1102230246251565e-16` |

其余字段精确一致，包括性能指标和原 4 组配对的验收结论。旧摘要比较要求残差也精确相等，
所以仍报 `recomputed comparison differs`。本轮 72 条回归的审计先调用这份旧归档审计，
因此也会被阻断；新摘要自己的容差不会绕过旧检查。

不同云端任务并非每次失败。上面的日志确认了触发失败的字段，后续同机对照才定位到 OpenBLAS 路径。
旧诊断曾尝试禁用一组 AVX512 特性，但 NumPy 2.4 仍报告部分分组启用，不能把这次设置
当作“已排除所有 AVX512 影响”的证据。MuJoCo 的版本上限也没有解决这一项问题。

可在仓库根目录运行只读诊断：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m tools.ci.diagnose_recovery_replay
```

命令输出精确差异路径与浮点十六进制值；有差异时退出码为 1。它不修复或重写归档，
也不把“性能判定相同”当作审计通过。源码哈希与原有验收门槛保持不变。

需要区分数值路径时，Linux x86_64、NumPy 2.2 及以上的环境可以运行：

```bash
python -m tools.ci.recovery_runtime_matrix --repo .
```

四个新进程分别检查原生路径、OpenBLAS Haswell、Sandybridge，以及禁用 NumPy 编译目标中的
AVX512 分组。每次只改一项，先清除调用终端已有的相关覆盖变量。工具会读取 OpenBLAS 实际
内核名称，检查 NumPy 实际启用的函数路径，并记录第一条轨迹的最早算术差异。
没有 AVX512 的机器无法据此检验 AVX512 假设；库不支持某个指定内核时会明确失败。
任意进程存在精确差异、超时或没有完成复算，整体都返回非零退出码。

GitHub 的 [`recovery-runtime`](../.github/workflows/recovery-runtime.yml) 工作流只接受手动触发。
它分别使用固定 core 环境和上述失败任务的 Python／NumPy／MuJoCo 版本，在每台运行器内做
四路对照。后一环境的其余依赖仍按安装范围解析，日志会记录实际版本，不能视为完整锁定。
工作流保存诊断日志，原生路径失败不会被某个替代路径的成功覆盖；正常 `tests` 工作流仍独立运行。

### 复现环境限定为 Haswell 数值内核

[同机四路对照](https://github.com/LYHrmer/robot-arm-compliant-control-lab/actions/runs/35867214077)
在两台云端 CPU 上得到相同结果。[证据摘录](evidence/recovery-runtime-2026-09-23.json)
保存实际内核信息与全部差异，并绑定原始日志的 SHA-256。下表是重算摘要与冻结摘要的精确差异数：

| 单变量对照 | Intel Xeon 8573C／固定 core | AMD EPYC 9V45／报错版本 |
|---|---:|---:|
| 原生 OpenBLAS `SkylakeX` | 16 | 16 |
| 仅指定 OpenBLAS `Haswell` | 0 | 0 |
| 仅指定 OpenBLAS `Sandybridge` | 0 | 0 |
| 仅禁用 NumPy AVX512 分派 | 16 | 16 |

BLAS 对照中的库哈希、线程数和 NumPy 分派都不变。最早调度差异位于第 837 拍、1.674 s
的投影力点积，相差 1 ULP；此前负载和 `alpha` 一致。两种 BLAS 内核切换都消除了该差异。
这将本次问题定位到 OpenBLAS 的内核选择，不是 `expm1` 系数变化。

固定环境和普通 CI 因而使用 `OPENBLAS_CORETYPE=Haswell`、单线程，并通过
`python -m tools.ci.check_replay_kernel` 检查实际加载的内核。只设置变量但库未采用它时，
检查仍会失败。上面的安装命令已包含这一步；变量须在启动 Python 前设置，只影响当前终端。
Haswell 是这里选用的数值内核名称，不要求 CPU 品牌为 Intel；检查器还要求 CPU 支持 AVX2 和 FMA3。
此限定不适用于 ARM 或其他 BLAS 后端，也不是跨所有 CPU 的逐位一致性保证。

零差异指摘要精确匹配。补偿力逐拍重构原本就有约 `8.88e-16` N 的残差，Haswell 路径
复现了这个残差，并没有让所有逐拍字段变成逐位相同。原轨迹、源码哈希、验收门槛与旧摘要
比较器都未修改；手动对照保留原生路径的失败。72 条换向回归仍为 34/36、整体 `FAIL`。

## 更新锁文件

仅在目标 Python/平台上重新解析。使用单独的维护环境安装 `pip-tools`；本次生成器为 7.6.1。它不属于实验运行依赖。先生成 core，再以 core 约束学习环境：

```bash
pip-compile --generate-hashes --allow-unsafe --no-emit-index-url \
  --no-emit-trusted-host --pip-args=--only-binary=:all: \
  --output-file=environment/core.lock environment/core.in
pip-compile --generate-hashes --allow-unsafe --no-emit-index-url \
  --no-emit-trusted-host --pip-args=--only-binary=:all: \
  --constraint=environment/core.lock \
  --output-file=environment/learning-cpu.lock environment/learning-cpu.in
```

需要更新已锁版本时再显式添加 `--upgrade`，审查生成差异。更新后必须在空虚拟环境中实际安装，并重跑 smoke、学习测试及相应回归。`pip --dry-run` 只验证解析，不能证明 MuJoCo 或 Torch 能加载。CPU wheel 的版本和平台改动也要重新核对[官方 PyTorch 下载索引](https://download.pytorch.org/whl/cpu/torch/)，不要改成默认 GPU 包源。

旧锁文件随 Git 提交保留。若新环境验证失败，旧环境和旧结果仍可继续使用；回退应恢复相应提交的锁文件，再创建一个新环境，不在原实验环境里反复升降级。
