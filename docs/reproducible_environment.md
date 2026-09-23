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
python tools/ci/install_locked.py --profile core --check-only
python tools/ci/install_locked.py --profile core
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
python tools/ci/install_locked.py --profile learning-cpu
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
