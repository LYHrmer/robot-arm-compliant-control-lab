# BC / PPO 从复现到离线部署

本页提供三条可独立选择的路径。第一次学习这个项目，建议先运行公开模型，读懂验收报告，
再从源码重新采集数据和训练。所有实验均为 Franka Panda 的 MuJoCo 仿真。

| 想做什么 | 从哪里开始 | 是否需要 Torch | 主要输出 |
|---|---|---|---|
| 从零采集示教、训练 BC/PPO、评估 | [路径 A](#路径-a从源码重训) | 需要，CPU 训练 | 数据、训练记录、模型、16 回合验收 |
| 下载 GitHub 源码后复核已公开的模型，或自行打包 | [路径 B](#路径-b运行公开模型并制作离线包) | 不需要 | 新仿真报告、可选离线发行目录 |
| 已经拿到完整离线发行目录 | [路径 C](#路径-c安装已有离线发行目录) | 不需要 | 独立虚拟环境、新仿真报告 |

公开模型位于 [`results/franka_bc_deployment_fix/bundle`](../results/franka_bc_deployment_fix/bundle)，
完整训练轨迹和依赖 wheel 保存在本地 `dist/`，不随 Git 提交。只克隆仓库不会获得
`dist/learning-release-bcfix-20261009/`；可走路径 B 自行制作，或完整复制交付包后走路径 C。

## 环境和磁盘

固定环境是 CPython **3.10.12**、Linux **x86_64**、glibc **≥ 2.35**，已验证系统为
Ubuntu 22.04。`python3 --version` 必须先显示对应版本；这里的脚本不负责安装系统 Python。
运行 `venv` 需要系统已有该 Python 的 venv 支持。MuJoCo 使用无窗口模式，不要求显卡或桌面。

| 锁文件 | 用途 |
|---|---|
| [`core.lock`](../environment/core.lock) | 源码控制实验、测试、构建工具；不含 Torch |
| [`learning-cpu.lock`](../environment/learning-cpu.lock) | core 全部依赖及 CPU Torch，供重新训练 |
| [`deployment-runtime.lock`](../environment/deployment-runtime.lock) | 离线推理的 23 个运行依赖；不含 Torch、pytest 或构建工具 |

固定运行版本包括 NumPy 2.2.6、MuJoCo 3.12.0、Gymnasium 1.3.0。启动 Python 前设置
`OPENBLAS_CORETYPE=Haswell`、单线程；Haswell 是数值内核名称，检查器还要求 CPU 支持
AVX2 和 FMA3。详见[固定环境说明](reproducible_environment.md)。

一次完整准备已有约 **633 MiB** 的工作区记录，离线发行目录约 **70 MiB**；这是已测运行的
规模，不是磁盘上限。新机器还需安装 Python 依赖，建议为训练环境及输出预留数 GiB。
运行时长取决于 CPU；训练和评估期间可以查看终端打印的 `stage:`，不要因为暂时没有
新日志就启动另一个进程写入同一目录。

## 获取固定源码

以下命令使用 Bash。已有仓库时先进入仓库并 `git fetch origin`，跳过 `git clone`。
新建分离工作区可以保留你当前分支的修改，同时让实验始终绑定同一份源码。

```bash
git clone https://github.com/LYHrmer/robot-arm-compliant-control-lab.git
cd robot-arm-compliant-control-lab
git worktree add --detach ../robot-arm-learning-b698e84 \
  b698e84dd02ece4063ef46812b318d9260195924
cd ../robot-arm-learning-b698e84
python3 --version
```

这个 commit 固定新版部署流程。公开 bundle 在原仓库的后续文档／证据提交中；路径 B 将从
相邻原仓库读取它，而在这个固定工作区运行代码。不要对运行中的工作区执行切分支、修改
`src/`、`tools/` 或更新依赖。`prepare` 会检查相关源码已经提交，并在阶段间检查来源未变。

## 路径 A：从源码重训

在上述固定工作区建立全新训练环境：

```bash
unset PYTHONPATH
python3 -m venv .venv-learning-cpu
source .venv-learning-cpu/bin/activate
export OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MUJOCO_GL=disable
python tools/ci/install_locked.py --profile learning-cpu --check-only
python tools/ci/install_locked.py --profile learning-cpu
python -m tools.ci.check_replay_kernel
python -m tools.learning_deployment doctor --training
python -c 'import torch; assert torch.version.cuda is None; print(torch.__version__)'
```

安装需要联网，随后训练和仿真不需要网络。`--check-only` 仅检查，不安装；`doctor` 报告
依赖和资源，不能代替后面的训练与验收。

先预览，再运行：

```bash
python -m tools.learning_deployment prepare \
  --workspace dist/reproduce-bcfix-seed11 --dry-run
python -m tools.learning_deployment prepare \
  --workspace dist/reproduce-bcfix-seed11 \
  --seed 11 --bc-epochs 100 --ppo-episodes 32 \
  --bc-input-mode drop_previous_residual
```

这会按以下顺序完成全流程：

1. 冻结 `plan.json`：24 个 12 秒示教回合，按 12 个物理任务组划分为 16 train、
   4 validation、4 development-test；同组不会跨分割。
2. 采集教师示教，训练 BC 100 epochs。仅按验证动作 MSE 选 checkpoint，完全相等时选更早的 epoch。
3. 在 friction nominal 上训练 bounded PPO 32 episodes。使用预先声明的最终 checkpoint，
   不按 development-test 选择 PPO。
4. 导出有限 JSON 模型，核对 NumPy 与 Torch 的推理输出，测量加载及调用耗时。
5. 分别运行 BC、BC 的 adaptive nominal、PPO、PPO 的 friction nominal；每种 4 个
   development-test 回合，共 16 回合，并保存工程与策略两类判定。

这是常规部署准备流程。第 2 步使用验证集上的离线动作 MSE 选 epoch，但 `prepare`
不会另外运行候选 BC 的 4 个闭环 validation 回合。本次研究交付还在开发测试前执行了
这 4 个闭环验证，通过后才固定候选；该资格流程的计划和判定保存在
[`study_plan.json`](../results/franka_bc_deployment_fix/study_plan.json) 与
[`eligibility.json`](../results/franka_bc_deployment_fix/eligibility.json)。路径 B 另外提供
公开候选的闭环验证重跑命令。常规 `prepare` 通过不能替代这份额外的研究资格记录。

新版 BC 将观测索引 `14、15、16`（上一步残差动作）屏蔽，训练使用剩下的 46 个通道。
导出时屏蔽合入第一层权重，模型仍保持 **49 → 32 → 32 → 3**，部署接收原始 49 维观测。
它没有降低门槛或把 BC 换成解析控制器。输入消融的动机和已有多种子实验见
[BC 闭环迁移说明](bc_closed_loop_transfer.md)。

交付模型的训练记录使用 Torch **2.12.1+cu130**，实际计算设备为 **CPU**；这里推荐的
干净训练锁使用 **2.12.1+cpu**。所以路径 A 复现的是从零采集、训练与部署验收，不能预先承诺
新模型文件与交付模型逐字节相同。每次以本次生成的 manifest、模型哈希和物理指标为准。
路径 B、C 则直接验证同一份已交付模型，推理环境没有 Torch。

### 输出应该怎样读

```text
dist/reproduce-bcfix-seed11/
├── plan.json                  固定源码、参数、分组、选择规则、门槛
├── dataset/                   示教 manifest 与 24 个回合的轨迹
├── bc_training/               曲线、选中模型、epoch 与来源
├── ppo_training/              训练回合、checkpoint 与失败记录
├── bundle/                    两个模型、两个基线、cases 和冻结评估协议
├── evaluation_bc/             BC 开发测试原始轨迹和报告
├── evaluation_bc_nominal/     adaptive nominal 对照
├── evaluation_ppo/            PPO 开发测试原始轨迹和报告
├── evaluation_ppo_nominal/    friction nominal 对照
└── acceptance/report.json    汇总判定、推理耗时、导出误差及各阶段哈希
```

部署验收使用四个 development-test case 的最差表现：接触率至少 99%、切向 RMSE 不超过 10 mm、力 RMSE
不超过 2 N，并同时满足冻结协议中的全部硬约束。`engineering_status` 表示完整执行、
导出和运行要求；`policy_acceptance.bc`、`.ppo` 分别表示策略是否达标。基线的失败也会保留，
不会替换策略自身的结论。相对基线的改善另报，绝对门槛通过不等于优于解析控制。

| 退出码 | 含义 | 下一步 |
|---:|---|---|
| `0` | 当前命令通过；完整部署验收要求工程和两个策略均通过 | 保留报告，再打包 |
| `1` | 来源、依赖、完整性、输入或执行错误 | 读 stderr 的 `error`，修复对应原因 |
| `2` | 完成检查但至少一项工程／策略验收未通过 | 查看报告字段和对应 case，不改旧结果 |

此表适用于 `tools.learning_deployment` 入口。路径 B 的底层独立评估入口
`tools.evaluate_surface_candidate` 只报告执行是否完成，必须另查报告才能判断门槛。

### 中断恢复和旧方案对照

恢复必须使用相同源码 **HEAD commit**、完整参数及运行环境，包含 Torch 构建版本和记录的
平台信息。验证通过的完整阶段才会复用；失败训练或已修改目录不会被自动覆盖。

```bash
python -m tools.learning_deployment prepare \
  --workspace dist/reproduce-bcfix-seed11 \
  --seed 11 --bc-epochs 100 --ppo-episodes 32 \
  --bc-input-mode drop_previous_residual --resume
```

即便只提交了文档导致 HEAD 改变，也应回到原固定工作区再续跑。不能用 CPU 锁环境直接
接续交付时的 cu130 训练工作区。更换种子、输入配置、训练预算或源码必须用新目录。

要复现原来使用全部 49 维输入的 BC 路径，在新目录显式指定：

```bash
python -m tools.learning_deployment prepare \
  --workspace dist/reproduce-full49-seed11 \
  --seed 11 --bc-epochs 100 --ppo-episodes 32 --bc-input-mode full49
```

这是保留失败方案的对照入口。旧发行的 BC 最差切向 RMSE 为 11.494 mm，原始验收为
FAIL，详见[历史交付结果](learning_deployment.md#历史交付full49-保持失败)。新运行仍需以自身
报告判定，不能把旧结果、不同环境结果或新版本结果混为同一份证据。

## 路径 B：运行公开模型并制作离线包

从固定工作区开始；如果已经激活训练环境，先 `deactivate`，再建一个不含 Torch 的 core 环境。

```bash
unset PYTHONPATH
python3 -m venv .venv-bundle-core
source .venv-bundle-core/bin/activate
export OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MUJOCO_GL=disable
python tools/ci/install_locked.py --profile core
python -m tools.ci.check_replay_kernel
python -m tools.learning_deployment doctor
BUNDLE_PATH="$(cd ../robot-arm-compliant-control-lab && pwd)/results/franka_bc_deployment_fix/bundle"
BUNDLE_SHA=1048a6280d3231ac661a3f0b0ad86b68a208eb18f82c785a227fc52218de6961
python -m tools.learning_deployment verify \
  --bundle "$BUNDLE_PATH" --expected-manifest-sha256 "$BUNDLE_SHA" --profile
```

上面的 SHA 来自独立发布的交付记录。不要在运行时读取待检查 bundle 的 `manifest.json`
再计算“预期 SHA”；那样无法发现整份 bundle 连同 manifest 一起被替换。公开模型必须与
绑定源码、运行库和接口一致，不能通过修改 manifest 绕过 `verify` 的来源检查。

### 复核公开 BC 的四个闭环验证任务

公开的 [`validation_protocol.json`](../results/franka_bc_deployment_fix/validation_protocol.json)
冻结了这份 BC 模型、4 个 validation case 和原有门槛。继续使用上面的无 Torch 环境：

```bash
python -m tools.evaluate_surface_candidate \
  --candidate "$BUNDLE_PATH/bc.json" \
  --protocol "$BUNDLE_PATH/../validation_protocol.json" \
  --cases "$BUNDLE_PATH/cases.json" \
  --expected-protocol-sha256 d899e9543c52cea5c8d7d69dde0bb4dced22c8873f3f38a0a7fbbf95a412783a \
  --purpose il_friction_teacher --nominal-kind adaptive \
  --output dist/public-bc-validation-01 && python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("dist/public-bc-validation-01/report.json").read_text())
checks = {name: report.get(name) is True for name in (
    "acceptance_met", "all_episodes_succeeded", "all_metrics_available"
)}
checks["validation_split"] = report.get("evaluation_split") == "validation"
checks["four_cases"] = report.get("expected_case_count") == 4
passed = all(checks.values())
print(json.dumps({"status": "PASS" if passed else "FAIL", "checks": checks}, indent=2))
raise SystemExit(0 if passed else 2)
PY
```

底层评估命令退出 `0` 只说明执行完成，**不代表策略门槛通过**。紧接着的检查要求
`acceptance_met`、`all_episodes_succeeded`、`all_metrics_available` 都为 `true`，并确认
验证分割及 4 个 case；任何一项不满足都会返回 `2`。本次发布的参考结果为 `PASS`，
最差切向 RMSE 为 8.100 mm，完整数值见
[`validation_report.json`](../results/franka_bc_deployment_fix/validation_report.json)。

此协议绑定已发布模型的 SHA，不能直接套用路径 A 新训练的另一份模型。新候选如果需要
同样的研究资格流程，应另行冻结绑定该候选的验证协议，先完成闭环验证再决定是否进入
开发测试；不得改写这份公开协议或把重跑结果伪装成原始资格记录。

### 复核部署的十六个开发测试回合

```bash
python -m tools.learning_deployment accept \
  --bundle "$BUNDLE_PATH" --expected-manifest-sha256 "$BUNDLE_SHA" \
  --output dist/public-bundle-acceptance-01
```

这一步运行 BC、PPO 及各自基线共 16 个 development-test 回合，保存新的轨迹和报告。
加上前面的独立 BC 验证，共重跑 4 + 16 个回合。它们是新产生的重跑轨迹，不保证与
原始归档逐字节一致；报告中的来源、实际数值和判定必须保留。

想只运行 BC，可把 `accept` 换成 `run --algorithm bc`；加 `--nominal` 则运行对应的
零残差基线。`run` 和 `accept` 都支持 `--dry-run`，只核对模型与协议，不创建仿真结果。
每次实际运行使用新的输出目录。

### 制作离线发行目录

制作离线发行目录需要本地 Git 中存在模型绑定的 commit，以及 core 环境中的构建工具。
先联网下载哈希固定的运行 wheel，再离线打包：

```bash
python -m pip download --only-binary=:all: --require-hashes \
  -r environment/deployment-runtime.lock --dest dist/runtime-wheelhouse
python -m tools.learning_deployment package \
  --bundle "$BUNDLE_PATH" --expected-manifest-sha256 "$BUNDLE_SHA" \
  --wheelhouse dist/runtime-wheelhouse --output dist/offline-release-01
```

打包器从 bundle 绑定的 Git commit 导出干净源码，构建 wheel，并复制 23 个运行依赖和
模型。将打印的 `release_manifest_sha256` **单独保存**，作为路径 C 的外部校验值。
自己打包的 wheel／发行 SHA 可能不同，不能套用本机交付包的 SHA。

如果打包路径 A 的新模型，将 `BUNDLE_PATH` 换成 `dist/reproduce-bcfix-seed11/bundle`，
将 `BUNDLE_SHA` 换成本次准备完成后 `acceptance/report.json` 中记录的
`bundle_manifest_sha256`，并保留该报告。其他步骤不变。

## 路径 C：安装已有离线发行目录

复制整个发行目录，保留 `install.py`、`runtime.lock`、`wheels/`、`bundle/`、`manifest.json`
和 `COMPLETE`；只复制 `bc.json` 不构成离线发行包。目标机使用 Python 3.10.12，网络可关闭。
以下以路径 B 生成的 `dist/offline-release-01` 为例，将占位值替换成打包时单独保存的 SHA：

```bash
python3 -I dist/offline-release-01/install.py \
  --expected-release-sha256 RELEASE_SHA_FROM_PACKAGE \
  --venv dist/offline-runtime-01 \
  --accept-output dist/offline-acceptance-01 --dry-run
python3 -I dist/offline-release-01/install.py \
  --expected-release-sha256 RELEASE_SHA_FROM_PACKAGE \
  --venv dist/offline-runtime-01 \
  --accept-output dist/offline-acceptance-01
```

安装器验证全部文件和外部 SHA，在独立虚拟环境中离线安装依赖与项目，然后检查实际加载的
源码、资源、版本及推理预算；指定 `--accept-output` 才会继续运行全部 16 个仿真回合。
只安装成功不等于策略验收通过。报告位于 `dist/offline-acceptance-01/report.json`。

同一发行可重跑安装，但每次验收要用新结果目录；环境属于其他发行时会拒绝覆盖。源码／
资源校验失败时使用新的环境目录重新安装。回退时直接选择旧发行目录和它的独立环境。
本机交付的一键脚本、外部 SHA 与实测结果见[部署说明](learning_deployment.md)。

## 常见问题

| 现象 | 检查和处理 |
|---|---|
| `Missing local release` | Git 不携带 `dist/`；走路径 B 打包，或复制完整交付包 |
| `requires CPython 3.10.12` | 换对应解释器重建虚拟环境，不修改版本检查 |
| `resume ... differs` | 回到原 HEAD、参数和环境；若有意改变配置，换新目录 |
| `runner/runtime ... differs` 或源码哈希不同 | 使用模型绑定的源码和锁定运行库；不要改 manifest |
| 输出目录已存在 | 恢复 prepare 时用 `--resume`；独立评估用新目录 |
| `engineering_status=PASS` 但 BC 为 false | 模型可运行但控制指标未达标；查看逐 case 指标 |
| `ModuleNotFoundError: torch` | 只有重训需要 learning-cpu；公开模型验收应在 core／离线环境执行 |

延迟结果针对测量主机；同步策略超时在调用返回后识别，不保证硬实时抢占。以上流程不会
连接真机。development-test 是已公开使用过的任务域，模型通过这里的门槛也不能扩展为
未知表面上的泛化结论。
