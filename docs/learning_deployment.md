# BC / PPO 的 MuJoCo 部署与验收

本入口把已有示教采集、BC、bounded PPO、有限 JSON 模型和独立闭环评估串起来。
目标是 Linux x86_64 上的本地 MuJoCo 仿真；没有连接真实机械臂。
训练使用 CPU PyTorch，部署只用 NumPy 推理。BC 使用 adaptive nominal，PPO 使用
friction nominal，两者的 49 维观测／3 维动作契约分别校验，不能交换模型。

## 本地交付（2026-10-09）

本机已有 `dist/learning-release-20261009/` 离线发行目录和独立运行环境。
在仓库根目录执行以下命令，可重新安装校验并完整复核两个模型及各自名义控制器：

```bash
./scripts/accept_learning_release.sh
```

[脚本](../scripts/accept_learning_release.sh)固定使用本次发行的外部 SHA256，并自动生成新的
结果目录。加 `--dry-run` 只检查，不写入、不仿真。发行目录、虚拟环境与大型训练轨迹保存在
本地 `dist/`，不随 Git 提交；其他机器须完整复制发行目录，或按下面步骤重新训练、打包。

本次发行源码：`36d1fd2a4fca3bfce4315fa4a605679f5fd77ae6`。
发行 manifest SHA256：
`ea5a1ccfbfc5c8c5d8be17334eecb01fca3ef51626efc9044176e53551f67f42`。
模型 bundle manifest SHA256：
`9317b374e95e4f5a61f9241cafb2f6e8d9148d4d726bfaa751513aa54acd6a25`。

源码版与离线安装版均完整运行 16 个回合，工程检查 `PASS`；所有物理指标、门槛判定和
完成状态完全一致，主机测得的调用耗时分别保留。[完整交付证据](evidence/learning_deployment_20261009.json)
记录源码、训练来源、依赖审计、原始指标及校验值。

| 策略／基线 | 最差切向 RMSE (mm) | 最差力 RMSE (N) | 策略门槛 |
|---|---:|---:|---|
| BC | 11.494 | 0.178 | FAIL：切向误差超过 10 mm |
| BC 的 adaptive nominal | 11.680 | 0.155 | FAIL：切向误差超过 10 mm |
| PPO | 1.702 | 0.158 | PASS |
| PPO 的 friction nominal | 1.673 | 0.159 | PASS |

BC 的其余六项门槛通过。PPO 虽通过绝对门槛，其四个 case 的切向 RMSE 相对自身基线
仍增加 0.024–0.029 mm，因此不作为优于解析控制的证据。整体验收命令返回 `2`，明确保留
BC 未达标结论；两个模型都可在独立仿真入口加载运行，默认控制器不变。

本轮 24 个示教回合、BC 100 epochs（验证集选择 epoch 68）、PPO 32 episodes
在 `708508d88e67e72163815c023ba5b2673ad17270` 完成，PPO 训练失败回合为 0。
后续只修复部署来源校验与安装隔离，未改 trainer、核心控制、案例或门槛。
逐项核验源码、参数和运行版本后，将数据与模型原样复制到新计划；旧计划及所有归档保留，
新源码版和安装版分别重新执行全部评估。复用依据位于本地
`dist/learning-run-20261009/reuse-evidence.json`。

验证：2605 项 Python 测试通过、零跳过；部署专项 49 项通过；独立安装审计 35/35 通过。
NumPy/Torch 导出对照最大误差分别为 BC `4.44e-16`、PPO `1.04e-17`。
安装版每个模型的 512 次推理测量中，最慢调用分别为 0.0405 ms、0.0319 ms，低于
20 ms 策略周期；冷加载约 78–84 ms，须在控制循环开始前完成。这是本机测量。
已有 1540 个受保护的核心／资源／历史证据文件字节未变。

完整报告：本地 `dist/learning-installed-acceptance-20261009/report.json`；
源码准备报告：`dist/learning-run-20261009/acceptance/report.json`；
验证日志与独立审计：`dist/learning-verification-20261009/`。
离线压缩包为 `dist/learning-release-20261009.tar.gz`（约 68.2 MiB），另附 `.sha256`。
压缩包 SHA256：`1e0640cd3297dd0ba9e37909d8b0bb11d5cd1e1d1a0418de190e48ab83fe7df9`。

## 从源码准备

先按[固定环境](reproducible_environment.md)安装 core 与 learning 依赖。
新增入口可直接在仓库根目录用 `python -m tools.learning_deployment` 调用；安装 wheel
后对应 `franka-learning-deploy`。准备必须从已提交源码执行，以便产物绑定具体 commit。

```bash
python -m tools.learning_deployment doctor --training
python -m tools.learning_deployment prepare --workspace dist/learning-run-01 --dry-run
OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=disable \
  python -m tools.learning_deployment prepare --workspace dist/learning-run-01
```

最后一条命令完成全链路：24 个 12 秒示教回合、BC 100 epochs、PPO 32 episodes、模型导出、
NumPy/Torch 输出对照，以及 BC、PPO 和各自名义控制器的完整 development-test 对照。
物理参数按组固定分为 16 train / 4 validation / 4 development-test；BC 仅按验证动作
MSE 选 checkpoint，PPO 固定使用训练前声明的最终回合，不按 development-test 选模型。
这是单种子工程验收，不能用来宣称收敛或跨种子优于解析控制。

`plan.json` 在任何训练前写入，记录源码、参数、分组、选择规则和原有性能门槛。
改变 `--seed`、`--bc-epochs`、`--ppo-episodes` 要使用新目录；PPO 回合数须为 4 的倍数。
同一个计划可加 `--resume`：逐项校验已完成阶段后复用，不覆盖已有数据或失败记录。
已有目录不属于本计划、来源不同或数据被修改时明确报错。

输出中的 `bundle/` 只含模型、接口对应的 cases、冻结评估协议与哈希清单。`acceptance/`
保存完整验收摘要；四个 `evaluation_*` 目录保存原始轨迹、指标和失效记录。
`engineering_status`、`policy_acceptance` 和相对名义控制器的误差变化分别报告。
退出码 `0` 表示当前命令通过，`1` 表示输入／环境／完整性或执行错误，`2` 表示验收未通过。
`prepare`／`accept` 返回 `2` 时，应分别检查工程与策略字段：工程通过但策略未达标，
仍保留失败判定；安装完成不代表策略验收通过。

## 打包成离线发行目录

下载步骤需要网络；部署后的推理不需要网络。runtime lock 是已核验 core lock 的 23 个
运行依赖子集，不包含 Torch、pytest、ruff 或构建工具，范围限定为 Linux x86_64 / Python 3.10。

```bash
python -m pip download --require-hashes -r environment/deployment-runtime.lock \
  --dest dist/learning-wheelhouse
python -m tools.learning_deployment package \
  --bundle dist/learning-run-01/bundle \
  --expected-manifest-sha256 BUNDLE_SHA_FROM_PREPARE \
  --wheelhouse dist/learning-wheelhouse --output dist/learning-release-01
```

将 `BUNDLE_SHA_FROM_PREPARE` 换成 prepare 成功或失败报告里记录的 `bundle_manifest_sha256`。
保留这份报告作为外部校验依据，不在使用时从待检查的文件重新计算“预期值”。
打包器从模型所绑定的 Git commit 导出干净源码，构建 wheel，携带所有运行依赖与模型。
它打印 `release_manifest_sha256`，应单独保存。

## 安装后的一键验收

完整复制发行目录后，在目标机器执行以下命令。Python 小版本须与模型记录完全相同。
把 `RELEASE_SHA_FROM_PACKAGE` 替换为打包时保存的哈希；环境与结果目录使用新路径。

```bash
python3 dist/learning-release-01/install.py \
  --expected-release-sha256 RELEASE_SHA_FROM_PACKAGE \
  --venv /tmp/franka-learning-env-01 \
  --accept-output /tmp/franka-learning-acceptance-01
```

加 `--dry-run` 可先核对发行目录和目标路径，不安装、不联网、不运行仿真。
安装器检查每个文件，创建独立虚拟环境，离线安装锁定依赖和 wheel，再核对真正加载的
源码／assets／库版本。它隔离 `PYTHONPATH`、用户级 Python 包与 pip 环境变量／配置。
目标机无需安装训练环境或 PyTorch，`verify`、`run`、`accept` 均使用 NumPy 推理。
同发行版本可重跑安装；已有环境没有对应标记时拒绝覆盖。pip 可能跳过已有同版本包，
因此源码或 assets 校验失败时请使用新的虚拟环境路径。重新验收也须指定新结果目录，
已有报告不会覆盖。

安装成功后也可分别运行：

```bash
/tmp/franka-learning-env-01/bin/franka-learning-deploy verify \
  --bundle dist/learning-release-01/bundle \
  --expected-manifest-sha256 BUNDLE_SHA_FROM_PREPARE --profile
/tmp/franka-learning-env-01/bin/franka-learning-deploy accept \
  --bundle dist/learning-release-01/bundle \
  --expected-manifest-sha256 BUNDLE_SHA_FROM_PREPARE \
  --output /tmp/franka-learning-acceptance-02
```

`accept` 不训练，完整重跑两种策略及两个名义控制器，共 16 个开发测试回合。
`run --algorithm bc` 或 `run --algorithm ppo` 可只运行一种策略；`--nominal` 使用同接口的
零残差基线。它们也支持 `--dry-run`，在创建环境前检查模型和协议。

## 回退与边界

每个版本使用独立发行目录和虚拟环境。回退就是重新使用上一版的目录、环境和外部哈希，
无需覆盖模型或卸载共享 Python。保留失败验收输出，修改参数后使用新计划重新执行。

推理延迟报告针对当前主机；同步超时在调用返回后发现，不提供硬实时抢占保证。
失败动作由已有环境执行零残差回退并终止回合。历史 BC/PPO、v0.5 和换向恢复失败结论保持不变，
当前验收通过也不构成真机安全认证或策略优越性证据。
