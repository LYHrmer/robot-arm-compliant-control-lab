# BC / PPO 的 MuJoCo 部署与验收

本入口把已有示教采集、BC、bounded PPO、有限 JSON 模型和独立闭环评估串起来。
目标是 Linux x86_64 上的本地 MuJoCo 仿真；没有连接真实机械臂。
训练使用 CPU PyTorch，部署只用 NumPy 推理。BC 使用 adaptive nominal，PPO 使用
friction nominal，两者的 49 维观测／3 维动作契约分别校验，不能交换模型。

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
