# 训练前的分层修正

最初的 `20260906` 切分将两个 development-test 组都分配为标称场景。
在尚未训练策略时，按场景配置增加条件：三个标称锚点分别进入三个 split。
从原 seed 起取第一个满足条件的 seed，结果为 `20260908`；搜索不接收性能指标。
这是查看经典基线结果后的公开数据整理，不是首次揭盲或一轮新的仿真。

`partition.json` 保存全部三个候选 seed、组身份、父 manifest 哈希和前后审计结果。
三个 `v1_*_manifest.json` 是原始文件的字节副本，对应父哈希。
新划分见 [v2 数据清单](../franka_surface_learning_preparation/dataset_manifest.json)。
NPZ 和事件 JSON 字节不改，物理参数、观测、动作、回报和性能指标不改。

计数仍为 8/2/2 组；validation 和 development_test 现在各包含一个标称组、一个联合扰动组。
每组有两个噪声重复回合。这个小规模公开开发集不提供充分的泛化或真机证据。

生成器：[`repartition_surface_preparation.py`](../../tools/repartition_surface_preparation.py)。
