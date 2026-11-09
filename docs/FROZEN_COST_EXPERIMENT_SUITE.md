# DB16：冻结 C 在线训练与两组标签重放

实验子分支 `experiment/frozen-cost-suite__from_frozen-cost`，父提交 `ccc57bb`。
用户于 2026-11-03 确认代码改动、启动、统一 replay seed=42，以及只计旧 cold_start 采集费用。

## 实验安排

默认输出根目录 `outputs/frozen-cost-suite-20261103/`：

- GPU6：`online9/`。直接导入原 run01 step_000500.pt，完整冻结 C；最多9轮在线 Y-only，
  每轮训练完成后先保存 `iterations/iteration-0001.pt` 等权重，第3/6/9轮评估 test50。
  不重复进行 iteration=0 评估。
- GPU4：`run01_replay/`。从原 run01 历史标签快照重新初始化并拟合，seed42。
- GPU5：`run02_replay/`。从原 run02 历史标签快照重新初始化并拟合，seed42。
- 两组 replay 各输出独立的 `300step/` 和 `500step/`，每个目录包含权重、50题原生评估、
  调用账本和逐题轨迹；不进行在线训练、不新采集冷启动轨迹、不调用 Teacher。

重放严格按历史100/200/300/400/500快照累计拟合 Y/C，保留原epochs、batch、learning rate等；
每次只用当时已有的有效标签。不会从最终500标签筛前300个调用。300评估前后恢复训练RNG，
从而不改变后续400/500拟合的随机序列。统一seed只控制训练初始化与采样，不保证远程API或
原生环境完全确定；四次结果属于单seed数据来源消融。

## 费用规则

online的50 CNY是轮间软停止阈值，包含原 run01 `scope=cold_start` 的唯一调用已知费用
**0.61204095 CNY**（500笔、预留0），加本次新增Actor、Teacher和评估的已知＋预留占用。
不搬入旧online/旧test费用。历史费用来源和调用SHA记录在 `historical-cold-cost.json`。

达到50后，完成已开始的轮次、保存权重、完整评估一次，再停止后续轮；因此最终支出可能超过50。
正常预算结束状态为 `budget_round_completed`。单task/策略版本1 CNY、单路径50次Actor调用及
未知费用安全检查继续生效。已有成本标签和预测C都不能代替实际唯一请求账本。

两组replay分别独立记账，每组两次评估合计50 CNY预算。没有新增冷启动采集费用。

## 入口与恢复

```bash
cd /DB/data/yujietang/Agent/code/mixed-pilot
PY=/DB/rhome/yujietang/miniconda3/envs/mixed-pilot/bin/python
"$PY" scripts/run_frozen_cost_suite.py --dry-run
"$PY" scripts/run_frozen_cost_suite.py --gpus 6,4,5 --min-free-mib 8192
```

每个worker独立进程、独立GPU和日志，启动前检查显存占用。实际命令及GPU写入 `suite.json`，
进程信息、退出码、stdout日志位于 `processes/`。已运行/正常完成的job不会重复启动。

检查异常原因并修复之后，恢复单个job：

```bash
"$PY" scripts/run_frozen_cost_suite.py --gpus 6,4,5 --min-free-mib 8192 --only online --resume
# --only 也可指定 run01_replay 或 run02_replay。
```

online具备轮次事务、rollout journal和评估progress；replay保存训练checkpoint、源快照hash、
阶段progress及评估账本。恢复保留原费用、权重、完成题目，不删除旧输出重新付费。
若涉及代码工程修复，应保留启动版本/执行指纹与修复记录，不能直接抹掉源码兼容检查。

`scripts/train_frozen_cost_online.py` 是单独online入口，
`scripts/replay_ksr_cold.py` 是单独replay入口。两者均支持 `--dry-run`，合成链路用 `--smoke`。
API凭据仅在进程内读取原配置，不写入新配置、权重或进程参数。

45分钟巡检检查pid、GPU、日志、进度、账本、评估覆盖与C模型指纹；正常无实质变化保持安静，
完成后分析并报告，异常时保留证据并修复/恢复。全部结束后停止巡检。

## 启动前验证

2026-11-03：MTRouter 全套145 passed、30 subtests passed；新增预算/重放13项行为测试。
三个正式配置dry-run通过；旧cold导入、online与resume、run01/run02真实快照的合成重放五条CLI
检查均通过。验证没有调用付费API。日志保存在 `outputs/frozen-cost-suite-validation-20261103/`。

正式启动前DB28空卡已被占用，用户改指DB16并允许共享其4/5/6卡，随后允许检查DB21并优先使用空卡。最终选择DB21 GPU2/3/6，启动器记录真实hostname；原DB28没有正式进程。显存共享需显式 --min-free-mib 8192，本次未启用。

最终执行位置更新：按用户进一步要求，online由DB21 GPU2中断并使用同一输出--resume迁移至DB16 GPU6；run01/run02重放位于DB16 GPU4/GPU5。suite.json保存完整迁移历史，原始已完成交互和取消请求费用记录均保留。45分钟自动巡检已更新为DB16。
