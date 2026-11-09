# 逐 step 标签采集

入口：`scripts/collect_steps.py`（安装包后也可用 `mixed-pilot-collect-steps`）。
复用 `infer.py` 的数据集加载、环境隔离、Agent、评测、费用记录、重试、恢复和 trace 导出。
已有的 `infer.py --mode ksr` / 内部 `ksr_step` 保持原语义；它们与本文的新策略不同。

## 随机采样

```bash
python scripts/collect_steps.py \
  --dataset /Users/liangjianfeng/Data/Router/SWE-neo-data \
  --split train --mode run --runs 1 \
  --sampling-strategy random \
  --output-dir run/step-random
```

默认直接使用 `harness.json` 的 `random-router` profile，包括模型池和 `possibility`。
每步重新选择；未配置 possibility 时均匀采样。可用 `--router-profile` 指定其他
RandomRouter profile，或用 `--models` / `--router-possibility` 覆盖其池与权重。
这里不接受 MTRouter profile。

## KSR-brunch-1

```bash
python scripts/collect_steps.py \
  --dataset /Users/liangjianfeng/Data/Router/SWE-neo-data \
  --split train --mode run --runs 1 \
  --sampling-strategy KSR-brunch-1 --ksr-config KSR_config.json \
  --max-depth 3 --max-nodes 15 \
  --output-dir run/step-ksr
```

兼容拼写 `ksr-brunch-1`、`ksr-branch-1`。

每个任务的每种起始模型分别运行一棵独立树，各有一个根节点。令该树的起始模型为 A、另一个模型为 B：

```text
A → A → KSR 触发（分枝前历史完全相同）
         ├─ A → A → … → 可再次 KSR 分枝
         └─ B → A → … → 可再次 KSR 分枝
```

A 在整个任务树中始终是该任务的起始模型。B 只接手一次 API step，包含该回复的工具动作；
然后恢复 A。两个子分枝都能继续分枝，并不是只给主线分枝。
每次后续分枝仍对比 A 与“B 一步后回到 A”。

先利用当前环境运行切换模型的子分枝，另一分枝通过重放共享前缀恢复。
重放不调用 LLM、不重复计费；工具观察、原生环境分数必须精确匹配，否则标记重放错误。
SWE 仍使用已隔离的工作区与独立评测，KSR 不读取隐藏测试、gold patch 或评测答案。

### KSR_config.json

```json
{
  "strong_model": "deepseek-flash",
  "weak_model": "qwen-flash-2025-07-28",
  "default_start": "weak",
  "strong_first_tasks": ["hard-task-id", "project-hard-*"],
  "weak_first_tasks": ["easy-task-id"],
  "KSR_strong2weak": {
    "signals": ["easy_progress"],
    "min_branch_gap": 3,
    "progress_window": 2
  },
  "KSR_weak2strong": {
    "signals": ["consecutive_errors", "repeated_command", "stalled_progress"],
    "min_branch_gap": 3,
    "error_window": 2,
    "repeat_window": 3,
    "stall_window": 4
  }
}
```

- 模型名是 `harness.json` 的具体 profile；强弱是用户定义，不自动评估模型能力。
- 起始任务名单按数据集 `task_id` 匹配，支持 shell 风格 glob。同一 task 可以同时匹配两份名单，每轮分别从强、弱模型开始运行两棵独立树。同一名单内多个模式匹配不会重复运行。
- 每轮先运行 strong-first 组，再运行 weak-first 组；每组内部保持数据集顺序。未匹配任何名单的任务按 `default_start` 加入相应组。
- 两份名单为空时，所有任务按 `default_start` 运行一次。
- job 由 `(task_id, run_id, base_profile)` 区分，两个方向的历史、树限制、标签和断点恢复独立。总费用与全局 budget 仍由整个采集共享。
- 每组规则的 `signals` 按顺序检测，任一命中即可触发，记录规则、方向和证据 call ID。
- `min_branch_gap` 是自本分枝开始以来至少执行的 step 数；不会刚分枝又在同一状态继续分枝。

当前规则是可配置的**启发式判断**，不保证下一步难度：

| 信号 | 条件 |
| --- | --- |
| easy_progress | 最近 progress_window 步均无错误，且每步有原生分数进展或明确成功输出 |
| consecutive_errors | 最近 error_window 步均格式错误或包含错误提示 |
| repeated_command | 最近 repeat_window 步执行同一个非空命令 |
| stalled_progress | 最近 stall_window 步无分数进展，且观察文本反复不变 |

也可配置 `error_markers`、`success_markers` 字符串列表。平坦分数本身不触发停滞，因为
SWE/HLE 在最终评测前通常没有中间分数。规则只读取模型已见到的过去动作、工具观察和
原生分数；不额外调用付费 judge。进一步替换判断算法的入口是
`src/mixed_pilot_harness/step_collection/ksr.py::detect`。

`--max-depth`、`--max-nodes` 限制树膨胀；达到上限后记录 suppressed_forks 并继续当前路径。
深度 3 的完整二叉树需要 15 个节点，默认 infer 的 14 个节点可能阻止最后一次分枝。

## Run / budget 与恢复

- `--mode run --runs N`：遍历所选 split N 轮；每轮先强起始组、再弱起始组。summary 的 total_tasks 为原始任务数，planned_jobs 为包含两组的运行任务数乘以 N。
- `--mode budget --budget 10`：循环遍历直到已知费用达到 10（币种由 pricing 配置决定）。
- `--split train|test|all`、`--step-limit`、`--trace-budget`、`--pricing` 与 infer 相同。
- 每完成一个叶子 trace 等待 3 秒；`--max-retries` / `--retry-wait` 控制恢复。
- `--dry-run` 只做配置/数据/价格检查，不调用 API、不创建 Docker。
- `--resume` 重放未完成的工作；`--resume-partial` 重启 partial 任务，保留旧费用并排除旧标签。
- 配置内容存入 plan/node snapshot；更改 KSR 规则、起始名单、模型角色或随机概率需新输出目录。
- budget 是下一次调用前检查，单次调用可能超支；默认未知费用部分按 0 参与预算累计并继续采样，保留 cost_complete=false；实际账单可能更高。传 --stop-on-unknown-cost 可恢复严格停止。
- 多次任务的相同初始状态是多次 rollout，分枝共享前缀在训练样本和账本中只计一次。

HLE 如需 LLM 评测，传 `--judge-profile PROFILE`；也可 `--hle-evaluator exact`。

## 输出与标签语义

保留 infer 的完整 `calls/`、`tree.json`、`traces/*.traj.json`、`steps.jsonl`、
`events.jsonl`、`summary.json`、`SUMMARY.md`。总通过率和各模型费用仍在 summary 顶部。
KSR 的通过率分母是已评测叶子数，相关分枝不应解释为独立任务成功率。

**MTRouter 训练入口是 `training/samples.jsonl`，不是原始叶子展开的 steps.jsonl。**

每行一个 unique call，包含调用前 state、profile、实际 API model、step cost、
remaining_cost、归一化 y、真实随机 propensity、分枝 ID、基准 profile、规则策略及有效性。

- Random：y 是该 rollout 的归一化终局得分，remaining_cost 是从本步到终局的 Agent API 费用。
- KSR：分枝处对两个后续分枝各取 1/2，递归求预期终局得分/费用。不是把所有叶子直接平均，
  也不是把采集整棵树的账单当作一条轨迹的费用。
- 每个节点内所有步骤都有标签；费用使用各自的剩余后缀，不只导出分枝第一步。
- ScienceWorld 使用 `(score+100)/200`；SWE/HLE 等沿用各自评测归一化结果。
- KSR 枚举两个候选动作，selection_probability 为 null；不伪造随机抽样概率。
- 该标签描述**配置的采样/继续策略**下的收益，不是模型本身的绝对能力或最优 Q 值。
- 任一未来子分枝未评测完，祖先预期标签无效；已完成子分枝仍可保留自身标签。
- API 费用未知、replay/error、superseded 数据不进入有效训练集；记录保留供诊断。
- remaining_cost 不含终局 evaluator 费用；总账仍包含 evaluator、失败尝试和 superseded 费用。

## MTRouter 对接

现有命令不变。训练 YAML 的 `model_pool` 和模型元数据键应使用 **profile 名**，
例如 `qwen-flash-2025-07-28`、`deepseek-flash`；即使某个 profile 别名与实际 API model 不同，
动作身份仍按 profile 区分。配置编码器路径和训练设备后：

```bash
mtr-core-audit --data-dir run/step-ksr/training \
  --models qwen-flash-2025-07-28 deepseek-flash

mtr-core-precompute --config MTRouter/configs/training.local.yaml \
  --data-dir run/step-ksr/training --output run/step-ksr/embeddings.pt --backend hf

mtr-core-train --config MTRouter/configs/training.local.yaml \
  --data-dir run/step-ksr/training --precomputed-path run/step-ksr/embeddings.pt \
  --output-dir run/step-ksr/model
```

直接把 `run/step-ksr` 传给 data-dir 也会优先选择其 `training/samples.jsonl`，
避免同时读取 raw traces、旧 samples、展开叶子而重复训练。同一次训练不要同时传根目录与其 training 子目录。
新的 JSONL loader 读取真实 benchmark、币种和 remaining_cost；审计拒绝混合币种，
跳过显式无效/费用不完整的样本。旧格式未提供新字段时保留原回退行为。
当前训练默认单头预测终局 y（lambda=0），这里没有改成每步 judge 分数。
任务分组划分仍按 benchmark/task_id，所有相关分枝与重复轮次保持同一训练/验证侧。
