# KSR 树状分枝、单步干预与 random step 数据构造

> 入口迁移：本文保留历史实验说明。批量推理使用 [scripts/infer.py 的新文档](INFERENCE.md)；旧实验包装脚本已删除。逐步 judge 仍可使用 `scripts/judge_trace_batch.py`。

入口：`scripts/construct_step_data.py`。
Python 函数：`mixed_pilot_harness.branch_sampling.core.collect_ksr` / `collect_ksr_step` / `collect_random`。
复用现有 provider、ScienceWorld episode、回复 parser 和冻结价格表；不引入新的训练依赖。
本功能不需要 MTRouter 网络或 per-step LLM judge。

## 模式和筛选

默认任务源 `/Users/liangjianfeng/Data/scienceworld_task`，当前包含 29 个 variation 0 实例。
默认强模型 `deepseek-flash`，弱模型 `qwen-flash-2025-07-28`；不读取 GLM 的轨迹。

- **ksr**：以 `(task_name, variation_idx, simplification_str)` 匹配 baseline 数据；至少一条强模型成功，且至少一条弱模型失败，就选中该任务。不同 run 不要求编号配对，也不要求弱模型每次都失败。目前筛出 13 个任务实例。
- **ksr_step**：沿用 ksr 的任务筛选，但从单条弱模型主轨迹开始；只在 KSR 点比较“当前步强、后续弱”和“当前步弱、后续弱”。根节点不分枝，干预侧不递归分枝。
- **random**：不依赖 baseline 的结果，遍历任务源全部任务；每次 API 调用独立均匀选择强/弱模型，各 0.5。目前 29 个任务。
- 默认 `--split all`，可显式设置 train/dev/test。`--task-filter`、`--limit` 在计划中可见；ksr / ksr_step 的 limit 在资格筛选之后应用。

## KSR 的分枝语义

```text
同一初始状态：分枝点 0
├─ strong 连续执行
│  └─ 发现关键状态 t（执行 step t 前）
│     ├─ strong 连续执行 → 后续关键状态可再分枝
│     └─ weak 连续执行   → 后续关键状态可再分枝
└─ weak 连续执行
   └─ 发现关键状态 u
      ├─ strong 连续执行
      └─ weak 连续执行
```

分枝点的两个 action 都实际调用模型。选中某模型后，直到下一分枝点保持该模型。
发现分枝后，**切换模型的子段立即沿用当前真实环境和历史继续执行**；保持原模型的子段记录不可变历史前缀，进入 FIFO 队列，稍后重放恢复。
每次只持有一个环境，不会为切换分枝重新启动 JVM。根节点仍按 strong、weak 的顺序初始化。
这是切换分枝优先的调度，不再是严格广度优先；节点数或预算受限时，先运行的路径可能先用掉分枝名额/预算。
`plan.json` 记录 `scheduling`，每个节点的 `state_source` 为 `initialization`、`live_continuation` 或 `replay`。
直接续跑节点的 `replay=[]` 表示无需恢复，不表示重放失败。环境在当前连续路径结束后统一关闭。

KSR 只用**此前已执行步骤**判定，不偷看未来成功标签、不预先调用另一个模型比较回复。
分枝在下一次模型调用前发生。默认规则按以下优先级：

1. 最近 2 步连续回复格式错误或反馈含无效动作标记。
2. 最近 3 步命令完全相同。

距上次分枝至少执行 3 步（`--min-branch-gap`）。所有窗口都有同名 CLI 参数。
规则只检查当前段，避免继承历史让新子分枝立即再次分枝。
两种 KSR 模式共用以上两条规则，不保证找出所有任务真正的关键步骤。
已移除“无分数进展”检测及 `--stagnation-window` / `STAGNATION_WINDOW` 参数；原生 score 只用于回报标签、记录和重放一致性检查。

默认每个 task/run 最多 14 个节点、3 层再分枝、每条路径 50 个 API-step。
节点是“模型保持不变的一段”，分枝点 0 的两个节点深度为 0。
触及树深/节点上限时不再分枝，继续当前模型直到结束或路径限制；被抑制的触发记录在节点中。

## ksr_step：弱模型主轨迹上的单步干预

```text
初始状态 → weak 连续执行（没有根分枝，也不采样初始点）
  └─ KSR 状态 t
     ├─ 当前步 strong → 后续全部 weak → 终局（不再分枝）
     └─ 当前步 weak   → 继续 weak 主轨迹
        └─ 下一个 KSR 状态 u
           ├─ 当前步 strong → 后续全部 weak → 终局（不再分枝）
           └─ 当前步 weak   → 继续 weak 主轨迹
```

每个 strong 干预侧只有第一次 API 调用使用强模型，后续调用恢复弱模型，即使该侧再次错误或重复也不触发 KSR。
格式错误同样消耗这一次强模型调用；若当前步已结束任务，则不额外调用弱模型。
恢复弱模型只改变模型选择，**保留干预后的环境状态和完整历史**。
弱侧继续检测 KSR，但它在所有真实执行步骤都使用弱模型；未来 strong 干预是独立的旁支，不进入弱侧的回报。

调度继续复用真实环境：先在当前环境执行 strong 单步干预并弱模型续跑至终局，
再重放分枝前的不可变前缀，恢复弱主轨迹。每次只持有一个环境。
两侧当前调用前的 history、previous_actions、路径预算必须一致。
预算不足时仍可能只完成一侧，因此训练比较时应检查 `paired_actions_complete`。

`tree.json` 的节点带有 `base_profile` 和 `node_role`（`mainline` / `probe`）。
`profile` 表示该段首次调用的模型；probe 后续实际模型以 calls 中的 `selected_profile` 为准。
根节点 `branch_point_id=null`；主输出只采样真正分枝点的两侧首步，初始段和后续连续步骤仅保存在 calls 中。

该模式最多使用 `1 + 2 × 分枝次数` 个节点；`--max-depth` 限制弱主轨迹的分枝次数，
`--max-nodes` 限制总节点数。默认 14 个节点、最大深度 3，最多触发 3 次比较。
`--max-depth 0` 或 `--max-nodes 1` 可关闭分枝，只运行弱模型且不生成比较样本。
两次分枝之间的规则窗口从新的弱模型段重新累计，受 `--min-branch-gap` 限制。

```bash
# 检查新模式计划；不调用模型、不创建输出目录。
python scripts/construct_step_data.py --mode ksr_step \
  --error-window 1 --repeat-window 2 --min-branch-gap 2 \
  --output-dir run/branch-data/ksr-step-v1 \
  --step-output-dir run/step-data/ksr-step-v1 --dry-run
```

将 `--dry-run` 换成 `--execute` 执行；支持原有 `--runs` / `--until-budget` 和三层预算。
现有两个 shell 采集脚本仍使用原 `ksr` 模式，新模式通过此入口显式选择。

## 环境复原与一致性

根分枝共享同一初始观察。切换模型的子分枝直接继承当前真实状态；保持原模型的子分枝通过重新初始化真实 ScienceWorld 环境，重放共享前缀回复恢复。
重放走原 `ScienceWorldEpisode.advance`：查询不消耗环境回合、格式错误不执行动作、完成标记结束任务。
重放不调用 LLM，因此没有模型 API 成本，但仍有本地模拟器/CPU 时间开销。

每一步比对命令、原始观察、格式错误、原生 score、done，初始观察也必须一致。
失败默认重试 3 次；仍不一致则节点标记 `replay_failed`，不调用该分枝的模型。
**不排序/归一化观察文字，不把顺序差异静默视为一致。** 物体顺序随机变化可能导致分枝被拒绝。
`tree.json` 保存各次重放结果。以上验证不等于证明隐藏模拟器状态完全一致；尚无原生快照克隆。

## 三层预算与账本

费用单位跟随 `configs/collection.prices.json`（当前 CNY）；使用返回的 usage、缓存命中和价格时段估计，非实际账户账单。

| 参数 | 范围 | 默认 |
|---|---|---:|
| `--total-cost-limit` | 本次采集全部 tasks/runs 的唯一实际调用 | 10 |
| `--task-cost-limit` | 一个 task/run 的整棵树（或 random 轨迹） | 1 |
| `--path-cost-limit` | 一条 root→leaf 路径，含继承的前缀成本 | 0.5 |

共享前缀存为 call ID 引用：它在路径成本中出现，但不在全局/任务账本中重复计费。
**预算是调用前检查的停止阈值**，最后一次调用可能超出阈值；串行执行后不再开启下一次调用。
当前实现不是预付费硬上限，也不提前精确预测输出 token 数。

超时/缺少 usage 等无法确定费用时：记录失败，`cost_complete=false`，`unpriced_calls>0`，终止后续采集。
`total_api_cost` 此时仅是已知费用下界，未知费用不会伪装成免费。
即使环境执行失败，已返回模型调用的费用也会保留。

`events.jsonl` 在发起调用前写入事件，返回后记录费用；每个调用的原始响应/usage 先落盘，再执行环境。
输出目录必须不存在，重复使用会报错；本版本不自动断点续跑，避免配置混用和重复付费。
发生进程崩溃时先检查事件与 calls，不应把不完整输出当成完整账单。

## 数据输出

```text
output/
  plan.json         任务、规则、价格、prompt、provider 配置（不保存 Key）、环境版本
  tree.json         根/父/子节点、共享前缀、分枝原因、重放结果、终局结果
  calls/c*.json     每次唯一模型调用的 state、回复、反馈、usage、费用和原生 score
  events.jsonl      调用开始/费用事件
  samples.jsonl     训练样本
  summary.json      样本数量、全局成本、task/run 成本、未知费用和失败节点数
```

- **ksr**：每个分枝点（含根）的 strong/weak 子段各取第一步。中间连续步骤保留在 calls 中，但不重复采样。
- **ksr_step**：只取主轨迹 KSR 分枝点的 strong/weak 首步；没有根节点样本。
- **random**：每个实际返回并进入环境处理的 step 都采样。
- 模型调用失败仍保存在 calls 与 events 中，但没有合法回复时不生成训练样本。
- `state` 是该步之前的历史/system/previous_actions/路径预算；记录原生 `score_before/score_after/done` 在 calls 中。

每条 sample 包含 `sample_id, task_id, run_id, node_id, parent_id, branch_point_id, step_idx,
state, action_profile, action_model, response, observation, cost, y, labels_valid,
selection_probability, paired_actions_complete`。
step_idx 从 0 开始；tree 中 split_before_step 是面向用户的从 1 开始步号。

### 标签 y

random 的全部步骤使用该轨迹归一化终局分数 `(final_score+100)/200`。

原 `ksr` 模式一个 action 后可能继续分枝，不能只拿最好的叶子当标签：

- 叶子正常结束、或达到预先设定的路径预算/步数限制：用当前原生终局分数归一化。
- 内部节点：递归取两个子节点回报的平均值，对应未来每个分枝点 50/50 选择的预期回报。
- 任一子树因全局/整树预算、重放失败、API/环境错误没有完整标签：该节点 `y=null`。

这是**分枝续跑回报估计**，不是即时奖励、judge 分数或最优动作价值。

`ksr_step` 使用 `single_step_then_weak_terminal_score`：

- strong 样本：当前调用 strong 一次后，一直使用 weak 得到的归一化终局分数。
- weak 样本：当前及后续全部使用 weak 得到的归一化终局分数；沿 `mainline` 子节点传递，**不平均后续 probe 回报**。
- 已完成的路径或预设路径预算/步数截断仍有有效标签；全局/整树预算、API/环境错误或重放失败导致缺失的续跑回报为 null。
- 某次 probe 失败不自动使正常完成的 weak 回报失效；weak 续跑失败也不抹掉已完成 probe 的回报，但这样的比较不是完整对。
- 额外保存 `base_profile`、`continuation_profile`（均为弱模型）、`action_scope=single_step`。
  `cost` 仍是当前调用费用；`remaining_api_cost` 是当前调用及后续 weak 续跑的总费用，不含共享前缀，也不含未来旁支费用；缺失回报时为 null。

同一分枝点的两个 sample 都有完整回报，才标记 `paired_actions_complete=true`。
构造强弱比较数据时优先过滤此字段；不要将 `null` 转成 0，也不要只保留成功叶子。
KSR 的动作是枚举执行，`selection_probability=null`；random 才记录真实抽样概率 0.5。

`samples.jsonl` 是显式 (历史, 模型, y) 表；现有 JSONL 适配直接使用 y，见
[训练适配说明](../KSR_DATASET_ADAPTATION_AND_TRAINING.md)。原 ksr 与 ksr_step 的续跑策略不同，
应分别评估，不能当成同一目标直接混合。当前 loader 的成本字段仍使用 `cost`，不会自动改用 `remaining_api_cost`。
不能把 tree.json 当普通轨迹输入，也不能把每个叶子展开后无权重复制共享前缀，否则会重复计数并改变采样分布。

### 可选：每个叶子路径及其全部 step

通过 `--step-output-dir PATH` 额外保存完整路径视角的数据。该目录必须不存在，且与主输出目录不同、互不嵌套。
不设置则保持原有输出。该导出不调用模型，不增加 API 费用，在采集结束的收尾阶段生成：

```text
step-output/
  traces.jsonl   每个叶子节点一条记录，包含 root→leaf 的 call_ids、最终分数和状态
  steps.jsonl    每条叶子路径的每个已有 step 一条记录（包括共享前缀）
  summary.json  路径数、展开后的 step 数、有效标签 step 数、唯一调用数
```

这里与原 `samples.jsonl` 的语义不同：原文件是每个分枝点每个 action 一条，包含内部节点，
原 ksr 的 y 为未来分枝平均回报，ksr_step 的 y 为对应单步选择后 weak 续跑回报；
`steps.jsonl` 每条记录的 y 来自所属叶子路径的终局分数。
同一个前缀 step 可以因后续路径结果不同，在两条 trace 中具有不同 y。

- `trace_id` / `leaf_node_id` 标识叶子，`sample_id=leaf_id:call_id` 唯一，`step_idx` 从 0 开始。
- `trace_complete=true` 表示路径自然结束或提交（不等于成功）；步数/路径预算截断的路径标记 false，但保留有效的预算内终局 y。
- 重放失败、全局/整树预算中止、API/环境错误的路径也保留已有前缀，`trace_status` 说明原因，`labels_valid=false, y=null`；它们不冒充完整轨迹。
- 无合法模型回复的失败调用仍只保存在主目录 `calls/` 和费用账本，不构造虚假 step。
- `call_id` 指向唯一实际调用，`call_multiplicity` 表示它在导出路径中重复出现的次数。
  `cost` 和 `path_api_cost` 是引用成本，**不能直接跨路径求和当总费用**；总费用以主目录 `summary.json` 为准。
- 训练时必须按任务划分训练/验证集，并根据训练目标处理前缀重复权重；这些展开数据不会自动适配 MTRouter。

```bash
python scripts/construct_step_data.py --mode ksr \
  --output-dir run/branch-data/ksr-v2 \
  --step-output-dir run/step-data/ksr-v2 --execute
```

## 使用

### KSR 和 random 各自持续采集到指定预算

统一入口 `scripts/collect_ksr_random_budget.sh`：`--budget B` 表示 **KSR 单独 B、random 单独 B**，
目标合计为 2B；默认价格表的单位是 CNY。先串行执行 KSR，再执行 random，不共享账本或挪用剩余额度。

```bash
conda activate opensquilla
# 只检查两种模式的计划，不创建输出目录、不调用 API：
bash scripts/collect_ksr_random_budget.sh --budget 5 --dry-run

# 真正执行：KSR 5 元，random 5 元。
bash scripts/collect_ksr_random_budget.sh --budget 5 --execute \
  --output-root run/budget-data/ksr-random-5yuan
```

输出根目录必须不存在，省略则自动使用 `run/budget-data/时间戳-进程号`：

```text
output-root/
  ksr-branches/  分枝点 samples.jsonl、tree、calls 和 KSR 费用账本
  ksr-steps/     traces.jsonl（每叶子一条）和 steps.jsonl（每路径每步一条）
  random/        samples.jsonl（逐 step）、calls 和 random 费用账本
  summary.json  两种模式的成本、停止原因和采集数量汇总
```

底层新增 `--until-budget`，与 `--runs` 互斥。它按轮次遍历所有选中任务：先全部任务 run 0，
再全部任务 run 1，以此类推，直到该模式的 `total-cost-limit` 用尽。KSR 当前每轮 13 个候选任务，
random 每轮 29 个任务。轮数预先未知，汇总中 `planned_jobs` / `unstarted_jobs` 为 null，
`started_jobs` 为实际启动次数；每个任务的 `run_id` 随轮次递增。

仍保留每个 task/run 1 元、每路径 0.5 元、每路径 50 步、KSR 每树 14 节点和最大深度 3 的限制。
单个任务或路径达到自身限制，不会直接结束整个采集，会继续后续任务/轮次。
KSR 使用错误窗口 1、重复窗口 2、分枝最小间隔 2。
可通过 `TASK_COST_LIMIT`、`PATH_COST_LIMIT` 等环境变量覆盖，完整列表见脚本 `--help`。

`stop_reason` 包含：

- `budget_exhausted`：达到总预算停止阈值；最后一次调用可能超出预算。
- `unknown_api_cost`：调用超时或 usage 不可用，停止该模式，防止在费用不明时继续调用。
- `no_cost_progress`：一整轮费用没有增长（例如全部环境初始化失败），提前停止，防止无限空转。
- `interrupted`：执行异常/被中断；保留已有数据，不自动续跑。

某种模式正常收尾但有失败节点时，脚本仍执行另一种模式，最后返回非零状态提示检查数据。
配置错误、异常退出或用户中断则停止整个脚本。失败/重放不一致不会自动丢弃或伪造标签。
运行结束时恰好被总预算截断的路径可能无有效 y，属于原有标签语义。

### 13 个候选任务各运行一次

`scripts/collect_ksr_13_tasks.sh` 调用现有构造入口，不截取任务列表；执行前验证资格筛选恰好得到 13 个任务。
每个 task 构造一棵树（根节点分别调用强弱模型），`runs=1`。在 opensquilla 环境、已有 provider keys 的终端中运行：

```bash
bash scripts/collect_ksr_13_tasks.sh --dry-run
bash scripts/collect_ksr_13_tasks.sh --execute
```

同时保存每条路径的所有 step：

```bash
bash scripts/collect_ksr_13_tasks.sh --execute \
  --output-dir run/branch-data/ksr-13tasks-v2 \
  --step-output-dir run/step-data/ksr-13tasks-v2
```

默认输出到带时间戳和进程号的新目录 `run/branch-data/ksr-13tasks-*`，可用 `--output-dir PATH` 指定不存在的目录。
默认规则比构造入口更敏感：错误窗口 1、重复窗口 2、分枝最小间隔 2。
因此小试里第 4 步已识别的无效动作就可触发分枝，无需等待第二条错误反馈；这不代表错误关键词已经全面覆盖。
较短的窗口也可能把正常准备动作识别为关键步骤，需检查实际回报差异。

预算默认总额 20、每 task 1、每路径 0.5（默认价格表单位 CNY）；总预算为全部 13 个任务留出余量，
但每任务预算仍可能截断树，API/未知费用错误也可能阻止后续执行，不能保证所有任务完整结束。
保留每路径 50 步、每树 14 节点、最大深度 3、重放 3 次的限制。可通过环境变量覆盖预算、profile 和规则：

```bash
TOTAL_COST_LIMIT=20 TASK_COST_LIMIT=1 PATH_COST_LIMIT=0.5 \
  ERROR_WINDOW=1 REPEAT_WINDOW=2 MIN_BRANCH_GAP=2 \
  bash scripts/collect_ksr_13_tasks.sh --execute
```

完整覆盖项见 `bash scripts/collect_ksr_13_tasks.sh --help`。默认仅 dry-run，不调用模型。

在项目根目录、已有 opensquilla/ScienceWorld 环境运行。
默认仅输出计划，不检查 API Key、不启动 Java、不调用模型：

```bash
python scripts/construct_step_data.py --mode ksr \
  --output-dir run/branch-data/ksr-v1

python scripts/construct_step_data.py --mode random \
  --output-dir run/branch-data/random-v1
```

准备好 keys 后加 `--execute` 执行。下面预算仅为用法示例：

```bash
python scripts/construct_step_data.py --mode ksr \
  --output-dir run/branch-data/ksr-v1 \
  --total-cost-limit 10 --task-cost-limit 1 --path-cost-limit 0.5 \
  --max-nodes 14 --max-depth 3 --step-limit 50 --execute

python scripts/construct_step_data.py --mode random \
  --output-dir run/branch-data/random-v1 \
  --total-cost-limit 10 --task-cost-limit 1 --path-cost-limit 0.5 \
  --step-limit 50 --execute
```

两次独立执行各有自己的总预算。若想两种模式合计不超过某个目标，分别分配额度，且考虑单次调用超额。
`--runs` 可对每个任务重复采集；`--limit 1` 可先做小规模运行。
自动发现当前 Python 环境的 `lib/jvm`；Java gateway 需要本机通信权限。

## 验证

单元测试覆盖：树结构、切换后模型保持、共享前缀不重复计费、random 全 step 采样、
全局预算截断标签、路径预算终局标签、重放不一致拒绝、未知费用中止、环境异常保留费用、任务匹配。
ksr_step 测试还覆盖：仅弱模型起步、根节点不采样、单步强干预后恢复弱模型、干预侧不分枝、
多次主轨迹分枝的状态对齐、弱侧标签不混入 probe、续跑成本、节点/深度限制、失败侧标签隔离及叶子路径导出。
计划验证为 13 个 KSR / 29 个 random 任务。开发验证不调用付费模型。

已完成真实 ScienceWorld（Java）+ 离线固定回复的集成验证：构造 6 个树节点、12 次固定回复调用、
6 条有效分枝点样本，所有共享前缀严格重放通过。测试使用虚构 usage 验证计费路径，未产生实际 API 费用。
新的 ksr_step 模式尚未执行真实 DeepSeek/Qwen 付费采集，规则及单步干预的训练效果需要后续评估。
