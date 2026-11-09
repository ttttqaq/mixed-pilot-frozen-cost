# 通用数据集推理

唯一批量推理入口：`python scripts/infer.py`，或安装后的 `mixed-pilot-infer`。
在项目根目录、已安装依赖的 `opensquilla` 环境运行。默认读取 `harness.json`，
不会替换 profile 的真实模型名、思考模式或 token 上限。

## 关键参数

| 参数 | 含义 |
|---|---|
| `--dataset PATH` | SWE-neo 风格的 `train.json` / `test.json` 索引，或 ScienceWorld/ALFWorld/HLE 的 `tasks/*/manifest.json` 目录 |
| `--split train\|test\|all` | `all` 为 train + test，不包含 dev；也接受 `train-test` |
| `--mode single-model\|random\|ksr` | 固定模型、每步均匀随机、关键步骤分支；`ksr-branch` 是 `ksr` 别名 |
| `--models PROFILE ...` | harness.json 的准确 profile 名；固定模式恰好一个，其余至少两个 |
| `--stop-mode run-mode` | 遍历所有选中任务，按 `--runs N` 完整运行 N 轮 |
| `--stop-mode budget-mode --budget B` | 一轮接一轮遍历，直到累计调用费用到达 B；也接受用户原称 `budget-model` |
| `--output-dir PATH` | 必须是新目录，防止覆盖数据或混合不同配置；续跑必须显式传 `--resume` |
| `--pricing PATH` | 默认 `configs/inference.prices.json`；费用币种取文件的 currency |
| `--step-limit N` | 每条路径的 API 步数上限，默认 50 |
| `--trace-budget B` | 可选：每条路径的费用阈值，包含继承前缀 |
| `--seed N` / `--limit N` | 随机种子 / 调试任务子集 |
| `--dry-run` | 校验并打印计划，不调用 API、不运行 Docker/Java、不下载数据、不创建输出 |

其他参数用 `--help` 查看；SWE 的 `--cache`、`--cpus`、`--memory`、`--tool-timeout`、
`--eval-timeout`、`--pull-timeout` 可独立配置。需要 Docker 和原生 benchmark 镜像；
真实运行时可下载镜像/评测依赖，dry-run 不检查它们是否可运行。

HLE 默认需要 `--judge-profile PROFILE`；或显式选择 `--hle-evaluator exact`。
judge 费用单独列出，也计入总预算。

价格文件是可编辑的冻结估算表，不是自动更新的报价或账单。所有选中的 agent/judge
profile 都必须有价格，缺价直接报错，不能默认为免费。

## 示例

```bash
# 先验证 SWE-neo：100 train + 20 test，不调用 API
python scripts/infer.py \
  --dataset /Users/liangjianfeng/Data/Router/SWE-neo-data \
  --split all --mode single-model --models deepseek-flash \
  --stop-mode run-mode --runs 2 --output-dir run/swe-deepseek --dry-run

# 每一步从三个 profile 均匀抽样，总预算 5 元（默认价表为 CNY）
python scripts/infer.py \
  --dataset /Users/liangjianfeng/Data/Router/SWE-neo-data \
  --split train --mode random --models deepseek-flash glm-5 glm-4.7-flash \
  --stop-mode budget-mode --budget 5 --output-dir run/swe-random --dry-run

# ScienceWorld 关键步骤分支，不需要预先运行 baseline
python scripts/infer.py \
  --dataset /Users/liangjianfeng/Data/Router/scienceworld_task \
  --split train --mode ksr --models deepseek-flash glm-5 \
  --runs 1 --max-nodes 14 --max-depth 3 --output-dir run/sw-ksr --dry-run

# HLE，指定独立 judge
python scripts/infer.py \
  --dataset /Users/liangjianfeng/Data/Router/hle_tasks \
  --split train --mode single-model --models glm-5 \
  --judge-profile deepseek-flash --output-dir run/hle-glm5 --dry-run
```

去掉 `--dry-run` 才执行真实推理。分别评估多个固定模型时，各自使用一次
`single-model` 调用和不同输出目录；`random` 是同一条轨迹内部切换模型。

## KSR 与预算的准确语义

KSR 对全部选中任务建立每模型一个根分支。在连续错误或重复动作处，每个候选模型
产生一个子分支，之后保持该模型，直到下个分支点或终止。
触发规则通过 `--error-window`、`--repeat-window`、`--min-branch-gap` 配置。
`--max-nodes` 和 `--max-depth` 控制树大小。没有沿用旧实验“强模型成功、弱模型失败”
的 baseline 过滤器；用户的 dataset/split 就是本次任务集合。

切换的首个子分支继续使用现场环境，其余分支从干净环境重放共享前缀。
重放不调用模型、不重复计费；观察和分数不一致时记录 `replay_failed`，不生成有效标签。
SWE 的 shell 输出若带时间、随机数等变化，也可能无法严格重放。

预算是停止阈值，每次新 agent 请求前检查；最后一次模型/终局 judge 调用可能超出预算。
默认费用未知时继续采样，仅按已知费用累计预算（未知部分暂按 0），保留费用不完整标记，实际账单可能超过预算。`--stop-on-unknown-cost` 开启严格停止。预算中断的未评分分支保留空标签。
免费模型导致完整一轮费用无增长时，以 `no_cost_progress` 停止，避免无限循环。

连续三次正文与工具调用都为空，记 `EmptyResponseLimit`，该路径计 0 分和失败，
保留费用与 trace，然后继续任务。环境/评测故障保留诊断与空标签。普通 run-mode 在 API 费用不明时继续下一题，
摘要标记 `cost_complete=false`；只要设置任意费用预算，费用不明就停止。
明确未计费的请求拒绝（例如 400/429）只结束当前路径。

## 输出与指标

```text
output-dir/
├── SUMMARY.md          # 第一行通过率，第二个标题为总费用；运行中更新
├── summary.json        # 总费用、agent/judge 费用、通过率、未评分数、停止原因
├── plan.json           # 任务、模型设置、价格、采集参数快照
├── traces/*.traj.json  # 完整 info/messages/steps，兼容已有逐步评估结构
├── steps.jsonl         # 有效轨迹每一步的终局分数、成功标签、cost-to-go
├── samples.jsonl       # KSR 分支点标签；random/single 每个步骤一条
├── calls/*.json        # 唯一 API 调用、usage、response、状态及计费
├── tree.json           # 分支关系和重放验证记录
├── events.jsonl        # 调用事件
├── leaf-data/          # 所有叶路径及所有步骤，含无效/中断标签
└── workspaces/         # HLE/SWE 工作目录、工具日志、原生评测与 patch
```

**通过率 = 成功叶轨迹数 / 有效评分叶轨迹数**，摘要同时列出未评分轨迹数。
固定/随机模式每 task/round 一条；KSR 一题可有多个叶轨迹，因此 KSR 通过率不是
“一题只算一次”的独立任务准确率，也不是挑最好分支的 best-of 指标。

ScienceWorld 采用原生终局分数（>=100 通过）；HLE 用答案匹配/judge；
SWE 用现有 Docker 原生 grader，不能把模型说“完成了”算作通过。

每条路径可引用同一共享前缀；**总费用只看 summary.json 的 total_api_cost**，
不能将所有 trace 的路径费用简单相加。`steps.jsonl` 的 `remaining_cost` 是该路径
从当前 agent step 到结束的 agent 费用，不含终局 judge。
ScienceWorld 的 KSR y=(score+100)/200；ALFWorld/HLE/SWE 的 y=score/100。

ALFWorld 通过原生 `won` 状态判分，成功为 100，其余为 0。
ScienceWorld + ALFWorld 混合数据集见 [VirtualWorld-1000](VIRTUALWORLD.md)。
KSR 分支标签取后续有效子分支均值；缺失子分支时不伪造标签。

## 代码与旧入口清理

- `scripts/infer.py`：仅启动器。
- `src/mixed_pilot_harness/inference/cli.py`：参数、校验、执行。
- `inference/dataset.py`：数据集适配；`inference/episodes.py`：运行及评测适配。
- `branch_sampling/core.py`：共用采集循环、预算、重放、KSR。
- `inference/report.py`：trace/step 导出与摘要。
- `inference/swe_runtime.py`：迁移后的原生 Docker 运行库。

已删除固定模型/固定任务数的 baseline、four-model、KSR Bash 包装、旧 SWE train 入口，
以及重复的 collect_data/construct_step_data 启动器。旧核心 Python 模块仍保留，
历史输出保持原样。旧格式实验需要另行处理，不能把它们当作新入口的续跑目录。

保留的数据构建/校验、ScienceWorld 重放、逐步 judge 和旧四模型报告汇总脚本有独立用途。
原始方法与旧实验说明仍在其他 docs 文件中，执行命令以本文为准。


## 预检、续跑、仅重评

这些入口仍使用 `scripts/infer.py`。重评与预检无需 API key，也不调用模型。

```bash
# Docker 环境修好后，先验证空补丁应失败、gold 补丁应通过
python scripts/infer.py --precheck \
  --dataset /Users/liangjianfeng/Data/Router/SWE-neo-data --split train \
  --output-dir run/swe-precheck

# 仅检查数据/参考补丁字段，不启动 Docker；结果为 static_checked，不代表环境可用
python scripts/infer.py --precheck --dry-run \
  --dataset /Users/liangjianfeng/Data/Router/SWE-neo-data --split train \
  --output-dir run/swe-static-check

# 使用已保存 prediction.patch 重新评测，不重新推理
python scripts/infer.py \
  --eval-only run_0929/swe-neo-deepseek-flash-0929 \
  --output-dir run_0929/swe-reevaluated

# 仅重新分类已有日志：不启动 Docker、不下载、不调用模型
python scripts/infer.py \
  --reparse-only run_0929/swe-neo-deepseek-flash-0929 \
  --output-dir run_0929/swe-log-review
```

- `--precheck` 输出 `precheck.json`，逐题保留 empty/gold 结果。只有 empty=`failed`、
  gold=`passed` 才标记 `ready`。不自动删除或筛除原数据。
- `--resume`：在原推理命令末尾追加此参数，保留原 output-dir、dataset、models、
  split、预算和采样参数。已完成的 job 跳过；未完成路径重放已保存动作后继续，
  重放本身不调用模型。随机状态和唯一调用账本会恢复；旧工作目录保留，使用新的 replay 目录。
  已有失败任务不会自动付费重跑。中断时若工具动作是否执行无法确认，保留为
  `interrupted_call`；重放输出不一致仍标记 `replay_failed`，不会强行制造有效标签。
- `--resume-partial`：替代 `--resume` 单独使用，保留原推理参数和输出目录。
  支持数据集仅新增任务（例如从 50 条扩容到 100 条）：按 `task_id` 验证所有旧任务
  仍存在且描述、已有内容哈希未变；禁止删除、替换旧任务或使用重复 ID。
  旧 complete 跳过，旧 partial 重跑，旧未开始任务和新增任务按当前数据集顺序运行。
  扩容后的清单原子写入 `plan.json`，`task_extensions` 保留扩容记录，供后续续跑和重评使用。
  `summary.json` / `SUMMARY.md` 更新总任务数和运行进度，费用包含历史尝试；
  run-mode 每轮包含新增任务，budget-mode 仍受原累计预算约束。
  允许调整 `--step-limit`，例如 `--resume-partial --step-limit 500`；新尝试使用新上限，
  旧结果及其配置、费用保留。步数上限改变时，尚在 `running` 的任务也从头重试。
  已是 `complete` 的任务仍跳过，包括达到旧步数上限后已正常评测的任务。
  在普通续跑基础上，把已有 `partial` job 从干净环境、第 1 步重新推理，每次启动每题最多重试一次。
  `complete` job 跳过，包括已经正常评分但模型未通过的任务。KSR 会重跑该 job 的整棵树。
  `running` job 仍按普通续跑处理；本次才转成 `partial` 的任务留到下次启动重试。
  原节点、call、trace、workspace 保留，旧节点/trace 标记 `superseded=true`，job 的
  `previous_attempts` 保存历史；新尝试使用新节点 ID。通过率、samples 和 step 导出只使用当前尝试，
  总费用及预算累计包含所有旧、新尝试。未知费用标记仍保留；默认可继续，使用 `--stop-on-unknown-cost` 时仍要求先核对未知费用。

  例如重试 0929 实验里的 partial，并继续未开始的任务：

  ```bash
  python scripts/infer.py \
    --dataset /Users/liangjianfeng/Data/Router/SWE-neo-data \
    --split train --mode single-model --models deepseek-flash \
    --stop-mode run-mode --runs 1 \
    --output-dir run_0929/swe-neo-deepseek-flash-0929 \
    --resume-partial
  ```

- `--eval-only` 输出新的评测日志、`summary.json`、`SUMMARY.md`，并从旧 trace 生成
  更新标签的 `traces/*.traj.json` 和 `steps.jsonl`；消息、动作和已知费用沿用原记录。
  找不到已导出的 patch 的任务保留为未评分。原目录不改动。
- `--reparse-only` 只生成分类审计报告；不能证明补丁冲突等问题已经运行修复。
  需要重新执行评测的条目保留 error，不凭猜测算通过。

原生 SWE 的评分条件为全部 FAIL_TO_PASS 和 PASS_TO_PASS 通过。
额外通过的测试不导致失败；官方测试路径由评测器管理，模型对这些路径的补丁被排除，
再应用官方测试。Dockerfile 的 ENV 由 Docker 原生继承，不转换成 export。

缺少本地实现、明确的编译或断言失败计为任务 failed；缺依赖、超时、未知零测试、
缺少必测结果且没有明确失败证据计为 error（空标签）。解析规则保守，不能替代
empty/gold 环境预检。初始化失败记录为 `initialization_error`，与重放不一致分开。

## Router 模式

支持 `--mode router --router-profile random-router` 和 `--router-profile mtrouter`，
用 `--models` 覆盖 profile 模型池；本地 MTRouter 用 `--router-checkpoint` 指定权重。
配置、完整命令、计费及续跑约束见 [ROUTER_INFERENCE.md](ROUTER_INFERENCE.md)。

## 工具时间与长上下文

`scripts/infer.py` 支持 `--max-retries 2 --retry-wait 5`：失败后最多额外重试 2 次，
每次重试前固定等待 5 秒。这也是默认行为；`--max-retries 0` 禁用重试，
`--retry-wait 0` 禁用重试等待。参数作用于环境初始化／历史重放及 API 临时错误
（例如连接故障、429、503），不自动重跑模型答错的任务或完整 partial trace。
认证等不可重试错误仍直接记录失败。已有 `--replay-attempts N` 保留，代表包含首次的
N 次尝试，不能和 `--max-retries` 同时使用；该入口也将 API 重试上限设为 N−1。
`--resume` / `--resume-partial` 允许调整重试次数、重试等待及 trace 间隔；
实际生效的恢复参数写入 `events.jsonl`，`plan.json` 中的原始推理设置保留；
扩容时仅更新任务清单并追加扩容记录。采样和预算设置的一致性检查不变。

每条完成并获得有效评测结果的 trace，在保存状态和关闭环境后固定等待 **3 秒**，
再继续后续推理；KSR 每个完整叶分支也适用。模型未通过但正常评分的 trace 同样等待；
续跑直接跳过的已完成任务不会重复等待。这个间隔独立于重试等待时间。

SWE shell 默认 `--tool-timeout 300` 秒，模型可申请更长时间；独立上限
`--max-tool-timeout 3600` 秒。原生最终评测仍使用 `--eval-timeout`。

子进程关闭标准输入，Git、pip 和 Corepack 使用非交互设置：确认提示直接收到 EOF，
不自动回答 yes；仍然阻塞的命令到上述时限后终止。宿主机超时或取消会清理整个进程组；
Docker shell 超时会删除当前任务容器及其进程，结束当前路径并记录环境错误，
保留已产生的 trace 和费用，由采集器继续处理其他任务。

SWE 源码快照来自当前选定镜像（包括 `SWE_NEO_PREPARED_IMAGES` 指定的已验证镜像），
按评测器的 base_commit 重置和清理后提取，不再从 GitHub 全量初始化子模块。
镜像中已有的子模块源码会保留，未初始化的可选子模块保持缺失。
缓存按镜像不可变 ID 和 base_commit 区分，旧缓存不删除也不复用。
隐藏测试仍在建立供 Agent 使用的单提交 Git 基线前移除，原始及子模块 Git 历史一并丢弃。
确实缺少的必要组件应在环境准备阶段定向补齐，再通过禁网 gold/empty 验证后用于推理。

OpenAI-compatible profile 可增加 `context_window`（整数、必须大于 `max_tokens`），
填写所用服务实际允许的窗口。未配置时保留旧行为，不猜测模型别名对应的窗口。
已配置时，按 UTF-8 字节加消息开销做保守估算，预留输出额度，整组移除最老的
assistant/tool 消息，保留 system、任务和最近交互；完整历史仍留在 trace 中。
`calls/*.json` 中的 `context_management` 记录删减数量。任务本身和最近交互仍装不下时，
在请求前以已知零费用的 `provider_error` 结束路径。这不是模型专用 tokenizer。

OpenAI-compatible 的 `reasoning_content` 保留在 trace，回传给同一 endpoint/model；
切换服务时不回传另一服务的推理内容。非法工具参数保留原文与 usage，工具返回纠错信息，
不执行该调用。这里不宣称支持 Anthropic/Gemini 的所有思考签名协议。

### Context compression and live profile costs

See [CONTEXT_AND_COST.md](CONTEXT_AND_COST.md) for official per-model windows,
local lossy compaction, retry cost uncertainty, `profile_costs` live reporting,
and migration of existing output directories with `--resume-partial`.

### 采集 MTRouter 逐 step 训练标签

使用 `scripts/collect_steps.py`，支持复用 RandomRouter 配置的随机采样以及
KSR-brunch-1 双向递归分枝；详见 [STEP_COLLECTION.md](STEP_COLLECTION.md)。
