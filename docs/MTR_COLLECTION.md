# MTRouter 数据读取与采样

> 入口迁移：本文保留历史实验说明。批量推理使用 [scripts/infer.py 的新文档](INFERENCE.md)；旧实验包装脚本已删除。逐步 judge 仍可使用 `scripts/judge_trace_batch.py`。

采集入口支持 ScienceWorld 和本地 HLE。固定模型与逐 step 随机模型共用采集循环，
环境交互和终局评测由各自适配器处理；模型调用复用 mixed-pilot 的 provider，不依赖
MTRouter 的 Router checkpoint、LiteLLM、PyTorch、Transformers 或 vLLM。

## 已接入的范围

| 功能 | 状态 |
|---|---|
| ScienceWorld 原生任务枚举、train/dev/test variation 读取 | 已接入 |
| 固定模型完整 Trace、逐 API-step 均匀随机模型采样 | 已接入 |
| 显式模型池、固定随机种子、重复运行、断点跳过 | 已接入 |
| 完整 history、历史模型选择、propensity、终局分数、逐步费用 | 已接入 |
| 原生 JSON/JSONL task 清单及 mixed-pilot manifest 读取 | 已接入 |
| 本地 HLE manifest 读取、原生工具调用、终局答案评分 | 已接入 |
| HLE shell/Python、文件工具与独立 LLM judge | 已接入；judge 显式指定 |
| HLE 搜索/浏览工具 | 未接入 |
| Hugging Face HLE 远程枚举、过滤图片题 | 保留只读入口 |
| AppWorld、learned router、SFT/RL 训练 | 未迁入 |
| 从中间状态恢复、单点或连续替代采样 | 未实现；当前采集完整 Trace |

上游固定版本：`ZhangYiqun018/MTRouter@ee9e4573a128c3e54aa8c83800318bfb783c4dfe`。
提取了任务读取、ScienceWorld 环境适配器、正则动作解析器和完整 ScienceWorld
提示模板；采集循环及策略依据上游 Collector、FlexibleAgent、BaselineMode、
RouletteMode / PropensityRouletteRouter 精简适配。

原始文件路径、SHA256 和本地修改清单见
[`PROVENANCE.json`](../src/mixed_pilot_harness/collection/upstream/PROVENANCE.json)，
MIT 许可证原文见 [`LICENSE.md`](../src/mixed_pilot_harness/collection/upstream/LICENSE.md)。

## opensquilla 环境依赖

2026-09-16 实际检查的环境为
`/Users/liangjianfeng/miniconda3/envs/opensquilla`，Python 3.12.13。

| 依赖 | 当前状态 | 用途 |
|---|---|---|
| PyYAML 6.0.3 | 已安装 | 读取提取的 Agent 配置 |
| Jinja2 3.1.6 | 已安装 | 渲染原始提示模板 |
| scienceworld 1.2.3 | 缺少 | 本地模拟器及随包 JAR |
| py4j | 缺少，安装 scienceworld 时自动带入 | Python–Java 通信 |
| Java runtime | 缺少，`/usr/bin/java` 只是不可用的系统入口 | 执行模拟器 |
| datasets | 缺少，仅 HLE 读取需要 | 读取 Hugging Face 数据 |

在仓库根目录执行：

```bash
conda install -n opensquilla -c conda-forge openjdk=17
conda run -n opensquilla python -m pip install -e '.[scienceworld]'
conda run -n opensquilla python scripts/collect_data.py --doctor
```

也可先 `conda activate opensquilla`，随后直接运行 `python`。这里选择 PyPI 已发布的
`scienceworld==1.2.3`，固定版本避免实验中途换环境。上游 GitHub 的 1.3.0 涉及
模拟器确定性变化，不应在已有采样过程中直接替换。原生环境要求 Java 8+，这里
建议用 Conda 管理的 Java 17。不需要安装 Scala 或 sbt 来重新编译 JAR。

如果 `--doctor` 仍找到 `/usr/bin/java`，请检查当前环境的 `PATH`/`JAVA_HOME`
是否指向 Conda 安装的 Java。`--doctor` 会实际运行 `java -version`，而非仅检查文件存在。

可选的 HLE 依赖：

```bash
conda run -n opensquilla python -m pip install -e '.[hle]'
```

2026-09-17 已在 `opensquilla` 安装 ScienceWorld 1.2.3、Py4J 0.10.9.9 和 OpenJDK 17.0.18，
并验证本地 ScienceWorld 任务加载。上表保留 09-16 初次检查记录。若只运行本地 HLE，
不需要 ScienceWorld、Java、datasets 或额外 Python 包。

## 先做不花 API 费用的检查

下面的 dry-run 从本地示例 JSON 读取任务，不需要 Java、ScienceWorld 或 API key：

```bash
conda run -n opensquilla python scripts/collect_data.py \
  --tasks-file configs/scienceworld.tasks.example.json --split all \
  --config configs/collection.providers.json \
  --model-pool qwen-flash-2025-07-28 deepseek-flash \
  --mode roulette --runs 2 --dry-run
```

预期为 2 个 task、4 条 Trace。示例 variation 的官方 split 未验证，因此故意标为
`all`，仅用于检查输入格式。正式实验应按下面的方法从已安装环境生成清单。

安装依赖后，读取并冻结 train/dev/test task（仍不调用模型）：

```bash
python scripts/collect_data.py --list-tasks --split train --limit 30 \
  --save-tasks run/collection/tasks.train.json
python scripts/collect_data.py --list-tasks --split dev --limit 10 \
  --save-tasks run/collection/tasks.dev.json
python scripts/collect_data.py --list-tasks --split test --limit 10 \
  --save-tasks run/collection/tasks.test.json
```

`--limit` 统计具体 task 实例，每个实例是 `(task_name, variation_idx)`；选择器按
任务类型轮流取样，并用 `--seed` 固定顺序，避免全取字典序最前面的任务。
`--task-filter 'boil*'` 可进一步过滤。`--save-tasks` 不会覆盖已有清单。
原生 split 与 MTRouter 论文自定义 split 不同；本实现不自动随机拆分 step 样本。

## 配置 API 与费用

[`configs/collection.providers.json`](../configs/collection.providers.json) 是单独的
低成本配置，不修改原 `harness.json` 和 `router.toml` 的默认模型。也可以通过
`--config harness.json` 使用已有 profile，但必须先替换 Qwen URL 中的
`{WorkspaceId}` 占位符，配置合适的生成参数，并为选中的 profile 提供价格。

示例包含北京地域 Qwen 快照、DeepSeek Flash 和智谱 GLM-5；API key 分别从环境变量
`DASHSCOPE_API_KEY`、`DEEPSEEK_API_KEY`、`ZHIPU_API_KEY` 读取，文件中不存 key。
三个模型均显式关闭 thinking，输出上限 1024 token；可在正式采集前调整。

**模型池变化：** DeepSeek [2026-09-10 官方公告](https://api-docs.deepseek.com/zh-cn/news/news260910/)
说明，9 月 14 日 12:00 起 `deepseek-v4-pro` 的请求暂时转到 V4.1 Flash。因此示例不
把它作为独立的第三个模型。采样支持任意数量的 profile；GLM-5 的 provider 和价格配置已补充。
增加其他模型时同样需要补充两份配置。每个 step 保存 requested/returned model，但服务端
若继续返回旧别名，仅凭这个字段无法证明实际模型身份。

[`configs/collection.prices.json`](../configs/collection.prices.json) 使用 **CNY**，包含
2026-09-16 核对的公开标价：Qwen 北京快照按输入长度分档，DeepSeek 按北京时间
工作日高峰/空闲时段计价，并区分缓存命中。费用是返回 token 用量乘冻结价格得到的
估算，不是供应商账单；免费额度、促销、汇率和失败请求计费需要单独对账。
跨高峰边界的请求使用本地请求开始时间估算。原始 usage 一并保留，可重新计费。

`--cost-limit` 和 `--total-cost-limit` 单位均为价格表币种。前者在下一次请求前检查，
可能超出最多一次请求的费用；后者在下一条 Trace 前检查，可能超出一条 Trace。
它们是停止阈值，不是供应商侧硬限额。provider 内部的传输重试仍属于同一个路由
step，未返回 usage 的请求无法精确核算；费用缺失或 API 失败时停止采集并标记不完整，
不把未知费用当成 0 成本训练标签。

## 固定模型与随机模型采样

先用一个 task 跑通完整链路，再扩展任务数。以下命令会调用付费 API：

```bash
python scripts/collect_data.py \
  --tasks-file run/collection/tasks.train.json --split train --limit 1 \
  --config configs/collection.providers.json \
  --model-pool qwen-flash-2025-07-28 \
  --mode baseline --profile qwen-flash-2025-07-28 \
  --runs 1 --step-limit 20 --cost-limit 0.1 --total-cost-limit 1 \
  --output-dir run/collection/smoke
```

30 task × 2 repeats 的一个固定模型组：

```bash
python scripts/collect_data.py \
  --tasks-file run/collection/tasks.train.json --split train \
  --config configs/collection.providers.json \
  --model-pool qwen-flash-2025-07-28 \
  --mode baseline --profile qwen-flash-2025-07-28 \
  --runs 2 --step-limit 50 --cost-limit 0.5 --total-cost-limit 10 \
  --output-dir run/collection/fixed-qwen
```

换成 DeepSeek 固定组时同时替换 `--model-pool`、`--profile` 和输出目录。
逐 step 随机模型组：

```bash
python scripts/collect_data.py \
  --tasks-file run/collection/tasks.train.json --split train \
  --config configs/collection.providers.json \
  --model-pool qwen-flash-2025-07-28 deepseek-flash \
  --mode roulette --runs 2 --step-limit 50 \
  --cost-limit 0.5 --total-cost-limit 10 \
  --output-dir run/collection/random
```

本实现严格使用 `--model-pool` 指定的 profile，不会读取上游默认六模型池或降级到
昂贵模型。固定组 propensity 为 1，随机组为 `1 / 模型数`。同一次 provider 的
内部重试不会重新选择模型。采集目前串行执行，便于低成本调试和避免 JVM 状态共享。

## 三模型全量基线脚本

两个独立入口分别执行 ScienceWorld 或 HLE 的三个模型组，组内串行运行：

- `scripts/collect_baseline_scienceworld.sh`：174 条 Trace。
- `scripts/collect_baseline_hle.sh`：180 条 Trace，不要求安装 ScienceWorld/Java。
- 原 `scripts/collect_baselines.sh` 保留为六组联合入口。

三个入口共用 `scripts/_collect_baseline.sh`，参数一致。

| 模型 | ScienceWorld | HLE |
|---|---|---|
| qwen-flash-2025-07-28 | 29 task × 2 = 58 Trace | 30 task × 2 = 60 Trace |
| deepseek-flash | 58 Trace | 60 Trace |
| glm-5 | 58 Trace | 60 Trace |

合计 354 条 Agent Trace。所有 HLE 组默认统一使用 deepseek-flash 做终局 judge；
judge 调用单独计费，不计入这 354 条 Agent Trace 或 Router step。

```bash
conda activate opensquilla
# 先检查计划，不需要 API key，也不会调用模型：
bash scripts/collect_baseline_scienceworld.sh --dry-run
bash scripts/collect_baseline_hle.sh --dry-run

# 在当前终端配置 DASHSCOPE_API_KEY、DEEPSEEK_API_KEY、ZHIPU_API_KEY 后运行：
bash scripts/collect_baseline_scienceworld.sh
bash scripts/collect_baseline_hle.sh
```

默认数据目录为 `~/Data/scienceworld_task` 和 `~/Data/hle_tasks`。脚本验证实验 train
划分分别恰好 29/30 个任务，任意一个配置或凭据缺失时会在付费调用前停止。dry-run
会保存任务清单和计划文件。独立入口分别输出到 `run/collection/baselines-scienceworld-3models-2runs/` 和
`run/collection/baselines-hle-3models-2runs/`；联合入口仍使用
`run/collection/baselines-3models-2runs/`。目录结构（仅包含所选数据集）：

```text
inputs/                         固定任务清单、配置指纹和实验设置
plans/                          所选组的 dry-run 计划及运行时环境检查
logs/<benchmark>-<model>.log     各组控制台日志
scienceworld/<model>/           完整 Trace、per-step 数据、组汇总
hle/<model>/                    完整 Trace、per-step 数据、组汇总和工作目录
batch-summary.json              所选组全部完成后的汇总（含费用与通过率）
```

默认每条 ScienceWorld 最多 50 API step，HLE 最多 8 step；每条 Agent Trace 费用
停止阈值 0.5 元，每组采集费用停止阈值 40 元（包含 judge，六组合计阈值 240 元）。
这些是估算费用的停止阈值，不是费用预测或硬上限。达到预算或遇到错误不会被报告为
全量完成。`GROUP_COST_LIMIT` 可在续跑时提高；其他实验设置或任务发生变化时，应使用
新的输出目录。相同输入重跑会跳过有效的既有 Trace，错误或中断记录需先人工检查。

```bash
# 用新的目录开始独立实验；详细参数见 --help。
bash scripts/collect_baselines.sh --output-dir run/collection/baselines-experiment-2
```

GLM-5 走智谱按量 API `https://open.bigmodel.cn/api/paas/v4`，使用 ZHIPU_API_KEY。
价格表按[智谱官方定价](https://bigmodel.cn/pricing)记录 32 千输入 tokens 前后两档，
缓存命中费用也分档；配置在 2026-09-18 添加，Qwen/DeepSeek 仍沿用 09-16 的冻结价格。
关闭 thinking 的参数参照 [GLM-5 官方文档](https://docs.z.ai/guides/llm/glm-5)。

### 失败与续跑

- **任务失败不等于采集异常**：答错、得分不足、正常达到 step/单条费用限制，若终局评测成功且费用完整，会保存为有效训练样本并继续下一条。格式错误通常作为观测反馈给 Agent，受步数限制约束。
- **请求重试**：网络故障、429 和可重试服务端错误最多尝试 3 次，间隔 0.5、1 秒；鉴权等不可重试错误直接失败。
- **异常停止**：API 重试耗尽、环境异常、judge 失败或费用不完整会保存异常 Trace，停止当前组；外层脚本随即非零退出，后续模型不运行。达到组费用阈值而未完成全部任务也会停止。
- **断点复用**：重新运行相同命令、相同输出目录，会跳过已有有效 Trace（包括模型答错的 Trace），累计其费用。不会从 Trace 中间某个 step 恢复。
- **错误不会自动重跑**：已存异常 Trace 会再次阻止继续；保留为 Running 的中断记录会报错。请先检查日志与 Trace；需要重跑时将异常文件移到组目录以外归档，切勿直接删除。归档后旧尝试的费用不会自动计入新组汇总，需另行累计。
- **进程锁**：同一个组输出目录禁止并发写入。强制终止可能遗留 `.collection.lock`，确认原进程已退出后才可移除。

独立脚本默认使用新目录，不会自动继承旧联合实验的进度；不要因此误启动重复付费实验。联合实验需续跑时仍使用原入口和原输出目录。

## 输入现有 mixed-pilot 数据

`--tasks-file` 支持 JSON 数组、`{"tasks": [...]}`、JSONL、单个 manifest、任务
目录、以及包含 `tasks/` 的数据根目录。

在现有 manifest 的常规字段之外增加：

```json
{
  "id": "scienceworld-boil-1",
  "source": {"dataset": "ScienceWorld", "split": "train", "revision": "scienceworld==1.2.3"},
  "environment": {
    "benchmark": "scienceworld",
    "task_name": "boil",
    "variation_idx": 1,
    "simplification_str": ""
  }
}
```

这里只展示新增/相关字段；完整 manifest 仍应符合仓库原 schema。variation 的 split
以原生枚举结果为准，读取手写清单时不会启动 JVM 来验证其声明。普通 JSON task
记录的字段格式见 `configs/scienceworld.tasks.example.json`。

```bash
python scripts/collect_data.py --list-tasks --split train \
  --tasks-file /Users/liangjianfeng/Data/mixed-pilot-data
```

ScienceWorld 采集不会读取 `reference/` 的答案，也不使用通用 rubric 当作 ScienceWorld 原生
得分；Agent 任务描述来自加载的环境，`prompt.md` 不参与这个独立执行入口。
向统一 task 目录添加任务后，如需使用旧 `scripts/validate.py`，还要维护原 catalog
和 checksum 清单。本次未修改外部数据目录。

## 输出与标签语义

输出目录包含：

- `*.traj.json`：完整 Trace、每次调用前 history、历史模型选择、实际选择概率、
  返回文本、环境 observation 或 tool_calls/tool_results、原始 token usage、估算 step cost、终局分数。
- `*.steps.jsonl`：逐 step 的 `(state, action, terminal_score, terminal_success,
  remaining_cost, propensity)` 单次观测，`remaining_cost` 包含当前 step。
- `*.summary.json`：通过率、平均原生分数、已知费用、有效/计划 Trace 数。

ScienceWorld 的 `terminal_success` 定义为原生分数达到 100；模型自行输出 `task completed` 不代表
成功。环境报告结束会终止 Trace；这一行为与上游仅凭提交标记结束的循环有所区别。
达到 step/cost 限制仍保留原生最终分数，可作为有预算约束的样本；API、环境或费用
记录失败的轨迹保存诊断但不导出监督标签，summary 也单独报告有效数量。

新采集格式为 `collector_schema=2.0`：不再写入 `environment_score` 或
`environment_done`，只保留 `info.final_score/success/exit_status`。历史 Trace 不会被改写。
格式升级改变采集签名，请使用新输出目录，避免误以为旧实验会按原签名跳过。

HLE 的终局评分为 0/100，方法与候选答案记录在 `info.evaluation`，评测失败为未知
（而不是任务失败），不导出标签。`info.model_stats.instance_cost` 与样本的
`remaining_cost` 只统计 Agent 推理费用；`info.evaluation.cost` 单独统计 judge 费用。
`info.collection_cost`、summary 的 `known_cost` 和 `--total-cost-limit` 统计两者之和；
summary 也分别输出 `agent_cost/evaluation_cost`。`--cost-limit` 仅约束 Agent 推理。
HLE 的 `workspaces/<trace_id>/` 保留各次执行的公共工作目录，便于人工检查。

这些是 **single_rollout** 标签，还不是对同一 `(s_t, a_t)` 多次采样得到的均值。
`state.previous_actions` 保留完整历史，`continuation_policy` 标明固定或随机后续
策略；不得只按 task ID + step 编号把不同 history 的结果平均。

输出保留了 MTRouter 的 `messages/info` 轨迹形状，并显式填充 `usage.cost`，但当前
费用币种是 CNY。若后续直接接其训练器，需要检查币种、模型名映射、归一化、原生
分数与 AEC 标签选择，不能沿用上游默认美元预算而不转换。

相同配置重复执行会跳过已保存 Trace；提高 `--runs` 可只补新增重复。模型池、
价格、提示、环境版本或每条 Trace 的限制变化会生成新签名。运行中断/失败记录
不会被静默覆盖；确认原因后移至归档目录再重试。并行进程不能写同一个输出目录。
`--total-cost-limit` 也计入本次任务清单中已有匹配 Trace 的费用。

## 本地 HLE 采集

无需修改 `/Users/liangjianfeng/Data/hle_tasks`：读取其中的 manifest、prompt.md 和公共
workspace，终局 evaluator 单独读取 reference/gold.json。`routing_metadata.router_split`
用作实验 split；`source.split` 保存在 `source_split`。当前 30 条记录为实验 train、来源 test。
使用 `--task-filter 'Math'` 等按 hle_category 筛选，`--limit` 按类别平衡选样。
加载本地 HLE 不需要 datasets 下载或 JVM。

先检查计划（不调用 API）：

```bash
conda activate opensquilla
python scripts/collect_data.py \
  --benchmark hle --tasks-file /Users/liangjianfeng/Data/hle_tasks \
  --split train --limit 3 \
  --config configs/collection.providers.json --pricing configs/collection.prices.json \
  --model-pool deepseek-flash --mode baseline --profile deepseek-flash \
  --judge-profile deepseek-flash \
  --runs 1 --step-limit 8 --cost-limit 0.5 --total-cost-limit 2 \
  --output-dir run/collection/hle-deepseek --dry-run
```

配置 DEEPSEEK_API_KEY 后，删除最后的 `--dry-run` 即可真实采样。当前 provider 示例
输出上限 1024 token；如长答案频繁截断，请在采集前调整配置。若需要随机模型采样，
指定多个 `--model-pool` 并改为 `--mode roulette`，judge 保持固定，避免评价标准随路由变化。

HLE 模型使用原生 tool calling，按 manifest.agent.tools 启用现有文件和 shell 工具。
一个模型响应可以请求多个工具，但仍然是一个 API step。没有工具调用、且输出包含
完整 `Answer:` 字段时才提交终局答案；输出截断或缺少答案会提示重试，达到步数/费用
上限仍未提交答案则记为失败。模型可能直接答题，采样器不会强制制造多步轨迹。

shell 可调用当前环境的 Python，超时由 `--tool-timeout` 控制（默认 30 秒）。仅复制
公共 workspace；参考答案不加入 Agent 的 prompt、history 或 workspace。shell 子进程
只继承必要的 PATH/临时目录等配置，不继承 API key。沿用项目的本地 process 工具，
其工作目录隔离不是 OS 沙箱；运行不可信任务需另配容器。当前没有专用搜索/浏览工具。

默认 `--hle-evaluator judge`：规范化空白后完全相同的答案直接判通过，其余答案调用
显式 `--judge-profile` 做语义比较。评测调用及其费用位于 info.evaluation，不混入
Router 的 steps/messages。该 evaluator 是本项目的答案比较实现，不声称复现官方
HLE 指定 judge 配置。为公平对比，应固定 judge 模型与参数。

`--hle-evaluator exact` 可省略 judge-profile，完全不调用 judge，只做保守文本匹配。
这种模式可能把数学等价或同义答案判为错误，适合检查链路，不建议当作正式 HLE 准确率。
评测输出格式错误或超时会保留记录并停止本批采集，不会悄悄标成失败样本。

完整 Trace → per-step 导出仍独立由 `step_samples()` 完成；两种 benchmark 使用同一
模型选择策略、费用核算与导出机制。新增 benchmark 可继续添加 episode 适配器，
提供 start/advance/evaluate/close，不必复制采集循环。

## HLE 远程只读入口

```bash
python scripts/collect_data.py --benchmark hle --list-tasks --split test --limit 10
```

首次可能下载 Hugging Face 数据，不调用 LLM。CLI 输出会去掉答案与原始带标签
字段。HLE 公开数据是测试集，本入口不将其自动重新命名为训练集；如要复现其
采集流程，请使用上面的本地 manifest 入口；远程枚举结果不是可直接执行的本地任务清单。

## 验证

```bash
PYTHONPATH=src conda run -n opensquilla python -m pytest -q tests
```

测试覆盖模型池与 propensity、官方 split 读取、manifest 导入、费用/缓存/峰谷
估算、格式错误恢复、费用停止、API 失败、完整 Trace 到 step 数据的导出及断点
续采，以及 HLE 工具协议、终局答案、judge 费用/异常、来源与实验 split 的区分。
测试使用可控 provider 和本地工具，不调用付费 API。
