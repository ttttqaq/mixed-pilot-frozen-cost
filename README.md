# mixed-pilot

`mixed-pilot` 是一个 7 条数据的最小混合 Agent 测试集：SWE-bench Verified、GAIA、BrowseComp、OSWorld-Verified、JobBench、Mind2Web 和 WebArena-Verified 各 1 条。选样偏向短任务和可人工核对结果。

仓库现在还包含一个精简的 OpenSquilla 式执行 harness，提供 `chat`、`agent` 和
`code-task` 三种模式。`chat` 用于单轮文本问答；Agent 模式支持单 Turn 多 Step 工具循环，以及
一次性的代码环境准备、评测和评测环境回收；详见 [`docs/HARNESS.md`](docs/HARNESS.md)。
`agent --restore-workspace --task-file <text>` 可在一次性 workspace 副本中执行，
结束后恢复环境，并同时保存带 Task/Turn/Leg/Iteration 父子关系的 JSONL 与 SSE 轨迹。
`code-task` 通过 TaskRuntime 隔离 agent、evaluator 和 rubric scorer workspace；任务
含 `rubrics.json` 时会在 Agent 完成后生成分项 `rubric_result.json`。
`code-task --task` 接受任意 `category`，统一执行 Agent → eval → score。
它自动读取任务的 `workspace/`、`prompt.md`、`agent_environment`、`rubrics.json`
和可选的 `reference/`；eval 命令需配置在 `harness.json` 的 `tasks.<task-id>` 中，
或通过 `--eval-command` 指定。例如：

```bash
mixed-pilot code-task --config harness.json --profile gpt-5.6 \
  --task jobbench-bookkeeping-task1 \
  --eval-command 'test -s deliverables/bank_reconciliation.md && test -s deliverables/controller_memo.md' \
  --attempt-root run/code-task
```

任务仍需提供可运行的 workspace 和相应环境；类别字段不负责安装浏览器或 GUI 环境。
存在 rubric 时，eval 与 rubric 都通过才返回退出码 `0`。

通用批量推理统一使用 `python scripts/infer.py`，支持 SWE-neo、ScienceWorld、ALFWorld、HLE，
提供 single-model / random / KSR 模式，按轮次或预算停止，输出完整 trace、step 标签、通过率和费用。
参数与示例见 [通用推理说明](docs/INFERENCE.md)。先加 `--dry-run` 验证计划。

迁移 SWE-neo 到新服务器时，可直接从数据目录生成公共镜像拉取计划，无需复制本机 Docker
存储或旧绝对路径；使用方法见 [SWE 服务器恢复流程](docs/SWE_SERVER_SETUP.md)。

推理日志的 profile 对比图使用 `python3 visualized/generate.py <日志目录> [更多目录] --open`。
生成 accuracy × 平均 cost / tokens / steps 的离线交互散点图，并导出 JSON/CSV；
使用与统计口径见 [visualized/README.md](visualized/README.md)。

VirtualWorld-1000 包含 ScienceWorld、ALFWorld 各 500 条原生交互任务，沿用统一任务
目录，并通过同一批量推理入口运行；抽样、原生验证与使用方式见
[VirtualWorld-1000 说明](docs/VIRTUALWORLD.md)。

`mixed-pilot per-step-judge --profile <judge> --trace <unique.traj.json> --step <1-based>`
可对采集轨迹的一个 API-step 评分（0~100），复用单轮 chat 调用。支持 prompt 预览、
可选标准答案和完整评分记录；用法及评分定义见 [`docs/PER_STEP_JUDGE.md`](docs/PER_STEP_JUDGE.md)。

## 统一目录

任务数据默认位于 `~/Data/mixed-pilot-data/tasks`，不通过符号链接放入本代码仓库，
以免编辑器自动发现数据中的嵌套 Git 仓库。可用 `MIXED_PILOT_DATA_ROOT` 或相关
命令的 `--data-root` / `--dataset-root` 覆盖数据根目录。每条任务都使用相同布局：

```text
tasks/<task-id>/
├── manifest.json   # 来源、workspace 模式、工具、输出契约、评测级别
├── prompt.md       # 统一四段式 Agent prompt
├── rubrics.json    # 统一 100 分制 rubric
├── workspace/      # Agent 可见；原始 repo、附件、HTML 或环境描述
└── reference/      # grader-only；gold、官方 evaluator 或原始带标签记录
```

`reference/` 绝不能挂载给 Agent。原生 benchmark 字段被保留在这里，Agent 可见 workspace 中会去掉答案和 evaluator。

## 快速使用

单轮 Chat，直接使用 `harness.json` 中的 profile：

```bash
mixed-pilot chat --config harness.json --profile gpt-5.6 --message '你好，请介绍一下自己'

# 也可以从 stdin 或文本文件读取问题
echo '解释一下模型路由' | mixed-pilot chat --profile claude-sonnet-4-6
mixed-pilot chat --profile gpt-5.6 --task-file prompt.txt --result chat-result.json
```

`chat` 每次发送一条 user message，只做一次逻辑模型调用，不提供工具、不执行
ReAct 循环，也不需要 workspace。默认 system prompt 为 `You are a helpful assistant.`，
可用 `--system-prompt` 覆盖（传空字符串可省略）；回复文本输出到 stdout。
`--result` 可保存回复、模型、stop reason 和 usage。省略 `--profile` 时使用
`default_profile`；选择 `router` 时会先路由一次。Provider 原有的网络重试保持生效。

校验集合：

```bash
export MIXED_PILOT_DATA_ROOT=/Users/liangjianfeng/Data/mixed-pilot-data
python3 scripts/validate.py
```

只准备一个隔离 attempt：

```bash
python3 scripts/run_agent.py \
  --data-root /Users/liangjianfeng/Data/mixed-pilot-data \
  --task swebench-flask-5014 --prepare-only
```

接入任意能从 stdin 读取 prompt 的 Agent CLI：

```bash
python3 scripts/run_agent.py --task gaia-paper-enrollment -- your-agent --flag
```

Runner 会复制 workspace，设置 `MIXED_PILOT_TASK_ID`、`MIXED_PILOT_TASK_DIR`、`MIXED_PILOT_WORKSPACE`、`MIXED_PILOT_OUTPUT_DIR`、`MIXED_PILOT_PROMPT`，并保存 stdout、stderr 和进程级轨迹。Agent 内部的 ReAct/tool/SSE 事件可继续追加到 attempt 的 `trace/events.jsonl`。

## 逐 API Step 的 Squilla Router

`router` 是一种组合 provider。`TurnRunner` 每次调用 `complete(messages, tools,
system)` 时，它先用当前 task 的完整 user/assistant/tool trail 做四档路由，再把原始
请求不变地交给该档位的实际 provider。目标 provider 自己的网络重试不会重新路由。

安装 V4 推理依赖：

```bash
python -m pip install -e '.[router]'
```

[`harness.json`](harness.json) 中的示例 profile：

```json
{
  "kind": "router",
  "model": "squilla-router-v4",
  "router_config": "router.toml",
  "api_key_optional": true
}
```

[`router.toml`](router.toml) 配置 OpenSquilla V4 bundle、R0-R3 到 c0-c3 的映射，
以及每档实际调用的 harness profile。档位也可以覆盖 `model`、`base_url`、
`api_key_env`、`max_tokens` 等 ProviderProfile 字段。API key 本身不写入 TOML，
只保存环境变量名。

`[decision].mode` 控制最终档位选择：`legacy`（默认）使用 OpenSquilla 完整的
turn-level 后处理，`raw` 直接使用融合概率 argmax，`api_step` 使用 raw argmax 并且
只保留可配置的 `high_risk_floor`（R2），不会应用 `repo_arch`、
`debug + long_context`、deep-conversation 等 turn-level 强制升级。示例：

```toml
[decision]
mode = "api_step"
high_risk_floor = true
```

运行时选择示例 profile：

```bash
mixed-pilot agent --config harness.json --profile router \
  --workspace /path/to/workspace --message 'inspect and fix the task'
```

当前适配直接继承 V4 bundle 内的 BGE/TF-IDF/SVD/PCA 特征、390 维 LightGBM、
1536 维 ONNX MLP 和概率融合；`legacy` 模式还继承完整后处理。无法原样继承的是
turn/session 输入语义：
本项目把 task objective 放入 current-user 通道，把最近 user/assistant/tool 轨迹投影到
history/assistant/context 通道；旧权重因此属于兼容冷启动，生产部署仍应使用逐 step
轨迹重新训练和校准。RouterProvider 预留了 `agent_trail_supplier` 接口；精简 harness
目前没有跨 agent trail 总线，所以默认只路由本 task 内的完整 trail。

## Workspace 完整性

- SWE-bench：保留 Flask 基准 commit 的完整浅克隆（含 `.git`）。
- JobBench：保留原始任务说明和两份 CSV 附件。
- Mind2Web：保留该训练样本两步的 raw/cleaned HTML；标签只在 grader reference。
- GAIA 与 BrowseComp：原任务无附件，workspace 即开放 Web；保留来源记录与答案在 reference。
- OSWorld 与 WebArena：完整 VM/网站容器体积巨大且依赖上游服务，因此保留精确 revision、原生 task config 和环境描述，不复制镜像。

## 重要边界

- GAIA 使用论文中公开展示的样例，不是对 gated validation/test split 的再分发。
- Mind2Web 使用 train 样本；未包含其禁止再分发的解压 test 文件。
- BrowseComp 原始行在本地 reference 中保存。若公开发布本目录，应先复核上游关于 benchmark contamination 的最新要求。
- `rubrics.json` 统一了表示方式，但不会把不同 benchmark 的证据强行等价：代码用隐藏测试，GUI 用状态检查，Office 用加权 rubric，事实题用精确匹配/官方 judge。

## 独立 MTRouter 训练与推理核心

`MTRouter/` 提供从上游移植的轨迹解析、embedding、单头 Q 网络训练、checkpoint 和 HTTP 评分服务。
使用独立 Python 环境安装，harness 通过 `engine = "mtrouter"` 接入，无需在现有环境安装 torch/transformers。
详见 [安装、训练与接入说明](MTRouter/README.md)。

## KSR 与随机 step 数据构造

通过 `python scripts/infer.py --mode ksr|random` 运行，见 [通用推理说明](docs/INFERENCE.md)。
旧方法细节保留在 [KSR 文档](docs/KSR_COLLECTION.md)，其中实验专用脚本已移除。

## 在线 Router 训练

`scripts/train_online_router.py` 实现独立的预算条件在线策略改进：当前 Router 访问状态、
同前缀单步 A/B 干预、短 rollout、真实结果训练的 Teacher（强模型提供辅助证据）、
选择性完整审计，以及收益/费用双头 Router 更新。
默认读取 `data/VirtualWorld-subset` 的 train/valid，训练和评估都不读取 test。
配置为 `configs/online-router.db94.yaml`，支持 `--dry-run`、`--smoke`、`--set`、
`--resume` 和新格式 checkpoint 的 `--evaluate-checkpoint`。
完整运行、方法和输出说明见 [在线训练说明](docs/ONLINE_ROUTER.md)，
字段与消融见 [配置参考](docs/ONLINE_ROUTER_CONFIG_REFERENCE.md)。

## 固定 reference Teacher 在线 Router

当前方法分支使用 `configs/online-router.reference-teacher.db94.yaml`，强模型为 NEW CLI Responses 的 gpt-6.1-terra（low），无需训练 Teacher。训练用 train 300 题，评估用 test 50 题。key 留空供填写，Teacher 请求默认间隔 15 秒。方法、费用口径和运行命令见 [运行说明](docs/REFERENCE_TEACHER_ROUTER.md)。
