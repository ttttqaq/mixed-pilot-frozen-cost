# Router 学术探索 Harness 审计

审计日期：2026-09-15。代码基线：`05b510be124c5dfbb73ef89725cef8e902e40ac5`，包含工作区当前配置。审计开始前已有 `harness.json` 修改和未跟踪的 `docs/`；本次仅新增报告与离线复现材料，没有修改执行代码、任务数据或模型配置。

## 1. 结论与适用范围

**当前是可用的逐调用 Router 工程原型；作为产出可信质量—成本比较的学术探索 harness，尚未合格。现有架构值得保留，优先补齐评测和计量。**

这里的最低门槛是：在相同可执行任务、工具、预算和裁判下替换路由策略，能完整保留成功与失败轨迹，正确计算质量与费用，并重建实验配置。探索阶段可以小样本起步，结果应明确标注探索性。

| 用途 | 当前判断 | 依据 |
|---|---|---|
| 检查 Router 是否逐 step 被调用、输出是否合法 | 合格 | Provider 边界、工具循环、路由事件和离线测试齐备 |
| 在单个代码任务上观察执行轨迹 | 基本合格 | 已有 SWE-bench 任务真实运行与分组测试结果 |
| 比较固定模型与 Router 的质量、成本 | 尚不合格 | 裁判随策略变化、费用信息丢失、超时配置不一致 |
| 支持七类 benchmark 的完整实验 | 尚不合格 | 本地仅六个任务目录；工具、环境和原生评测接入不完整 |
| 支持方案中的同状态分支干预 | 尚不合格 | 当前只有任务级新建副本，没有调用前环境快照与续跑接口 |
| 支撑论文中的稳定收益或泛化结论 | 证据不足 | 两条真实记录来自同一题，路由记录没有实际切换模型 |

训练器、RL 和生产服务能力不属于本次最低验收范围；中途快照仅在研究同状态干预时成为必要条件。

## 2. 已经做对的部分

- `TurnRunner` 统一模型调用与工具执行；`RouterProvider` 每次逻辑请求选择一个执行 provider，再传递原始 messages/tools/system。网络重试留在目标 provider 内。这是清晰、可保留的实验边界。[runner.py](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/runner.py:68)、[router/provider.py](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/router/provider.py:96)
- 调用前 messages、system、工具定义、工具输入输出和路由概率均有轨迹；Task/Turn/Leg/Iteration 有关联字段。恢复数据采集可以从这里扩展。[runtime.py](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/runtime.py:384)
- 评测与 scorer 使用独立目录，隐藏测试在 Agent 结束后应用，FAIL_TO_PASS/PASS_TO_PASS 可以单独检查；不会把隐藏评测结果反馈给 Agent 再修一轮。[code_task.py](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/code_task.py:517)
- `legacy/raw/api_step` 分离模型概率与后处理规则；README 明确承认 V4 的输入语义迁移是兼容冷启动。这种边界说明准确。

以上目录分离是工作流隔离，尚不等同于文件访问权限隔离。

## 3. 必须优先修复的发现

P1 表示会影响比较有效性或结果正确性，应在正式采集前修复。以下均区分代码事实、离线复现和历史结果。

### F1 · P1：裁判与被测策略绑定

`scorer_runner_factory` 仍调用 `_make_runner(args, config, ...)`，只改 system prompt。`args.profile`、模型与 router 配置仍来自被测 Agent。因此 A、B、Router 三个实验条件会使用不同裁判；选 router 时，裁判自身也会路由。独立 scorer workspace 没有解决测量标准随策略变化的问题。

证据：[cli.py:351](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/cli.py:351)。历史 Sonnet 运行确实为 scope 项执行了额外 scorer turn。

**验收：**提供独立、冻结的 scorer profile、参数和 prompt；同一提交在不同被测策略标签下得到同一确定性评价，或由同一固定 judge 按同一重复采样协议评价。单列 scorer 费用。

### F2 · P1：质量与实际成本尚无法同时审计

`Usage` 只有输入/输出两个整数。Anthropic adapter 丢弃缓存读写 token，OpenAI adapter 丢弃 token 明细，HTTP 层没有保存物理重试次数、请求标识与原始 usage。缺失 usage 默认成零。运行记录也没有冻结价格表、实际账单对账或 Router 推理耗时。

离线复现：模拟 Anthropic 返回普通输入 10、缓存写入 100、缓存读取 900、输出 5，经 adapter 后只剩 `10/5`。这证明返回信息会丢失；本次没有核实历史网关请求是否使用缓存，也没有假定它一定产生了缓存费用。Claude 原生 API 将缓存读写单列计量，实际网关费用还需按网关账单审计。[官方缓存文档](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)

证据：[types.py:38](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/types.py:38)、[providers.py:347](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/providers.py:347)、[providers.py:126](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/providers.py:126)。`api-routing-cost-model.html` 是参数化成本演示，未与运行日志计费接通。

**验收：**保留各 provider 原始 usage 和请求元信息，区分 unknown 与 0；记录逻辑调用与每次物理尝试；冻结计价规则，记录缓存/思考 token 的 provider 语义，避免重复计费；失败请求费用未知时明确标注。完整任务费用应可按执行、Router、工具与评测分别汇总，并记录账单对账方式。

### F3 · P1：可变 HEAD 会破坏提交与测试基准

`collect_changes` 对 Agent 当前 `HEAD` 做 diff；evaluator 克隆 Agent 仓库；`_restore_test_files` 从 evaluator 的 `HEAD` 恢复测试。Agent 可以提交 commit，所以这个 HEAD 不是冻结的 benchmark base。

**已离线复现：**在临时仓库中，Agent 将程序的回归值从 0 改成 9，同时把回归断言改成 9 并提交；evaluator 保留这个断言，应用另一个官方测试文件后评测通过，而导出的 `change.patch` 为 0 字节。普通 Agent 提交修复 commit 也会导致相对 HEAD 的补丁为空。这是 harness 通用行为；不表示历史 Flask 运行曾这样操作。

证据：[code_task.py:79](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/code_task.py:79)、[code_task.py:137](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/code_task.py:137)、[runtime.py:260](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/runtime.py:260)。

**验收：**启动时固定可信 source/base commit；从可信基准准备 evaluator；按该基准导出 Agent 的最终改动；从可信基准恢复评测拥有的测试。Agent commit 与否应得到一致的提交产物和分数。

### F4 · P1：原生评测方法退化成通用 LLM 判分，缺失测试还可能得满分

Scorer 只直接实现 `native_test` 和部分 gold patch 快捷检查。`exact_match`、`trajectory_match`、`agent_response_evaluator` 等全部进入同一 LLM scorer；GAIA 没有确定性 exact-match 实现，Mind2Web 没有独立 JSON/action matcher，WebArena 没有调用其原生 evaluator。`run-tasks.sh` 中的 `true` 仅让前置命令通过，质量实际由 rubric 决定。

另外，已声明的 FAIL_TO_PASS 组缺失时，`_native_criterion` 会回退到 broad evaluator 状态；离线输入 `evaluation.status=passed`、无 native_checks，会给该组 70/70。judge JSON 的 `NaN` 也会经当前 clamp 变成 100/100。

证据：[rubrics_scorer.py:70](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/rubrics_scorer.py:70)、[rubrics_scorer.py:152](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/rubrics_scorer.py:152)、[rubrics_scorer.py:303](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/rubrics_scorer.py:303)。

**验收：**按 method 实现独立 evaluator，明确区分原生分数、自定义 rubric、LLM 代理评分；声明的证据缺失时返回 unverified/error；拒绝非有限数值和不合规结构。可确定比较的答案/动作使用确定性规则，开放式产物使用固定 judge。

### F5 · P1：隐藏答案与运行产物仍可被 Agent 访问

Shell 只设置 cwd，仍拥有宿主进程的文件访问权限。任务 reference 虽未复制进 Agent workspace，仍在相同宿主文件系统；run 目录上层还复制了 rubrics。JobBench rubric 包含具体答案细节。`search_files` 遍历时也没有对每个文件重新检查 resolved path，会跟随指向 workspace 外的文件符号链接。

**已离线复现：**shell 读取 workspace 父目录的合成 grader marker 成功；search_files 从指向该 marker 的符号链接读取成功。只使用自建临时文件，没有读取真实私密信息。

证据：[tools.py:190](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/tools.py:190)、[tools.py:158](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/tools.py:158)、[code_task.py:444](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/code_task.py:444)。

**验收：**执行环境只能访问允许的 workspace 和工具资源，grader/reference/控制配置从权限上不可达；评测与执行分别挂载；符号链接遵守同一访问范围。先在一个受控代码域做到即可。SWE-bench 官方采用容器环境来实现可复现评测，可作为接入参考。[官方 harness 说明](https://github.com/SWE-bench/SWE-bench/blob/main/docs/reference/harness.md)

### F6 · P1：固定模型与 Router 的超时条件不一致

`_profile_and_defaults` 将全局 `provider_timeout=360` 写入选中的顶层 profile。选固定模型时生效；选 router 时，这个 timeout 没传给 router.toml 引用的子 profiles，子模型继续使用默认 120 秒。

**已按当前配置离线复现：**outer router 360 秒；c0/c1/c2/c3 均为 120 秒。相同模型直接跑和被 router 调用会有不同失败概率，污染质量比较。

证据：[cli.py:113](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/cli.py:113)、[providers.py:519](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/providers.py:519)、[router/config.py:124](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/router/config.py:124)。

**验收：**生成并保存最终生效的执行配置；同一 action 在所有策略下的 timeout、max tokens、采样参数与重试规则一致。

### F7 · P1：终止与失败产物口径不完整

只要无 tool_calls，runner 就结束，未按 `length/max_tokens/refusal` 等 stop reason 区分正常完成。达到 max_steps 或遇到 provider 错误时，code-task 直接写 `evaluation=not_run`、空 patch，未收集已经产生的工作区改动，result 也缺少累计 usage。已有成功调用 usage 可部分从 trace 找回，但没有统一任务结果。

**已离线复现：**`stop_reason=length` 返回普通完成结果；max_steps=1 时 Agent 已写入修复文件，最终 result 仍无 patch、无 usage 且不评测。无 tool_calls 不等于任务质量成功，后续 evaluator 能拦截部分错误，但当前终止分类不足以分析失败。

另外，shell/evaluator 超时仅 kill 父进程，未清理进程组。`sleep 0.8 & wait` 设置 0.05 秒，实际约 0.8 秒后返回；长期子进程可继续等待或产生后续副作用。

证据：[runner.py:89](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/runner.py:89)、[code_task.py:494](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/code_task.py:494)、[tools.py:203](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/tools.py:203)、[code_task.py:202](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/code_task.py:202)。

**验收：**冻结终止协议，区分模型完成、拒绝、截断、步数耗尽、API/工具/评测故障；所有路径均保留改动与费用。预算耗尽后是否评价当前产物预先规定，所有 attempt 都进入实验分母；超时取消整个进程组并验证没有残留子进程。

## 4. 数据、工具和复现能力

### F8 · P1（完整项目范围）：声明的任务覆盖尚未成为可执行评测覆盖

| 任务 | 本地可核查状态 | 研究口径 |
|---|---|---|
| SWE-bench Flask | 仓库、隐藏 patch、分组测试及两次历史结果存在 | 最适合作为首个完善域 |
| JobBench | CSV/任务材料存在，产物存在检查 + LLM rubric | 可做受控探索，先固定裁判 |
| Mind2Web | 离线 HTML 存在，产物存在检查 + LLM 评价动作 | 实现确定性 action matcher 后使用；标注离线任务 |
| GAIA / BrowseComp | prompt-only，manifest 声明 web_search/browser | 内置工具仅文件、shell、git；尚无标准化检索/浏览器通道 |
| WebArena | 环境描述存在，所需站点服务未在本项目接入 | 不能将目录运行结果解释为完整 WebArena 实验 |
| OSWorld | 本机默认数据根下任务目录缺失 | catalog 和运行脚本仍列出，应修正状态或补齐环境 |

Shell 可以临时调用网页工具，但缺少统一能力、网络数据版本与观测接口，会使检索/浏览能力成为额外实验变量。GUI 任务还缺少图像消息与动作执行通道。证据：[tools.py:39](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/tools.py:39)、[types.py:22](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/types.py:22)、[run-tasks.sh](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/run-tasks.sh:49)。

### F9 · P2：实验输入尚未冻结，验证入口已过期

- `scripts/validate.py` 实跑因 OSWorld manifest 缺失抛 FileNotFoundError；单独审计 SHA256SUMS 得到 **301 个匹配、6 个缺失、19 个不匹配**。这些不匹配包含迭代后的代码与 manifest，不应直接解释成数据损坏，但清单已经不能证明当前版本完整。
- validator 没有执行目录中的 JSON Schema 校验；CLI 运行也没有完整的任务 preflight。
- `request.json` 未冻结完整有效 provider/Router 配置、base commit、bundle checksum、包版本与价格版本；router.toml 使用本机绝对 bundle 路径，依赖只有最低版本约束。bundle 本机存在，但运行没有将其资产哈希写入实验记录。
- 核心使用文档 `docs/HARNESS.md` 被 gitignore；其中 `--profile anthropic` 与当前配置 profile 名不符。批处理脚本硬编码另一台机器的数据路径，标注“不用跑”的 OSWorld 命令仍会执行。
- 多数非 coding 命令绕过 `--task` 使用 `--repo`，因此不会自动读取并应用 manifest 中的环境与工具约束。

**验收：**产出可重建的 experiment manifest；分别冻结代码、数据、环境、模型配置、Router 资产、评测器与预算。记录 task family/split/repetition/policy 的关联；数据校验失败应输出完整问题列表并阻止计入正式实验。

## 5. Router 研究接口的边界

完整请求已到达 `RouterRequest`，这是可扩展的正确边界。但 V4 实际输入只取原始目标、最近最多 4 条消息、最后 assistant 和统计特征；system 与工具完整描述没有进入 V4 特征，旧 turn-level 权重也没有 step-level 校准。这是当前 baseline 的输入定义，不能把它表述成完整长历史编码。证据：[input_adapter.py:118](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/src/mixed_pilot_harness/router/input_adapter.py:118)。

当前通用请求还缺少预算剩余、累计费用、前一实际模型、有效执行参数与环境快照标识。trace 的调用前消息可用于日志分析，但 task 级新建 clone 不等于恢复任意中间状态。若实施 `ROUTER_REBUILD_PLAN.md` 的分支实验，应增加：

1. 调用前状态、workspace/外部环境快照及其校验。
2. 从指定状态恢复 messages、工具环境、Router 历史和剩余预算的续跑入口。
3. 固定 continuation policy、动作与采样标识，保存分支来源与重复编号。
4. 初始实现限制在可靠可恢复的代码/文件任务域。

当前引擎配置只支持 fixed/V4 且动作固定为四档，足以作为初始基线；后续任意模型/effort 动作注册表与学习型策略接入可以单独扩展。c0/c1 当前指向同一 Sonnet，因此实际是三个不同模型动作，不能把档位变化计为模型切换。

## 6. 实际验证与结果边界

### 已执行

- 读取全部运行模块、Router 模块、测试范围、配置、脚本、schema、研究文档，并检查外部默认数据目录及本地两条历史运行。
- Python 3.12.14：`PYTHONPATH=src python -m unittest discover -s tests -q`，**40/40 通过，8.252 秒**。初始系统 Python 3.9 缺 pytest/tomli，其测试失败属于未安装依赖，不计作代码缺陷。
- 运行数据 validator；另行完整审计校验清单。
- 使用 synthetic provider、临时 Git 仓库与合成 grader marker 完成上述边界复现。原项目测试全过，说明现有覆盖尚未触及这些实验有效性条件。

### 历史记录能证明什么

| 运行 | 真实执行模型 | Steps | Native checks | Rubric |
|---|---|---:|---|---:|
| `20260910T051822Z-…-ce817ce6` | 26 次均 Sonnet；c0=23，c1=3 | 26 | 1 FAIL_TO_PASS + 59 PASS_TO_PASS 通过 | 98 |
| `20260910T075420Z-…-635f3783` | 7 次均 gpt-5.6；无 Router 事件 | 7 | 同上通过 | 100 |

可以确认任务执行、路由打点和评测流程曾运行。第一条记录虽有档位变化，但没有真实模型切换；两条记录的 token 总量不能直接比较为实际花费，98/100 也不是路由稳定收益证据。

本次没有重新调用付费模型、加载真实 V4 checkpoint 推理、运行完整 benchmark 容器或核查第三方网关账单；没有验证论文创新性。40 个离线测试不能替代这些验证。

## 7. 最小合格路线

| 顺序 | 交付 | 完成判据 |
|---|---|---|
| 1 | 可信提交与评测 | 固定 base；grader 不可达；原生 matcher；固定 judge；F1/F3/F4/F5 复现转为通过的回归验收 |
| 2 | 统一计量和终止 | 同模型有效参数一致；完整 usage/重试/费用；终止分类；成功失败均保留产物；子进程回收 |
| 3 | 可重建运行 | 一个可执行任务域；锁定环境和 manifest；validator 通过；每次运行关联 task/policy/repetition |
| 4 | 最小对照实验 | 同一模型池下固定模型、固定 seed 的随机混合、规则策略、V4 适配；对任务做配对运行，保留全部失败 |
| 5 | 学术探索结果 | 在一批独立任务上重复，报告任务级质量差、实际费用、延迟及置信区间；步骤不当独立样本，数据按任务/仓库家族划分 |
| 6（条件项） | 同状态分支 | 仅在开展干预研究时增加快照恢复与 continuation 标识，并验证相同初始状态 |

探索集先用于发现效果与方差，正式样本量再依据目标质量容忍度、效果大小和不确定性确定。每个数据集一题只能做冒烟检查；扩充数量之前先确保测量可信。

**最终建议：保留 Provider/Runner/Trace 的主结构，先完善一个可恢复代码域的评测与费用闭环，再扩展任务域和 Router 学习。当前最紧迫的工作是让比较结果可信。**

## 8. 可复查材料

- [离线复现脚本](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/docs/audit-2026-09-15/probes.py)：使用 Python 3.11+ 直接运行，所有模拟任务在临时目录内。
- [离线复现结果](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/docs/audit-2026-09-15/probe-results.json)
- [任务、校验与历史运行清单](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/docs/audit-2026-09-15/inventory.json)
- [40 项测试日志](/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/docs/audit-2026-09-15/unittest.log)
