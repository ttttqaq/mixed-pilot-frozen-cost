# KSR 冷启动与 Y/C 独立监督的 Reference Router

父分支：`feature/reference-teacher__from_online-router`，父提交 `94d7512`。
新分支：`feature/ksr-coldstart__from_reference-teacher`。
新入口 `scripts/train_ksr_reference_router.py`，新配置
`configs/online-router.ksr-reference.db94.yaml`，旧 Reference 方法和输出保留。

## 冷启动

`cold_start.trajectory_steps=500` 指 **500 次新的 Actor 调用及其环境交互**，
包括分枝新增调用；共享前缀的环境重放不调用 Actor、不重复计数。
500 不是优化器次数、epoch、任务数，也不是展开后的重复路径记录数。
API 失败且没有环境转移的请求不作为轨迹 step，但费用仍进入唯一调用账本。

KSR 使用现有 `KSR-brunch-1` 定义：由 `KSR_config.json` 确定强/弱起始模型，
根据错误、重复、停滞、进展规则在关键状态枚举两个候选模型各执行一步，
然后回到该树的基础模型；后续允许递归分枝。任务在两个 benchmark 间交替，
同一 task 的两种起始模型共享采集费用限额。每树默认深度 3、节点数 14。

每 100 个实际轨迹 step，保存数据快照和 Router checkpoint：

```text
cold_start/checkpoints/step_000100.pt 及 .json
cold_start/checkpoints/step_000200.pt 及 .json
cold_start/checkpoints/step_000300.pt 及 .json
cold_start/checkpoints/step_000400.pt 及 .json
cold_start/checkpoints/step_000500.pt 及 .json
```

每个采集 checkpoint 用当时已有的有效标签更新 Router；未终止分枝不填伪造的失败零。
权重文件保存模型、优化器、RNG、embedding cache 和采集计数；数据 cache 每步保存，
`--resume` 从最近已提交权重和已保存采集进度继续，不重新购买已完成的调用。
若一个 milestone 恰被中断，恢复时先补齐该 checkpoint，再继续采集。

停止在第 500 个 step 时，末尾可能还有未终止分枝。原始 trace 和未知标签照常保存，
不会为补齐终局而偷偷超过 500 次交互。采集数量不等于有效训练标签数量，两者分别报告。

## 标签和缓存

- `cold_start/cache.json`：完整采集树、唯一调用、配置/数据/源码/价表指纹和恢复位置。
- `cold_start/samples.jsonl`：每个唯一实际调用一条记录，保存所有有效或未知 Y/C 标签。
- `cold_start/traces.jsonl`：每个叶子/未完成路径的全部 step，含共享前缀引用。
- `trajectories/`、`calls/`、`ledger.json`：原始交互与唯一计费记录，失败费用也保留。

Y 的叶子标签使用原生 success 0/1，父节点按未来分枝等权递归平均。
旧 KSR 的归一化分数不能直接当成功概率。C 是从当前调用开始的预期 Actor 后缀费用，
未来分枝费用取平均，不把全部旁支的采集支出相加当单一路径费用。
共享前缀不因不同叶子重复加权训练；路径展开记录只供追踪分析。

缓存可以复用到下一次冷启动，无新增冷启动 Actor 请求；要求采集 step 数恰好匹配且
模型池、规则、数据、价表、源代码和特征/预算语义兼容。旧 Reference/Option checkpoint
不能当新方法的 resume checkpoint；旧 KSR normalized-score 缓存也不能直接混入。

## 独立监督和在线训练

保留冻结 Qwen3 历史编码器、模型编码器和三网络 ensemble。两个输出头分别预测：

```text
Y：本次选模后按指定后续策略完成任务的成功概率
C：本次调用开始到终局的 Actor 剩余费用（CNY）
```

损失是 Y 的加权平方误差 + C/C_ref 的加权平方误差 + 模型 embedding 正则。
两个头都训练；无 U、Delta-U 或 pair-ranking loss，也无旧预测 anchor。
`decision.beta=0.5` 仅用于 Router 选模 `U=Y-beta*C`。
为复用 Reference 的决策器，程序派生 `lambda_=beta*cost_reference`，不把它加入训练目标。
主动查询用 Y/C 各自的分歧和收益差；自适应 rollout/审计分别比较收益阈值和费用阈值，
不再使用 beta 或效用差。A/B 仍保留，但只提供两侧独立 Y/C 标签。

冷启动结束先进行一次完整 test 50 题评估（iteration=0），之后执行 5 个在线周期。
每周期默认仍为 4 个任务、4 个查询根，Actor 首步干预及固定 reference Teacher 保留。
Teacher 只补训练尾部，不更新参数；真实标签权重 1，Teacher 标签权重 .25。
在线数据按当前后续策略版本过滤，KSR 回报不冒充新 Router 策略回报。
默认还在第 3、5 轮评估 test。test 不进入训练/Teacher，但参与 best_eval 选择，
不能称为完全未参与模型选择的最终留出集。

## 费用和调用限制

每条路径最多 50 次 Actor 调用（包括继承前缀）；同一 task 在同一采集/策略版本下，
所有分枝合计最多 1 CNY Actor 费用，强/弱起始树也共享这笔额度。
Teacher 支出单独进入全流程账本，总预算为用户指定的 80 CNY，包括测试。

每次 Actor 请求前，按 UTF-8 输入上界、最大价表费率及输出 token 额度计算保守费用包络，
必要时缩小最大输出额度；连最小输出都无法负担时，不发送请求。
单路径预算结束由原生环境评分；旁支采集总预算不足造成的未完成分枝保持未知。
供应商 usage 超出费用包络或费用不确定时停止，保留费用和原始记录，不把异常当免费。
这个限制针对配置价表及 provider token 契约；不承诺供应商异常计费时仍能保证真实账单上限。
全局 80 CNY 仍为请求前停止阈值；Teacher 的最后一次请求可能超过该阈值。

## 运行

```bash
cd /DB/data/yujietang/Agent/code/mixed-pilot
git switch feature/ksr-coldstart__from_reference-teacher
PY=/DB/rhome/yujietang/miniconda3/envs/mixed-pilot/bin/python
CFG=configs/online-router.ksr-reference.db94.yaml

# 静态预检，无模型/环境/API 初始化，不创建运行目录。
"$PY" scripts/train_ksr_reference_router.py --config "$CFG" --dry-run

# 完整离线链路检查：合成环境、回复及 hash 编码器；不是任务效果测试。
"$PY" scripts/train_ksr_reference_router.py --config "$CFG" \
  --smoke --output-dir outputs/ksr-reference-my-smoke

# 正式执行：先选择空闲 GPU，将 7 换成该卡编号。
# 仅在进程内复用已填写的 Reference 配置凭据，新 YAML 的 key 保持留空。
CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
"$PY" scripts/train_ksr_reference_router.py --config "$CFG" \
  --credentials-config configs/online-router.reference-teacher.db94.yaml
```

默认输出：`outputs/ksr-reference-router-run01`。
同一配置、源码和数据恢复时，在正式命令末尾增加 `--resume`。
只做冷启动及其测试时增加 `--cold-only`；之后原命令加 `--resume` 继续在线部分。

复用本次完整冷启动 cache 到新运行：在正式命令末尾增加
`--reuse-cold-start outputs/ksr-reference-router-run01 --output-dir outputs/ksr-reference-router-run02`。
该新运行恢复时也要保留同一 `--reuse-cold-start` 参数。复用后重新训练 Router 并做冷启动测试，
冷启动数据采集费用为零；后续在线与测试仍正常计费。复用运行只生成 step_000500 权重，
原采集运行的 100/200/300/400 快照仍在来源目录。

独立评估新 checkpoint：原命令加
`--evaluate-checkpoint outputs/ksr-reference-router-run01/cold_start/checkpoints/step_000500.pt`
及一个新的 `--output-dir`；无需 Teacher key，也不读取 train reference。

## 验证边界

新增测试覆盖：精确 500 次 Actor/environment 交互与五个 checkpoint、断点续采、
缓存复用零新增请求、递归分枝成功率/费用标签、未完成标签保留、强弱树共享任务费用、
原生成功目标、beta 不影响监督梯度、两个头均更新，以及冷启动评估后完成五轮在线训练。
离线测试与 scripted 原生环境检查只验证实现，不证明真实任务成功率或费用改善。

本次验证（2026-11-02）：13 项新增测试通过；完整回归为 142 passed、56 subtests passed，
另有 1 项旧失败 `RouterTests::test_random_pool_probabilities_billing_and_all_step_exports`，
在父提交 94d7512 的隔离源码上复现相同 trace 枚举顺序断言失败。
完整 CLI smoke、cold-only→resume、缓存复用运行均通过，复用运行 cold_start scope 的 Actor 调用数为 0。
原生 train/test 的 ScienceWorld/ALFWorld 共 4 项 scripted 重放检查通过（12 次合成回复），
GPU7 上真实本地 Qwen3-Embedding-0.6B 冻结前向通过。以上验证未发起付费 Actor/Teacher API 请求，未启动正式训练。

## 运行异常恢复（2026-11-02）

预算截断时，在当前环境关闭之前直接进行原生免费评分，不为了评分重新重放整条路径。
实际 A/B 分枝和续接仍严格验证公开历史、环境 transition、score、done；不忽略状态差异。

Teacher 的空响应、缺失用量和 JSON/schema 错误最多尝试 3 次，并受原来的每轮
Teacher 调用数及 80 元总预算约束。缺少计费用量时保存原始响应并保留保守费用预留，
不标作免费或已结算；该回答不产生训练标签。重试后仍不可用时返回缺失证据，触发已有的
真实 continuation rollout 补标签流程，不能伪造 Y/C。有效缓存继续复用，错误记录保留在
teacher_predictions.jsonl/calls 中，重试及降级记录在 teacher_retries.jsonl。

修复版本恢复旧运行时，原命令加：

```bash
--resume --runtime-repair-from 679d73bcdf0d16efe851a91033f34aece5c0a67c
```

这会从指定祖先提交重建原运行指纹，核对当前基础运行时、价格、方法配置、数据和参考，
确认一致后保留原缓存身份；执行代码的实际指纹与提交另记在 runtime_repairs.jsonl。
原 split_manifest、冷启动缓存、标签、权重和账本不重写。此选项用于已审阅的工程修复，
不允许借此改变预算、beta、模型池、数据划分或训练目标。

Actor 请求的费用未知但已有保守预留时，每任务预算统一按“已知费用 + 预留上界”计算，
与全局账本一致；无上界的未知费用仍阻止后续调用。这样 HTTP 5xx 的有限重试可以在
原 1 元任务预算内继续，而不是把一次有界预留直接当作耗尽整笔任务预算。
评估路径因预算停止时可对当前活环境原生评分；训练分枝的共享采集预算截断仍保留未知。
原生 outcome_valid 与费用完整性分开：未知费用保留预留和原始调用，C/联合训练标签
仍无效，成功率可使用已确认的原生结果。恢复评估也检查旧前缀中的未知费用，费用报告
分别给出已知金额与预留占用。

Teacher 的 SSE/传输错误在底层重试结束后也按缺失证据处理；确认费用已知或有保守
上界且未耗尽全局预算后，切换到已有真实 rollout 标签流程。为避免同一轮反复调用
已不可用的服务，本轮后续未缓存的 Teacher 查询直接使用真实 rollout；有效缓存仍可
复用。下一轮 begin_phase 重置此状态，再尝试 Teacher。传输错误、受影响的 call_id
及跳过请求都记入 teacher_retries.jsonl；无上界的未知费用仍使流程停止。
