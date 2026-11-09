> 本文记录父分支的可训练 Teacher 方法。当前固定 reference Teacher、test 评估及新配置见 [REFERENCE_TEACHER_ROUTER.md](REFERENCE_TEACHER_ROUTER.md)。

# DB94 在线路由训练

项目：`/DB/data/yujietang/Agent/code/mixed-pilot`。
方法分支：`feature/online-router__from_main`。
入口：`scripts/train_online_router.py`；配置：`configs/online-router.db94.yaml`。

这是独立在线方法，不读取旧离线 MTRouter checkpoint。新权重使用
`mixed-pilot-online-router-v1` 格式，不应交给旧 `mtr-core-serve` 加载。
候选模型默认是 Qwen Flash 和 GLM-5.3，强模型 GLM-5.3 同时担当训练期辅助评估员。

## 运行

DB94 的 `mixed-pilot` Conda 环境已有依赖，无需重新安装。脚本会把本仓库的
`src/`、`MTRouter/src/` 加入导入路径。命令中的路径均基于以下项目根目录。

```bash
cd /DB/data/yujietang/Agent/code/mixed-pilot
git switch feature/online-router__from_main

# 静态预检：只读 train/valid、配置和价格，不加载编码器/环境，不调用 API。
/DB/rhome/yujietang/miniconda3/envs/mixed-pilot/bin/python \
  scripts/train_online_router.py --config configs/online-router.db94.yaml --dry-run

# 离线合成 smoke：使用真实划分清单，但环境、模型回复和 embedding 都是测试替身。
# 此结果只检查程序链路，不代表原生 benchmark 成功率，也不产生真实 API 费用。
/DB/rhome/yujietang/miniconda3/envs/mixed-pilot/bin/python \
  scripts/train_online_router.py --config configs/online-router.db94.yaml \
  --smoke --output-dir outputs/my-online-smoke
```

正式运行前，在当前 shell 设置 `DASHSCOPE_API_KEY` 和 `ZHIPU_API_KEY`。
程序不会从技能文件读取候选模型 key，不会把 key 写进配置或结果。

```bash
# 交互读取，避免 key 进入 shell history；输入时不回显。
read -rs -p 'DashScope API key: ' DASHSCOPE_API_KEY; echo
read -rs -p 'Zhipu API key: ' ZHIPU_API_KEY; echo
export DASHSCOPE_API_KEY ZHIPU_API_KEY

# 选择当前空闲物理 GPU，例如 7；进程内可见设备为 cuda:0。
CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
/DB/rhome/yujietang/miniconda3/envs/mixed-pilot/bin/python \
  scripts/train_online_router.py --config configs/online-router.db94.yaml \
  --output-dir outputs/online-router-run01
```

正式命令会调用候选 LLM、强模型辅助评估员及 valid 评估，产生 API 费用。
默认全流程已知费用阈值为 30 CNY，下一次请求前检查；最后一次请求可能超出阈值。
价表是冻结估算表，不是实际账单。费用不明的请求会保留并停止后续付费调用。

```bash
# 修改参数不必改源码。YAML 和 --set 二选一，--set 优先。
CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
/DB/rhome/yujietang/miniconda3/envs/mixed-pilot/bin/python scripts/train_online_router.py \
  --config configs/online-router.db94.yaml --output-dir outputs/online-router-run02 \
  --set online.iterations=10 --set budget.total=50 \
  --set short_rollout.horizon=3 --set budget.lambda_=0.05

# 中断恢复：配置和 output-dir 保持相同。
/DB/rhome/yujietang/miniconda3/envs/mixed-pilot/bin/python scripts/train_online_router.py \
  --config configs/online-router.db94.yaml --output-dir outputs/online-router-run01 --resume

# 独立评估新格式权重：只运行 valid，不训练、不调用 Teacher。
/DB/rhome/yujietang/miniconda3/envs/mixed-pilot/bin/python scripts/train_online_router.py \
  --config configs/online-router.db94.yaml --output-dir outputs/online-router-valid01 \
  --evaluate-checkpoint outputs/online-router-run01/best_valid.pt
```

恢复以完整迭代为边界。中断轮可能重新采集，旧调用和费用全部保留；它不是精确到 API
步的无重复续跑。`state.json` 原子指向一代 checkpoint，避免网络和 replay 交叉版本。
完成的最后一轮 valid 已评估时，resume 不会重复付费评估。
改变消融、价格、provider、源码或数据后，使用新输出目录；resume 会拒绝不一致。

## 数据隔离

数据为 `data/VirtualWorld-subset`：train 300，valid 50，test 50。
训练入口只打开显式的 `train.json` / `valid.json`，不 glob 全任务、不读取 test 索引。

- Teacher 校准、单模型冷启动、访问轨迹、短分枝、完整审计、梯度更新全部使用 train。
- valid 仅用贪心 Router 运行原生任务，记录成功率/费用；不产生训练或 Teacher 样本。
- 原始 `source_split` 是上游出处；当前实验划分以 `routing_metadata.router_split` 为准。
- 检查 train/valid 原生实例及任务身份无交叉；保存索引、manifest、任务内容指纹。
- 每次网络更新前，`assert_training_rows` 再检查所有行的 split。

默认 `evaluation.max_tasks: null`，每隔两轮及结束时评估全部 valid 50 个任务。
调试可显式缩小 max_tasks；覆盖不完整会记录，不能当作完整 50 题成功率。
不提供切换到 test 的训练参数，test 也不参与 checkpoint 选择。

## 网络与目标

沿用冻结本地 Qwen3-Embedding-0.6B，但重新序列化完整 role/content、工具参数、反馈和
可见推理记录，保留任务与最近上下文。8192-token 为硬上限，使用 FP32 编码。
1024 维历史向量附加 3 个数值：剩余预算比例、剩余步数比例、当前步数比例。
历史 embedding 按实际文本缓存，在线增量生成，不需要旧离线预计算文件。

Router 为动作条件 ensemble：状态 1027 维 + 模型属性/残差向量 64 维，
经过默认 256/256 MLP，输出终局收益和剩余 Agent 费用两个头。
收益使用 sigmoid，费用使用 softplus；内部费用除以 cost_reference 后训练。

效用为：

`U(x,a) = E[terminal reward] - lambda_ * E[remaining agent cost] / cost_reference`。

默认 `rollout.reward: success`，目标是原生成功(0/1)；可改为 normalized_score，
ScienceWorld 此时使用 `(score+100)/200`，其他使用 `score/100`。
它们是不同目标，不能把归一化分数解释成成功概率。

训练策略使用冻结的 epsilon-greedy Router，默认 epsilon=0.1。
A/B 后续和主轨迹均遵循同一策略分布；valid 关闭探索。
`policy_version` 是 Router 参数、编码语义、预算尺度、lambda 和 epsilon 的 SHA256
指纹，不是迭代编号。Teacher 更新不改变该指纹，Router 更新会产生新指纹。

## Teacher 的设置与实际作用

Teacher 是用真实结果训练的状态价值 ensemble，预测当前冻结策略的尾部收益和费用。
它不是直接把 GLM 的打分当标签。

强模型辅助路径：

1. 在 train 的部分真实返回校准状态、A/B 短展开尾状态，调用 strong_model (GLM-5.3)。
2. 不给工具，要求 JSON：expected_reward、remaining_cost、confidence、reason。
3. 明确继续策略版本、epsilon、模型池、剩余预算/步数、价表与收益定义。
4. 预测值构成 4 个辅助输入：收益、归一化费用、自报 confidence、可用性 mask。
5. Teacher 以本地真实终局收益及真实后缀费用为监督，学习如何使用/纠正这些辅助信息。
6. Teacher 默认以 25% 概率屏蔽整组辅助输入，学习无 LLM 证据时的预测。

LLM 自报 confidence 只是输入，不是可靠置信区间。错误 JSON、截断、工具调用、
越界数值等记为无效证据，但已发生费用仍计入 teacher scope。禁用 llm.enabled 时，
Teacher 仍可用真实结果训练，只是不接收强模型辅助输入。

每个阶段默认最多 16 次 assessor 调用；anchor_fraction=0.5 限制校准 anchor 最多使用
一半，预留其余额度给 A/B 截断尾部。相同状态/策略的证据缓存可复用，不重复调用。
缓存键包含策略指纹；Router 更新后不把旧证据自动当作新策略预测。

辅助调用默认 max_tokens=8192、temperature=null(继承候选强模型设置)，不强制关闭思考。
GLM-5.3 官方说明其始终启用思考，因此不能只给很小的总输出额度并假定一定返回 JSON。
实际截断会作为无效证据记录，可通过该参数调节。接口依据：
https://docs.bigmodel.cn/cn/guide/models/text/glm-5.3 。

Teacher 只用 `source=real` 且继续策略指纹匹配的样本训练，单模型冷启动数据不混为
当前 Router 的 V 标签。至少 16 个当前策略真实 anchor 后才 ready。
该门控是最低数据量检查，不是准确率保证；ensemble disagreement 也仅是经验代理。
强制 A/B 前的状态是动作 Q 监督；Teacher 的 V anchor 从强制动作之后开始提取。

## 完整在线流程

冷启动：

1. 从 train 选 4 个任务，分别运行两个固定模型的完整轨迹。
2. 用真实收益/费用初始化 Router；这些固定策略的 return 只用于此阶段。
3. 在候选关键状态构造 4 组 A/B 完整对照，第一步干预，后续按冻结 Router 继续。
4. 拟合初始 Teacher，再更新初始 Router。下一轮会按新策略指纹重新校准。

每轮在线：

1. 固定 Router 参数与 epsilon，记录实际策略指纹。
2. 从 train 抽默认 4 题运行主轨迹，收集访问状态和原生终局结果。
3. 从真实终局后缀提取 Teacher anchor，调用部分强模型证据，拟合当前策略价值。
4. 从访问池选默认 4 个状态。按价值差、ensemble 分歧、潜在影响和预测成本打分，
   其中默认 25% 为随机查询。这是可消融的启发式 acquisition，不宣称无偏 VOI。
5. 每个状态执行下面的 A/B 短展开、必要的长度扩展和完整审计。
6. 用新真实续跑结果再次拟合 Teacher，重新生成本轮伪标签/权重。
7. 用当前策略版本的真实和估计 Q 样本更新 Router。旧策略 MC 不直接作为本轮 Q 目标；
   模型参数延续，且损失包含更新前预测锚定。
8. 按间隔仅在 valid 评估，保存本轮日志和恢复 checkpoint。

Teacher 在一轮对照采集中不更新，Router 直到该轮全部采集完成才更新。
因此每个标签都有明确继续策略；固定单模型旧 return 不会静默污染在线目标。

## A/B 短 rollout 与标签

从相同保存前缀恢复两次原生环境；逐步核对初始历史、观察、分数、格式状态和 done。
这是经过验证的前缀重放，不声称复制了全部隐藏模拟器位。
重放复用已有回复/动作，不调用 LLM、不再次计费。

两侧从相同历史/剩余预算开始：第一步分别强制 A/B；之后均按冻结 Router 继续。
默认 H=3 表示总共 3 次新 API 调用(含强制第一步)，不是 A/B 各保持 3 步。

未终止分枝的目标：

`reward_target = Teacher.reward(endpoint)`

`cost_target = 已实际发生的分枝后缀费用 + Teacher.cost(endpoint)`。

终止分枝使用真实原生收益与真实后缀费用，不再添加 Teacher 尾部。
继承前缀与其他反事实分枝、Teacher、valid 费用均不进入此 cost-to-go。
短截断不是失败，不能赋 0 标签；环境故障/重放失败也保留空标签。

默认自适应展开：价值差未明显超过分歧代理时，继续 3 步，最多 9 步。
随机完整审计先于不确定性审计预留名额，防止自信错误永远不被验证。
默认每轮最多 2 个主动/随机审计 pair；Teacher 未 ready 时的真实 grounding 不受此
可选审计上限限制，但仍受总费用阈值限制。随机审计会跳过不必要的中间长度扩展。

关闭 audit 不会允许未 ready 的 Teacher 伪造标签；这种情况仍需要真实完整续跑。
一侧失败时放弃 pair，但另一侧独立有效的真实后缀仍可保留。
真实/估计混合 pair 中，真实动作的点回归权重仍为 1。

## Router 如何更新

当前策略的真实样本权重为 1；伪标签默认权重为 `0.25/(1+utility_disagreement)`。
Teacher 校准后会刷新伪目标、差值与权重，目标生成不参与梯度。

损失包括收益/归一化费用的加权 MSE、成对效用差的 MSE、更新前预测锚定、模型残差 L2。
成对项拟合 `U(A)-U(B)`，无需把很小的估计差硬转换成 winner。
默认 AdamW，lr=0.001，10 epochs/batch64；Router 与 Teacher 均为小网络更新。
冻结历史编码器和候选 Qwen/GLM 不进行微调。

## 参数与消融

完整字段见 `docs/ONLINE_ROUTER_CONFIG_REFERENCE.md`。最常用参数：

| 参数 | 意义 |
|---|---|
| budget.total / rollout.task_budget | 全流程训练费用阈值 / 单条实际路径费用阈值 |
| budget.lambda_ / cost_reference | 收益与费用权衡及尺度 |
| online.iterations / tasks_per_iteration / states_per_iteration | 在线轮数、主轨迹任务数、对照状态数 |
| online.exploration_epsilon | 冻结训练策略的模型探索概率 |
| short_rollout.horizon / max_horizon / extension | 初始短长度、上限、扩展步数 |
| teacher.llm.profile | 强模型，默认要求等于 strong_model |
| teacher.llm.max_calls_per_iteration / anchor_fraction | assessor 总配额及 anchor 配额占比 |
| teacher.min_real_samples / evidence_dropout | 当前策略真实 anchor 门控及辅助特征屏蔽 |
| audit.random_fraction / max_pairs_per_iteration | 随机审计比例与可选完整审计上限 |
| optimization.pairwise_weight / anchor_weight / pseudo_weight | 成对差值、更新锚定、伪标签权重 |
| evaluation.every_iterations / max_tasks | valid 评估间隔与显式调试子集 |

独立输出目录对应独立消融，例如：

```bash
# 随机选状态
--set selection.enabled=false
# 固定短长度
--set short_rollout.adaptive=false
# Teacher 不使用强 LLM 辅助，但仍以真实结果训练
--set teacher.llm.enabled=false
# 完整 rollout 基线：Teacher 关闭会自动关闭 LLM 辅助并强制真实终局
--set teacher.enabled=false --set short_rollout.enabled=false
# 不用可选完整审计(仍保留未 ready 的必要 grounding)
--set audit.enabled=false
# 不用成对差值损失 / 不用更新锚定 / 不惩罚费用
--set optimization.pairwise_weight=0
--set optimization.anchor_weight=0
--set budget.lambda_=0
```

## 输出

| 文件/目录 | 内容 |
|---|---|
| config.resolved.json | 实际生效配置，含 CLI 覆盖与绝对路径 |
| split_manifest.json | train/valid 身份、内容指纹、代码提交与运行源文件指纹 |
| calls/*.json / ledger.json | 唯一实际调用、usage、回复、状态、各 scope/profile 费用 |
| trajectories/*.json | 原生历史、动作、重放证据、继承前缀、实际新后缀 |
| teacher_evidence.jsonl | 强模型原始辅助输入、有效性、调用 ID 和缓存身份 |
| teacher_real_samples.jsonl | 当前策略真实终局 anchor，作为 Teacher 监督 |
| router_samples.jsonl | Router 点监督；sample_id 更新行以最后有效值为准 |
| pairs.jsonl | A/B 身份、同状态指纹、目标、优势、置信权重和审计记录 |
| teacher_audits.jsonl | 审计前预测差与实际续跑差的误差/排序比较 |
| policy_decisions.jsonl | 实际策略选择、概率、收益/费用预测与版本 |
| training_log.jsonl | 每轮样本数、Teacher/Router loss、更新次数、费用 |
| validation.jsonl | valid 成功率、收益、费用、覆盖率，不含训练更新 |
| checkpoints/*.pt / state.json | 原子提交的迭代恢复代，含优化器/RNG/replay |
| latest.pt | 最新在线 Router/Teacher，可用于独立 valid 评估 |
| best_valid.pt | 完整 valid 成功率最高；同成功率时费用最低的权重 |
| iterations/iteration-*.pt | 按 checkpoint.every_iterations 另存的归档 |
| summary.json / SUMMARY.md | 运行停止原因、总费用、样本与最佳 valid 汇总 |

训练结果保存小网络、优化器、模型元数据和 embedding 缓存，不打包完整 Qwen 权重；
换服务器仍需本地冻结编码器。新 checkpoint 是在线独立格式，不兼容旧离线评分入口。

## 验证与限制

离线单测验证单步干预、严格重放、预算与未知费用、真实 Teacher 标签、梯度更新、
策略版本、train/valid 隔离、强模型证据配额、checkpoint 与已完成恢复不重复计费。
真实原生环境检查使用合法 `?all` 固定回复，检查 train/valid 各一个 ScienceWorld 和
ALFWorld 的启动与前缀恢复；它不测模型任务成功率。真实 Qwen 编码器检查另验证
1027 维状态特征、冻结参数和保存/重新编码预测一致。

验证入口：`scripts/check_online_native.py`、`scripts/check_online_encoder.py`。
检查报告位于 `outputs/online-router-verification-20261028/`。

目前没有真实付费在线训练的收益证明。主动选择、ensemble 分歧、自适应 horizon 和
伪标签降权都是待实验检验的方法选择；Teacher readiness 不能代替校准质量测量。
整体费用统计含所有冷启动、主轨迹、分枝、审计、Teacher 和 valid 调用；
单路径 cost-to-go 则只含该路径的 Agent 费用。预算是请求前停止阈值，不是硬账单保证。
