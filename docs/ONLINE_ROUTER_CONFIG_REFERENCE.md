> 本文记录父分支的可训练 Teacher 方法。当前固定 reference Teacher、test 评估及新配置见 [REFERENCE_TEACHER_ROUTER.md](REFERENCE_TEACHER_ROUTER.md)。

# Online Router 配置参考

默认配置为 `configs/online-router.db94.yaml`。配置由
`mixed_pilot_mtrouter.online.config.load_config` 读取，未知字段、重复 YAML 键、
非法类型、非有限数值及越界参数均会报错。

相对路径以 `project_root` 为基准，与启动命令时的工作目录无关。命令行
`--set section.key=value` 使用 YAML 标量/列表语法，可重复指定，例如：

```bash
--set online.iterations=3 \
--set teacher.llm.enabled=false \
--set model.hidden_dims='[128,128]' \
--set output_dir=outputs/online-router-no-llm
```

一次实验的生效值写入 `config.resolved.json`。恢复实验要求生效配置与数据、
provider/价格/模型元数据指纹保持一致；改变消融开关、预算或模型参数应使用新输出目录。

## 数据、模型与路径

| 字段 | 默认值 | 含义 |
|---|---|---|
| `project_root` | `/DB/data/yujietang/Agent/code/mixed-pilot` | 项目根目录；未声明时从配置文件位置查找项目。 |
| `harness_config` | `harness.json` | A/B 和强模型辅助评估所用的 provider profiles；API key 仍由 profile 指定的环境变量提供。 |
| `pricing` | `configs/inference.prices.json` | 冻结价格快照；币种及所有付费作用域的计费依据。 |
| `output_dir` | `outputs/online-router` | 新实验结果目录；不能无意覆盖已有实验。 |
| `seed` | `42` | 任务选择、随机状态查询、探索和网络训练的随机种子。外部 LLM 回复不保证可复现。 |
| `dataset.root` | `data/VirtualWorld-subset` | 原生 ScienceWorld/ALFWorld 任务目录。 |
| `dataset.train_split` | `train` | 只能为 `train`。训练及真实终局校准只读取 `train.json`。 |
| `dataset.validation_split` | `valid` | 只能为 `valid`。周期评估只读取 `valid.json`。 |
| `model_pool` | `[qwen-flash-2025-07-28, glm-5.3]` | 恰好两个不同的具体 harness profile；顺序决定 pair 中 A/B 的顺序。 |
| `strong_model` | `glm-5.3` | 必须属于 A/B 模型池；开启 LLM 辅助时必须用此 profile。 |

`source_split` 仅保留上游来源；本实验的划分取 manifest 的
`routing_metadata.router_split`，并与索引严格核对。在线入口不会打开 `test.json`，
也不会遍历所有任务来发现 test。train/valid 的任务 ID 与原生实例身份必须互不重叠。
`split_manifest.json` 保存所选索引、manifest、内容指纹与原生身份。

## 预算、状态特征与网络

| 字段 | 默认值 | 含义 |
|---|---|---|
| `budget.total` | `30.0` | 全实验已知 API 费用停止阈值，包含冷启动、主轨迹、反事实分枝、完整审计、Teacher LLM 和 valid。每次新请求前检查，最后一次在途请求可能越过阈值。 |
| `budget.cost_reference` | `0.1` | 内部费用目标的归一化单位，币种取价格文件。改变它会改变已训练 head 的单位，不能原地恢复旧 checkpoint。 |
| `budget.lambda_` | `0.05` | 路由效用 `reward - lambda_ * cost / cost_reference` 中的费用系数。设为 `0` 是只看收益的消融，不会关闭实际计费。 |
| `encoder.backend` | `hf` | `hf` 使用冻结本地文本模型；`hash` 只用于显式离线 smoke，不能据此报告真实语义路由效果。 |
| `encoder.model_name` | DB94 本地 Qwen3-Embedding-0.6B 路径 | 历史编码器位置或 HuggingFace 模型 ID。HF 加载为 FP32，编码器不参与梯度训练。 |
| `encoder.max_tokens` | `8192` | 编码历史时的 token 上限，保留任务与较新交互。 |
| `encoder.device` | `auto` | `auto`、`cpu`、`cuda`、`cuda:N` 或 `mps`；自动模式优先 CUDA。 |
| `encoder.hash_dim` | `32` | 仅 hash smoke 编码器使用的维度。 |
| `model.model_metadata` | `MTRouter/configs/models.ksr-db94.yaml` | A/B 元数据，必须覆盖模型池；价格是冻结特征，不是实时报价。 |
| `model.hidden_dims` | `[256,256]` | 路由 action-value MLP 的隐藏层；Teacher 由 `teacher.hidden_dims` 单独配置。 |
| `model.dropout` | `0.1` | 路由 MLP 的训练 dropout，范围 `[0,1)`。 |
| `model.ensemble_size` | `3` | 路由 ensemble 成员数；其分歧用于查询排序和不确定性启发式。 |

新模型显式编码完整可见消息内容、工具结构与已有 reasoning，并附加剩余预算、剩余
步数和当前步数等数值特征。路由器分别预测收益与剩余费用。ensemble 分歧是经验性
采集信号，不是已经校准的统计置信区间。

## 环境运行与冷启动

| 字段 | 默认值 | 含义 |
|---|---|---|
| `rollout.task_budget` | `1.0` | 一条任务路径的费用阈值，含继承前缀；A/B 两侧从相同剩余预算开始。 |
| `rollout.step_limit` | `50` | 一条任务路径的 Agent API 步数上限，含继承前缀。 |
| `rollout.replay_attempts` | `2` | 含首次在内的环境初始化/严格前缀重放尝试次数。 |
| `rollout.retry_wait` | `0.0` | 前缀恢复失败后的尝试间隔，秒。不是完整任务答错后的自动重跑。 |
| `rollout.trace_delay` | `0.0` | 有效终局轨迹完成后的间隔，秒。 |
| `rollout.reward` | `success` | `success` 使用原生成功的 0/1 标签；`normalized_score` 使用归一化终局分数。ScienceWorld 的后者为 `(score+100)/200`，不代表成功概率。 |
| `cold_start.enabled` | `true` | 是否先构造初始 Router；关闭后直接从初始化网络进入在线迭代。 |
| `cold_start.tasks` | `4` | 从 train 中抽取的冷启动任务数，实际不超过 train 大小。 |
| `cold_start.strategy` | `single_models` | `single_models` 对每个任务分别跑 A/B 固定模型完整轨迹；`random` 跑每步均匀随机模型轨迹。 |
| `cold_start.anchor_states_per_trace` | `4` | 每条有效真实终局轨迹最多保留多少 Teacher 状态锚点；也用于在线主轨迹/完整分枝，并非只作用于冷启动。 |
| `cold_start.queries` | `4` | 初始 Router 拟合后，从冷启动访问状态中追加的完整 A/B 对照数量。 |

固定模型冷启动轨迹只初始化 Router，不作为当前策略的 Teacher 真值。追加 A/B
比较只干预第一步，之后均按冻结初始 Router 继续；Teacher 锚点从干预之后的策略
后缀提取。前缀通过保存回复重放，不重新调用 LLM，不重复计费。

## 在线采样与短展开

| 字段 | 默认值 | 含义 |
|---|---|---|
| `online.iterations` | `5` | 冷启动后的在线策略更新轮数。 |
| `online.tasks_per_iteration` | `4` | 每轮抽取多少 train 任务运行当前冻结策略。 |
| `online.states_per_iteration` | `4` | 每轮最多购买多少组访问状态 A/B 对照。 |
| `online.exploration_epsilon` | `0.1` | 主轨迹与反事实分枝后缀中的 epsilon-greedy 动作探索；valid 使用贪心路由。 |
| `selection.enabled` | `true` | 开启主动状态查询；关闭时从同一访问状态池中均匀随机抽样。 |
| `selection.random_fraction` | `0.25` | 查询组数中向上取整的随机覆盖比例，其余按 acquisition 排序。 |
| `selection.impact_weight` | `1.0` | acquisition 中候选收益差的影响权重。 |
| `selection.uncertainty_weight` | `1.0` | acquisition 中 ensemble 效用分歧的权重。 |
| `selection.cost_weight` | `1.0` | 对预测查询费用代理的惩罚指数；设为 `0` 去掉该费用分母。 |
| `short_rollout.enabled` | `true` | 开启短展开＋尾部价值；关闭时直接获得完整 A/B 终局。 |
| `short_rollout.horizon` | `3` | 初始展开的新 Agent API 调用数，含 A/B 干预首调用。 |
| `short_rollout.adaptive` | `true` | 优势不明确或 Teacher 未就绪时，逐段增加展开长度。 |
| `short_rollout.max_horizon` | `9` | 自适应短展开最多新增的步数，不含继承前缀。 |
| `short_rollout.extension` | `3` | 每次自适应增加的调用数。 |

两个孩子从完全相同的调用前历史与剩余预算开始，分别仅强制 A/B 一步，其后使用
同一冻结策略。短展开截止是 `truncated`，不是失败标签。部署的决策单位仍是一次
API 调用，而非承诺连续使用某模型 H 步。

## Teacher 与完整审计

| 字段 | 默认值 | 含义 |
|---|---|---|
| `teacher.enabled` | `true` | 开启真实终局监督的尾部价值网络；关闭时自动关闭 LLM 辅助，并用完整续跑获得目标。 |
| `teacher.ensemble_size` | `3` | 尾部收益/费用 ensemble 数量。 |
| `teacher.hidden_dims` | `[128,128]` | 每个 Teacher 尾部价值 MLP 的隐藏层，必须是正整数列表。 |
| `teacher.dropout` | `0.1` | Teacher MLP 的训练 dropout，范围 `[0,1)`，独立于 Router 的 `model.dropout`。 |
| `teacher.epochs` | `10` | 每次 Teacher 拟合的 epoch 数；一轮可以在采样前后各拟合一次。 |
| `teacher.learning_rate` | `0.001` | Teacher 的 AdamW 学习率，独立于 Router 的 `optimization.learning_rate`。 |
| `teacher.min_real_samples` | `16` | 当前策略版本的真实锚点达到该数量后，才允许该 Teacher 为相应版本提供伪标签。 |
| `teacher.evidence_dropout` | `0.25` | Teacher 训练时随机遮掉强模型辅助证据的比例，范围 `[0,1)`，支持无辅助证据的尾部状态。 |
| `teacher.real_replay_iterations` | `1` | 固定只允许 `1`，防止把旧策略收益当作当前策略的 Teacher 真值；历史记录仍归档。 |
| `teacher.llm.enabled` | `true` | 是否请求强模型的辅助价值证据；关闭后保留真实结果训练的神经 Teacher。 |
| `teacher.llm.profile` | `glm-5.3` | 开启时必须等于 `strong_model`；设为 null 时自动采用 strong_model。 |
| `teacher.llm.max_calls_per_iteration` | `16` | 一个冷启动阶段或在线轮次内的付费 assessor 请求上限，缓存命中不消耗配额。 |
| `teacher.llm.anchor_fraction` | `0.5` | 真实锚点请求最多使用 `floor(max_calls * fraction)` 次，余量留给短展开尾部查询。 |
| `teacher.llm.max_tokens` | `8192` | 辅助评估输出上限，为强模型的推理和最终 JSON 留出空间；截断回复仍标记无效。 |
| `teacher.llm.max_history_chars` | `16000` | 辅助 prompt 中历史文本的截断长度，保留头部与最近尾部；另有格式标记开销。 |
| `teacher.llm.temperature` | `null` | null 不额外覆盖强模型 profile 的厂商采样设置；显式数值仅覆盖 assessor 温度，不修改 A/B Actor。 |
| `teacher.llm.confidence_floor` | `0.0` | 低于该自报置信度的辅助回复标记无效；此值不是校准概率。 |
| `audit.enabled` | `true` | 是否购买可选的高不确定性／随机完整审计。 |
| `audit.uncertainty_threshold` | `0.1` | 成对效用差的不确定性阈值，同时用于自适应延长和选择性完整审计。 |
| `audit.random_fraction` | `0.2` | 对每组对照额外随机选取完整审计的概率，防止只校准自认不确定的样本。 |
| `audit.max_pairs_per_iteration` | `2` | 每轮可选完整审计组数上限；Teacher 未就绪时的必要真实续跑不受此可选配额限制。 |

强模型只输出预期收益、剩余费用、置信度及简短理由。这些值进入 Teacher 的辅助
输入，**不是训练真值**。Teacher 的目标来自 train 任务原生真实终局和实际尾部费用；
valid 不参与 Teacher 拟合。坏 JSON、截断回复和非法工具调用标记为无效证据，已发生
API 费用仍保留；价格不明会使整轮停止，不能悄悄计为免费。

默认沿用强模型的 thinking 设置，不主动关闭推理。对 GLM-5.3 等厂商模型，
输出预算需要容纳推理与最终 JSON，不能仅按短 JSON 的长度设置；默认温度留空，
避免额外假设其采样参数行为。实际有效回复仍以严格 JSON 解析和完整性检查为准。

数据中的 `policy_version` 是实际冻结路由权重及其继续策略配置的指纹，不是简单的
在线轮次整数。同一轮发生 Router 参数更新后，Teacher 就绪状态和证据缓存需要按
新策略指纹重新校准；不同轮次若实际策略相同，也不应仅因轮次编号不同而视为新策略。

`audit.enabled=false` 仅关闭可选审计，不会允许未就绪 Teacher 伪造标签；必要时仍
完整续跑。要测试完全不使用 Teacher 的方案，使用 `teacher.enabled=false`。

## 优化、评估与保存

| 字段 | 默认值 | 含义 |
|---|---|---|
| `optimization.epochs` | `10` | 每次 Router 拟合的 epoch 数；Teacher 由 `teacher.epochs` 单独配置。 |
| `optimization.batch_size` | `64` | 小网络优化 batch size，与 HF 编码 batch 无关。 |
| `optimization.learning_rate` | `0.001` | Router 的 AdamW 学习率；Teacher 由 `teacher.learning_rate` 单独配置。 |
| `optimization.weight_decay` | `0.01` | AdamW 参数衰减。 |
| `optimization.grad_clip` | `1.0` | 梯度范数裁剪，`0` 关闭。 |
| `optimization.anchor_weight` | `0.1` | Router 更新前后，在回放状态上的预测稳定性正则；目标是更新前的冻结预测。 |
| `optimization.pairwise_weight` | `0.2` | A/B 预测效用差与成对目标差之间的 MSE 权重；`0` 关闭成对损失。 |
| `optimization.cost_loss_weight` | `1.0` | 归一化剩余费用 head 的回归权重；不影响实际计费。 |
| `optimization.replay_limit` | `50000` | 单次学习最多使用的近期输入样本数量。过期策略的伪标签不会继续作为当前监督。归档记录可能更多。 |
| `optimization.pseudo_weight` | `0.25` | 估计目标相对真实目标的基础权重，还会随估计分歧降权。 |
| `evaluation.enabled` | `true` | 是否执行 valid-only 在线评估；关闭后不产生 valid API 调用。 |
| `evaluation.every_iterations` | `2` | 周期 valid 评估间隔；训练结束另补评一次，已评估同一版本不重复付费。 |
| `evaluation.max_tasks` | `null` | null 使用全部 valid 任务；整数仅选其前若干条，并在报告中标明覆盖数。 |
| `checkpoint.every_iterations` | `1` | 周期命名归档快照的间隔；为保证边界恢复，事务性 generation checkpoint 仍在每轮完成后保存。 |

valid 仅计算原生成功率、平均收益、费用与覆盖率，不生成 Teacher 证据、不写入训练
样本、不做梯度更新。`best_valid.pt` 只在选定 valid 集合完整评分后更新，先比较成功率，
相同时比较费用；`latest.pt` 是最新轮次，二者用途不同。

恢复以 `state.json` 指向的 generation checkpoint 为准，并恢复 optimizer、Torch/
Python RNG、回放与实验账本。在一轮内部中断时，从最后完成的边界重新开始该轮，
历史已付费用仍保留，可能购买新的续跑；这不是对外部 API 的 exactly-once 执行。

## 常见消融组合

每个组合都必须使用新的 `output_dir`，并按相同总采集预算比较。

| 消融 | 覆盖项 | 仍保留什么 |
|---|---|---|
| 无 LLM 辅助证据 | `teacher.llm.enabled=false` | 原生真实结果训练的神经 Teacher、短展开和审计。 |
| 完整 A/B 在线基线 | `teacher.enabled=false` | 同状态单步 A/B 干预、冻结路由后缀；全部目标来自真实完整续跑。 |
| 随机选择查询状态 | `selection.enabled=false` | 查询数、短展开、Teacher 和审计。 |
| 固定短展开长度 | `short_rollout.adaptive=false` | 初始 horizon，及允许的完整审计／必要真实续跑。 |
| 无可选完整审计 | `audit.enabled=false` | 就绪 Teacher 的估计监督；未就绪时仍必须真实续跑。 |
| 无费用效用惩罚 | `budget.lambda_=0` | 费用 head 仍可训练，费用账本与运行预算仍有效。 |
| 无成对差值损失 | `optimization.pairwise_weight=0` | 单动作收益/费用回归。 |
| 无预测稳定性正则 | `optimization.anchor_weight=0` | 真实/估计目标与成对监督。 |
| 无固定模型冷启动 | `cold_start.enabled=false` | 从初始化网络直接开始在线主轨迹及当前策略校准。 |

仅设置 `short_rollout.enabled=false` 会停用估计尾部目标，但若 Teacher 仍开启，
它仍可能产生校准计算和辅助评估费用。完整在线基线应使用 `teacher.enabled=false`
明确去掉不需要的 Teacher 费用。
