# Router 学术创新性：基于实现的核查与研究定位

核查日期：2026-09-13。本文区分当前代码、已提出方案和待验证研究，不宣称已实现算法或取得性能收益。

## 1. 结论

当前已实现部分主要是 OpenSquilla V4 推理 bundle 的逐 API step 接入；上一版方案中的 history 编码、候选模型评分、终局监督、剩余费用、数据迭代和 RL，也分别已有先例。仅将这些组件组合起来，方法创新性仍然有限。

最值得进一步研究的主线是：**通过同状态干预估计模型替换相对参考策略的长期质量损失，并在策略实际访问的状态上，将逐步损失与整条任务的质量约束联系起来。**

建议研究定位为“面向长程 Agent 的干预式路由与轨迹级质量损失控制”。这是一项研究提案，既不是已经成立的新算法，也不是经过全面文献查重后的首创声明。

## 2. 核查范围与证据边界

- MTRouter：作者公开仓库当日可读取的 `main`，核对训练脚本、配置、轨迹解析与混合采集代码。远端代码未运行；不把论文默认实验与代码可选功能混为一谈。
- OpenSquilla：本地 `/Users/liangjianfeng/Acadamic/open-squilla`，commit `c16be6206fa104c0a2f84d40d1f8097e7810e74f`；检查的 Router 和 policy 文件没有未提交修改。主要分析 V4 Phase 3 推理及 self-learning 路径。
- Fugu：Sakana 官方技术报告与公开产品资料。官方仓库目前所见主要是文档、报告和接入脚本，不能将这些当作生产训练代码的完整公开。
- OpenFugu：同时检查本地 `/Users/liangjianfeng/Acadamic/OpenFugu-main`。它是独立重实现，不是 Sakana 官方源码；其中某段训练实现不能反推生产 Fugu 的全部行为。
- Ours：当前 `mixed-pilot` 中的适配代码，以及 `ROUTER_REBUILD_PLAN.md` 中尚未实施的方案。

## 3. MTRouter：与我们的重叠比“终局得分回归”更广

### 3.1 已有实现

| 代码位置 | 核查到的行为 | 对创新性的影响 |
| --- | --- | --- |
| `config/research/training.yaml` | 冻结 Qwen3-Embedding-0.6B，history 上限 8192 tokens，模型属性与 residual 表示，小型 Q 网络 | history + candidate scorer 不是新结构 |
| `scripts/train_q_function.py::build_value_targets` | 返回 `score_targets - lambda_ * cost_targets`；默认 `lambda_=0` | 显式加入费用已是可选路径 |
| `research/trajectory/parser.py::get_remaining_cost` | 累加当前 step 到终局的费用 | cost-to-go 不是我们的新增定义 |
| 同文件 `to_training_samples` | 生成终局得分、剩余费用、已执行动作、history 等样本，支持 AEC 调整 | 逐步拆 trace、终局监督、错误信号已有 |
| `research/collection/modes.py::MixedMode` | 混合 learned policy 与随机路由，保留选择概率相关流程 | 不能概括为只支持一次性随机数据采集 |
| 训练脚本的可选 AWR / margin | 按 `(task_id, step_idx)` 分组构造相对分数信号，默认关闭 | 相对评分/排序损失这个名称本身也不足以区分 |

来源：[训练配置](https://github.com/ZhangYiqun018/MTRouter/blob/main/src/miniagenticrouter/config/research/training.yaml)、[训练脚本](https://github.com/ZhangYiqun018/MTRouter/blob/main/scripts/train_q_function.py)、[轨迹解析](https://github.com/ZhangYiqun018/MTRouter/blob/main/src/miniagenticrouter/research/trajectory/parser.py)、[采集模式](https://github.com/ZhangYiqun018/MTRouter/blob/main/src/miniagenticrouter/research/collection/modes.py)。

需要修正上一版文档的表述：不仅 Ridge/XGBoost 对照考虑显式成本，当前神经训练脚本也保留非零 lambda 的成本惩罚变体。默认主方法使用终局得分，不代表实现中没有这条路径。

### 3.2 可以研究的差异

日志中的 `(history, 已选模型, 终局结果)` 可以学习预测价值，但并不自动形成“相同状态下换另一个模型”的成对干预数据。即使随机选择当前动作，状态覆盖、后续策略混合和有限样本仍会影响估计；不能笼统把全部随机日志称为混杂数据。

特别是相同任务的相同步数不一定对应同一 history 与环境。我们的分支方案应使用完整调用前状态、环境恢复验证和明确 continuation，区分任务难度与替换动作的边际影响。

这是采集协议与估计目标的区别，还需要在相同采集费用下证明样本效率和决策收益。仅恢复一次快照、增加 pairwise loss，尚不足以建立强创新性。

## 4. OpenSquilla：完整系统远多于四类分类器

### 4.1 已有实现

V4 Phase 3 的 `build_feature_bundle()` 融合当前用户输入、历史用户输入、前一 assistant、路由历史和手工特征，形成 390 维向量；`InferenceCore.predict()` 调用 LightGBM 与 ONNX MLP，融合四类概率并后处理。它并非只看当前用户问题。

本地 `engine/routing/policy.py` 已包括置信门控、投诉升级、缓存窗口防降级、能力检查和预算门控；`squilla_router/controller.py` 还涉及 thinking mode 与 prompt policy。因此，history、缓存、预算、effort 相关控制都不能作为宽泛的新增功能声明。

本地 self-learning 的具体含义是：

1. `alignment.py::align_session()` 根据实际服务档位、用户反馈、当前/后续投诉等构造修正档位标签。
2. `dataset.py` 对这些标签进行证据加权和去噪。
3. `train.py` 继续训练四类 LightGBM，复用其他 bundle 部件。
4. `evaluate.py::route_metrics()` 计算标签一致率、关键档位不足率和平均预测档位；晋级成本门槛使用档位代理。

这些逻辑与“真实环境分支后，按终局成功和剩余账单监督”有区别。但只能据此描述已核查 self-learning 路径，不能凭当前推理资产推断原始 checkpoint 全部训练数据的来源。

本地证据：[特征构造](/Users/liangjianfeng/Acadamic/open-squilla/src/opensquilla/squilla_router/models/v4.2_phase3_inference/runtime_src/src/router/inference/features.py:15)、[标签对齐](/Users/liangjianfeng/Acadamic/open-squilla/src/opensquilla/squilla_router/self_learning/alignment.py:90)、[增量训练](/Users/liangjianfeng/Acadamic/open-squilla/src/opensquilla/squilla_router/self_learning/train.py:86)、[晋级评估](/Users/liangjianfeng/Acadamic/open-squilla/src/opensquilla/squilla_router/self_learning/evaluate.py:47)、[运行策略](/Users/liangjianfeng/Acadamic/open-squilla/src/opensquilla/engine/routing/policy.py:301)。

### 4.2 可以研究的差异

我们可以将“模仿修正档位”改为“预测动作对终局的影响”，将平均档位代理改为实际任务费用，并验证局部质量信号能否支撑轨迹级目标。这相对该 self-learning 实现是清晰差异，但价值预测和实际费用本身又与 MTRouter 重叠。

另外，当前上游产品文档中的 C3 可以调用多模型融合。比较学习算法时必须固定为相同动作注册表；否则 C3 的额外 proposer/aggregator 改变了执行系统。[官方 Router 文档](https://github.com/TokenRhythm/opensquilla/blob/main/docs/features/squilla-router.md)

## 5. Fugu：必须区分普通版、Ultra 和第三方重实现

### 5.1 官方 Fugu 普通版

技术报告 §3.1 描述的是 decision-only 模型选择：预训练骨干的 hidden state 接轻量选择头，并适配部分奇异值尺度。SFT 阶段根据各 worker 多次运行的平均得分构造 softmax 软目标，以 KL 训练；随后在真实多轮 Agent 任务上，用 sep-CMA-ES 优化终局结果。普通版不分配 Trinity 的角色。

因此，轻量选择头、soft-label SFT、读取多轮状态、SFT 后做终局优化，都已有直接先例。报告 §3.2 的 Fugu-Ultra 才是生成子任务、worker 和通信关系的 GRPO 编排路径。[Fugu 技术报告](https://arxiv.org/html/2606.21228v1)

当前官方产品还包含明确面向成本—性能的 Fugu Max，所以不能根据早期报告宣称整个 Fugu 产品线只优化准确率。其最新内部算法未完整核实，不能凭产品描述断言它使用或不使用某种具体风险约束。[官方模型说明](https://console.sakana.ai/models)

### 5.2 本地 OpenFugu 的实际实现

`openfugu/mini.py::FuguRouter.route()` 从 hidden state 产生 agent/role logits；`Coordinator.run()` 依据演化中的观测逐步选择 worker 与角色。`train/train_trinity_perstep.py` 用整个多轮循环的 GSM8K 终局数值匹配率作为 sep-CMA-ES fitness。因此，这份实现已有真正的逐步重选，并非全部是任务开始时只选一次。

但具体脚本的实验能力应审慎解释：默认 8 个训练问题、4 轮；示例 worker 缓存 key 只取末条消息前 200 字符，可能把不同完整状态合并。正式对照应修复完整输入缓存键，并建立独立测试集，不能把训练集 fitness 当成泛化证据。[本地逐步训练脚本](/Users/liangjianfeng/Acadamic/OpenFugu-main/train/train_trinity_perstep.py:63)

本地 `train_conductor.py` 所接的 `toolscale_data.py` 奖励由计划格式、工具名称/顺序和参数匹配组成；该奖励不是执行整个真实工具环境后的成功率。因此，不能把这一路径的 reward 上升等同于部署任务成功率上升。[本地奖励实现](/Users/liangjianfeng/Acadamic/OpenFugu-main/train/toolscale_data.py:110)

上述是第三方代码的具体边界，不是对官方 Fugu 能力的判断。[OpenFugu 项目声明](https://github.com/trotsky1997/OpenFugu)

## 6. 哪些主张目前不足以成为核心创新？

| 候选主张 | 判断 |
| --- | --- |
| 每次 API 调用根据 history 选模型 | MTRouter、Fugu 和相关逐步路由已有 |
| 从四档变成模型/effort 配置 | 动作表示或动作空间扩展，本身较弱 |
| 小编码器 + MLP / 双头 | 常规结构；双头有实用价值但不足以单独构成贡献 |
| 终局监督、剩余费用、质量减费用 | MTRouter 实现已有相关目标和数据字段 |
| SFT 冷启动后 RL/ES | Fugu 已有；更换优化器通常只是实验变量 |
| 自学习、数据聚合 | OpenSquilla 和其他模仿学习/路由工作已有先例 |
| 缓存、预算、失败回退 | 已有系统功能；需提出并验证新的决策机制才可能升级为研究贡献 |
| 强模型轨迹单步替换 | 与 AggreVaTe 等后续损失监督思想重叠，需承认来源 |
| 加一个质量置信阈值 | 风险控制路由已有；不能直接声称首次可靠路由 |
| 新数据、更多任务、公开 checkpoint | 可成为实证/资源贡献，但不自动构成方法创新 |

相关先例：[AggreVaTe](https://arxiv.org/abs/1406.5979)、[Conformal LLM Routing](https://aclanthology.org/2026.acl-srw.70/)、[Conformal Arbitrage](https://proceedings.neurips.cc/paper_files/paper/2025/hash/65a655c5a267f678fd3e897e4137ef53-Abstract-Conference.html)。这不是全面查重清单。

## 7. 建议主贡献：从单步替换标签走向轨迹级质量损失控制

### 7.1 先明确所估计的量

选定固定参考策略 `π_ref`，在相同调用前 history 和环境状态下，比较“当前用参考动作”与“当前用候选动作”，之后都继续使用 `π_ref`。定义：

\[
d^{ref}(h,a)=V_Y^{ref}(h)-Q_Y^{ref}(h,a).
\]

`d` 是候选动作相对参考策略的预期终局质量损失，可以为负；它不是当前输出相似度，也不是局部工具成功率。确定性参考策略下，参考动作的真实 `d` 等于零。

训练侧使用同状态分支重复采样获得差值及不确定性，部署侧只用调用前可观测特征预测。真实环境状态用于恢复和验证，不能把隐藏答案变成输入。

同时估计成本-to-go 时，必须标记 continuation。强参考续跑测得的费用不是当前混合策略的部署费用；费用头可通过当前策略 rollout 更新，最终按实测完整任务费用比较。

### 7.2 强 continuation 并非只能用于冷启动

上一版方案正确指出 `Q^ref ≠ Q^π`。但还有一条可研究的路线：通过参考策略 advantage 将局部差值连接到整体质量，而不是直接把 `Q^ref` 当成混合策略价值。

在共同初始分布、相同环境与终止规则、有限时域、状态包含必要时间信息、Q/V 为真实期望回报等条件下，性能差分恒等式给出：

\[
J_Y(\pi_{ref})-J_Y(\pi)
=\mathbb E_{\tau\sim\pi}\left[\sum_t d^{ref}(h_t,a_t)\right].
\]

关键是右侧状态来自待部署策略 `π`，不是仅来自强模型自己的成功轨迹。这解释了为什么“强轨迹上每次替换都安全”不能推出“连续替换仍安全”。

这个恒等式属于已有策略改进理论，不是我们的新定理；此处给出的是有限时域、终局质量回报下的应用形式。[Kakade–Langford 原论文](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf)

### 7.3 尚待设计的方法部分

可以研究一个显式累计质量损失预算的 Router：在策略访问状态上学习 `d`，利用可信上界对高风险替换保守处理，优先把质量预算用于具有较大费用收益的动作。

理想情况下，若有覆盖所有实际访问状态/动作的有效上界 `u(h,a) ≥ max(0,d(h,a))`，并使每条路径满足 `Σu ≤ δ`，由上述恒等式可得到期望质量退化不超过 `δ`。真实学习系统还必须处理估计误差、上界失效概率、状态分布漂移、可变时域以及与货币预算的冲突；不能直接把一个 sigmoid 概率或普通验证集阈值当成这种上界。

因此，可能的研究贡献在于：**构造可用的估计与控制算法，明确有限采样下的条件和误差项，并证明其比固定每步阈值或固定 lambda 更有效。** 仅重新表述性能差分恒等式不构成创新。

与单请求 conformal gate 的比较也必须清楚：我们的目标是动作会改变后续状态的长程 Agent，控制相对参考的终局质量下降；已有单请求方法所控制的错误事件、条件概率与采样假设不必与之相同。不能直接套用其保证，也不能忽略已有风险控制工作。

## 8. 配套贡献：采集预算下的分支选择与替换交互分析

### 8.1 分支采集不必在所有状态全模型穷举

可先粗采样，再把额外 rollout 分配给“动作排序可能改变、风险接近阈值、且潜在节省较大”的状态。目标是单位采集费用带来的部署质量—成本改善，而不是单纯降低价值回归 MSE。

主动采样本身已有大量研究。只有给出针对终局路由风险的分配准则，并在相同总采集费用下优于均匀分支、随机日志和不确定性采样，才可能形成独立贡献。重复采样次数自适应变化时，还需避免不恰当的固定样本置信区间和可选停止偏差。

### 8.2 测量“单次替换安全、组合替换失败”

以下只是说明性假设，不是实验结果：参考全强成功率 1.00，单独替换某两个位置分别为 0.99 和 0.99，同时替换却只有 0.60。强模型后续补救掩盖了单次替换的潜在影响。

应比较单次干预、连续干预，以及 Router 自己访问状态上的干预。多步干预的位置/触发事件和提前终止处理必须事先定义，不能将不同轨迹的同一步数当作相同环境状态。

若能系统刻画这种交互、发现它在何类工具/任务上显著，并证明所提控制器能够缓解，就有实证价值。反之，若该效应很弱，研究应转向更简单的评分或任务级路由，不必强行维持复杂方法。

## 9. 最小论文实验包

| 实验 | 必须回答的问题 |
| --- | --- |
| 同编码器、同训练数据：普通 Q / 双头 Q / 相对损失预测 | 改进来自目标还是模型容量？ |
| 同采集费用：随机日志 / 均匀分支 / 主动分支 | 干预数据是否值得额外费用？ |
| 同状态 vs `(task_id, step_idx)` 分组 | 精确恢复对信用分配是否重要？ |
| 仅强轨迹状态 vs learner 访问状态 | 分布覆盖能否解释连续路由收益？ |
| 固定 lambda / 固定每步阈值 / 累计损失控制 | 轨迹级控制是否真正降低质量违规？ |
| 参考/当前策略 continuation 与费用更新消融 | 是否混淆后续策略导致价值估计失准？ |
| 缓存关闭/开启，短/长 horizon，修复密集任务 | 控制器对真实执行成本是否稳健？ |
| 多任务域、固定模型池、独立最终测试集 | 收益能否泛化，而非单任务记忆？ |

强对照至少包括 MTRouter 默认与成本变体、统一动作的 OpenSquilla、Fugu-style SFT+ES 重训版本，以及固定模型/随机/规则基线。Fugu-style 是方法级重实现，应标注与生产服务差别；生产 Fugu API 若加入则单列黑盒系统比较。

质量目标在冻结策略后做任务级配对评估，报告置信区间；有限样本证据不足时明确写探索性结果。训练账单、Router 运行开销、失败任务费用和采集摊销都要报告。

## 10. 建议写进开题/论文的表述

> 我们研究固定 Agent 执行框架中的逐调用模型路由，重点解决局部替换评价与整条任务质量之间的偏差。我们拟通过可恢复环境中的配对分支估计参考策略相对损失，在学习策略实际访问的状态上更新估计，并设计累计质量损失约束下的成本优化机制。实验将检验该机制在相同采集预算和模型池下，是否比终局价值回归、软标签路由及固定权衡策略更可靠地控制长程任务的质量退化。

当前可确认的是工程可行性与研究问题，不是学术创新已经成立。建议先完成一个验证替换交互与参考损失估计的小实验，再决定是否投入完整风险控制算法与理论分析。
