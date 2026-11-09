# KSR 冷启动后冻结 C，在线只学习 Y

分支：`feature/frozen-cost__from_ksr-coldstart`。父分支：
`feature/ksr-coldstart__from_reference-teacher`，父提交 `028e684`。

入口为 `scripts/train_frozen_cost_router.py`，配置为
`configs/online-router.frozen-cost.db94.yaml`，默认输出为
`outputs/ksr-frozen-cost-router-run01`。原 KSR 入口和配置仍保留原方法。

## 方法与冷启动停止规则

冷启动仍使用 KSR 分枝、原生 success 标签、预期 Actor 后缀费用标签和原 Y/C 联合损失。
`trajectory_steps=500` 在新入口中是**启动新任务的停止阈值**：达到 500 次新增 Actor/environment
交互后，完成当前 task 队列中的所有 KSR 分枝及其强/弱起始树，再结束采集；不启动下一个 task。
重放共享前缀仍不算新增 step。最终数量可超过 500，以实际数量为准。

100/200/300/400/500 等里程碑仍保存权重和数据快照；如果收尾到第 516 步，再保存
`step_000516.pt/.json`。500 步时尚未收尾的权重仍处于联合训练阶段；**最终收尾并完成最后一次
Y/C 拟合之后**才冻结 C。断点恢复保留当前任务队列和唯一调用账本，不重复购买已完成交互。

每条路径 50 次 Actor 调用、同 task＋策略版本全部分枝合计 1 CNY、总预算 80 CNY 仍有效。
“跑完当前任务”受这些限制约束。共享采集预算不足或重放失败的未完成分枝仍保留未知，
不会伪造失败标签；这类截断与人为卡在第 500 步的截断分别处理。

冷启动结束后复制完整三网络 ensemble，包括 model embedding、共享 MLP 和 C 输出层，
设为 eval、关闭梯度，作为固定 C 预测器；在线模型的共享 MLP、model embedding 和 Y 输出层
继续接受 Y 监督，在线 C 层不更新。历史编码器继续冻结并共用。固定的是整个函数：
相同历史、预算、步数和模型输入，在不同在线轮次给出相同 C；不同输入可给出不同 C。

在线损失只有加权 Y 平方误差及原有模型 embedding 正则，没有 C、U 或 pair-ranking 损失。
真实 Y 权重 1，Teacher Y 权重 0.25。Teacher 只输出 expected_reward、assessment_status、reason；
保留原 train 参考隔离、有限重试及缺失证据时的真实 continuation rollout 兜底。
主动查询、短 rollout 扩展和审计依据 Y 的分歧/差值，不要求 Teacher 费用输出。
真实 Actor/Teacher 费用仍完整记账，用于预算和报告，不能用预测 C 替代账单。

选模：`U = Y_online - beta * C_cold`，C 单位为 CNY。
建议第一组继续 `decision.beta=0.5`：多花 0.1 元需要增加 5 个百分点预测成功率。
`optimization.cost_loss_weight` 仅作用于冷启动。`audit.cost_threshold` 在新方法在线阶段不使用。
冻结后的 C 是冷启动成本估计，不再随在线后续策略校准；是否改善真实效果需要正式对照验证。

## 在 DB94 运行

```bash
cd /DB/data/yujietang/Agent/code/mixed-pilot
git switch feature/frozen-cost__from_ksr-coldstart
PY=/DB/rhome/yujietang/miniconda3/envs/mixed-pilot/bin/python
CFG=configs/online-router.frozen-cost.db94.yaml

# 静态预检：不初始化模型或环境，不发起 API，不创建输出。
"$PY" scripts/train_frozen_cost_router.py --config "$CFG" --dry-run

# 完整离线 smoke：合成环境/回复，hash 特征，不是任务效果测试。
"$PY" scripts/train_frozen_cost_router.py --config "$CFG" \
  --smoke --output-dir outputs/frozen-cost-my-smoke

# 正式新运行：先用 nvidia-smi 选择空闲 GPU，将 7 改为所选卡。
# 凭据仅在进程内读取，避免写入新配置和 checkpoint。
CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
"$PY" scripts/train_frozen_cost_router.py --config "$CFG" \
  --credentials-config configs/online-router.reference-teacher.db94.yaml
```

正式命令增加 `--cold-only` 可以先采集、收尾、拟合并评估冷启动；随后去掉 `--cold-only`，
增加 `--resume` 继续在线训练。恢复必须使用相同参数、配置、源码、价格和数据。

调 β 时在首次运行增加 `--set decision.beta=1.0` 并指定新的 `--output-dir`；恢复时保留相同
override。推荐只用训练/验证划分调参，避免使用重复查看的 test 选择 β。默认仍沿用旧评估安排：
冷启动结束、第 3 和第 5 轮评估 test；test 不进训练/Teacher，但参与 best_eval 选择。

## 显式复用以前的冷启动权重

如需直接从原 run01 的 500-step 冷启动权重开始，可在**新的**正式命令追加：

```bash
--init-cold-checkpoint outputs/ksr-reference-router-run01/cold_start/checkpoints/step_000500.pt \
--output-dir outputs/frozen-cost-from-old-cold-run01
```

这不会追加采集原缓存中未完成的任务；它显式接受以前的冷启动快照，原样复制其参数及优化器，
冻结 C 后重新评估并开始新在线训练。新的“500 后完成当前任务”规则适用于新采集的冷启动。
入口会校验原运行的数据/参考划分、harness/价格/metadata 指纹、模型池、编码器、费用单位、
调用限制、KSR 规则及配套数据快照。旧 online/latest/best 权重不能代替该冷启动快照。
权重及来源 SHA256 记入新 checkpoint；原运行费用不搬入新账本，初始化的新增冷启动 Actor
请求为 0，后续评估/在线仍计费。恢复时保留相同 `--init-cold-checkpoint` 参数。

新方法自己产生的完整 cache 也可用 `--reuse-cold-start <source-run>` 重新拟合联合冷启动，
不会重新采集；要求新的任务收尾语义、源码和配置指纹兼容。原方法 cache 不能直接混用。

## 结果检查

- `training_log.jsonl`：在线 objective 为 `Y_only_frozen_C`，cost_loss 为 0。
- `summary.json`：实际 cold_trajectory_steps、阈值 cold_step_threshold、cost_frozen 和
  cost_model_fingerprint。各在线轮次应使用同一成本模型指纹。
- `teacher_predictions.jsonl`：有效估计只包含 Y，不包含 cost。
- `pairs.jsonl`：targets 只有 Y；frozen_costs 是根状态的冻结预测，明确标为预测值。
- `latest.pt`、`iterations/`、`best_eval.pt`：都内嵌完整冻结副本；恢复不从在线模型重建 C。
  用 `FrozenCostLearner.load` 读取；旧 LearningSystem loader 会拒绝静默忽略冻结副本。

独立评估用新入口加 `--evaluate-checkpoint <new-method-checkpoint>` 和新的 `--output-dir`，
不提供 `--init-cold-checkpoint`。无需 Teacher key，模型池、限制和 β 必须匹配 checkpoint。

离线测试命令：

```bash
PYTHONPATH=src:MTRouter/src "$PY" -m pytest \
  MTRouter/tests/test_frozen_cost_router.py MTRouter/tests/test_ksr_reference_router.py -q
```

离线验证只证明数据流、冻结、梯度、恢复和任务收尾行为，不证明真实成功率或费用改善。

## 本次验证（2026-11-03）

MTRouter 全套：132 passed、30 subtests passed，其中新方法 14 项测试覆盖 C 参数/输出不变、
Y/共享层可更新、Y-only Teacher、五轮在线、保存恢复、冷启动收尾及缓存/旧权重初始化。
合成 500-step 阈值测试在 504 步结束，504 条标签全部有效；中间 500-step 权重未提前冻结。
正式配置 dry-run、完整 CLI smoke、cold-only→resume、缓存复用、独立评估和 β override 均通过。
验证记录在 `outputs/frozen-cost-validation-20261103/`，没有新增付费 API 请求，也未启动正式训练。

额外运行全仓测试得到 410 passed、141 subtests passed、14 failed；失败涉及旧 SWE 工具的
Python/Linux 兼容、缺失的 SWE 数据和既有 harness 断言。在父提交 `028e684` 的隔离源码上
复现相同 14 项失败，未修改这些无关模块。此全仓检查之后只增加了一项新方法缓存阈值测试，
并完成上述最终 MTRouter 全回归。
