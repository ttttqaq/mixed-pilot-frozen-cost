# KSR 数据集适配与 MTRouter 训练总结

本文记录 `ksr` 分支上为 mixed-pilot 驱动的 MTRouter 增加的 KSR `samples.jsonl` 训练支持，以及之后采集数据时需要遵守的格式约定。

## 这次适配改了什么

- `MTRouter/src/mixed_pilot_mtrouter/research/trajectory/parser.py`：增加 `find_training_sample_files()`，递归发现数据目录中的 `samples.jsonl`。原有 `.traj.json` 发现逻辑保留。
- `MTRouter/src/mixed_pilot_mtrouter/research/training/dataset.py`：将 JSONL 每行转换为一条训练样本：读入对话 history、执行模型、`y` 监督标签和调用成本；忽略无效标签、超出范围的标签和模型池外的模型。兼容 `action_model` / `action_profile` 两种模型字段。
- `MTRouter/src/mixed_pilot_mtrouter/audit.py`：支持预检查 `samples.jsonl`，检查模型名、标签、成本、币种一致性并报告样本量。
- `MTRouter/src/mixed_pilot_mtrouter/config/research/training.yaml`：当前训练池配置为 `qwen-flash-2025-07-28` 和 `deepseek-flash`；Embedding 模型指向本地 Qwen3-Embedding-0.6B；训练设备、early stopping 和 embedding 预计算等参数集中在此配置。
- `MTRouter/src/mixed_pilot_mtrouter/cli/train_q_function.py`、`research/training/encoders.py` 与 `cli/precompute_embeddings.py`：新增 `auto`、`cpu`、`cuda`、`cuda:N` 设备选择，训练与本地 HuggingFace embedding 预计算使用同一设备。命令行 `--device` 可覆盖 YAML 配置。
- `MTRouter/tests/test_core.py`：同步当前双模型配置并覆盖相关核心训练流程。

上述改动均在 `ksr` 分支；没有合并到 `main`。当前工作区改动尚未提交。

## 训练数据文件布局

训练入口会递归查找每个 `--data-dir` 下的 `samples.jsonl`，例如：

```text
ksr-random-5yuan/
├── ksr-branches/samples.jsonl
└── random/samples.jsonl
```

文件是 UTF-8 编码的 JSON Lines：**每一行是一个独立 JSON 对象**，不是一个 JSON 数组。可将 KSR 和 random 等来源放在不同子目录下共同训练；避免把相同文件通过重叠目录重复传入。

## 每行记录的格式

建议采集时至少提供以下字段：

| 字段 | 要求与用途 |
| --- | --- |
| `state.history` | 必填的对话历史数组；元素使用 `{"role":"user", "content":"..."}` 这样的消息结构，按实际时间顺序排列。它会被 Qwen embedding 编码，是路由状态的主要文本输入。 |
| `action_model` | 必填；本次实际调用的模型名，必须与训练配置 `model_pool` 中某一项完全一致，或使用能被解析为同名的 provider 前缀形式。也兼容旧字段 `action_profile`。 |
| `y` | 必填的目标标签，必须是有限数值且位于 `[0, 1]`。当前代码将它直接作为该状态-模型对的监督目标，不会从整条 trajectory 自动推导标签。 |
| `task_id` | 强烈建议；同一任务实例应保持稳定，用于按任务划分训练集和验证集，避免同一任务的数据同时泄漏到两边。缺失时会被记为 `unknown`，不适合作为正式数据。 |
| `run_id`、`step_idx`、`sample_id` | 强烈建议；用于标识一次运行、决策步和样本，帮助追踪样本与避免身份混淆。缺少时 loader 会使用默认值或生成 sample ID。 |
| `labels_valid` | 建议显式记录；若值为 `false`，该行会被跳过。若标签未知，使用 `y: null`，同样不会用于训练。 |
| `cost` | 建议记录非负的有限数值及明确币种。当前 JSONL audit 将 KSR/random 数据视作 CNY；JSONL loader 把它当作当前模型调用成本，不是整条轨迹未来剩余成本。 |

推荐同时保留 `state.system`、`response`、`observation`、`selection_probability` 等字段，便于数据追溯和后续扩展。当前实现中 `state.system` 会在 history 尚无 system 消息时补入；`response`、`observation` 和 `selection_probability` 会被读取或保留为样本元数据，但不是当前 Q 网络的主要监督输入。

示例（每条记录单独占一行）：

```json
{"sample_id":"task-001:run-01:step-0","task_id":"task-001","run_id":"run-01","step_idx":0,"state":{"system":"Solve the task.","history":[{"role":"user","content":"Please solve ..."}]},"action_model":"qwen-flash-2025-07-28","response":"...","observation":"...","y":0.8,"cost":0.012,"labels_valid":true,"selection_probability":0.5}
{"sample_id":"task-001:run-01:step-0-deepseek","task_id":"task-001","run_id":"run-01","step_idx":0,"state":{"system":"Solve the task.","history":[{"role":"user","content":"Please solve ..."}]},"action_model":"deepseek-flash","response":"...","observation":"...","y":0.6,"cost":0.004,"labels_valid":true,"selection_probability":0.5}
```

## 采集标签时尤其要注意

1. **同一决策状态的候选模型要可比较。** 如果目标是让 Router 选择两个模型中更合适的一个，应尽量为同一任务、同一运行上下文和同一决策状态收集不同候选模型的结果；对应样本的 `task_id`、`step_idx`、history 应能明确对齐。不同模型各自处理不同输入时，模型效果差异可能与任务难度混在一起。
2. **明确 `y` 的含义并统一标度。** 当前代码不推断 `y` 是最终成功率、奖励还是人工评分；它只接受 `[0,1]` 并直接回归。若原始分数不是该范围，应先定义一致、可复现的归一化方式，并在数据说明中记录。
3. **不要把未知标签伪装成 0。** 0 是合法监督值。无标签/无效标签请用 `labels_valid: false` 或 `y: null`，避免模型把缺失数据学成失败结果。
4. **确保模型名一致。** 新增候选模型前，先同步更新 `training.yaml` 的 `model_pool`，并让数据里的 `action_model` 使用对应名称。模型池不一致的数据会被跳过或在 audit 阶段报错。
5. **成本必须语义统一。** 当前配置 `lambda_: 0.0`，所以训练目标暂时不扣除成本；若未来要训练质量-成本权衡，需要明确 `cost` 单位/币种、范围和它代表当前调用还是整条剩余成本，并同步设置目标构造逻辑，不能直接混用不同定义或币种。
6. **保留完整 history 和稳定身份。** 对话消息缺失、乱序或 role/content 不规范会改变 embedding 输入；任务 ID 不稳定则会削弱按任务隔离验证集的作用。

## 预检查与训练

先对每个数据目录执行审计（模型列表须与训练配置一致）：

```bash
PYTHONPATH=MTRouter/src python -m mixed_pilot_mtrouter.audit \
  --data-dir ksr-random-5yuan/ksr-branches \
  --data-dir ksr-random-5yuan/random \
  --models qwen-flash-2025-07-28 deepseek-flash
```

再训练：

```bash
conda activate mixed-pilot
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m mixed_pilot_mtrouter.cli.train_q_function \
  --config MTRouter/src/mixed_pilot_mtrouter/config/research/training.yaml \
  --data-dir ksr-random-5yuan/ksr-branches \
  --data-dir ksr-random-5yuan/random \
  --output-dir outputs/mtrouter-ksr-random-5yuan \
  --device cuda:0
```

显卡也可以在配置文件 `training.device` 设置；`--device` 优先级更高。`precompute.batch_size`（或 `--precompute-batch-size`）只控制 embedding 预计算一次编码多少条，不是 Q 网络训练的 batch size。

## 如何判断训练产物

- `training_log.jsonl`：逐 epoch 的训练/验证损失和最佳轮次标记。
- `q_function_best.pt`：验证损失最低的 checkpoint；部署/后续评估通常优先使用它。
- `q_function.pt`：训练停止时最后一轮的 checkpoint，不保证是最佳模型。
- `split.json`：训练/验证任务划分；验证是按 `benchmark + task_id` 分组，减少同一任务跨集合泄漏。

本次已有运行的最佳验证损失约为 `0.2297`（第 5 轮），训练集均值常数预测在该验证集上的 MSE 约为 `0.2312`，提升很小；因此当前只能确认训练流程运行完成，不能据此证明路由效果良好。验证集只有 3 个任务，且现有日志记录的是回归 MSE，不是路由准确率或 regret。正式判断效果前应增加任务/运行覆盖，并评估同一状态不同模型间的选择准确率、平均 regret 等路由指标。
