# Router profile 与逐步推理

`harness.json` 已提供 `random-router` 和 `mtrouter` 两个 profile。
两者都是 `kind: "router"`，模型池成员引用已有的具体 LLM profile。
原 `router`（OpenSquilla V4）、普通单模型、`--mode random` 和 KSR 入口保留。
本文的 dataset Router 入口支持 RandomRouter 和 MTRouter；OpenSquilla V4 继续使用原有 chat/code-task 入口。

## RandomRouter

```bash
python scripts/infer.py \
  --dataset /Users/liangjianfeng/Data/Router/SWE-neo-data --split train \
  --mode router --router-profile random-router \
  --models qwen-flash-2025-07-28 deepseek-flash glm-5 \
  --seed 42 --output-dir run/random-router
```

默认每次 LLM 调用前均匀随机选一个 profile，概率为 `1 / 模型池大小`。
可省略 `--models`，使用 profile 内的模型池。`--seed` 控制 infer 的随机序列，默认 42。
也可用 `--mode random-router --models ...` 而不指定 Router profile。
旧的 `--mode random --models ...` 仍按原逻辑运行。

### 按 profile 配置 possibility

在 `harness.json` 的 RandomRouter profile 的 `router` 对象中配置：

```json
{
  "engine": "random",
  "models": ["qwen-flash-2025-07-28", "deepseek-flash", "glm-5"],
  "possibility": {
    "qwen-flash-2025-07-28": 6,
    "deepseek-flash": 3,
    "glm-5": 1
  }
}
```

每个 step 的概率分别是 60%、30%、10%。`possibility` 是相对权重，
也可填 `0.6、0.3、0.1`，不要求总和为 1。省略整个字段时均分概率。
配置该字段时必须覆盖模型池全部 profile，不允许多余名称、负数、非有限数或全零；
单个权重为 0 表示不选该模型。全部权重相等时仍使用原来的均匀随机序列。

infer 命令行可覆盖配置，也支持不使用 router profile 的 `--mode random-router`：

```bash
--router-possibility qwen-flash-2025-07-28=6 deepseek-flash=3 glm-5=1
```

若 `--models` 更换模型池，已有 possibility 必须与新模型池匹配，或同时传入新的
`--router-possibility`。该参数仅用于 RandomRouter，旧 `--mode random` 不变。
TOML 配置可在 `[router.possibility]` 中按 profile 名设置权重。

`router_decision.probabilities`、`selection_probability`、导出 step 的 `propensity`
均记录实际归一化概率。权重与随机数状态随 checkpoint 保存；恢复不会重抽历史步骤。
修改权重（包括从均匀改成非均匀）需要使用新的输出目录，避免一次实验混用不同策略。

## MTRouter：直接加载 checkpoint

```bash
python scripts/infer.py \
  --dataset /Users/liangjianfeng/Data/Router/SWE-neo-data --split train \
  --mode router --router-profile mtrouter \
  --router-checkpoint /absolute/path/q_function_best.pt \
  --router-device cpu \
  --models qwen-flash-2025-07-28 deepseek-flash glm-5 \
  --output-dir run/mtrouter
```

也支持 `--mode mtrouter --models ... --router-checkpoint ...`。
checkpoint 在开始任务前加载一次，每一步根据当前完整消息历史计算 Q 值，选择 argmax 对应的 profile。
不是每步重新加载权重。当前 Python 环境需要安装本项目 MTRouter 包及其依赖：

```bash
python -m pip install -e ./MTRouter
```

checkpoint 不包含冻结的历史 encoder 全部权重，所引用的 encoder/模型元数据仍需可用。
`--router-device` 可选 `cpu`、`cuda`、`mps`。
vLLM checkpoint 可使用 `--router-embedding-url`，迁移模型元数据文件可使用
`--router-model-metadata /absolute/path/custom_models.yaml`。

**模型池必须与 checkpoint 的模型名称、顺序一一对应。** 默认用各 profile 的 `model` 字段匹配。
如果训练使用的名称与实际 API model 不同，显式提供对应名称，不能随意冒充另一个训练模型：

```bash
--models profile-a profile-b --router-model-names trained-model-a trained-model-b
```

当前不支持对任意新模型零样本扩池，也不自动裁剪 checkpoint 模型池。名称/顺序不一致会报错。
本地加载会在调用付费 LLM 前验证；HTTP 服务会在每次路由时验证返回名称、全部评分及 argmax。

## MTRouter：独立评分服务

依赖需要隔离时，在安装 MTRouter 的 Python 环境启动：

```bash
mtr-core-serve --checkpoint /absolute/path/q_function_best.pt --device cpu --port 8765
```

infer 使用 `--router-service-url http://127.0.0.1:8765`，或直接使用 `mtrouter` profile
内的默认服务地址。它和 `--router-checkpoint` 互斥，显式参数会替换 profile 内的另一种运行方式。
服务只负责评分，harness 调用选中的 LLM。外部服务的权重生命周期由启动服务的一方管理；
续跑期间应固定服务 checkpoint，harness 无法仅凭 URL 校验远程权重是否被替换。

## 在 harness.json 配置

```json
"my-router": {
  "kind": "router",
  "model": "my-router",
  "router": {
    "engine": "mtrouter",
    "models": ["qwen-flash-2025-07-28", "deepseek-flash", "glm-5"],
    "checkpoint": "run/mtrouter/model/q_function_best.pt",
    "device": "cpu"
  }
}
```

RandomRouter 使用 `"engine": "random"`、`"models": [...]` 和可选 `"seed": 42`。
不填写 checkpoint 或 service_url。禁止 Router 嵌套、重复 profile、未知 profile。
配置中的相对路径基于 harness.json；CLI 路径基于当前工作目录。
旧 `router_config` TOML 仍支持，dynamic engine 的 `[tiers.*]` 引用 pool profile。
infer 要求 tier 不覆盖 profile 参数；需要不同模型参数时另建一个有名字的具体 profile。

Router profile 也可交给原有 `mixed-pilot chat/code-task --profile my-router`。
infer 同样接受 `--mode single-model --models my-router`，会识别并转为逐步 Router 模式。

## 记录、计费与续跑

- 池内每个具体 profile 必须有 API 配置、凭据及价格。Router profile 本身不需要虚构一个 token 单价。
- `calls/*.json`、trace、`steps.jsonl` 保存实际 profile/model、选择概率和 `router_decision`；MTRouter 额外保存 Q 值。
- 费用、模型使用统计按实际调用的具体 profile 计算。Q 值不是概率；argmax 的条件选择概率记录为 1。
- 本地 Router 算力和外部 embedding 服务的费用不计入 LLM token 账本。
- HTTP 临时路由错误遵循 `--max-retries` / `--retry-wait`。非法评分、模型池不匹配不重试，不静默切换默认模型。
  路由失败而尚未调用 LLM 时费用已知为零，当前任务保留 partial，不因未知 API 费用而中止整批。
- RandomRouter 的 RNG 状态随 checkpoint 保存。重放历史动作不重新路由、不再调用 LLM、不重复计费。
- 续跑保持 Router 算法、模型池、模型映射和本地 checkpoint 内容不变；文件 SHA-256 纳入运行配置，原地替换权重也会拒绝续跑。
  数据集扩容、partial 重试、预算及 trace 间隔沿用现有规则。
- 追加 `--dry-run` 可验证配置、价格、checkpoint 文件存在并记录文件哈希；不会加载神经网络或调用评分服务/LLM。
  checkpoint 内部结构和训练模型池在实际启动时进一步验证。

MTRouter 沿用训练端的 role/content 历史表示，包含任务文本和已有工具观测；结构化
`tool_calls` 不额外拼进 content，避免悄悄改变已有 checkpoint 的输入分布。
若要编码完整工具参数，需要同步修改训练与推理编码并重新训练。
trace 消息补充模型、概率及费用字段，便于 MTRouter 读取；训练仍需筛选有效、非 superseded 的轨迹。
