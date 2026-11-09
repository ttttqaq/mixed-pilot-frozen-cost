# MTRouter core for mixed-pilot

已从本地 `/Users/liangjianfeng/Acadamic/MTRouter-main` 移植训练与推理核心。
这是可独立安装的包 `mixed-pilot-mtrouter`，导入名为 `mixed_pilot_mtrouter`。
运行时不再依赖原仓库路径，不占用上游 `miniagenticrouter` 命名空间。

## 范围与依赖隔离

```text
现有 harness 环境                     独立 MTRouter 环境
provider / 工具 / ScienceWorld    HTTP    PyTorch / Transformers
RouterProvider -----------------------> NeuralQFunction.predict
               模型名 + Q-values <----- checkpoint + history encoder
```

迁入：轨迹解析、HistoryEncoder、ModelEncoder、单头 QNetwork、NeuralQFunction、
embedding 预计算、训练循环、可选 AEC 错误惩罚、checkpoint 保存/加载。
附加：只读数据审计、按任务实例划分训练/验证、本地 HTTP 评分服务、harness bridge。

保留上游终局监督目标：`(final_score + 100) / 200`，默认 `lambda_=0`。
没有把 per-step judge 分数替换为训练标签。Q-values 不是概率，不输出伪造的 softmax 置信度。

没有迁入上游 provider、LiteLLM、CLI agent、环境采集器、XGBoost/Ridge 比较基线。
这些不是神经训练和评分所需的依赖。harness 继续负责实际模型调用和工具执行。
已有 collection CLI 的 baseline/roulette 采集语义保持不变；本次没有新增 `collect --mode learned`。

依赖注意事项：

- 根项目仍支持 Python 3.9+；独立包要求 Python 3.10+，建议单独用 3.12。
- 根项目 `setup.cfg` 不增加 torch/transformers 等依赖，也不要求安装原 MTRouter。
- 独立包需要 torch、transformers、numpy、PyYAML、tqdm；vLLM 客户端和 TensorBoard 为可选 extras。
- 不安装 vLLM 服务端；Mac 默认使用 HuggingFace CPU。CUDA 可用时预计算/训练自动用 CUDA。
- 本地预计算已去除上游硬编码 `.cuda()`，无 CUDA 时使用 CPU；默认 FP32 与在线 HF 编码一致。
- 完整上游安装会额外引入 litellm/openai/scienceworld 等；现有项目的 ScienceWorld extra 固定 1.2.3，独立核心不声明 ScienceWorld 依赖，避免安装时改变现有环境。
- `constraints-tested-macos-py312.txt` 记录验证时已有的核心依赖版本，不代表已验证所有平台、所有可选依赖组合。

## 安装（在项目根目录执行）

使用独立 venv，不要向现有 opensquilla 环境安装这个包的依赖：

```bash
# 换成你的 Python 3.12 路径；如果系统 python3 是 3.9，请勿使用它。
PYTHON_BIN=/Users/liangjianfeng/miniconda3/envs/opensquilla/bin/python
"$PYTHON_BIN" -m venv MTRouter/.venv
MTRouter/.venv/bin/python -m pip install -e ./MTRouter
```

venv 默认不继承源环境的 site-packages。以上使用 opensquilla 的 Python 可执行文件建环境，
不会往 opensquilla 安装或升级依赖。实际安装需要联网下载依赖；移植过程中没有运行该安装命令。

可选：`MTRouter/.venv/bin/python -m pip install -e './MTRouter[vllm-client,tensorboard]'`。

## 1. 审计已有数据（不加载模型、不调用 API）

```bash
MTRouter/.venv/bin/mtr-core-audit \
  --data-dir run/collection/baselines-scienceworld-3models-2runs/scienceworld \
  --models qwen-flash-2025-07-28 deepseek-flash glm-5
```

也能在无需任何第三方依赖的环境中使用：
`PYTHONPATH=MTRouter/src python3 -m mixed_pilot_mtrouter.audit ...`。

审计会拒绝未知模型、无终局标签、明确无效标签/不完整费用、混合币种、重复目录输入。
当前 174 条轨迹 / 3924 步已验证，币种为 CNY。
费用原值保留；不把 CNY 冒充 USD、不静默按汇率转换。模型价格特征也使用现有冻结 CNY 快照。
如果使用上游 USD 轨迹，应先显式统一币种与模型价格配置。

## 2. 配置与预计算

默认模型池就是现有三个模型，编码器默认 `Qwen/Qwen3-Embedding-0.6B`、8192 tokens、HF backend。
第一次实际运行可能下载编码器权重；CPU 能运行但速度与内存开销取决于上下文长度。

```bash
cp MTRouter/src/mixed_pilot_mtrouter/config/research/training.yaml \
   MTRouter/configs/training.local.yaml

MTRouter/.venv/bin/mtr-core-precompute \
  --config MTRouter/configs/training.local.yaml \
  --data-dir run/collection/baselines-scienceworld-3models-2runs/scienceworld \
  --output run/mtrouter/embeddings.pt --backend hf --batch-size 4
```

模型元数据位于 `src/mixed_pilot_mtrouter/config/models/custom_models.yaml`。
**价格来自现有项目快照；上下文上限、输出上限、知识截止日目前使用相同的占位特征，
不是已核实的模型规格。** 正式实验前补全；可在 `model_encoder.custom_models_path` 指定自定义文件。
自定义路径建议写绝对路径，避免切换工作目录后无法加载。修改元数据后须重新训练。

远程 embedding 可把 `history_encoder.backend` 改为 `vllm`，设置自己的
`vllm_base_url` 和 `vllm_model_id`，安装 `vllm-client` extra；预计算也要显式传
`--backend vllm --vllm-base-url http://...`。训练和推理必须使用同一编码器与截断/pooling 设置。

## 3. 训练

```bash
MTRouter/.venv/bin/mtr-core-train \
  --config MTRouter/configs/training.local.yaml \
  --data-dir run/collection/baselines-scienceworld-3models-2runs/scienceworld \
  --precomputed-path run/mtrouter/embeddings.pt \
  --output-dir run/mtrouter/model --epochs 100 --batch-size 64 --lambda_ 0
```

按 `(benchmark, task_id)` 分组划分，固定 seed 42；同一任务的全部模型/run 保持在同一侧。
`split.json` 保存任务划分，日志为 JSON，可选 TensorBoard。
产出 `q_function_best.pt` 和 `q_function.pt`。`--val-ratio 0` 可关闭验证，此时只使用最终 checkpoint。

先用较少 epoch 验证流程，再进行正式实验。原有 baseline 数据可用于起步，但未覆盖跨模型接手；
仍建议补充 roulette 轨迹和更多 variation。仅 variation 0 的任务分组验证更接近任务间泛化。
AEC 是可选 `--aec-enabled`，默认关闭。AWR/margin 上游实验选项保留，默认关闭；
它们按 task+step 分组，不代表不同模型该步拥有相同历史，不建议无验证地启用。

## 4. 启动评分服务

```bash
MTRouter/.venv/bin/mtr-core-serve \
  --checkpoint run/mtrouter/model/q_function_best.pt --port 8765 --device cpu
```

服务只监听 `127.0.0.1`，串行使用一个常驻模型。`GET /health` 返回模型池；
`POST /route` 接收 `messages` 与有序 `models`，返回各模型 Q-value 和 argmax 模型名。
不接收 API key，不代替 harness 调用被选模型。模型池不匹配或非有限评分会报错，不静默降级。
迁移机器后，checkpoint 依赖的 encoder 权重/路径仍需可用；HF 权重不包含在 router checkpoint 中。
vLLM checkpoint 可用 `--embedding-url` 覆盖服务地址，自定义元数据路径可用 `--model-metadata` 重定位。

## 5. 在现有 harness 环境调用

统一 dataset 推理现在支持 Router profile、直接加载 checkpoint 及 RandomRouter。
详见 [Router 推理说明](../docs/ROUTER_INFERENCE.md)。本节的独立 HTTP 服务接口继续兼容。

示例 `MTRouter/configs/harness.example.json` 复用现有 provider 配置与 key 环境变量；
`router.toml` 的三个 tier 顺序与 checkpoint 一致，不要求补出第四个 tier。

在原 harness 环境、项目根目录执行：

```bash
mixed-pilot chat --config MTRouter/configs/harness.example.json \
  --profile mtrouter --message '请解释水为什么会沸腾'
```

实际 chat 会调用被选模型，需要你配置对应 API key。
用 ScienceWorld 轨迹训练的 router 的有效范围主要是类似的环境历史；普通 chat 仅用于接口连通验证。
代码任务也可以指定该 profile，但不代表已验证跨任务路由效果。

harness bridge 只使用 Python 标准库，完整历史用于评分；实际模型请求的 messages/tools/system 原样传递。
`router_decision` 事件记录模型名和 Q-values；`confidence_available=false`，confidence=0 为接口占位值。
历史编码沿用上游 role/content 表示，不把结构化 tool_calls 额外编码；当前 ScienceWorld 文本动作适用。

## 验证与来源

```bash
PYTHONPATH=MTRouter/src MTRouter/.venv/bin/python -m unittest discover -s MTRouter/tests -v
# 原 harness 环境
PYTHONPATH=src python -m unittest discover -s tests -v
```

核心测试在临时目录创建随机初始化的小型 BERT，离线预计算、训练一轮、加载 checkpoint、
验证真实 HTTP 评分与保存/加载一致性。不下载大模型、不调用付费 API；小模型只验证工程路径，不验证路由质量。
已验证现有 3924 个 step 的解析；未对全部真实数据完成训练，未验证真实 Qwen embedding 或远程 vLLM 服务。

`UPSTREAM.json` 记录每个源文件 SHA-256，`LICENSE.upstream.md` 保留上游 MIT 许可。
关键修改：包命名与依赖边界、任务分组 split、审计、CPU 预计算、冻结 encoder 保持 eval 模式、
可选 TensorBoard、默认本地 embedding 配置、HTTP 服务和 harness 接口。

## 逐 step 采集入口

`scripts/collect_steps.py` 支持 RandomRouter（复用 profile/possibility）和递归单步切换的
KSR-brunch-1，输出去重的 `training/samples.jsonl`。现有 audit/precompute/train 命令可直接
读取该目录；传采集运行根目录也会优先选择该训练导出。JSONL 支持真实 benchmark、币种、
有效性及预期 remaining_cost，详见 [逐 step 采集说明](../docs/STEP_COLLECTION.md)。
