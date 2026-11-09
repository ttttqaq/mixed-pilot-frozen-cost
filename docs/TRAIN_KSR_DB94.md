# DB94：训练 step-KSR-brunch-1

项目目录：`/DB/data/yujietang/Agent/code/mixed-pilot`。
本说明只针对现有 KSR 训练导出，保留采集数据、默认配置、网络结构和监督目标。

## 本次适配

- `MTRouter/configs/training.ksr-db94.yaml`：独立训练配置，模型池为
  `qwen-flash-2025-07-28`、`glm-5.3`，顺序与当前 harness 的 mtrouter profile 一致。
  使用 DB94 本地 Qwen3-Embedding-0.6B；device=auto，保留 8192-token 输入、
  原有 MLP 和训练参数，将 FP32 embedding 预计算 batch size 设为 1。
- `MTRouter/configs/models.ksr-db94.yaml`：显式提供这两个模型的元数据，避免
  GLM-5.3 静默使用默认值。价格取目标采集运行的冻结 CNY 快照，输入/输出限制取
  同一运行的 profile 配置；知识截止日保留原方案的相同中性占位值，不代表真实规格。
- `MTRouter/src/mixed_pilot_mtrouter/research/training/encoders.py`：在线 HF 编码器
  显式使用 FP32，与现有预计算路径保持一致。当前 Transformers 会按权重配置自动加载
  BF16；原实现会造成在线/预计算 embedding 的数值与 dtype 不一致。
- `MTRouter/tests/test_core.py`：现有端到端用例改为使用 BF16 保存的小型模型，
  检查在线加载为 FP32，并保留预计算、训练、checkpoint、HTTP 评分与 embedding 一致性检查。
- Embedding 权重位于 `/DB/data/yujietang/huggingface_models/Qwen3-Embedding-0.6B`。
  上游为 `Qwen/Qwen3-Embedding-0.6B`，固定 revision
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`。
  `mixed-pilot-source.json` 记录来源、文件大小和上游 LFS SHA-256。

原有 generic training.yaml、custom_models.yaml、harness.json 和 KSR_config.json 均不改动。
这次训练学习的是路由器，不会微调 Qwen Flash、GLM-5.3。梯度训练只使用已有样本；
启用 VirtualWorld-20 在线评估后，评估阶段会调用候选模型 API 并计费。

## 数据检查结果

训练只读取 `data/step-KSR-brunch-1/training/samples.jsonl`。
不要把该运行根目录和 training 子目录同时作为两个 data-dir，也不要加入叶子展开数据。

- 共 12,775 行，12,395 行有效，380 行因标签/费用不完整跳过。
- Qwen Flash：10,658 条；GLM-5.3：1,737 条。
- ScienceWorld：1,194 条；ALFWorld：11,201 条；均为 train split。
- 默认按 benchmark + task_id、seed=42 分组：90 个训练任务 / 10 个验证任务，
  对应 11,153 / 1,242 条样本，任务没有交叉。
- 保留已经导出的 KSR 预期终局 y，不重新从叶子计算标签。
- lambda_=0、AEC/AWR/margin 关闭；训练回归终局收益，不对成本施加惩罚。
  模型元数据中的价格特征仍然保留。数据中的实际 remaining_cost 不变。

## 正式训练

DB94 已有 `mixed-pilot` Conda 环境，无需重新安装。
先运行以下预计算命令，完成后再运行下文“训练期间评估 VirtualWorld-20”的训练命令。
命令不依赖前一个终端的 export 或 KSR_CONFIG 等变量；GPU 0 可换成空闲 GPU 编号。
8192-token FP32 embedding 预计算耗时比后续 MLP 训练长，建议在 tmux 中执行。

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
/DB/rhome/yujietang/miniconda3/envs/mixed-pilot/bin/python \
  -m mixed_pilot_mtrouter.cli.precompute_embeddings \
  --config /DB/data/yujietang/Agent/code/mixed-pilot/MTRouter/configs/training.ksr-db94.yaml \
  --data-dir /DB/data/yujietang/Agent/code/mixed-pilot/data/step-KSR-brunch-1/training \
  --output /DB/data/yujietang/Agent/code/mixed-pilot/outputs/mtrouter-ksr-full/embeddings.pt \
  --backend hf --device cuda:0 --batch-size 1
```

CUDA_VISIBLE_DEVICES 选物理 GPU；命令中的 cuda:0 表示选中后可见的第一张卡。
例如使用物理 GPU 3 时只需将 CUDA_VISIBLE_DEVICES 改为 3，命令仍写 cuda:0。

训练最多 100 epochs、batch size 64、learning rate 0.001、验证比例 0.1，
连续 3 epochs 验证损失未改善时 early stopping。
Embedding 的 batch size 与 MLP 训练的 batch size 是两个独立参数。

训练产物：

- `outputs/mtrouter-ksr-full/model-online-valid/q_function_best.pt`：验证损失最低的 checkpoint。
- `outputs/mtrouter-ksr-full/model-online-valid/q_function.pt`：最后一轮 checkpoint。
- `outputs/mtrouter-ksr-full/model-online-valid/split.json`：任务划分。
- `outputs/mtrouter-ksr-full/model-online-valid/training_log.jsonl`：逐 epoch 损失。
- `outputs/mtrouter-ksr-full/embeddings.pt`：可以复用的冻结历史 embedding。

同一数据文件、模型顺序、编码器、max_tokens、pooling 保持不变时，可以复用 embedding
调整训练 epochs/lr/batch size。数据身份包含输入 samples.jsonl 的绝对路径，移动文件后应重新预计算。
新实验使用新输出目录；现有训练入口不提供优化器状态的断点续训。

部署 checkpoint 时也需要该本地 embedding 目录和元数据文件。
验证 MSE 只衡量回归误差，路由效果仍需在独立测试任务上测量成功率和费用。

## 验证记录

2026-10-18 在 DB94 的 mixed-pilot 环境完成：

- 真实数据审计、12,395 条样本完整加载及 90/10 个任务分组验证。
- 从两个模型、两个 benchmark 的 8 个任务抽取 31 条真实样本，完成 RTX 3090 上的
  Qwen3-Embedding 预计算，输出形状为 31 × 1024。
- 完成一轮 GPU 训练、6/2 个训练/验证任务划分、最佳和最终 checkpoint 保存及重新加载。
- 用真实模型核对在线/预计算 embedding，检查样本的最大绝对差为 0.0；路由评分为有限值。
- 新增 BF16 场景在修复前失败，原因是在线模型为 torch.bfloat16；修复后 18 项相关测试全部通过。
- `git diff --check` 通过。没有启动全量正式训练。

验证日志、子集、checkpoint 和 verification.json 保存在：
`/DB/data/yujietang/Agent/code/mixed-pilot/outputs/ksr-db94-smoke-20261018/`。
该小样本 checkpoint 仅用于工程验证，不能用于判断正式路由效果。

## 训练期间评估 VirtualWorld-20

该功能不影响训练 embedding。预计算入口、编码器参数、输入数据和 embedding 缓存
均保持不变；YAML 新增的 online_evaluation 段不参与预计算。
当前预计算可以继续，完成后直接使用同一个 embeddings.pt。
评估中的新任务历史会实时编码，只用于评估，不写回训练缓存或训练样本。

默认建议每 **500 个优化器更新 step** 评估一次：训练集 11,153 条、batch size 64，
每 epoch 约 175 次更新，因此约每 3 个 epoch 评估一次。训练结束/早停时补评估一次；
若最后一次更新刚好已经评估，则复用该次结果，不重复调用 API。
每个任务最多 50 个 Agent API step，与训练优化器 step 是不同计数。

启用后，训练暂停优化器更新，通过临时本地 HTTP 服务复用当前编码器和 Q 网络，
由推理子进程运行评估任务。评估结束恢复训练模式，不额外加载一份编码器。
原有每 epoch 的 MSE 验证、早停和 q_function_best.pt 选择规则保留；外部成功率仅监控。
评估失败/超时保存错误和覆盖率，继续训练；不把未评分任务伪装成正常的完整评估。

评估参数统一放在 `MTRouter/configs/training.ksr-db94.yaml` 的 `online_evaluation` 段：

```yaml
online_evaluation:
  dataset: "/DB/data/yujietang/Agent/code/mixed-pilot/data/VirtualWorld-20"
  every_steps: 500
  split: "test"
  step_limit: 50
  timeout: 7200
  harness_config: "/DB/data/yujietang/Agent/code/mixed-pilot/harness.json"
  pricing: "/DB/data/yujietang/Agent/code/mixed-pilot/configs/inference.prices.json"
  python: "/DB/rhome/yujietang/miniconda3/envs/mixed-pilot/bin/python"
  model_profiles: null
```

修改 YAML 即可调参；同名用途的 `--eval-*` 命令行选项优先于 YAML。
例如临时追加 `--eval-every-steps 1000`，只覆盖本次运行的评估间隔。
将 `dataset` 改成 `null` 即关闭在线评估；没有此段的其他训练配置保持原有行为。
路径建议使用绝对路径；相对路径和原有命令行参数一样按启动时的工作目录解析。
训练时会先检查数据、模型映射、API key 和解释器，并拒绝与训练任务 ID 重叠的评估任务。

**等预计算完成后**，在 DB94 执行以下命令。把两处 key 占位符替换为真实 API key。
如果当前 Shell 已有这两个变量，可删除命令最前面的两行赋值。评估会实际调用模型并计费。

```bash
DASHSCOPE_API_KEY='你的 DashScope API key' \
ZHIPU_API_KEY='你的智谱 API key' \
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
/DB/rhome/yujietang/miniconda3/envs/mixed-pilot/bin/python \
  -m mixed_pilot_mtrouter.cli.train_q_function \
  --config /DB/data/yujietang/Agent/code/mixed-pilot/MTRouter/configs/training.ksr-db94.yaml \
  --data-dir /DB/data/yujietang/Agent/code/mixed-pilot/data/step-KSR-brunch-1/training \
  --precomputed-path /DB/data/yujietang/Agent/code/mixed-pilot/outputs/mtrouter-ksr-full/embeddings.pt \
  --output-dir /DB/data/yujietang/Agent/code/mixed-pilot/outputs/mtrouter-ksr-full/model-online-valid \
  --device cuda:0
```

YAML 中 `python` 指定评估解释器；现在与训练共用 `mixed-pilot` 环境。
`timeout` 默认为每轮 7200 秒，可按需要调整。
`model_profiles: null` 表示沿用训练模型池的名称和顺序；API 模型别名不同可指定一一映射列表。
重复实验使用新 --output-dir；已有 online_eval 的目录会拒绝复用，以免混合实验或重复计费。

新增输出均在训练 output-dir 内：

- online_validation.jsonl：各次评估的 global_step、epoch、通过率、覆盖率、已知费用及错误。
- online_eval/step-XXXXXXXX/q_function.pt：本次评估对应的训练权重。
- online_eval/step-XXXXXXXX/inference/：原有推理格式的 summary、轨迹、费用和逐步记录。
- online_eval/step-XXXXXXXX/inference.log 和 result.json：执行日志与本次评估结果。
- 安装 TensorBoard 时，完整评估另记录 OnlineValidation/pass_rate 和 known_api_cost。

VirtualWorld-20 持续用于开发验证后，最终效果应另用独立测试任务报告。

本次代码验证：28 项 CPU 离线测试通过，包括 YAML 参数读取、命令行覆盖、真实小型编码器的
预计算 → 训练 → HTTP 评分 → 周期/结束评估链路。训练前后 embedding 缓存 SHA-256 相同；
相同随机种子下，启用和关闭在线评估得到的最终模型参数逐项完全相同。
这些测试不调用付费 API，也不衡量真实任务成功率。

## VirtualWorld-20 环境配置记录

2026-10-18 按要求在 `mixed-pilot` 中补齐 ALFWorld 0.4.2、TextWorld 1.7.0、
fast-downward-textworld 20.6.4 及依赖；原有 ScienceWorld 1.2.3、Java 11 可直接使用。
安装锁定此前已有包的版本，记录位于 `outputs/virtualworld-env-20261018/`，
包括 requirements.txt、constraints-before.txt、安装报告和安装前后包清单。
TextWorld 上游 wheel 的 WHEEL 标签前多出空行，已修正该安装元数据及 RECORD；
包版本、Python 代码与二进制内容保持不变，原始和修复 wheel 及校验值一并保留。

配置后验证：VirtualWorld-20 的 20 个原生任务均启动并执行观察动作；
另外确认 ALFWorld 能执行环境给出的合法动作。28 项 CPU 离线回归全部通过，
pip check 通过，原有 125 个包的版本与完整 embedding 缓存 SHA-256 均保持不变。
验证过程不调用付费模型 API。报告为同目录 verification.json。
