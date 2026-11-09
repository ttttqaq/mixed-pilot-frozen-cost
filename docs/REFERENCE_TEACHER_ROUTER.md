# 固定 reference Teacher 在线 Router（DB94）

分支：`feature/reference-teacher__from_online-router`，父提交 `2289454`。
入口仍为 `scripts/train_online_router.py`；使用新配置
`configs/online-router.reference-teacher.db94.yaml`。旧在线配置描述旧的可训练 Teacher，
请在旧方法分支使用它。本方法使用独立输出目录和 `mixed-pilot-reference-router-v2`
checkpoint，不能 resume 旧 run01/run02。

## 方法与数据

- train 保持 `data/VirtualWorld-subset/train.json` 的 300 题；评估默认改为 `test.json`
  的 50 题。不读取 valid 索引。`dataset.validation_split` 是历史字段名，现在设置为 test。
- test 仅贪心评估；不调用 Teacher、不进入 Router replay。日志的 split 明确为 test。
  该集合用于周期评估和 best_eval 选择，不再称为未参与选择的最终 holdout。
- 仅 Teacher 读取 train 的 `reference/gold_path.json` 成功动作路径。路径作为一种可行解；
  偏离 reference 不等于失败。reference 不进入 actor request、Router 特征和共享历史。
- Teacher 是固定 `gpt-6.1-terra`，无需小网络、optimizer、anchor、ready 样本量或拟合。
  输入实际历史、reference、当前策略描述/端点选模概率、预算/步数、候选价表和 train 调用
  token 统计。估计的是候选模型按当前策略继续的最终收益和剩余 Agent 费用，不是假设
  candidate 知道 reference。策略上下文不是未来路由行为的精确模拟，LLM 标签是估计。
- 冷启动为第 0 轮：uniform 选模运行 4 条 train 主轨迹，随机选 4 个访问状态，执行
  单步 A/B 干预及短 rollout（H=3），使用真实/LLM 标签训练初始 Router。
- 正式每轮：冻结 Router → 4 条主轨迹 → 选 4 个状态（25% 随机）→ A/B 短 rollout
  → 固定 Teacher 估计尾部/必要真实审计 → 更新 Router → 周期 test 评估。
- A/B 仅第一步分别强制 Qwen/GLM，之后按同一冻结 epsilon 策略继续。H 是新 API
  调用次数。未终止收益为 Teacher 的尾部终局预测；费用为真实新后缀费用加预测尾部费用。
- 自适应 H=3→6→9 使用效用差阈值，不伪造 Teacher ensemble 标准差。保留随机/主动
  完整审计；缺估计需真实续跑，不能把短截断、故障或配额耗尽标为失败 0。
- Router 网络、冻结编码器、收益/费用双头及 pair/anchor 损失保留。真实样本权重 1，
  LLM 估计权重 0.25，自报 confidence 不参与加权；真实结果覆盖估计。
  每轮仅用当前继续策略版本标签，Teacher 参数固定不意味着旧策略 return 可混用。

## API 与 key

以用户桌面 `API_test/test_api.py` 为协议参考：
`POST https://code1.newcli.com/v1/responses`，Bearer key，`input_text`、`instructions`、
`store=false`、`stream=true`；增加 `reasoning={"effort":"low"}` 和输出额度。
不调用 Chat Completions，不修改共用 actor provider。

打开上述新 YAML，在 `api_keys` 中填写 dashscope、zhipu、newcli，三个字段默认均为空；
已有环境变量也可继续使用（DASHSCOPE_API_KEY、ZHIPU_API_KEY、NEWCLI_API_KEY）。
CLI 对应参数优先于 YAML，YAML 优先于环境变量。key 在保存 resolved config、日志和
checkpoint 前移除。不会从桌面测试脚本读取或复制已填 key。

Teacher 串行调用，默认完成/失败后至少等 15 秒再发下一请求，首次及 resume 后也等待，
重试同样限速。可设置 `--set teacher.llm.min_interval=30` 进一步放缓。
429 遵守退避/Retry-After；重试耗尽、非法 JSON、截断会记录费用并暂停，不自动重启。
仅收到完整 Responses 终止事件后处理回复；input/output（含 reasoning）和 cache usage
统一进入 ledger，流断开/缺 usage 保留不确定费用，绝不计为免费。
JSON mode 默认关闭以接近桌面脚本，通过提示要求 JSON；可按 relay 支持情况显式开启。

Teacher 预算价表来自 2026-10-30 模型广场：基准输入 $1/M、输出 $5/M、缓存读取 $0.05/M，
渠道倍率 1.5。默认用假定 7.3 CNY/USD 得到 10.95、54.75、0.5475 CNY/M。
这是可编辑的保守预算口径，不等于订阅额度的实际人民币账单。全局预算 30 CNY 包含
所有 scope；每路径预算 1 CNY、50 次调用。末次请求可能超过请求前停止阈值。

## 运行

```bash
cd /DB/data/yujietang/Agent/code/mixed-pilot
git switch feature/reference-teacher__from_online-router
PY=/DB/rhome/yujietang/miniconda3/envs/mixed-pilot/bin/python

# 静态预检：不加载编码器/环境，不调用 API。
$PY scripts/train_online_router.py --config configs/online-router.reference-teacher.db94.yaml --dry-run

# 离线链路检查：合成环境/回复/hash 编码器，无任何真实网络请求。
$PY scripts/train_online_router.py --config configs/online-router.reference-teacher.db94.yaml \
  --smoke --output-dir outputs/reference-router-my-smoke

# 填 key 后正式训练；选择空闲 GPU。结束时评估全部 test 50 题。
CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
$PY scripts/train_online_router.py --config configs/online-router.reference-teacher.db94.yaml

# 使用完全相同配置恢复。迭代内部分工作可能重新采集，已发生费用全部保留；
# 已完成的 test 任务和完整轮次不会重复评估。
CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
$PY scripts/train_online_router.py --config configs/online-router.reference-teacher.db94.yaml --resume
```

独立评估：加 `--evaluate-checkpoint outputs/reference-router-run01/latest.pt`
和一个新的 `--output-dir`。此模式不需要 Teacher key，不加载 reference。
更换策略/价表/Teacher 设置/源码/参考路径内容时使用新输出目录；resume 校验指纹。

主要结果：`evaluation.jsonl`（split=test、完整覆盖、费用）、`best_eval.pt`、`latest.pt`、
`teacher_predictions.jsonl`、`teacher_audits.jsonl`、`router_samples.jsonl`、`pairs.jsonl`、
`training_log.jsonl`、`ledger.json`/`calls/`、`state.json`/`checkpoints/`、`summary.json`。
Teacher 预测缓存包含策略、reference、提示、模型设置及价表指纹；跨策略不直接复用。
审计只记录误差并修正 Router 标签，不更新 Teacher。离线 smoke 分数不代表 benchmark 成绩。

## 本次验证（2026-10-30）

静态预检：train 300、test 50、全部 300 条 train reference 可用。离线完整 CLI smoke 完成
2 轮，Teacher 更新次数为 0；原生 train/test 各一个 ScienceWorld/ALFWorld，共 4 个
环境及前缀重放通过，全部使用 scripted provider，真实付费 API 调用数为 0。

回归命令：`PYTHONPATH=src:MTRouter/src python -m pytest MTRouter/tests tests/test_providers.py tests/test_mtrouter_bridge.py tests/test_inference_router.py tests/test_router_provider.py -q`。
结果 129 passed、56 subtests passed；有 1 项旧测试失败：
`tests/test_inference_router.py::RouterTests::test_random_pool_probabilities_billing_and_all_step_exports`。
在父提交 2289454 的隔离源码上也复现相同 trace 文件枚举顺序断言失败，因此未修改无关实现。
真实 gpt-6.1-terra 请求尚未执行；请填写新配置中的 key 后自行运行。
