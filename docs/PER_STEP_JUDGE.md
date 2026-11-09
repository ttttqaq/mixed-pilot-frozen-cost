# Per-step judge

> 入口迁移：本文保留历史实验说明。批量推理使用 [scripts/infer.py 的新文档](INFERENCE.md)；旧实验包装脚本已删除。逐步 judge 仍可使用 `scripts/judge_trace_batch.py`。

对采集得到的唯一 `.traj.json` 的一个 API-step 进行 0~100 评分。指定 judge
profile 后，使用与 `mixed-pilot chat` 相同的 `chat_once` 函数发送一次请求；
不给 judge 工具，不执行 agent 循环，不重跑环境，不修改源 trace。

## 使用

```bash
TRACE=run/collection/baselines-scienceworld-3models-2runs/scienceworld/qwen-flash-2025-07-28/scienceworld-task-8-inclined-plane-determine-angle-var0-run0-b846d8d9ad1f5d3b.traj.json

# 先预览输入，无 API 请求、无需 Key。--step 从 1 开始。
mixed-pilot per-step-judge \
  --config harness.json --profile deepseek-flash \
  --trace "$TRACE" --step 8 --dry-run \
  --result run/step-judgments/qwen-slope-step8.prompt.json

# 正式评分：使用所选 profile 的凭据、模型和 generation 参数。
mixed-pilot per-step-judge \
  --config harness.json --profile deepseek-flash \
  --trace "$TRACE" --step 8 \
  --result run/step-judgments/qwen-slope-step8.json

# 若掌握标准答案/目标状态，可额外提供 UTF-8 文本文件。
mixed-pilot per-step-judge \
  --config harness.json --profile deepseek-flash \
  --trace "$TRACE" --step 8 \
  --reference-answer-file /path/to/reference-answer.txt \
  --result run/step-judgments/qwen-slope-step8-with-reference.json
```

也可直接使用 `--reference-answer '标准答案或期望最终状态'`，与文件参数互斥。
上述未提供标准答案的命令照常工作，`reference_source` 会明确标为 `task_goal_only`。
若命令未安装，可用 `PYTHONPATH=src python -m mixed_pilot_harness.cli per-step-judge ...`。

`--step 1` 对应 trace 中 `step_idx=0`；`--step 8` 对应 `step_idx=7`。
这里一步是一次模型调用，可能包含多次工具调用。

## 最终答案、环境与信息边界

- 历史：直接读取目标 step 的 `state.history`，保留完整历史，不做静默截断。
- 当前环境：目标 step 开始前最近一条 user/tool 反馈，另提供完整历史、任务说明、
  agent system 和可用工具。这是可见观察，不是完整模拟器内部状态。
- 被评步骤：该 step 的 `response`、`command` / `tool_calls`、`observation` /
  `tool_results` 和格式错误标记；没有环境反馈的提交会明确标记 `recorded=false`。
- 默认完成标准：ScienceWorld 初始消息中的 `<task_description>`；其它采集任务使用初始 user 输入。
- 标准答案：仅使用显式传入的参考文本。不会自动读取 trace 中的外部文件路径，
  不会把 trace 最后一次回答当成正确答案，也不会用终局分数猜测答案。
- 不输入后续 step、trace 的 `info.success` / `final_score` / `evaluation`，
  或候选模型的名称、成本与路由元数据。输出结果仍记录源 trace 身份以供关联。

因此斜面任务中，仅记录 “You focus on the inclined plane B” 并不告诉 judge
这个动作是否正确完成任务。若未提供参考答案，judge 应报告不确定，不能凭空判为 100。
已知标准答案可作为 grader 的参考，但它来自外部，并非该时刻 agent 已知信息。

支持 collection 的 ScienceWorld 文本动作和 HLE 工具反馈格式。
不支持把 `code-task/.../trace/events.jsonl` 或导出的 `.steps.jsonl` 直接当作输入；
应传入带 `info`、`steps[].state.history` 的完整 `.traj.json`。

## 分数含义和输出

分数表示 **执行该步后的状态价值**：假设后续 agent 合理行动，成功可能性越大、
所需剩余 API-step 越少则越高。它不是本步骤相对前一步的增量奖励，也不是整条轨迹
最终成功与否的回填标签。评分区间和具体规则位于
`src/mixed_pilot_harness/per_step_judge/prompt.py`，修改尺度时应递增 `PROMPT_VERSION`。

成功时 `status="scored"`，`judgment` 结构示例（以下是格式示例，不是真实评分）：

```json
{
  "score": 65,
  "success_probability": 0.8,
  "estimated_remaining_steps": 3,
  "confidence": 0.7,
  "reason": "已找到目标，但尚未完成比较。",
  "evidence": ["环境列出了两块斜面；尚无角度测量结果。"]
}
```

`score` 为有限的 0~100 数值，概率与置信度为 0~1，剩余步骤为非负整数或 `null`。
100 分要求确认完成，概率为 1、剩余步骤为 0。未知步骤数不填假数字。
概率、步数和综合分数是 judge 的主观估计，未经统计校准；综合分数使用提示中的尺度，
没有硬编码的概率/步数换算公式。

stdout 及可选 `--result` 文件还保存：trace 路径和 SHA-256、task/run/step 身份、
prompt 版本及全文、参考答案来源、judge profile、实际返回模型、时间与耗时、
原始回复、stop reason 和完整 token usage。不计算账单费用。

格式错误、越界分数、不完整/截断回复或意外工具调用会返回
`status="invalid_response"`、`judgment=null` 和错误说明，CLI 退出码为 2；
原始回复和 usage 仍可在结果文件查看，不会偷偷重试评分或执行工具。
传输错误沿用 harness 的报错及 provider 网络重试机制。
超长输入可能触发 provider 上下文限制；系统不会静默删掉历史。

## Python 函数

```python
import asyncio
from pathlib import Path

from mixed_pilot_harness.config import load_config
from mixed_pilot_harness.per_step_judge import judge_step, prepare_judgment

trace = Path("/path/to/unique.traj.json")
prepared = prepare_judgment(trace, step=8)  # 不调用 API

result = asyncio.run(judge_step(
    trace, step=8,
    config=load_config(Path("harness.json")),
    profile="deepseek-flash",
    reference_answer=None,  # 或显式参考答案文本
))
if result["status"] == "scored":
    print(result["judgment"]["score"])
else:
    print(result["error"])
```

在已有异步程序里直接 `await judge_step(...)`。`prepare_judgment` 可以单独用于人工
审阅 prompt；`judge_step` 返回字典，不自动写文件。

## 批量：随机 10 条 Qwen trace 的全部步骤

```bash
# 默认从 ScienceWorld 的 58 条 Qwen trace 中用 seed=42 抽 10 条，共 202 步。
# 使用 configs/collection.providers.json 的 deepseek-flash（关闭原生思考）。
bash scripts/judge_qwen_traces.sh --dry-run

# 在已设置 DEEPSEEK_API_KEY 的终端中运行；默认 3 个并发请求。
bash scripts/judge_qwen_traces.sh

# 原命令可断点续跑，成功结果不重复计费。
# 失败、格式错误和中断记录默认保留，显式指定以下参数才重试：
bash scripts/judge_qwen_traces.sh --retry-failed
```

可用 `PYTHON_BIN=/path/to/python` 指定解释器；`--config`、`--profile`、`--trace-dir`、
`--count`、`--seed`、`--workers`、`--output-dir` 均可覆盖默认值，`--help` 查看完整选项。
批量脚本逐步调用 `python -m mixed_pilot_harness.cli per-step-judge`，每个 step 独立请求。
没有提供标准答案时沿用任务目标，原生终局结果只用于最终报告对照。

默认产物目录 `run/step-judgments/qwen10-deepseek-seed42/`：

- `selection.json`：冻结的抽样名单、配置/trace/prompt 哈希和预期步骤数。
- `results/<trace>/step-NNN/attempt-NNN.json`：每次完整评分结果。
- 同目录 `*.started.json` / `*.finished.json` / `*.stderr.txt`：调用记录、退出码及错误。
- `summary.json`：全部逐步分数、理由及已记录回复的 token 总量（包括无效回复）。
- `scores.md`：10 条 trace 的完整分数序列，以及每步概率、预计剩余步骤和理由。

缺失或无效评分用 pending/failed 和 null 明确标记，不填零、不插值。
若 trace、config、prompt 或抽样发生变化，必须选择新的输出目录，避免混用结果。
同一输出目录有进程锁，避免两个批次同时发起重复请求。中断请求可能已产生费用，
没有返回 usage 的请求无法计入 token 汇总。

## 验证

离线测试覆盖未来结果隔离、step 编号、标准答案、无反馈提交、工具反馈、评分校验、
无效原始回复保留、dry-run、源文件保护，以及使用 scripted provider 的完整 CLI 流程。

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_per_step_judge.py' -v
PYTHONPATH=src python -m unittest discover -s tests -p 'test_chat.py' -v
```
