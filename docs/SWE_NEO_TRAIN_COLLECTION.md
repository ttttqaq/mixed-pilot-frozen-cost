# SWE-neo train：四模型 code-task 批量推理与原生评测

> 入口迁移：本文保留历史实验说明。批量推理使用 [scripts/infer.py 的新文档](INFERENCE.md)；旧实验包装脚本已删除。逐步 judge 仍可使用 `scripts/judge_trace_batch.py`。

入口：`scripts/run_swe_neo_train.sh`（Bash 包装）或 `scripts/run_swe_neo_train.py`。
默认读取 `/Users/liangjianfeng/Data/Router/SWE-neo-data/train.json`，仅使用 100 条 train，
其中 SWE-rebench V2 和 SWE-Bench Pro Verified 各 50 条。
四个模型各运行一次，共 **400 次 code-task**；20 条 test 不参与。

未调用任何真实模型。已用离线 scripted provider 检查 code-task、trace、补丁导出、续跑、统计及测评错误处理。
默认继续在当前 Mac 的 Docker 中运行；已实际验证前两条任务的 gold/空补丁检查，
以及推理容器的依赖读取和 shell 工具。其余 98 条尚未逐条检查，模型账号权限尚未通过真实 API 调用验证。

## 1. 环境

- Python 3.10+，建议 3.11/3.12；宿主机有 Git、Docker CLI，Docker daemon 正在运行。
- Mac Docker Desktop 可直接使用，镜像默认 `linux/amd64`；Apple Silicon 使用 Docker 的 amd64 仿真。
- 每个并发任务默认 4 CPU、8 GiB 内存；磁盘要能容纳所选任务镜像、100 个缓存仓库和 400 份工作区。
  镜像大小不固定，脚本保留镜像和工作区，不会清理 Docker 共享镜像。
- 控制端需要访问 GitHub、Docker Hub，以及模型 API；任务容器的网络默认关闭。

脚本自动把本仓库 `src/` 加入 Python 路径，无需先安装其他 agent 框架。
在当前机器上可指定已有 Python 3.12：

```bash
cd /Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot
export PYTHON=/Users/liangjianfeng/miniconda3/envs/opensquilla/bin/python
```

迁到服务器时，改成服务器的 Python 路径，并用 `--data-root /server/path/SWE-neo-data`。

## 2. 先看计划，无模型调用、无下载

```bash
bash scripts/run_swe_neo_train.sh plan
```

不带参数也是 `plan`。输出 `plan.json`，包含 400 个计划任务、实际模型名、预算和数据 SHA256。
`harness.json` 原先将 `glm-5.3` 的 `model` 写成了 `glm-5`；本批处理在内存中改成
`glm-5.3`，避免两个 profile 实际运行同一个模型。

默认 provider：

| Profile / 实际 model | API base URL | API key 环境变量 |
| --- | --- | --- |
| deepseek-flash | https://api.deepseek.com | DEEPSEEK_API_KEY |
| glm-5 | https://open.bigmodel.cn/api/paas/v4 | ZHIPU_API_KEY |
| glm-5.3 | https://open.bigmodel.cn/api/paas/v4 | ZHIPU_API_KEY |
| glm-4.7-flash | https://open.bigmodel.cn/api/paas/v4 | ZHIPU_API_KEY |

这些模型 ID 按你的请求发送；没有通过付费 API 探测账号是否支持。
若使用其他供应商或账号的模型 ID 不同，可显式覆盖，例如：

```bash
bash scripts/run_swe_neo_train.sh plan \
  --base-url glm-5.3=https://your-provider.example/v1 \
  --model glm-5.3=your-actual-model-id \
  --output run/collection/swe-neo-custom-provider
```

`plan`、`check` 和 `run` 要使用相同覆盖参数。同一个输出目录拒绝混入不同任务、模型或预算。
没有自动降级或替换模型。

## 3. 建议先验证两个任务的评测环境

`--limit 2` 交替选择两个来源，因此各取一条 train；不重新抽样原始数据集。

```bash
bash scripts/run_swe_neo_train.sh check \
  --limit 2 --output run/collection/swe-neo-smoke
```

该命令会下载这两条的 base commit、任务镜像和固定版本的测评脚本；
在独立评测容器里分别运行 **gold patch** 和 **空 patch**，不调用模型。
只有 gold 通过、空补丁未通过，才返回成功。结果在 `environment-checks/result.json`。
每次检查保存到 `environment-checks/<task-id>/attempt-NNN/gold/` 和 `empty/`，保留之前的日志。
汇总记录具体结果路径、耗时和错误原因。`check` 事件的 `profile=null` 表示未调用模型。

可选的全量预下载也不调用模型：

```bash
bash scripts/run_swe_neo_train.sh prepare
```

`run` 本身也会按需准备，无需先执行全量 `prepare`。
可以用 `bash scripts/run_swe_neo_train.sh check` 对全部 100 条执行同样的无模型校准。

## 4. 执行推理

先在终端设置 API key；不要把 key 写入脚本或提交到 Git：

```bash
export DEEPSEEK_API_KEY='你的 DeepSeek key'
export ZHIPU_API_KEY='你的智谱 key'
```

先跑上述两个任务 × 四模型，共 8 次：

```bash
bash scripts/run_swe_neo_train.sh run \
  --limit 2 --output run/collection/swe-neo-smoke
```

确认 trace、API 模型名和 native evaluation 正常后，跑完整 400 次：

```bash
bash scripts/run_swe_neo_train.sh run
```

默认串行、`max_steps=64`、每个 shell 最多 240 秒、每次 API 请求超时 360 秒、
整条 code-task 超时 7200 秒、原生评测超时 3600 秒。
`max_tokens` 和 `temperature` 默认沿用对应 profile（当前四模型均为 `max_tokens=8192`）。
可以用 `--max-steps`、`--max-tokens`、`--temperature` 等统一覆盖；
如果已经生成旧计划，改变预算请用新的 `--output`。

例如更长预算、两个模型同时运行：

```bash
bash scripts/run_swe_neo_train.sh run \
  --workers 2 --max-steps 128 --max-tokens 16384 \
  --output run/collection/swe-neo-128steps
```

`--workers` 只控制同一任务的模型并发，最大有效值为所选模型数；
两个并发容器需预留约 8 CPU / 16 GiB。默认串行方便中断和定位环境错误。
连续三条 `ProviderError` 会停止整个批次，保留已产生的结果；不会对失效端点持续请求。

## 5. 中断续跑、重试与仅重评

- 重复同一个 `run` 命令：跳过已有最终结果，对未完成任务创建新 attempt，保留旧 trace。
- 对推理或测评报错的任务重新推理：在同一命令后加 `--retry-errors`；正常测试未通过的任务不重试。
- 已有补丁，只想重做评测：

```bash
bash scripts/run_swe_neo_train.sh evaluate
```

`evaluate` 不调用模型，新建 `native-eval-NNN`，保留之前的评测日志。
对 smoke 目录操作时仍应传 `--limit 2 --output run/collection/swe-neo-smoke`。
重建统计也不调用模型，无需 API key、Docker 或原数据集：

```bash
bash scripts/run_swe_neo_train.sh summarize
```

## 6. 输出与 pass rate

默认输出：`run/collection/baselines-swe-neo-4models-train/`。
下载缓存：`run/swe-neo-cache/`，与原始 SWE-neo-data 分开保存。

```text
plan.json
progress.jsonl
summary.json                 # 四模型总体，以及每个来源各自的统计
summary.csv
task_results.csv
runs/<profile>/<task-id>/
  result.json                # 最新 attempt 的状态和 artifact 路径
  attempt-001/
    result.json
    prediction.patch         # 包含新文件、已提交修改、超时前的部分修改
    code-task/<timestamp-id>/
      request.json
      result.json            # 原 harness 结果；内置 evaluation 为 skipped
      trace/events.jsonl     # 模型消息、请求/响应事件、工具参数与结果
      trace/events.sse
      trace/tool-outputs/    # 完整 shell stdout/stderr；模型上下文仍遵循 harness 输出上限
      output/change.patch
      output/untracked/
      environments/agent/workspace/
    native-eval/
      result.json            # 真正的通过判断，以及缺失 F2P/P2P 测试
      prediction.patch
      entryscript.sh
      container.stdout.log
      container.stderr.log
      tests.*                # Pro 的结构化测试结果和原始日志
```

`summary.json.groups` 中：

- `passed`：原生测试通过数；推理完成本身不算通过。
- `failed`：原生测试未达到通过条件，也包括明确识别的代码编译失败（如空补丁缺少新字段或本地模块）。
- `unscored_errors`：环境、下载、运行或解析错误，未得到有效测试判断。
- `pass_rate`：全部任务有最终记录时，`passed / total_tasks`，包括基础设施错误在分母内；
  全量运行每模型分母 100，按来源分母 50。未跑完为 `null`。
- `scored_pass_rate`：`passed / (passed + failed)`，排除未评测错误；配合 `scoring_coverage` 使用。
- token、耗时来自已完成的 API 调用事件；中途中断、供应商未返回 usage 的调用不能完整计量。
  不估算费用。

达到步数上限但已有补丁时，保留推理错误并评测已有修改；只要原生测试确实通过，就计入 `passed`。
基础设施错误和真实测试失败分开记录，避免把脚本退出码当作统一通过指标。

## 7. 测评依据与可比性

调用的是现有 `mixed_pilot_harness.code_task.run_code_task`，保持单 turn、多 step、
现有七种工具和 trace 生命周期。批处理替换 shell/git 工具执行位置为 Docker。
内置 rubrics/LLM scorer 未调用；内置 evaluation 显式记录为 skipped，随后执行原生评测。
私有 `reference/`、gold patch、hidden test patch 不挂载给推理容器。

- **SWE-rebench V2**：使用 [官方 eval.py 与 log parsers](https://github.com/SWE-rebench/SWE-rebench-V2/blob/c71902a8cf8d2b725f63d51f199f4d3e56f68d2d/scripts/eval.py)，
  固定版本 `c71902a…`。应用提交补丁和原始 test patch，按照官方 `passed_match` 的精确集合规则判分。
- **Pro Verified**：使用 [AgentCompass Verified 的评测顺序与判分规则](https://github.com/open-compass/AgentCompass/blob/d409251420bf97b97d0b103287759353f925adc3/src/agentcompass/benchmarks/swebench_pro_verified/swebench_pro_verified.py)，
  加载 [ScaleAI 官方实例测试脚本](https://github.com/scaleapi/SWE-bench_Pro-os/tree/66f92766bba642462d4bbe5479e83f91f9211862/run_scripts)，
  固定版本 `66f9276…`。在新镜像内 reset 到 base，应用提交补丁、恢复隐藏测试、运行 parser；
  全部 F2P 与 P2P 必须出现于 passed 集合。

推理仓库仅取 base commit，重建 Git 历史；Pro 隐藏选定评测文件。
本脚本采用完全断网的推理/评测容器，区别于官方 blacklist/host-network 配置；
可能影响需要联网取依赖的任务，因此提供 `check` 校准命令。
这是一套本 harness 下的实验，尚未证明与官方运行环境、匿名化和全部防泄漏处理完全等价，
结果不应直接标成官方排行榜分数。

### Mac Docker 环境适配

推理和评测容器都加载只读 `scripts/docker_runtime/system-chromium.cjs`。
在安装了系统 Chromium 的 Alpine 镜像中，它让 Puppeteer 使用该浏览器，
避免调用依赖 glibc 的下载版 Chrome；仓库文件、隐藏测试和上游 parser 保持原始内容。
容器的 `runtime.json` 和原生评测 `result.json` 记录适配版本。
shell 工具使用 GNU/BusyBox 均支持的 `timeout -s KILL`。
推理挂载代码时，用每个容器独立的 Docker volume 保留镜像内已安装且不在源码快照中的
`node_modules`、虚拟环境等依赖目录，避免挂载遮住这些依赖；关闭容器时清理其临时 volume。

编译失败与环境故障分别记录：Go 新字段未实现或 TypeScript 缺少本地实现可判为 `failed`；
浏览器启动失败、动态库缺失、未知原因的零测试和评测超时仍为 `error`。
即使存在通过的测试，检测到浏览器启动故障也不会将环境检查标记为有效。
