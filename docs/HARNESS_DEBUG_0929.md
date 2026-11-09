# run_0929 评测故障修复与离线验证

## 实际日志结论

分析来源：`run_0929/swe-neo-deepseek-flash-0929`，共 69 条记录，其中 47 条有原生评测结果。
其余 21 条初始化/镜像问题和 1 条中断记录不在本次评测修复的统计范围内。
未重新推理、未启动 Docker 评测、未产生 API 费用。

| 日志证据 | 根因 | 修复 |
|---|---|---|
| 13 条 rebench 的 test.patch 冲突；Pro Ansible n000061 的测试文件 already exists | 模型修改/新增测试与官方测试合并冲突 | 应用 prediction.patch 时排除官方测试路径，再由原生评测器应用/还原官方测试；保留源代码修改 |
| OpenLibrary n000058：`export: en_US.UTF-8: not a valid identifier` | 把 Dockerfile 的旧式 ENV 直接改成 export | 继承容器镜像的 ENV，删除重复的 shell 转换 |
| Gophercloud n000009、PennyLane n000019、Sopel n000027：必测项均通过 | 使用 passed 集合完全相等规则，额外测试通过反而失败 | 按全部 FAIL_TO_PASS、PASS_TO_PASS 是否通过评分，保留 extra_passed_tests 诊断 |
| Downlevel n000020：Jest 明确 1 failed，parser 得到零项 | 解析器漏掉真实断言失败 | 明确失败的测试摘要可以证明 failed，但不能凭摘要推断 passed |
| GraphHopper n000014、Qutebrowser n000054 | Maven 依赖下载失败 / 缺 adblock 被算作模型失败 | 分类 runtime_dependency，移出有效评分分母 |
| Deku、SWC、Fabrikt、Flipt | 测试依赖下载失败，原来统称 zero_tests | 分类 runtime_dependency，并保留诊断行 |
| Tutanota n000066 | TS2322 本地源码类型错误，parser 只有 Build/Runtime Error 占位符 | 使用明确的源码编译证据判 failed；无失败证据的占位符仍保持 error |

特别防止误判：Qutebrowser n000063 的断言文本中包含 `ModuleNotFoundError` 字符串，
那是测试输出的值，不是初始化异常。回归测试验证该记录仍为任务 failed。

## 只重分类已有日志的结果

| 47 条已有评测 | 原始结果 | 修复规则复核 |
|---|---:|---:|
| passed | 12 | 15 |
| failed | 15 | 11 |
| error / 无有效评分 | 20 | 21 |
| 有效评分通过率 | 12/27 = 44.44% | 15/26 = 57.69% |

这是日志分类审计，不是重新执行测试所得的 benchmark 新成绩。
14 条补丁冲突和 ENV 错误仍保留 error，需要环境就绪后用 `--eval-only` 重评。
全目录计入未启动/中断记录后，共 43 条未评分。原已知 API 费用约 9.3354554 CNY，新增 0。
完整审计见 `run_0929/swe-neo-deepseek-flash-0929/provenance/evaluation-audit-v3/summary.json`。

## 运行与恢复

- 本地导出/评测异常不再误设 `cost_complete=false`，可以继续下一题。
- 无预算 run-mode 遇未知 API 费用继续下一题，摘要保留费用不完整标记；设有费用预算时仍停止。
- 初始化失败为 `initialization_error`；已有动作重放不一致才为 `replay_failed`。
- `--resume` 校验原任务/采样/模型/价格/提示配置，恢复 RNG 与唯一调用账本，跳过已完成 job。
  保留旧 workspace，使用新的 replay 目录；已经持久化 usage 的响应可恢复费用。
  无法确认工具是否执行的调用保留 `interrupted_call`，不盲目重复执行。
- `--precheck` 跑 empty/gold 评测，`--precheck --dry-run` 只做静态检查。
- `--eval-only` 重评保存的 patch，并在新目录导出更新标签的 trace/step；不调用模型。
- `--reparse-only` 只审计已有日志，不启动容器。

## 协议与限制

OpenAI-compatible 适配器保留 reasoning_content，同服务/模型回传；非法工具参数保留
原文和 usage，工具返回错误信息而不执行。显式配置 profile 的 context_window 后，
按保守字节估算裁掉完整旧交互组，并保留原始 trace；未配置时不会猜测窗口。
此保护不是模型专用 tokenizer，也未扩展到 Anthropic/Gemini 的所有思考签名协议。

SWE shell 默认 300 秒；模型可请求更长时间，独立上限默认 3600 秒。
不在本次修复中处理镜像拉取、依赖安装和联网策略。

离线测试涵盖真实 Git 补丁冲突、日志分类、ENV 继承、工具超时、畸形工具参数计费、
reasoning 回传、完整交互组裁剪、续跑去重/RNG/费用恢复、仅重评的标签导出和 empty/gold 预检。
命令及参数说明见 `docs/INFERENCE.md`。

## 本轮验证结果

- 新增 20 项离线回归测试全部通过（`tests/test_harness_debug.py`）。
- 全量测试 165 项：164 项通过；1 项因外部测试数据
  `/Users/liangjianfeng/Data/mixed-pilot-data/tasks/swebench-flask-5014/manifest.json`
  缺失报错，该项是已有环境缺项。
- 实际 SWE-neo 数据集前 2 题的推理 `--dry-run`、预检 `--dry-run` 均通过。
  静态预检不代表镜像和依赖已经可用。
- `git diff --check` 通过。
- 未执行真实模型推理或 Docker 重评；补丁冲突修复通过本地临时 Git 仓库验证，
  容器内端到端结果仍需环境恢复后确认。
