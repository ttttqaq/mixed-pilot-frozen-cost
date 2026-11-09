# Profile 推理结果可视化

输入一个或多个 `scripts/infer.py` 生成的日志目录，每个目录在图中对应一个散点：

- 主图：横轴平均 cost / task，纵轴 accuracy（`pass_rate`）。
- 副图：横轴分别为平均 tokens / task、平均 steps / task，纵轴保持相同的 0–100% accuracy 坐标。

只需 Python 3.9+ 标准库，不需要安装画图库或访问网络。生成的页面可直接在浏览器打开。

## 使用

在仓库根目录执行，支持相对路径或绝对路径：

```bash
python3 visualized/generate.py \
  run_0929/virtualworld-100-glm-5.3-0929 \
  run_0929/virtualworld-100-deepseek-flash-0929 \
  run_0929/virtualworld-100-qwen-flash-0929 \
  --open
```

默认写入 `visualized/output/`。`--open` 自动用默认浏览器打开报告，也可以手动打开生成的 `index.html`。

```bash
python3 visualized/generate.py \
  /path/to/run-a /path/to/run-b \
  --label 'GLM 5.3' --label 'Random router' \
  --title 'VirtualWorld-100 profile 对比' \
  --output-dir visualized/my-comparison
```

`--label` 可省略（默认目录名）；指定时必须按输入顺序为每个目录提供一个。每次执行重新读取日志，覆盖该输出目录中的报告。

输出：

| 文件 | 内容 |
| --- | --- |
| `index.html` | 内嵌数据、离线交互的三张散点图、汇总表与统计口径 |
| `metrics.json` | 每个运行的均值、资源总量、评分覆盖、profile 与注意事项 |
| `metrics.csv` | 可在 Excel 打开的 UTF-8 BOM CSV；多币种运行每个币种一行 |

页面支持悬浮/键盘聚焦查看散点、点击散点或表格查看详情、图例显隐、币种切换，以及各图单独导出 SVG。图中的编号对应图例与表格；完全重叠的点可从表格分别选中。空心点和 `≥` 表示对应资源数据不完整；没有 pass_rate 的运行只保留表格记录。

## 日志要求与统计口径

输入目录需要 `summary.json` 和 `calls/*.json`，可选 `plan.json`。推荐在运行结束后生成报告；调用数量与 summary 不一致时会报错，避免混用写入过程中的快照。

1. **Accuracy**：直接使用 `summary.json` 的 `pass_rate`，保留 harness 的 `passed / scored leaf trajectories` 定义。未评分轨迹不进入分母；KSR 的多个叶子可能属于同一个 task，不能当作独立 task 准确率。
2. **平均 cost / task**：`summary.cost_by_currency[币种] / summary.total_tasks`。旧日志回退到 `summary.total_api_cost` 和 `summary.currency`。使用已知总费用，包含 evaluator；遵循日志中的价格快照，不重新查询价格。不同币种不相加，可在页面切换；多币种运行在 cost 图中只展示所选币种的费用部分。
3. **平均 tokens / task**：`sum(call.usage.input_tokens + call.usage.output_tokens) / total_tasks`。优先使用标准化字段，也兼容 `prompt_tokens` / `completion_tokens`。输入包括 cached tokens；输出已有 reasoning tokens，不重复加。缺失 usage 不伪造为完整零消耗，页面会标出缺失调用数。
4. **平均 steps / task**：`calls/ 中唯一 call_id 数量 / total_tasks`。一个逻辑 API call 算一个 step；包含失败调用、被替代的重试轨迹及所有分支，HTTP transport retries 不另算 step。这是 API step，不是环境动作或工具执行次数。
5. **Task 分母**：默认使用 `summary.total_tasks`（唯一 task 数），缺失时从 `plan.tasks` 或 `summary.jobs` 获取。多轮、分支、重跑的资源一起除以 task 数；未启动 task 仍计入分母。详情会展示覆盖情况，未完成运行的均值不能视作最终结果。

cost、tokens、steps 均基于整个运行的资源消耗；费用使用 summary 的唯一调用账单，tokens 和 steps 读取 `calls/`，不会把不同 trace 的共享前缀重复计数。不同输入的 task 集合/数量不一致时，报告会提示比较限制。

## 验证

```bash
python3 -m unittest discover -s tests -p 'test_visualized.py'
```
