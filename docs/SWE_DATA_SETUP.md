# 原版 SWE-bench 数据集与服务器环境

`/Users/liangjianfeng/Data/Router/SWE-data-1000` 的任务布局以 `SWE-neo-data-1000` 为模板。
它从 `princeton-nlp/SWE-bench` 官方 full test 的 2294 条中无放回抽取 1000 条，种子为 20261014。
本地均标为 train，官方来源仍为 test；这些任务用于训练后，需要从独立评测集合中排除重叠实例。

数据版本固定为 `e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6`。`sampling.json` 包含原始行号和随机顺序，
`checksums.json` 包含全部文件的 SHA-256。`source_record.json` 完整保留官方字段，包括 JSON 字符串形式的测试清单。

## 构建

Python 3.10+，在 mixed-pilot 项目根目录执行：

```bash
python -m pip install '.[swebench]'
python scripts/build_swe_data.py \
  --cache /tmp/swebench-source --download \
  --output /data/SWE-data-1000 --count 1000 --seed 20261014 --verify-official
```

目标目录必须尚不存在。构建在临时目录完成校验后整体发布；失败时不会留下半成品目标目录。
`--verify-official` 逐条检查官方 TestSpec、评测脚本和镜像映射，不启动 Docker。

## 迁移与运行

复制数据集和更新后的 mixed-pilot 项目到 Linux x86_64 服务器，安装上述 extra，然后：

```bash
unset SWE_NEO_PREPARED_IMAGES
python scripts/pull_swe_images.py --dataset /data/SWE-data-1000 \
  --split train --output /data/swe-original-images --dry-run
python scripts/pull_swe_images.py --dataset /data/SWE-data-1000 \
  --split train --output /data/swe-original-images
python -m mixed_pilot_harness.inference.cli --dataset /data/SWE-data-1000 \
  --split train --precheck --limit 1 --output-dir /data/swe-original-precheck
```

`workflow/images.json` 记录公开的 `swebench/sweb.eval.x86_64.<instance>:latest` 引用；镜像名遵循官方规则，
其中 `__` 编码为 `_1776_`。不需要本机 Docker.raw、镜像导出 tar 或 `swe-neo-prepared:*`。
镜像在 `/testbed` 内提供仓库和已安装的 conda `testbed` 环境，agent shell 自动选择该环境。
公开镜像已经安装依赖，不要直接使用 SWE-neo 的依赖重建流程。

拉取脚本按任务断点续跑、重试，输出 `images.json`。先检查空补丁失败、gold 补丁通过，再扩展运行。
推理继续使用 `mixed-pilot-infer --dataset /data/SWE-data-1000 --split train` 及项目现有模型/路由参数。

## 原版评测适配

原版 SWE-bench 使用独立适配器，固定官方 `swebench==4.1.0`
（源码 revision `726c5461e2ef52d83cf1ea2107870a8bb3328d57`），避免评测接口更新改变数据含义。
只从官方库生成评测脚本并调用 `get_eval_report`；不使用 Pro 或 rebench 的测试解析器。
实例镜像已安装环境，因此不生成环境构建脚本，也不额外从 GitHub 下载历史 requirements。

每次评分使用独立容器，保留官方补丁应用顺序、隐藏测试补丁和测试日志解析。
stdout/stderr 合并写入 `test_output.txt`，官方报告保存为 `report.json`，统一结果保存为 `result.json`。
测试超时、补丁无法应用或测试脚本出错记录为未评分 error。运行保留项目的 CPU/内存上限和禁网策略。
`--eval-only` 和 `--reparse-only` 均可用于该数据集；后者使用官方原版日志解析器。
空补丁基线跳过补丁应用，测试未修改镜像的实际表现。

本次构建仅验证数据、镜像命名和评测配置，没有下载镜像或执行 Docker 测试。
`evaluation.ready=false`，服务器需另行验证 registry 可用性和空/gold 基线。
上游 `latest` 标签尚未固定 digest；同名镜像未来可能变化。
