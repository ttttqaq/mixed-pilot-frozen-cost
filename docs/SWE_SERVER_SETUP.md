# 在新服务器恢复 SWE-neo 环境

本项目的当前 train 包含 SWE-rebench V2 50 条、SWE-bench Pro v1 50 条。
两类任务都在 `reference/source_record.json` 中保留远程镜像引用：

- rebench：`image_name`（`docker.io/swerebenchv2/...`）。
- Pro v1：`dockerhub_tag`（完整地址为 `jefzda/sweap-images:<tag>`）。

这沿用 [SWE-bench Pro 的原始 Docker Hub 方案](https://github.com/scaleapi/SWE-bench_Pro-os#docker-images)：
目标服务器通过网络拉取每条任务的环境。复制本项目代码和 SWE-neo-data 后，在服务器安装
Python 依赖、Git、Docker，确保 Docker daemon 可访问。当前镜像按 `linux/amd64` 使用，优先
选择 Linux x86_64 服务器；ARM 主机需要兼容的仿真环境。

在项目根目录安装控制端基础依赖：`python -m pip install -e '.[collection]'`。

新版 Pro V2 的官方镜像在 GHCR，使用新的任务目录和 verifier。当前数据与 grader 保持
Pro v1 协议，不将 V2 镜像直接混接到原有任务。

## 1. 按服务器本地的数据路径生成计划并拉取镜像

以下命令在服务器上的 mixed-pilot 项目目录执行。修改数据路径；不复制 Mac 的旧 plan，
因为它包含 `/Users/...` 绝对路径。

```bash
export SWE_DATA=/srv/data/SWE-neo-data
unset SWE_NEO_PREPARED_IMAGES

# 预览并生成服务器自己的 plan.json/image-plan.json。
# 不调用模型，也不启动 Docker 或下载镜像。
python scripts/pull_swe_images.py \
  --dataset "$SWE_DATA" --split train \
  --output run/server-environments/images --dry-run

# 拉取 100 个原始远程镜像，复用本服务器已缓存的镜像，失败按配置重试。
python scripts/pull_swe_images.py \
  --dataset "$SWE_DATA" --split train \
  --output run/server-environments/images
```

拉取工具始终使用 source_record 中的原始引用，即使 shell 中遗留了 Mac 的
`SWE_NEO_PREPARED_IMAGES` 设置，计划生成也会忽略它。`plan.json` 中的 record_path/task_dir
根据当前服务器数据目录生成。拉取结果写到 `images.json`，各次尝试保存独立日志。

镜像地址可恢复环境的前提是上游仓库仍提供这些 tag，且服务器能访问镜像仓库。
`--dry-run` 只验证本地记录与路径，不验证远端可访问性或执行环境。

## 2. 在服务器上验证、补齐缺失依赖

原始镜像拉取成功不代表禁网测试的所有依赖缓存都齐全。Mac 上最终有 9 个本地补依赖
镜像；它们不是额外的必需搬运物，可以在服务器重新准备。是否需要补齐，以服务器实测为准。

```bash
python scripts/prepare_swe_dependencies.py \
  --plan run/server-environments/images/plan.json \
  --output run/server-environments/dependencies \
  --cache run/inference-cache
```

脚本先在原始镜像中执行 gold / empty 的禁网基线，原始镜像通过时直接使用。
缺少依赖时，联网安装到临时容器，再提交为本服务器的 `swe-neo-prepared` 镜像，
随后用禁网基线验证。gold 必须 passed、empty 必须 failed 才可用于可信评分。
请检查 preparation.json 的实际结果和日志；脚本没有替你迁移本机的旧验证结论。

准备阶段需要访问上游代码仓库和语言包仓库，不需要模型 API key。涉及大型项目时，
根据服务器资源调整 `--prepare-timeout`、`--eval-timeout`、`--cpus`、`--memory`。

## 3. 运行推理

确认本服务器所选任务全部 ready 后，启用服务器新生成的映射：

```bash
export SWE_NEO_PREPARED_IMAGES="$PWD/run/server-environments/dependencies/ready-images.json"

python scripts/infer.py \
  --dataset "$SWE_DATA" --split train \
  --mode single-model --models deepseek-flash \
  --output-dir run/swe-server-inference --dry-run
```

模型凭据、harness.json 和价表按服务器实际配置准备，确认计划后去掉 --dry-run。
若直接使用原始镜像，先 unset SWE_NEO_PREPARED_IMAGES；不要启用数据包中带 Mac 本地标签的
environment-images.json，除非这些镜像已在本服务器准备/导入并验证。

不需要迁移 Docker.raw、旧源码缓存、旧日志目录或本机 Docker 镜像库存。
服务器仍需要 Docker 运行这些 Linux 环境；此流程去除的是对原机器镜像缓存的依赖。
拉取和现场构建不会保证与 Mac 缓存逐字节相同；依赖版本、镜像 tag 更新和运行平台变化应以
本服务器验证结果为依据。要严格复现同一镜像内容，可另外保存镜像 digest 或发布已准备镜像。
