# SWE 环境准备（不调用模型）

在新服务器上从公共镜像重建环境，见 [服务器恢复流程](SWE_SERVER_SETUP.md)。
`pull_swe_images.py --dataset ...` 会生成当前服务器的路径，避免依赖 Mac 的旧 plan 和本地镜像映射。

范围为已有 `run_0929/swe-neo-deepseek-flash-0929/plan.json` 的 100 个 train 任务。
镜像下载完成不等于环境就绪；以禁网基线预检为准。

## 1. 下载镜像

在项目目录执行（工作目录必须是 mixed-pilot）：

```bash
python3 scripts/pull_swe_images.py \
  --plan run_0929/swe-neo-deepseek-flash-0929/plan.json \
  --output run_0930/environment-preparation/images
```

已有 linux/amd64 镜像直接复用，缺失镜像最多尝试 5 次，单次超时 1800 秒，
失败后有递增等待。重复执行会继续复用已下载镜像。可用 `--attempts` 和 `--timeout` 调整。
不会清理镜像或改写原始数据。进度写入 `images/images.json`，每次下载尝试保留独立日志。

## 2. 下载缺失依赖并禁网验证

确认镜像批次完成后执行：

```bash
/Users/liangjianfeng/miniconda3/envs/opensquilla/bin/python -u \
  scripts/prepare_swe_dependencies.py \
  --plan run_0929/swe-neo-deepseek-flash-0929/plan.json \
  --output run_0930/environment-preparation/dependencies \
  --proxy http://host.docker.internal:8902
```

8902 是此次检查确认可用的本机代理端口。客户端端口改变时修改此参数。
这个地址供容器访问宿主机代理；不要在容器中使用 `127.0.0.1:8902`。

每个任务执行以下流程：

1. 对原镜像运行禁网 gold / empty 基线。已就绪镜像直接复用。
2. 对未就绪任务启动独立联网容器，在原始 base_commit 上下载依赖。
   使用现有工具链和项目锁文件；不在这个容器内放入标准答案、测试补丁或宿主机目录。
3. 保存为 `swe-neo-prepared:<任务哈希>` 本地镜像，保留原始镜像标签。
4. 在新的禁网容器中再次运行 gold / empty。gold 必须通过、empty 必须失败才标为 ready。
   标准答案只用于一次性评测容器，不写入准备镜像。

该流程默认串行，CPU 上限 4、内存 8g；联网准备超时 1800 秒，每次评测超时 1200 秒。
参数为 `--cpus`、`--memory`、`--prepare-timeout`、`--eval-timeout`。
下载子进程使用非交互环境并关闭标准输入，不等待人工确认；仍阻塞时按准备时限终止，
清理宿主机进程组和本次准备容器，记录错误后跳过该任务。
不要统一下载全部子模块；只有验证表明确实需要时才定向补齐对应路径，并重新验证镜像。
推理源码从最终选定镜像提取，保持该镜像实际具备的子模块状态；隐藏测试和原 Git 历史
不会进入供 Agent 使用的源码基线。
网络准备完成仍可能有工具链、浏览器、编译或测试解析问题：这些任务保留 error 和诊断，
不会被误报为 ready。没有明确安装规则的任务也会保留错误，不擅自猜测依赖版本。

准备工具可重复执行。已经 ready 且镜像仍在的任务会跳过；已有错误默认保留，
使用 `--retry-failed` 重试。指定单个任务：

```bash
/Users/liangjianfeng/miniconda3/envs/opensquilla/bin/python -u \
  scripts/prepare_swe_dependencies.py \
  --plan run_0929/swe-neo-deepseek-flash-0929/plan.json \
  --output run_0930/environment-preparation/dependencies \
  --proxy http://host.docker.internal:8902 \
  --only google-go-safeweb --retry-failed
```

## 3. 查看结果并用于下一次推理

- `images/images.json`：镜像下载进度。
- `dependencies/preparation.json`：任务准备状态、错误、日志位置。
- `dependencies/ready-images.json`：通过禁网基线的镜像映射。
- 每个任务目录保存原镜像和新镜像的评测日志，以及依赖下载日志。

确认目标任务全部 ready 后，在**新推理进程**中启用映射，再运行原有推理命令：

```bash
export SWE_NEO_PREPARED_IMAGES="$PWD/run_0930/environment-preparation/dependencies/ready-images.json"
```

推理及评测都通过同一映射选取镜像，仍然禁网。映射会核对原始镜像引用和 base_commit。
映射缺少的任务会在环境选择阶段报错，避免悄悄用未准备的环境消耗模型 token。
若数据集中还有未就绪任务，应先解决错误或准备只含 ready 任务的数据集，再开始推理。
已有运行的结果、API 账本和失败记录不会被这个流程修改。

取消该映射后恢复原有镜像选择：

```bash
unset SWE_NEO_PREPARED_IMAGES
```
