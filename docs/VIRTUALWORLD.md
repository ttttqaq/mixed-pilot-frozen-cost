# VirtualWorld-1000

数据目录为 `outputs/VirtualWorld-1000`，以
`/Users/liangjianfeng/Data/Router/scienceworld_task` 的五部分目录格式构建。
每条任务包含 `manifest.json`、`prompt.md` / `prompt.txt`、`workspace/`、
`reference/` 和 `rubrics.json`。实际评分通过 `scripts/infer.py` 的原生环境适配器进行。

| 数据来源 | train | test | 合计 |
| --- | ---: | ---: | ---: |
| ScienceWorld | 400 | 100 | 500 |
| ALFWorld | 400 | 100 | 500 |
| 合计 | 800 | 200 | 1000 |

ScienceWorld 使用官方 train / test variation 划分，未抽取 dev；ALFWorld 的
test 来自官方 valid_seen 50 条和 valid_unseen 50 条，`source.split` 保留原始名称，
`routing_metadata.router_split` 保存本数据集的 train / test。这个 test 是本地实验划分，
不等同于 ALFRED 的隐藏 leaderboard test。训练集预先排除 ALFWorld 官方评测任务组，
避免同组的不同 trial 跨越本地 train/test。valid_unseen 的官方可运行游戏只有 47 个组，
因此按不同 trial 补足 50 条，同一组最多两条，不重复实例。

固定随机种子 42，类型分层并以轮询方式抽样；优先不同任务组。每条入选任务必须
通过原生 reset 和 oracle 路径重放，达到 100 分。失败候选保存在
`scienceworld.rejections.json` 和 `alfworld.rejections.json` 中，再从剩余候选补选。
这是筛选可运行实例的成功路径数据集，不能把通过率直接作为原始全量 benchmark 的结果。
具体各类型计数和验证范围见 `dataset.json`、`validation_report.json`。

## 运行环境

本机已准备 `outputs/virtualworld-venv/bin/python`，继承 opensquilla 环境中的
ScienceWorld 1.2.3 和 Java，并独立安装 ALFWorld 0.4.2、TextWorld 1.7.0、
fast-downward-textworld 20.6.4。没有修改 opensquilla 环境。

TextWorld 的 macOS 安装脚本会下载与 PDDL 环境无关的 Inform7 编译器。
本次使用 PDDL 专用 wheel，跳过这个构建步骤；官方 TextWorld Python 环境代码未修改。
wheel、来源、依赖版本和构建说明保存在数据集的 `runtime/` 中。跨平台重建需针对平台
重新编译 fast-downward-textworld 的动态库。

```bash
# 结构、样本身份、split、答案隔离、1000 条 oracle 证据、全部文件校验和
outputs/virtualworld-venv/bin/python scripts/validate_virtualworld.py \
  outputs/VirtualWorld-1000

# 先验证完整运行计划；不调用模型
outputs/virtualworld-venv/bin/python scripts/infer.py \
  --dataset outputs/VirtualWorld-1000 --split all \
  --mode single-model --models deepseek-flash --step-limit 500 \
  --output-dir run/virtualworld-deepseek --dry-run

# 实际运行：去掉 --dry-run，需要相应模型 API key，会产生调用费用
outputs/virtualworld-venv/bin/python scripts/infer.py \
  --dataset outputs/VirtualWorld-1000 --split test \
  --mode single-model --models deepseek-flash --step-limit 500 \
  --output-dir run/virtualworld-test
```

`random` / `ksr` 同样支持这个混合集，按现有文档设置多个模型和预算。
ALFWorld 使用与 ScienceWorld 相同的单个 `text` 代码块命令协议，`?all` 免费查询动作。
系统根据 benchmark 自动分派环境。ScienceWorld 保留原生分数（包括失败负分），
ALFWorld 仅在原生 `won` 为真时给 100 分，其余为 0；提前提交不能伪造成功。

`--step-limit 500` 是模型调用上限，ScienceWorld 的原生动作上限为 500，ALFWorld
为 200。免费查询和格式错误会消耗模型调用，因此 oracle 路径长度不表示一定能在
相同数量的模型调用中完成。

## workspace 兼容入口

复制单条任务的 `workspace/` 到独立 attempt 后，可以通过 `python vw.py start`、
`info`、`valid`、`step "动作"`、`stop` 交互；ScienceWorld 同时有 `sw.py`，
ALFWorld 同时有 `alf.py`。它们使用隔离的回环端口、随机 session token、启动日志、
有界启动等待、原子写入状态和 15 分钟空闲回收。

推荐使用 `infer.py` 进行评测。普通 `code-task` 的文件/rubric 评分并未接入这两个
原生评分方法；workspace 的 `score.json` 是可写文件，不能用于可信 benchmark 判分。
shell 模式下 ALFWorld 的 PDDL 初始状态可以被读取，属于不同的观测条件。
`infer.py` 仅向模型提供原生文本观察和动作接口，不提供 PDDL 文件和 reference。

## 来源与复现

官方来源：[ScienceWorld](https://github.com/allenai/ScienceWorld)、
[ALFWorld](https://github.com/alfworld/alfworld)。来源版本、下载地址、文件哈希和许可证
保存在每条 manifest / reference 和数据集根目录的 `provenance.json`、`licenses/` 中。
ScienceWorld gold 由固定 1.2.3 运行时为该 variation 生成并重放验证；没有把旧版本
公开 goldpaths 文件中的描述或动作混接到新运行时。ALFWorld 保留原始游戏和
ALFRED traj_data 字节，公开游戏只保留 domain、problem、grammar，移除 walkthrough。

下载的官方源包位于 `outputs/virtualworld-sources`。从一个**新的**输出目录重建：

```bash
outputs/virtualworld-venv/bin/python scripts/build_virtualworld.py \
  --sources outputs/virtualworld-sources --output outputs/VirtualWorld-rebuild --seed 42
```

ScienceWorld 1.2.3 在不同 JVM/重置下可能存在对象排序等差异，筛选失败也可能变化；
seed 固定候选顺序，但不承诺不同机器的生成结果逐字节相同。
已交付数据通过 SHA256SUMS 固定内容，并与当前 harness 的 ScienceWorld 版本一致。
每条任务的验证记录代表本机实际运行证据，不代表任意运行环境都已验证。
