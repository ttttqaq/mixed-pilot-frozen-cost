# 移植验证记录

日期：2026-09-21。本地源码来源与 SHA-256 见 `UPSTREAM.json`。

- 已有真实采集数据：174 个 `.traj.json`，3924 个 step，三个模型名称与每步标签解析通过。
- harness 全套 unittest：88 项通过，包含新 HTTP bridge 的模型选择、原请求转发、错误响应阻断、事件记录测试。
- 独立 core unittest：3 项通过。其中一个集成测试覆盖本地随机小型 BERT 的离线 embedding 预计算、分组划分、一轮训练、checkpoint 保存/加载、编码一致性、真实 HTTP 评分。
- 原始数据的币种是 CNY，已保留；混合币种与未知模型会被入口审计拒绝。
- 成功构建 wheel，检查了配置 YAML、训练 CLI、评分服务均包含在 wheel 中。
- harness bridge 导入检查：未导入 torch、transformers、openai 或独立核心包。
- `git diff --check` 通过。

测试使用已有 Python 3.12 环境中的依赖，版本见 `constraints-tested-macos-py312.txt`。
没有安装/升级现有环境依赖，没有修改原 MTRouter 仓库，没有调用付费模型 API。
真实 HTTP 测试在 127.0.0.1 临时端口运行，测试结束已关闭。

尚未验证：新建空 venv 后的联网依赖解析、完整 Qwen3 embedding 权重运行、全量训练质量、CUDA/MPS、远程 vLLM 服务。
Mac CPU 的小模型训练路径已验证，不能据此推断完整模型在该机器上的速度或内存占用。
