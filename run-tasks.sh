cd /Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot
export PYTHON=/Users/liangjianfeng/miniconda3/envs/opensquilla/bin/python

export PYTHON=/Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot/outputs/virtualworld-venv/bin/python

export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
export ZHIPU_API_KEY="${ZHIPU_API_KEY:-}"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-}"

# 确认计划后去掉 --dry-run 执行；使用新输出目录。
"$PYTHON" scripts/infer.py \
  --dataset /Users/liangjianfeng/Data/Router/VirtualWorld-subset \
  --split test \
  --mode single-model \
  --models qwen-flash-2025-07-28 \
  --stop-mode run-mode \
  --runs 1 \
  --output-dir run_1017/virtualworld-test-qwen-flash \
  --max-retries 5 \
  --retry-wait 10 \
  --step-limit 50 \
  --resume-partial

python scripts/infer.py \
  --dataset "$SWE_DATA" \
  --split train \
  --mode single-model \
  --models glm-4.7-flashx \
  --stop-mode run-mode \
  --runs 1 \
  --output-dir run_1010/swe-neo-100-glm-4.7-flashx-1010 \
  --max-retries 10 \
  --retry-wait 10 \
  --step-limit 200

# 采样 Step 标签数据
python scripts/collect_steps.py \
  --dataset /Users/liangjianfeng/Data/Router/SWE-neo-data \
  --split train --mode run --runs 1 \
  --sampling-strategy random \
  --output-dir run/step-random

"$PYTHON" scripts/collect_steps.py \
  --dataset /Users/liangjianfeng/Data/Router/VirtualWorld-100 \
  --split train --mode budget --budget 20 \
  --sampling-strategy random \
  --output-dir run_1002/step-random \
  --ksr-config KSR_config.json \
  --max-depth 10 --max-nodes 50 \
  --resume-partial