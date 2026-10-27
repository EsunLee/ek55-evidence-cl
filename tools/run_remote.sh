#!/usr/bin/env bash
# run_remote.sh —— 在 AutoDL（Linux）上跑一个实验：环境记录 + 日志落盘 + 断线保护
#
# 用法（先在 tmux 里，或让脚本自己提醒你）：
#   ./run_remote.sh exp1 --synthetic
#   ./run_remote.sh exp1 --features data/ek55_past30.npz --epochs 50
#   PYTHON=/root/miniconda3/envs/cv/bin/python ./run_remote.sh exp2
#
# 约定：日志 → runs/<exp>_<时间戳>.log；环境 → results/env_<时间戳>.json；
#       训练结束后用 tools/sync_features.sh push-results 回传。

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "用法: $0 <exp1|exp2|...> [脚本参数...]"
  exit 1
fi

EXP="$1"
shift

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG="$PROJECT_DIR/runs/${EXP}_${RUN_ID}.log"
SCRIPT="$PROJECT_DIR/code/${EXP}.py"

mkdir -p "$PROJECT_DIR/runs" "$PROJECT_DIR/results"

if [ ! -f "$SCRIPT" ]; then
  echo "[错误] 找不到脚本：$SCRIPT"
  exit 1
fi

if [ -z "${TMUX:-}" ]; then
  echo "[提醒] 当前不在 tmux 里。AutoDL 上强烈建议先 tmux new -s exp，避免 SSH 断开导致任务中断。"
fi

echo "[环境] 记录运行环境（GPU/驱动/torch/CUDA + 代码版本）"
"$PYTHON" "$PROJECT_DIR/tools/env_report.py" --quiet | tail -1

echo "[运行] $EXP $*"
echo "[日志] $LOG"
started=$(date +%s)

# nohup 让任务不随终端退出而终止；tee 同时写日志与屏幕
nohup "$PYTHON" "$SCRIPT" "$@" 2>&1 | tee "$LOG"

finished=$(date +%s)
echo "[完成] 用时 $((finished - started)) 秒"
echo "[下一步] 1) ./tools/sync_features.sh push-results   2) 把摘要登记进研究工作区的实验索引"