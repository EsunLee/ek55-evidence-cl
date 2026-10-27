#!/usr/bin/env bash
# sync_features.sh —— 特征/帧/结果的同步模板（rclone 与 ossutil 两种写法都给了）
#
# 为什么要脚本化：AutoDL 实例随时可能被回收，OSS 才是长期存储；同步必须可重复、可核对。
#
# 用法：
#   ./sync_features.sh pull            # 从 OSS 拉到本地 data/（默认 dry-run，先看清单）
#   ./sync_features.sh push-results    # 把结果与日志推回 OSS
#   DRY_RUN=0 ./sync_features.sh pull  # 真正执行
#
# 配置（按你的对象存储改这一行即可）：
#   rclone：先 `rclone config` 建一个远端（例如名字叫 oss），然后
#           export REMOTE="oss:你的桶名/ek55-evidence"
#   ossutil：export OSSUTIL_BUCKET="oss://你的桶名/ek55-evidence"
#
# 纪律：原始视频只在实例数据盘或 OSS 上，**永不进仓库**；仓库只收代码、文档、结果 JSON。

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${REMOTE:-oss:ek55-evidence}"
OSSUTIL_BUCKET="${OSSUTIL_BUCKET:-oss://ek55-evidence}"
DRY_RUN="${DRY_RUN:-1}"
TOOL="${TOOL:-rclone}"          # rclone | ossutil

flags=(--progress --transfers 8 --checkers 16)
if [ "$DRY_RUN" = "1" ]; then
  flags+=(--dry-run)
fi

run_rclone() {
  local source="$1"
  local target="$2"
  echo "[rclone] ${source} -> ${target}（DRY_RUN=${DRY_RUN}）"
  if [ "${DRY_RUN}" = "1" ]; then
    echo "  （预演：未实际调用 rclone）"
    return 0
  fi
  if ! command -v rclone >/dev/null 2>&1; then
    echo "[错误] 未安装 rclone。可改用 TOOL=ossutil，或先安装 rclone。"
    exit 1
  fi
  rclone copy "${source}" "${target}" "${flags[@]}"
}

run_ossutil() {
  local source="$1"
  local target="$2"
  if [ "${DRY_RUN}" = "1" ]; then
    echo "[ossutil] 预演：${source} -> ${target}（未实际调用 ossutil）"
    return 0
  fi
  if ! command -v ossutil >/dev/null 2>&1; then
    echo "[错误] 未安装 ossutil。可改用 TOOL=rclone。"
    exit 1
  fi
  if true; then
    echo "[ossutil] $source -> $target"
    ossutil cp -r -u "$source" "$target"
  fi
}

sync() {
  local source="$1"
  local target="$2"
  if [ "$TOOL" = "ossutil" ]; then
    run_ossutil "$source" "$target"
  else
    run_rclone "$source" "$target"
  fi
}

case "${1:-}" in
  pull)
    mkdir -p "$PROJECT_DIR/data"
    sync "$REMOTE/features/" "$PROJECT_DIR/data/"
    ;;
  pull-frames)
    mkdir -p "$PROJECT_DIR/data/frames"
    sync "$REMOTE/frames/" "$PROJECT_DIR/data/frames/"
    ;;
  push-features)
    sync "$PROJECT_DIR/data/" "$REMOTE/features/"
    ;;
  push-results)
    mkdir -p "$PROJECT_DIR/results" "$PROJECT_DIR/runs"
    sync "$PROJECT_DIR/results/" "$REMOTE/results/"
    sync "$PROJECT_DIR/runs/" "$REMOTE/logs/"
    ;;
  *)
    echo "用法: $0 {pull|pull-frames|push-features|push-results}"
    echo "环境变量: REMOTE / OSSUTIL_BUCKET / TOOL=rclone|ossutil / DRY_RUN=0|1"
    exit 1
    ;;
esac

echo "[完成] 提醒：结果回传后把摘要登记进研究工作区的实验索引（远程运行也要留痕）。"
echo "[可选] 不想用 rclone/ossutil，也可以直接用 AutoDL 官方工具（如他们的文件存储/网盘）或 sftp。"
echo "[可选] 若装了 ossutil，也可用：ossutil cp -r -u ./data oss://桶名/ek55-evidence/features/"