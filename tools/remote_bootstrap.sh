#!/usr/bin/env bash
# remote_bootstrap.sh —— 在 amax（或任意 Linux 训练机）上一键完成"盘点三件事"
#
#   ① 环境：Python/torch/CUDA/GPU 可见性 + 谁在用卡（共享机器礼仪）
#   ② 数据：EK55 帧与标注到底有什么（inventory_data.py）
#   ③ 资产：采样帧生成**干跑**（不写文件，只报告会跳过多少段）
#
# 全部不写大文件、不占 GPU（除一次 torch.cuda 探测）。跑完把末尾"回传区"发出来即可。
#
# 用法：
#   bash tools/remote_bootstrap.sh
#   bash tools/remote_bootstrap.sh <帧根目录> <标注目录> <产物目录>
#   PYTHON=/data/conda_envs/insight_env/bin/python bash tools/remote_bootstrap.sh
#
# 环境变量：
#   PYTHON      Python 解释器（默认 /data/conda_envs/insight_env/bin/python）
#   GPU_ID      用于探测的显卡编号（默认 0；**不要用 GPU 2**——它在该机有硬件故障）

set -uo pipefail

PYTHON="${PYTHON:-/data/conda_envs/insight_env/bin/python}"
GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${1:-/data/datasets/EPIC-KITCHENS/rgb_frames}"
ANN_DIR="${2:-/data/datasets/EPIC-KITCHENS/insight_annotations}"
OUT_DIR="${3:-/data/datasets/EPIC-KITCHENS/ckpt_assets}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT="$PROJECT_DIR/results/bootstrap_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$PROJECT_DIR/results"

# 主体包进函数，最后用 `main | tee` 落盘（比 exec > >(tee) 更可移植）
main() {
echo "======================================================================"
echo " remote_bootstrap @ $(hostname)  $(date '+%F %T')"
echo " 项目目录：$PROJECT_DIR"
echo " 解释器  ：$PYTHON"
echo "======================================================================"

# ===== ① 环境 =====
echo
echo "---------- ① 环境 ----------"
if [ ! -x "${PYTHON}" ]; then
  echo "[警告] 解释器不是可执行文件：${PYTHON}（若给的是命令名，下面自动解析）"
  PYTHON="$(command -v "${PYTHON}" 2>/dev/null || command -v python3 || true)"
fi
if [ -z "${PYTHON}" ]; then
  echo "[错误] 找不到任何 python，请用 PYTHON=<绝对路径> 重跑"
  exit 1
fi
echo "[python] $($PYTHON -V 2>&1)"
echo "[磁盘] /home 与 /data 使用情况（产物必须写 /data 或 /tmp）："
df -h /home /data /tmp 2>/dev/null | sed 's/^/        /'

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[GPU] nvidia-smi 摘要（注意别用 GPU 2）："
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
             --format=csv,noheader 2>/dev/null | sed 's/^/        /'
  echo "[GPU] 正在用卡的进程（共享机器，先看再决定）："
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/        /'
else
  echo "[GPU] 没有 nvidia-smi"
fi

echo "[torch] 探测（仅用 GPU ${GPU_ID}，几秒）："
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" - <<'PY' 2>&1 | sed 's/^/        /'
try:
    import torch
    print("torch", torch.__version__, "| cuda_available", torch.cuda.is_available(),
          "| cuda_version", getattr(torch.version, "cuda", None))
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0),
              "| memory_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
        x = torch.randn(1024, 1024, device="cuda")
        print("矩阵乘自检通过：", float((x @ x).sum()) == float((x @ x).sum()))
    else:
        print("[警告] CUDA 不可用——检查驱动/CUDA_VISIBLE_DEVICES")
except ImportError as error:
    print("[警告] 该解释器没有 torch：", error)
PY

echo "[环境 JSON] 写入 results/："
"$PYTHON" "$PROJECT_DIR/tools/env_report.py" --quiet 2>&1 | tail -2 | sed 's/^/        /'

# ===== ② 数据清点 =====
echo
echo "---------- ② 数据清点 ----------"
echo "[检查路径]"
for path in "$DATA_ROOT" "$ANN_DIR"; do
  if [ -e "$path" ]; then
    echo "        ✅ 存在：$path"
  else
    echo "        ❌ 不存在：$path  ← 需要你确认真实路径后重跑"
  fi
done

echo "[找标注 CSV]（前 20 个，用于确认真实目录）"
find /data/datasets/EPIC-KITCHENS -maxdepth 4 -name "*.csv" 2>/dev/null | head -20 | sed 's/^/        /' || true

if [ -d "$DATA_ROOT" ]; then
  "$PYTHON" "$PROJECT_DIR/tools/inventory_data.py" \
      --data-root "$DATA_ROOT" \
      --annotations "$ANN_DIR" \
      --out "$PROJECT_DIR/data/inventory_$(date +%Y%m%d_%H%M%S).json" 2>&1 | sed 's/^/        /'
else
  echo "        [跳过] 帧目录不存在"
fi

# ===== ③ 资产干跑 =====
echo
echo "---------- ③ 资产生成（干跑，不写文件）----------"
ANNOTATION_CSV="$(find "$ANN_DIR" -maxdepth 2 -name "*.csv" 2>/dev/null | head -1 || true)"
if [ -n "$ANNOTATION_CSV" ] && [ -d "$DATA_ROOT" ]; then
  echo "[用标注] $ANNOTATION_CSV"
  "$PYTHON" "$PROJECT_DIR/tools/build_ek55_assets.py" \
      --annotations "$ANNOTATION_CSV" \
      --frames-root "$DATA_ROOT" \
      --out-dir "$OUT_DIR" \
      --frames-per-segment 8 --image-size 224 --workers 16 \
      --limit 200 --dry-run 2>&1 | sed 's/^/        /'
  echo "        （--limit 200 只试前 200 段；全量干跑去掉 --limit 即可）"
else
  echo "        [跳过] 需要同时存在帧目录与至少一个标注 CSV"
fi

# ===== 回传区 =====
echo
echo "======================================================================"
echo " 回传区（把下面这段发给研究助手即可）"
echo "======================================================================"
echo "1) 主机与时间：$(hostname) $(date '+%F %T')"
echo "2) 解释器与 torch 版本：见上方 [torch] 两行"
echo "3) GPU 现状：见上方 [GPU] 摘要（用的哪张卡、显存占用）"
echo "4) 数据：见上方 [数据清点] 的『视频/帧/标注』三行 + 体检提示"
echo "5) 资产干跑：见上方 [完成：{...}] 计数（ok / partial / no_frames）"
echo "6) 报告文件：$REPORT"
echo
echo "下一步（等确认后再做，且只需跑一次）："
echo "  a) 全量干跑： 去掉 --limit 200"
echo "  b) 正式生成：去掉 --dry-run（CPU/IO 活，不抢卡；产物约 6–10 GB，写 ${OUT_DIR}）"
echo "  c) 提特征：  脚本待补（需先定骨干：ViT-B/16 还是 EgoVideo ViT-L/14）"
echo "======================================================================"

}

main 2>&1 | tee "${REPORT}"
echo "[报告已落盘] ${REPORT}"
