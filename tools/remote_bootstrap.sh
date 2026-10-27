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
OUT_DIR="${3:-/data/ldy/ek55_assets}"          # 产物写自己的目录，绝不写共享数据集
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

GPU_LIST="${GPU_LIST:-0,1,3}"
if command -v nvidia-smi >/dev/null 2>&1; then
  # 只逐张查询可用卡：本机万一有卡掉总线，整机查询会整体失败（历史故障：GPU 2, PCI 0000:67:00.0）
  echo "[GPU] 只查可用卡（GPU_LIST=${GPU_LIST}）："
  nvidia-smi -i "${GPU_LIST}" --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
             --format=csv,noheader 2>&1 | sed 's/^/        /'
  echo "[GPU] 这些卡上的**计算**进程（桌面/远程控制不算）："
  nvidia-smi -i "${GPU_LIST}" --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
             --format=csv,noheader 2>&1 | sed 's/^/        /'
  echo "[GPU] 整机状态（若出现 Unable to determine the device handle = 某张卡掉总线，可忽略）："
  nvidia-smi 2>&1 | grep -iE "unable|error" | head -3 | sed 's/^/        /'
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

echo "[找标注 CSV]（限深度 3，避免长时间扫描）"
find /data/datasets/EPIC-KITCHENS -maxdepth 3 -name "*.csv" 2>/dev/null | head -20 | sed 's/^/        /' || true
if [ ! -d "${DATA_ROOT}" ]; then
  echo "[找帧目录] --data-root 不存在，尝试定位 rgb_frames（限深度 4；若仍很慢可 Ctrl-C 跳过）："
  find /data -maxdepth 4 -type d -name "rgb_frames" 2>/dev/null | head -5 | sed 's/^/        /'
  echo "        备选：看 EPIC-KITCHENS 下体积最大的目录"
  du -sh /data/datasets/EPIC-KITCHENS/* 2>/dev/null | sort -hr | head -8 | sed 's/^/        /'
fi

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
