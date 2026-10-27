# build_ek55_assets.py —— 在 amax 上生成"可搬运资产"：采样帧 + 段表 +（可选）特征
#
# 为什么需要：原始帧体量大且只存在于数据所在机器；后续需要图像输入的阶段不应重复解码，
# 因此预先采样并 resize 成小体量资产（采样帧 + 段表 +（可选）特征）。
#
# 在 amax 上运行（采帧是 CPU/IO 活，不抢 GPU；提特征那一步才用卡）：
#   python tools/build_ek55_assets.py \
#       --annotations /data/datasets/EPIC-KITCHENS/annotations/EPIC_55_train.csv \
#       --frames-root /data/datasets/EPIC-KITCHENS/rgb_frames \
#       --out-dir     /data/datasets/EPIC-KITCHENS/ckpt_assets \
#       --frames-per-segment 8 --image-size 224 --workers 16
#
# 先干跑（只报告会做什么、跳过多少段，不写文件）：
#   python tools/build_ek55_assets.py ... --dry-run
#
# 产物（全部可上传到对象存储，总体积约 6–10 GB）：
#   <out>/frames/<narration_id>/000.jpg .. 007.jpg   采样帧（已 resize）
#   <out>/segments.csv                               narration_id, video_id, verb/noun 类, 帧区间
#   <out>/manifest.json                              配置、跳过清单、统计、sha1
#
# 设计取舍（重要）：
#   - 采样规则固定为"在 [start_frame, stop_frame] 上等距取 N 帧"，写进 manifest，保证可复现。
#   - 缺帧的段不猜、不补：记入 manifest 的 skipped 列表，由上层决定丢掉还是重下。
#   - 提特征是可选项（--with-features），因为那需要 torch 与 GPU；采帧不需要。

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# ===== 1. 依赖检查（采帧只需要 Pillow；提特征才需要 torch）=====

try:
    from PIL import Image
except ImportError:
    print("[错误] 需要 Pillow：pip install pillow")
    raise SystemExit(1)

# EK55 官方标注里我们需要的列（不同版本列名可能不同，故做同义词匹配）
COLUMN_ALIASES = {
    "narration_id": ["narration_id", "narration id", "uid", "id"],
    "video_id": ["video_id", "video id", "video"],
    "participant_id": ["participant_id", "participant id", "participant"],
    "start_frame": ["start_frame", "start frame"],
    "stop_frame": ["stop_frame", "stop frame"],
    "verb_class": ["verb_class", "verb class", "verb"],
    "noun_class": ["noun_class", "noun class", "noun"],
}


def guard_dataset_write(target_path, dataset_root="/data/datasets/EPIC-KITCHENS", allow=False):
    """护栏：禁止把产物写进共享数据集目录（数据集是多人共用，只读）。

    默认拦截 dataset_root 下的任何写入；确实需要时显式传 --allow-dataset-write。
    """
    if allow or not dataset_root:
        return
    resolved = os.path.abspath(target_path)
    root = os.path.abspath(dataset_root)
    if resolved == root or resolved.startswith(root + os.sep):
        raise SystemExit(
            f"[护栏] 拒绝写入共享数据集目录：{resolved}\n"
            f"        数据集根：{root}（多人共用，只读）\n"
            f"        请把产物写到自己的项目目录（例如 /data/ldy/ek55_assets），"
            f"确实要写数据集目录时加 --allow-dataset-write"
        )


# ===== 2. 标注读取与列名归一 =====

def normalize_row(row, mapping):
    result = {}
    for key, column in mapping.items():
        result[key] = row.get(column)
    return result


def build_column_mapping(fieldnames):
    mapping = {}
    lowered = {name.strip().lower(): name for name in fieldnames or []}
    for key, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                mapping[key] = lowered[alias]
                break
    return mapping


def load_segments(annotation_paths):
    """读取一个或多个标注 CSV，返回 (段列表, 列映射, 缺失列)。"""
    segments = []
    mapping = None
    missing = []
    for path in annotation_paths:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            file_mapping = build_column_mapping(reader.fieldnames)
            if mapping is None:
                mapping = file_mapping
                missing = [key for key in COLUMN_ALIASES if key not in file_mapping]
            for row in reader:
                normalized = normalize_row(row, file_mapping)
                normalized["source_csv"] = os.path.basename(path)
                segments.append(normalized)
    return segments, mapping or {}, missing


# ===== 3. 帧路径解析 =====

def _has_frames(directory, probe=50):
    """目录里是否有帧文件（只看前若干项，避免大目录全量 listdir）。"""
    try:
        with os.scandir(directory) as entries:
            checked = 0
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    return True
                checked += 1
                if checked >= probe:
                    break
    except OSError:
        return False
    return False


def candidate_dirs(frames_root, participant_id, video_id):
    """按真实布局给出的候选目录（按优先级排列）。

    已知真实布局（EPIC-KITCHENS 在 amax 上）：
        <root>/<participant>/rgb_frames/<video_id>/frame_0000000001.jpg
    同时也兼容"帧平铺在一个 rgb_frames 里"的历史布局。
    """
    return [
        os.path.join(frames_root, participant_id, "rgb_frames", video_id),
        os.path.join(frames_root, participant_id, video_id),
        os.path.join(frames_root, participant_id, "rgb_frames"),
        os.path.join(frames_root, video_id),
        frames_root,
    ]


def detect_frame_pattern(frames_root, participant_id, video_id, max_extra_depth=3):
    """返回 (目录, 前缀, 帧号位数, 扩展名)。找不到返回四个 None。

    先按候选路径找；再在 <root>/<participant> 下做**限深**搜索（避免全树扫描）。
    """
    for directory in candidate_dirs(frames_root, participant_id, video_id):
        if os.path.isdir(directory) and _has_frames(directory):
            return describe_frame_dir(directory)

    base = os.path.join(frames_root, participant_id)
    if os.path.isdir(base):
        for current, subdirs, files in os.walk(base):
            depth = current[len(base):].count(os.sep)
            if depth >= max_extra_depth:
                subdirs[:] = []
            if video_id and video_id in os.path.basename(current) and _has_frames(current):
                return describe_frame_dir(current)
    return None, None, None, None


def describe_frame_dir(directory):
    """从一个真实存在的帧目录里解析命名规则（前缀/位数/扩展名）。"""
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return None, None, None, None
    for name in names:
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem, extension = os.path.splitext(name)
        prefix = "".join(ch for ch in stem if not ch.isdigit())
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits:
            return directory, prefix, len(digits), extension
    return None, None, None, None


def count_frames(directory):
    """数目录里的图片数（只用 scandir，不做 stat；大目录下也快）。"""
    total = 0
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    total += 1
    except OSError:
        pass
    return total


def probe_layout(frames_root, pairs):
    """探针：对给定的 `participant:video` 列表打印脚本**实际会用的**目录与命名。

    用途：在不生成任何文件的前提下，确认布局与帧号能否正确定位。
    """
    print("[探针] 报告脚本实际解析到的目录（不写任何文件）")
    for pair in pairs:
        if ":" in pair:
            participant_id, video_id = pair.split(":", 1)
        else:
            participant_id, video_id = pair, ""
        print(f"\n  ── {participant_id}:{video_id or '(未指定)'} ──")
        for directory in candidate_dirs(frames_root, participant_id, video_id):
            exists = os.path.isdir(directory)
            marker = "✅" if exists else "  "
            extra = f"（图片 {count_frames(directory)} 张）" if exists else ""
            print(f"    {marker} 候选: {directory} {extra}")
        directory, prefix, digits, extension = detect_frame_pattern(frames_root, participant_id, video_id)
        if directory is None:
            print("    ❌ 未找到可用帧目录——请把上面候选里存在的那个路径发我")
            continue
        total = count_frames(directory)
        first = frame_path(directory, prefix, digits, extension, 1)
        print(f"    → 采用: {directory}")
        print(f"      命名: 前缀 '{prefix}' + {digits} 位数字 + '{extension}'；该目录共 {total} 张")
        print(f"      探测第 1 帧是否存在: {'✅ 存在' if os.path.isfile(first) else '❌ 不存在'}  ({first})")
        if video_id and video_id not in os.path.basename(directory):
            print("      ⚠️ 注意：目录名不含 video_id → 可能是多视频**平铺**在同一目录，"
                  "帧号会串（不同视频都从 1 开始），需要你确认")


def frame_path(directory, pattern_prefix, digits, extension, index):
    return os.path.join(directory, f"{pattern_prefix}{index:0{digits}d}{extension}")


def sample_indices(start_frame, stop_frame, count):
    """在闭区间 [start, stop] 上等距取 count 个帧号（含两端）。"""
    start_frame = int(start_frame)
    stop_frame = int(stop_frame)
    if stop_frame <= start_frame:
        return [start_frame] * count
    if count == 1:
        return [start_frame]
    step = (stop_frame - start_frame) / (count - 1)
    return [int(round(start_frame + step * i)) for i in range(count)]


# ===== 4. 采帧与保存 =====

def save_one_segment(segment, frames_root, out_dir, frames_per_segment, image_size, dry_run,
                     cache):
    """处理一个段：采 N 帧、resize、存 JPEG。返回 (状态, 记录)。"""
    participant = str(segment.get("participant_id") or "").strip()
    video = str(segment.get("video_id") or "").strip()
    cache_key = (participant, video)
    if cache_key not in cache:
        cache[cache_key] = detect_frame_pattern(frames_root, participant, video)
    directory, prefix, digits, extension = cache[cache_key]

    record = {
        "narration_id": segment.get("narration_id"),
        "video_id": video,
        "participant_id": participant,
        "verb_class": segment.get("verb_class"),
        "noun_class": segment.get("noun_class"),
        "start_frame": segment.get("start_frame"),
        "stop_frame": segment.get("stop_frame"),
        "source_csv": segment.get("source_csv"),
    }
    if directory is None or digits is None:
        return "no_directory", record

    indices = sample_indices(segment.get("start_frame") or 0, segment.get("stop_frame") or 0,
                             frames_per_segment)
    saved = 0
    target_dir = os.path.join(out_dir, "frames", str(record["narration_id"]))
    if not dry_run:
        os.makedirs(target_dir, exist_ok=True)
    for order, frame_index in enumerate(indices):
        path = frame_path(directory, prefix, digits, extension, frame_index)
        if not os.path.isfile(path):
            continue
        if dry_run:
            saved += 1
            continue
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                # 短边缩放到 image_size 后中心裁剪，保证与 ViT 输入一致
                width, height = image.size
                scale = image_size / min(width, height)
                image = image.resize((max(image_size, int(round(width * scale))),
                                      max(image_size, int(round(height * scale)))), Image.BILINEAR)
                left = (image.width - image_size) // 2
                top = (image.height - image_size) // 2
                image = image.crop((left, top, left + image_size, top + image_size))
                image.save(os.path.join(target_dir, f"{order:03d}.jpg"), quality=90)
            saved += 1
        except OSError:
            continue
    record["frames_saved"] = saved
    if saved == 0:
        return "no_frames", record
    if saved < frames_per_segment:
        return "partial", record
    return "ok", record


# ===== 5. 主流程 =====

def main():
    parser = argparse.ArgumentParser(description="生成 EK55 可搬运资产（采样帧 + 段表）")
    parser.add_argument("--annotations", nargs="*", default=[], help="标注 CSV（可多个：train/val/test）；探针模式可省略")
    parser.add_argument("--frames-root", required=True, help="EK55 帧根目录")
    parser.add_argument("--out-dir", default=None, help="产物目录（建议 /data/... 下）；探针模式可省略")
    parser.add_argument("--frames-per-segment", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 段（调试用，0 表示全部）")
    parser.add_argument("--dry-run", action="store_true", help="只报告，不写文件")
    parser.add_argument("--probe", type=str, default=None,
                        help='探针模式：只报告会用到哪些帧目录，如 --probe "P01:P01_01,P02:P02_03"')
    parser.add_argument("--forbid-dataset-root", type=str, default="/data/datasets/EPIC-KITCHENS",
                        help="禁止写入的共享数据集根目录（护栏）")
    parser.add_argument("--allow-dataset-write", action="store_true", help="显式允许写数据集目录（不推荐）")
    parser.add_argument("--with-features", action="store_true",
                        help="同时提 CLIP 特征（需要 torch 与 GPU；默认关闭）")
    args = parser.parse_args()

    if args.probe:
        probe_layout(args.frames_root, args.probe.split(","))
        return 0
    if not args.out_dir:
        print("[错误] 非探针模式必须给 --out-dir（写到自己的目录，例如 /data/ldy/ek55_assets）")
        return 1
    guard_dataset_write(args.out_dir, args.forbid_dataset_root, args.allow_dataset_write)
    if not args.annotations:
        print("[错误] 非探针模式必须给 --annotations")
        return 1
    for path in args.annotations:
        if not os.path.isfile(path):
            print(f"[错误] 找不到标注文件：{path}")
            return 1
    if not os.path.isdir(args.frames_root):
        print(f"[错误] 找不到帧目录：{args.frames_root}")
        return 1

    print("[1/4] 读取标注")
    segments, mapping, missing = load_segments(args.annotations)
    print(f"      段数 {len(segments)}；列映射 {mapping}")
    if missing:
        print(f"      [警告] 缺少列：{missing} —— 这些字段会留空，请在 manifest 里核对")
    if args.limit:
        segments = segments[: args.limit]
        print(f"      只处理前 {len(segments)} 段（--limit）")

    print("[2/4] 采帧" + ("（干跑）" if args.dry_run else ""))
    started = time.time()
    cache = {}
    counters = {"ok": 0, "partial": 0, "no_frames": 0, "no_directory": 0}
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(save_one_segment, segment, args.frames_root, args.out_dir,
                        args.frames_per_segment, args.image_size, args.dry_run, cache)
            for segment in segments
        ]
        for index, future in enumerate(futures, start=1):
            status, record = future.result()
            counters[status] += 1
            records.append(record)
            if index % 2000 == 0:
                print(f"      已处理 {index}/{len(segments)}（{time.time() - started:.0f}s）")

    print(f"      完成：{counters}（用时 {time.time() - started:.0f}s）")
    skipped = [r for r in records if r.get("frames_saved", 0) == 0]

    print("[3/4] 写段表与 manifest")
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "annotations": [os.path.abspath(p) for p in args.annotations],
        "frames_root": os.path.abspath(args.frames_root),
        "config": {
            "frames_per_segment": args.frames_per_segment,
            "image_size": args.image_size,
            "sampling_rule": "在 [start_frame, stop_frame] 上等距取 N 帧（含两端）",
            "crop_rule": "短边缩放后中心裁剪",
        },
        "counts": counters,
        "segments_total": len(records),
        "skipped": skipped[:200],
        "skipped_total": len(skipped),
        "column_mapping": mapping,
        "missing_columns": missing,
        "with_features": bool(args.with_features),
    }
    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)
        segments_csv = os.path.join(args.out_dir, "segments.csv")
        with open(segments_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()) if records else [])
            if records:
                writer.writeheader()
                writer.writerows(records)
        manifest["segments_csv"] = segments_csv
        manifest_path = os.path.join(args.out_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        # manifest 自身的 sha1，便于对账
        digest = hashlib.sha1(open(manifest_path, "rb").read()).hexdigest()[:12]
        print(f"      manifest: {manifest_path}（sha1 {digest}）")
        print(f"      segments: {segments_csv}")

    print("[4/4] 下一步")
    if args.with_features:
        print("      --with-features 已在参数里，但本脚本暂未实现提特征（见下方 TODO）")
    print("      TODO(待实现)：接 CLIP ViT-L/14 提每段特征 → features.npz（~GB）供 Stage 1")
    print("      然后把 <out-dir> 打包上传对象存储：frames/ + segments.csv + manifest.json")
    print("      后续：frames/ 供需要图像输入的阶段，features.npz 供只吃特征的阶段")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())