# inventory_data.py —— 清点本地 EK55 拷贝：有什么、缺什么、标注在不在
#
# 为什么需要：数据"有"不等于"知道手上是什么"。下载可能不完整、只覆盖部分 kitchen、
# 标注文件可能缺失或版本不一致。这些必须在提特征之前查清，否则后面所有数字都不可信。
#
# 用法（在**数据所在的那台机器**上跑）：
#   python tools/inventory_data.py --data-root E:\epic-kitchens-55 --annotations E:\epic-kitchens-annotations
#   python tools/inventory_data.py --data-root /data/EK55 --annotations /data/annotations --out data/inventory.json
#
# 只依赖标准库；Windows / Linux / macOS 都能跑；不读取视频内容（只看文件名与体积），所以很快。

import argparse
import json
import os
import re
import time

# ===== 1. 常量 =====

VIDEO_SUFFIXES = (".mp4", ".MP4", ".mkv", ".avi", ".webm")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
# EK55 官方标注里我们关心的列（用于判断"这是不是能直接用的标注"）
# 必备列（缺一不可）
REQUIRED_COLUMNS = ["narration_id|uid", "participant_id", "video_id"]
# 二选一列（官方 CSV 用 timestamp；转换后的数据集可能只保留 frame 号）
ALTERNATIVE_COLUMNS = [
    ("start_timestamp", "start_frame"),
    ("stop_timestamp", "stop_frame"),
    ("verb", "verb_class"),
    ("noun", "noun_class"),
]


def looks_like_participant(name):
    """判断一个目录名是否形如 participant（P01、P12）。"""
    return bool(re.fullmatch(r"P\d{1,2}", name or ""))


def guard_dataset_write(target_path, dataset_root="/data/datasets/EPIC-KITCHENS", allow=False):
    """护栏：禁止把产物写进共享数据集目录（多人共用，只读）。"""
    if allow or not dataset_root:
        return
    resolved = os.path.abspath(target_path)
    root = os.path.abspath(dataset_root)
    if resolved == root or resolved.startswith(root + os.sep):
        raise SystemExit(f"[护栏] 拒绝写入共享数据集目录：{resolved}\n"
                         f"        请写到自己的项目目录（例如项目内 data/ 或 /data/ldy/ek55_assets）")


# ===== 2. 目录扫描 =====

def scan_videos(data_root):
    """按 participant 统计视频文件数量与体积（EK55 目录形如 <root>/P01/P01_01.MP4）。"""
    per_participant = {}
    total_bytes = 0
    total_files = 0
    for current_dir, _, filenames in os.walk(data_root):
        for name in filenames:
            if not name.endswith(VIDEO_SUFFIXES):
                continue
            full_path = os.path.join(current_dir, name)
            try:
                size = os.path.getsize(full_path)
            except OSError:
                continue
            relative = os.path.relpath(full_path, data_root)
            participant = relative.split(os.sep)[0] if os.sep in relative else "(root)"
            entry = per_participant.setdefault(participant, {"files": 0, "bytes": 0})
            entry["files"] += 1
            entry["bytes"] += size
            total_files += 1
            total_bytes += size
    return per_participant, total_files, total_bytes


def scan_frames(data_root):
    """统计预提帧，并区分"真实文件"与"符号链接"。

    为什么要区分：真实数据里出现过"同一层既有 per-video 子目录、又有平铺的符号链接"，
    不区分会让体积/数量统计翻倍（本工具踩过这个坑）。
    """
    per_video = {}
    total_images = 0
    total_symlinks = 0
    for current_dir, _, filenames in os.walk(data_root):
        images = [name for name in filenames if name.endswith(IMAGE_SUFFIXES)]
        if not images:
            continue
        key = os.path.relpath(current_dir, data_root)
        links = sum(1 for name in images if os.path.islink(os.path.join(current_dir, name)))
        per_video[key] = {"images": len(images), "symlinks": links}
        total_images += len(images)
        total_symlinks += links
    return per_video, total_images, total_symlinks


def read_csv_header_and_rows(path, max_rows=200000):
    """读 CSV 表头与行数（不引 pandas，避免环境依赖）。"""
    columns = []
    rows = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for index, line in enumerate(handle):
                if index == 0:
                    columns = [item.strip().strip('"') for item in line.rstrip("\n").split(",")]
                    continue
                if line.strip():
                    rows += 1
                if rows >= max_rows:
                    break
    except OSError as error:
        return {"error": str(error)}
    return {"columns": columns, "rows": rows}


def summarize_json(path):
    """JSON 标注只报结构（条数/字段/键），不做列校验。"""
    info = {"bytes": os.path.getsize(path)}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            info["entries"] = len(data)
            if data and isinstance(data[0], dict):
                info["sample_keys"] = sorted(data[0].keys())[:12]
        elif isinstance(data, dict):
            info["keys"] = sorted(data.keys())[:12]
            first_list = next((value for value in data.values() if isinstance(value, list)), None)
            if first_list is not None:
                info["first_list_length"] = len(first_list)
    except (OSError, ValueError) as error:
        info["error"] = str(error)
    return info


def scan_annotations(annotations_dir):
    """列出标注目录里的 CSV，并判断哪些文件含我们需要的列。"""
    found = []
    if not annotations_dir or not os.path.isdir(annotations_dir):
        return found
    for current_dir, _, filenames in os.walk(annotations_dir):
        for name in sorted(filenames):
            lower = name.lower()
            if lower.endswith(".json"):
                info = summarize_json(os.path.join(current_dir, name))
                info["file"] = os.path.relpath(os.path.join(current_dir, name), annotations_dir)
                info["kind"] = "json"
                info["has_expected_columns"] = False
                info["missing_columns"] = ["(JSON：需人工核对字段结构)"]
                found.append(info)
                continue
            if not lower.endswith(".csv"):
                continue
            full_path = os.path.join(current_dir, name)
            info = read_csv_header_and_rows(full_path)
            info["file"] = os.path.relpath(full_path, annotations_dir)
            info["bytes"] = os.path.getsize(full_path)
            columns = info.get("columns", [])
            missing_required = [
                col for col in REQUIRED_COLUMNS
                if not any(candidate in columns for candidate in col.split("|"))
            ]
            missing_alternatives = [
                "/".join(pair) for pair in ALTERNATIVE_COLUMNS
                if not any(candidate in columns for candidate in pair)
            ]
            info["has_expected_columns"] = not missing_required and not missing_alternatives
            info["missing_columns"] = missing_required + missing_alternatives
            info["using_frame_columns"] = ("start_frame" in columns) and ("start_timestamp" not in columns)
            found.append(info)
    return found


# ===== 3. 体检结论 =====

def participants_from_frames(frames_per_video):
    """从帧目录相对路径里抠出 participant（形如 P01/P01_01/...）。"""
    found = set()
    for relative in frames_per_video:
        head = relative.split(os.sep)[0]
        if head.startswith("P") and head[1:].isdigit():
            found.add(head)
    return sorted(found)


def build_warnings(videos_per_participant, video_files, frames_per_video, annotations, data_root=""):
    warnings = []
    participants = sorted(set([key for key in videos_per_participant if key.startswith("P")]
                              + participants_from_frames(frames_per_video)))
    # 以单个 participant 目录为 --data-root 时（如 .../P01），目录名本身就是 participant
    root_name = os.path.basename(os.path.normpath(data_root))
    if looks_like_participant(root_name) and root_name not in participants:
        participants = sorted(set(participants + [root_name]))
    if not participants:
        warnings.append("没有找到形如 P01 的 participant 目录：确认 --data-root 指向的是 EK55 的根目录")
    elif len(participants) < 32:
        warnings.append(f"只找到 {len(participants)} 个 participant（官方为 32 个厨房）：可能是子集，需在协议里写明")
    if video_files == 0:
        warnings.append("没有找到视频文件：若你只有预提帧，可忽略；否则确认路径或扩展名")
    if not frames_per_video:
        warnings.append("没有找到预提帧目录：Stage 2（提示微调要跑 ViT 前向）需要帧，届时要么解帧要么用视频解码")

    usable = [item for item in annotations if item.get("has_expected_columns")]
    frame_based = [item for item in usable if item.get("using_frame_columns")]
    if not usable:
        warnings.append("没有找到同时含 narration_id/participant_id/video_id/start_timestamp/stop_timestamp/verb/noun 的标注 CSV："
                        "请从 epic-kitchens/annotations 仓库补齐")
    else:
        names = " ".join(item["file"].split("/")[-1] for item in usable[:5])
        warnings.append(f"含完整列的标注文件：{names}（请人工确认 train/val/test 划分是否齐）")
        if frame_based:
            warnings.append(f"其中 {len(frame_based)} 个用 start_frame/stop_frame（无 timestamp）"
                            f"→ 采帧脚本按帧号定位，正好匹配")
    return warnings


# ===== 4. 主流程 =====

def main():
    parser = argparse.ArgumentParser(description="清点本地 EK55 拷贝")
    parser.add_argument("--data-root", type=str, required=True, help="EK55 数据根目录（含 P01/P02... 或 frames/）")
    parser.add_argument("--annotations", type=str, default=None, help="标注 CSV 所在目录（epic-kitchens/annotations）")
    parser.add_argument("--out", type=str, default=None, help="输出 JSON（默认写到项目 data/inventory_<时间戳>.json）")
    parser.add_argument("--forbid-dataset-root", type=str, default="/data/datasets/EPIC-KITCHENS")
    parser.add_argument("--allow-dataset-write", action="store_true")
    parser.add_argument("--top", type=int, default=10, help="打印体积最大的前 N 个 participant")
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        print(f"[错误] 找不到数据目录：{args.data_root}")
        return 1

    print(f"[扫描] {args.data_root}（只看文件名与体积，不解码视频）")
    videos_per_participant, video_files, video_bytes = scan_videos(args.data_root)
    frames_per_video, frame_images, frame_symlinks = scan_frames(args.data_root)
    annotations = scan_annotations(args.annotations) if args.annotations else []
    warnings = build_warnings(videos_per_participant, video_files, frames_per_video, annotations,
                              data_root=args.data_root)
    frame_participants = participants_from_frames(frames_per_video)
    root_name = os.path.basename(os.path.normpath(args.data_root))
    if looks_like_participant(root_name) and root_name not in frame_participants:
        frame_participants = sorted(set(frame_participants + [root_name]))

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": os.path.abspath(args.data_root),
        "annotations_dir": os.path.abspath(args.annotations) if args.annotations else None,
        "videos": {
            "files": video_files,
            "bytes": video_bytes,
            "gigabytes": round(video_bytes / 1024 ** 3, 2),
            "participants": len(videos_per_participant),
        },
        "frames": {
            "directories": len(frames_per_video),
            "images": frame_images,
            "symlinks": frame_symlinks,
            "participants": frame_participants,
        },
        "per_participant": videos_per_participant,
        "annotations": annotations,
        "warnings": warnings,
    }

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        f"inventory_{time.strftime('%Y%m%d_%H%M%S')}.json",
    )
    guard_dataset_write(out_path, args.forbid_dataset_root, args.allow_dataset_write)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    # --- 打印人类可读摘要 ---
    print(f"\n视频：{video_files} 个文件，{summary['videos']['gigabytes']} GB，"
          f"覆盖 {summary['videos']['participants']} 个 participant")
    print(f"帧  ：{frame_images} 张图片（其中符号链接 {frame_symlinks} 张），"
          f"分布在 {len(frames_per_video)} 个目录，覆盖 {len(frame_participants)} 个 participant"
          + (f"（{frame_participants[0]}–{frame_participants[-1]}）" if frame_participants else ""))
    if frame_symlinks:
        print("      注：符号链接通常是「平铺副本」，统计体积时会翻倍，算存储时只算真实文件")
    csv_count = sum(1 for a in annotations if a.get("kind") != "json")
    json_count = sum(1 for a in annotations if a.get("kind") == "json")
    print(f"标注：{csv_count} 个 CSV（{sum(1 for a in annotations if a.get('has_expected_columns'))} 个含完整列）"
          f" + {json_count} 个 JSON（只报结构）")
    largest = sorted(videos_per_participant.items(), key=lambda item: item[1]["bytes"], reverse=True)[: args.top]
    if largest:
        print(f"\n体积最大的 {len(largest)} 个 participant：")
        for name, entry in largest:
            print(f"  {name:<8}{entry['files']:>5} 个文件{entry['bytes'] / 1024 ** 3:>8.2f} GB")
    if warnings:
        print("\n体检提示：")
        for text in warnings:
            print(f"  - {text}")
    print(f"\n[落盘] {out_path}")
    print("[下一步] 把该 JSON 提交进仓库（data/），并在 data/README.md 登记：覆盖哪些 kitchen、子集还是全量")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())