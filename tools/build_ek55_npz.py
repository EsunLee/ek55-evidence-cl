# build_ek55_npz.py —— 把官方标注 + 现成 patch-token 特征组装成 exp1.py 要的 npz
#
# 输入（服务器上的真实资产）：
#   labels           官方 EPIC_train_action_labels.csv
#                    （列：uid, participant_id, video_id, start_frame, stop_frame, verb_class, noun_class, ...）
#   features-train   features/frame_features/train/   每个 .pt = <video_id>_<uid>.pt，张量 [4, 256, 1024] float16
#   features-val     同上，val 目录 —— **这两份目录本身就是官方 train 按视频划出的 train/val**
#   exclude-manifest 可选：clean_v2_manifest.json（其中的 excluded_videos 是帧不全的视频，应剔除）
#
# 输出：一个 .npz，字段与 `data/README.md` 的约定一致
#   x_*        [N, 1024]           过去窗口的**池化**特征（token 均值 → 帧均值）
#   x_seq_*    [N, past, 1024]     过去窗口的**序列**（float16，供后续时序模型用）
#   x_future_* [N, future, 1024]   未来窗口序列（float16，供 L_dist 的 teacher 用）
#   y_*        [N]                 目标段（未来第 1 段）的动作类 = verb_class * 1000 + noun_class
#   meta                           配置与统计（JSON 字符串）
#
# 运行（服务器上）：
#   python tools/build_ek55_npz.py \
#       --labels /data/datasets/EPIC-KITCHENS/annotations/EPIC_train_action_labels.csv \
#       --features-train /data/datasets/EPIC-KITCHENS/features/frame_features/train \
#       --features-val   /data/datasets/EPIC-KITCHENS/features/frame_features/val \
#       --exclude-manifest /data/datasets/EPIC-KITCHENS/insight_annotations_clean_v2/clean_v2_manifest.json \
#       --out /data/ldy/ek55_assets/ek55_past30.npz --past 30 --future 8 --workers 8
#
# 先小样本试跑（几分钟内出结果，用来确认口径）：
#   ... --video-limit 10 --out /data/ldy/ek55_assets/ek55_pilot.npz

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None                      # 纯逻辑测试（build_windows）不依赖 torch

# ===== 1. 读段表 =====

REQUIRED_COLUMNS = ["uid", "video_id", "start_frame", "stop_frame", "verb_class", "noun_class"]


def load_segment_table(labels_path):
    """读官方标注 CSV → 段列表（按 video 分组后由调用方排序）。"""
    segments = []
    with open(labels_path, "r", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"[错误] 标注缺少列：{missing}；实际列：{reader.fieldnames}")
        for row in reader:
            segments.append({
                "uid": int(row["uid"]),
                "participant_id": row["participant_id"].strip(),
                "video_id": row["video_id"].strip(),
                "start_frame": int(row["start_frame"]),
                "stop_frame": int(row["stop_frame"]),
                "verb_class": int(row["verb_class"]),
                "noun_class": int(row["noun_class"]),
            })
    return segments


def group_by_video(segments):
    """按 video 分组并按 start_frame 排序——窗口必须建立在**同一视频的时间顺序**上。"""
    grouped = {}
    for segment in segments:
        grouped.setdefault(segment["video_id"], []).append(segment)
    for video_id in grouped:
        grouped[video_id].sort(key=lambda item: item["start_frame"])
    return grouped


def load_excluded_videos(manifest_path):
    if not manifest_path:
        return set()
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return set(manifest.get("excluded_videos", []))


# ===== 2. 组窗口（纯逻辑，可在无 torch 环境下单测）=====

def build_windows(ordered_segments, past=30, future=8, has_feature=None):
    """对每个目标段 i 组装窗口。

    定义（写清楚，避免口径漂移）：
      过去窗口 = 同一视频里紧邻目标段之前的 past 段（不含目标段）
      未来窗口 = 从目标段开始的 future 段（含目标段本身）
      标签 y   = 目标段的动作类（即"未来第 1 段"的动作）
    返回 (样本列表, 统计)。has_feature 为 None 时视为全部可用。
    """
    samples = []
    stats = {"too_early": 0, "missing_feature": 0, "too_late_future": 0, "kept": 0}
    has_feature = has_feature or (lambda segment: True)

    for index, target in enumerate(ordered_segments):
        if index < past:
            stats["too_early"] += 1
            continue
        if index + future > len(ordered_segments):
            stats["too_late_future"] += 1
            continue
        past_segments = ordered_segments[index - past:index]
        future_segments = ordered_segments[index:index + future]
        window = past_segments + future_segments
        if not all(has_feature(segment) for segment in window):
            stats["missing_feature"] += 1
            continue
        samples.append({
            "past": past_segments,
            "future": future_segments,
            "target": target,
        })
        stats["kept"] += 1
    return samples, stats


# ===== 3. 特征读取与池化 =====

def pool_feature_file(path):
    """读一个 .pt（[frames, tokens, dim]）→ (池化向量 [dim], 每帧向量 [frames, dim])。

    池化 = 先对 patch token 取均值，再对帧取均值。dtype 统一转 float32。
    """
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(tensor, "float"):
        tensor = tensor.float()
    if tensor.dim() == 3:
        per_frame = tensor.mean(dim=1)            # [frames, dim]：token 均值
        pooled = per_frame.mean(dim=0)            # [dim]：帧均值
    elif tensor.dim() == 2:
        per_frame = tensor
        pooled = tensor.mean(dim=0)
    else:
        raise ValueError(f"无法识别的特征形状：{tuple(tensor.shape)}")
    return pooled.numpy(), per_frame.numpy()


def feature_path(features_dir, segment):
    """特征文件名 = <video_id>_<uid>.pt（已由服务器上的实际数据确认）。"""
    return os.path.join(features_dir, f"{segment['video_id']}_{segment['uid']}.pt")


# ===== 4. 主流程 =====

def collect_split(segments, features_dir, past, future, workers, limit=0):
    """处理一个划分：组窗口 → 多线程读特征 → 返回数组与统计。"""
    grouped = group_by_video(segments)
    video_ids = sorted(grouped.keys())
    if limit:
        video_ids = video_ids[:limit]

    def has_feature(segment):
        return os.path.isfile(feature_path(features_dir, segment))

    all_samples = []
    stats_total = {"too_early": 0, "missing_feature": 0, "too_late_future": 0, "kept": 0}
    for video_id in video_ids:
        samples, stats = build_windows(grouped[video_id], past=past, future=future,
                                       has_feature=has_feature)
        for key in stats_total:
            stats_total[key] += stats[key]
        all_samples.extend(samples)

    print(f"    组窗口：保留 {stats_total['kept']}，"
          f"开头不足 {stats_total['too_early']}，缺特征 {stats_total['missing_feature']}，"
          f"结尾不足 {stats_total['too_late_future']}")

    started = time.time()
    results = [None] * len(all_samples)

    def process(index):
        sample = all_samples[index]
        past_vectors = []
        future_vectors = []
        for segment in sample["past"]:
            pooled, _ = pool_feature_file(feature_path(features_dir, segment))
            past_vectors.append(pooled)
        for segment in sample["future"]:
            pooled, _ = pool_feature_file(feature_path(features_dir, segment))
            future_vectors.append(pooled)
        return index, np.stack(past_vectors), np.stack(future_vectors)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (index, past_array, future_array) in enumerate(
                pool.map(process, range(len(all_samples))), start=1):
            results[index] = (past_array, future_array)
            if done % 2000 == 0:
                print(f"    已读取 {done}/{len(all_samples)}（{time.time() - started:.0f}s）")

    past_stack = np.stack([item[0] for item in results]).astype(np.float16)
    future_stack = np.stack([item[1] for item in results]).astype(np.float16)
    labels = np.array([sample["target"]["verb_class"] * 1000 + sample["target"]["noun_class"]
                       for sample in all_samples], dtype=np.int64)
    pooled = past_stack.mean(axis=1).astype(np.float32)
    return {
        "x": pooled,
        "x_seq": past_stack,
        "x_future": future_stack,
        "y": labels,
        "stats": stats_total,
        "uids": [sample["target"]["uid"] for sample in all_samples],
    }


def main():
    parser = argparse.ArgumentParser(description="组装 EK55 预测窗口 npz")
    parser.add_argument("--labels", required=True, help="官方 EPIC_train_action_labels.csv")
    parser.add_argument("--features-train", required=True)
    parser.add_argument("--features-val", default=None)
    parser.add_argument("--exclude-manifest", default=None, help="clean_v2_manifest.json（剔除帧不全的视频）")
    parser.add_argument("--out", required=True)
    parser.add_argument("--past", type=int, default=30)
    parser.add_argument("--future", type=int, default=8)
    parser.add_argument("--video-limit", type=int, default=0, help="每个划分只取前 N 个视频（试跑用）")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if np is None or torch is None:
        raise SystemExit("[错误] 需要 numpy 与 torch（请用 /data/conda_envs/insight_env/bin/python）")

    print("[1/4] 读标注与排除表")
    segments = load_segment_table(args.labels)
    excluded = load_excluded_videos(args.exclude_manifest)
    if excluded:
        before = len(segments)
        segments = [segment for segment in segments if segment["video_id"] not in excluded]
        print(f"    按 clean_v2 剔除 {len(excluded)} 个视频：{before} → {len(segments)} 段")
    print(f"    段数 {len(segments)}，视频 {len({s['video_id'] for s in segments})}")

    # 划分：train 目录 / val 目录本身就是按视频划好的两份
    train_videos, val_videos = set(), set()
    for split_dir, target_set in ((args.features_train, train_videos), (args.features_val, val_videos)):
        if not split_dir or not os.path.isdir(split_dir):
            continue
        for name in os.listdir(split_dir):
            if name.endswith(".pt"):
                target_set.add(name.rsplit("_", 1)[0])
    if not val_videos:
        print("    [警告] 未发现 val 特征目录——所有数据都进 train 划分（后续自行按视频划分）")
    train_segments = [s for s in segments if s["video_id"] in train_videos] or segments
    val_segments = [s for s in segments if s["video_id"] in val_videos]
    print(f"    划分：train {len(train_segments)} 段 / {len(train_videos)} 视频；"
          f"val {len(val_segments)} 段 / {len(val_videos)} 视频")

    print("[2/4] 处理 train 划分")
    train_data = collect_split(train_segments, args.features_train, args.past, args.future,
                               args.workers, args.video_limit)
    print("[3/4] 处理 val 划分")
    if val_segments:
        val_data = collect_split(val_segments, args.features_val, args.past, args.future,
                                 args.workers, args.video_limit)
    else:
        val_data = None

    print("[4/4] 落盘")
    payload = {
        "x_train": train_data["x"],
        "x_seq_train": train_data["x_seq"],
        "x_future_train": train_data["x_future"],
        "y_train": train_data["y"],
    }
    if val_data is not None:
        payload.update({
            "x_val": val_data["x"],
            "x_seq_val": val_data["x_seq"],
            "x_future_val": val_data["x_future"],
            "y_val": val_data["y"],
        })
    meta = {
        "labels": os.path.abspath(args.labels),
        "features_train": os.path.abspath(args.features_train),
        "features_val": os.path.abspath(args.features_val) if args.features_val else None,
        "excluded_videos": sorted(excluded),
        "past": args.past,
        "future": args.future,
        "window_rule": "过去=past 段（不含目标）；未来=future 段（含目标）；标签=目标段动作类",
        "label_rule": "verb_class * 1000 + noun_class",
        "pooling": "token 均值 → 帧均值（x_* 为池化向量；x_seq_* 保留序列）",
        "train_samples": int(len(train_data["y"])),
        "val_samples": int(len(val_data["y"])) if val_data is not None else 0,
        "train_window_stats": train_data["stats"],
        "val_window_stats": val_data["stats"] if val_data is not None else None,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    payload["meta"] = np.array(json.dumps(meta, ensure_ascii=False))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **payload)
    size_gb = os.path.getsize(args.out) / 1024 ** 3
    print(f"    已写出：{args.out}（{size_gb:.2f} GB）")
    print(f"    train 样本 {meta['train_samples']}，val 样本 {meta['val_samples']}")
    print(f"    x_train 形状 {payload['x_train'].shape}，x_seq_train 形状 {payload['x_seq_train'].shape}")
    print("    下一步：python code/exp1.py --features " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())