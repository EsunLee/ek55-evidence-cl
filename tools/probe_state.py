# probe_state.py —— 静态状态可读性探针（现有缓存特征）
#
# 两个子命令：
#   sample  从标注表里挑候选片段，产出人工标注模板（CSV）
#   train   读标注结果 + 冻结特征，视频级划分，线性探针
#
# 标注协议（第一轮只做静态状态）：
#   只纳入【整段内状态稳定、清晰可见】的片段；变化/遮挡记为 unstable/unknown，不参与训练。
#   同一类物体上尽量同时采正负例（都看得到容器，但一个被持握、一个在台面），
#   用来区分"读出的是状态"还是"读出的是物体类别"。
#
# 用法：
#   python3 tools/probe_state.py sample --labels <csv> --features <dir> --n 240 --out probe_template.csv
#   python3 tools/probe_state.py train  --labels probe_template.csv --features <dir> --state holding

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="静态状态可读性探针")
    parser.add_argument("mode", choices=["sample", "train"])
    parser.add_argument("--labels", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--min-frames", type=int, default=45, help="段长下限（约 1.5 秒 @30fps）")
    parser.add_argument("--max-frames", type=int, default=150, help="段长上限（约 5 秒）")
    parser.add_argument("--per-video", type=int, default=6)
    parser.add_argument("--state", default="holding", help="train 时用哪个状态列")
    parser.add_argument("--out", default="probe_template.csv")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def seconds(text):
    h, m, s = text.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def mode_sample(args):
    random.seed(args.seed)
    rows = list(csv.DictReader(open(args.labels)))
    by_video = defaultdict(list)
    missing = 0
    for row in rows:
        span = int(row["stop_frame"]) - int(row["start_frame"]) + 1
        if args.min_frames <= span <= args.max_frames:
            path = Path(args.features) / f"{row['video_id']}_{row['uid']}.pt"
            if not path.is_file():
                missing += 1
                continue
            by_video[row["video_id"]].append(row)

    chosen, videos = [], sorted(by_video)
    random.shuffle(videos)
    for video in videos:
        pool = by_video[video]
        random.shuffle(pool)
        chosen.extend(pool[: args.per_video])
        if len(chosen) >= args.n:
            break
    chosen = chosen[: args.n]

    if not chosen:
        raise SystemExit("[中止] 没有同时满足段长限制和特征存在条件的片段")
    print(f"[采样] 段长合格但缺少特征 {missing} 条")
    fields = ["uid", "video_id", "start_timestamp", "stop_timestamp", "narration",
              "state_holding", "state_tilt", "near_object", "judgeable", "note"]
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in chosen:
            writer.writerow({
                "uid": row["uid"], "video_id": row["video_id"],
                "start_timestamp": row["start_timestamp"],
                "stop_timestamp": row["stop_timestamp"], "narration": row["narration"],
                "state_holding": "", "state_tilt": "", "near_object": "",
                "judgeable": "", "note": "",
            })
    print(f"[采样] {len(chosen)} 条，来自 {len({r['video_id'] for r in chosen})} 个视频 → {args.out}")
    print("  填写约定：1=是，0=否，unknown=看不清；judgeable=1 表示整段状态稳定、可用")
    print("  note 可写：unstable（段内变化）/ occluded（遮挡）/ cut（跨截止）")
    return 0


def mode_train(args):
    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    from metrics import auroc

    state_col = {"holding": "state_holding", "tilt": "state_tilt"}.get(args.state, args.state)
    reader = csv.DictReader(open(args.labels, encoding="utf-8-sig", newline=""))
    required = {"uid", "video_id", "judgeable", state_col}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise SystemExit(f"[中止] 标注表缺少列：{', '.join(sorted(missing))}")

    rows, labels, groups, usable = [], [], [], 0
    for row in reader:
        if row.get("judgeable", "").strip() != "1":
            continue
        value = row.get(state_col, "").strip()
        if value not in ("0", "1"):
            continue
        path = os.path.join(args.features, f"{row['video_id']}_{row['uid']}.pt")
        if not os.path.isfile(path):
            raise SystemExit(f"[中止] 已标注片段缺少特征：{path}")
        tensor = torch.load(path, map_location="cpu", weights_only=False)
        array = tensor.float().numpy() if hasattr(tensor, "float") else np.asarray(tensor)
        if array.ndim == 3:                 # [frames, tokens, dim] → 池化
            vector = array.mean(axis=(0, 1))
        elif array.ndim == 2:
            vector = array.mean(axis=0)
        else:
            raise SystemExit(f"[中止] 特征形状不支持：{path} {array.shape}")
        if not np.isfinite(array).all() or not np.isfinite(vector).all():
            raise SystemExit(f"[中止] 特征包含非有限值：{path}")
        rows.append(vector); labels.append(int(value)); groups.append(row["video_id"])
        usable += 1

    if usable < 20:
        raise SystemExit(f"[中止] 可用标注样本仅 {usable} 条，至少需要 20 条")

    X = np.stack(rows).astype(np.float64)
    y = np.array(labels)
    g = np.array(groups)
    print(f"[数据] 可用 {usable} 条｜视频 {len(set(g))} 个｜正例 {int(y.sum())} / 负例 {int((1-y).sum())}")
    if len(set(g)) < 5:
        raise SystemExit("[中止] 视频数少于 5，视频级划分不可靠")

    # 视频级划分（不打乱样本顺序，按视频分组）
    rng = np.random.default_rng(args.seed)
    videos = sorted(set(g)); rng.shuffle(videos)
    cut = max(1, int(round(len(videos) * 0.7)))
    train_videos = set(videos[:cut])
    train_mask = np.array([v in train_videos for v in g])
    test_mask = ~train_mask
    if test_mask.sum() < 5:
        raise SystemExit("[中止] 测试集样本过少")

    for split, mask in (("train", train_mask), ("held-out", test_mask)):
        pos = int(y[mask].sum())
        neg = int(mask.sum() - pos)
        print(f"[划分] {split}｜视频 {len(set(g[mask]))}｜正例 {pos} / 负例 {neg}")
        if not pos or not neg:
            raise SystemExit(f"[中止] {split} 缺少一个类别，补充跨视频标注后再运行")

    mean, std = X[train_mask].mean(0), X[train_mask].std(0) + 1e-8
    Xn = (X - mean) / std

    def fit_logistic(Xtr, ytr, l2=1e-3, iters=600, lr=0.5):
        w = np.zeros(Xtr.shape[1]); b = 0.0
        for _ in range(iters):
            z = Xtr @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            grad_w = Xtr.T @ (p - ytr) / len(ytr) + l2 * w
            grad_b = float((p - ytr).mean())
            w -= lr * grad_w; b -= lr * grad_b
        return w, b

    w, b = fit_logistic(Xn[train_mask], y[train_mask].astype(float))
    score = Xn[test_mask] @ w + b
    pred = (score > 0).astype(int)
    yt = y[test_mask]
    acc = float((pred == yt).mean())
    majority = float(max(yt.mean(), 1 - yt.mean()))

    auc = auroc(score, yt == 1)

    print(f"\n[结果] state = {state_col}")
    print(f"  测试样本 {int(test_mask.sum())}（{len(set(g[test_mask]))} 个视频）")
    print(f"  探针准确率 {acc:.4f}｜多数类基线 {majority:.4f}｜提升 {acc - majority:+.4f}")
    print(f"  AUC {auc:.4f}（恒定分数基线 0.5）")
    print("  单次视频级划分的探索性点估计。")
    return 0


def main():
    args = parse_args()
    return mode_sample(args) if args.mode == "sample" else mode_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
