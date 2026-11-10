# audit_features.py —— 特征统计与结构假设审计
#
# 本脚本只做三件事，并严格区分三档结论：
#   实测事实：形状、有限值比例、范数分布、重复切片比例、各轴相似度
#   结构假设：空间布局、时间顺序（报告支持程度，不当作结论）
#   未知来源：提取层、采样时刻、位置编码、是否已做时空融合
#
# 仓库内没有提特征脚本（remote_bootstrap.sh 注明"提特征：脚本待补"），
# 因此上述来源信息无法从代码确认，只能标注为未知。
#
# 用法：
#   python3 tools/audit_features.py --features <dir> --labels <csv> --n 60

import argparse
import csv
import os
import random
from collections import defaultdict

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="特征统计与结构假设审计")
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--n-pairs", type=int, default=30, help="跨段配对数量")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def cos(a, b):
    a = a.reshape(-1); b = b.reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb + 1e-12))


def main():
    args = parse_args()
    random.seed(args.seed)
    rows = list(csv.DictReader(open(args.labels)))
    sample = random.sample(rows, min(args.n, len(rows)))

    shapes, norms, finite, uniq = [], [], [], []
    dist_sim = defaultdict(list)     # 索引距离 → 相似度
    per_pos_norm = []

    for row in sample:
        path = os.path.join(args.features, f"{row['video_id']}_{row['uid']}.pt")
        if not os.path.isfile(path):
            continue
        tensor = torch.load(path, map_location="cpu", weights_only=False)
        x = tensor.float().numpy() if hasattr(tensor, "float") else np.asarray(tensor)
        shapes.append(x.shape)
        if x.ndim != 3:
            continue
        frames, tokens, dim = x.shape
        finite.append(np.isfinite(x).mean())
        norms.append(np.linalg.norm(x, axis=2).mean())
        per_pos_norm.append(np.linalg.norm(x, axis=2).mean(axis=1))
        frame = x[0]
        uniq.append(len(np.unique(np.round(frame, 4), axis=0)) / tokens)

        # ---- 索引距离相似度（不预设是网格） ----
        for distance in (1, 2, 16):
            vals = []
            for _ in range(200):
                i = random.randrange(tokens - distance)
                vals.append(cos(frame[i], frame[i + distance]))
            dist_sim[distance].append(np.mean(vals))
        randoms = []
        for _ in range(200):
            i, j = random.randrange(tokens), random.randrange(tokens)
            if i != j:
                randoms.append(cos(frame[i], frame[j]))
        dist_sim["random"].append(np.mean(randoms))

    shapes = np.array(shapes)
    print("=" * 60)
    print("【实测事实】")
    print(f"  抽样文件 {len(shapes)}｜frames {sorted({int(s[0]) for s in shapes})}"
          f"｜tokens {sorted({int(s[1]) for s in shapes})}"
          f"｜dim {sorted({int(s[2]) for s in shapes})}")
    print(f"  有限值比例 {np.mean(finite):.4f}｜向量 L2 范数 {np.mean(norms):.3f}"
          f"（范围 {np.min(norms):.3f}–{np.max(norms):.3f}）")
    print(f"  单帧内去重切片比例 {np.mean(uniq):.3f}（1.0=无重复 token）")
    print("  索引距离相似度（余弦）：")
    for key in (1, 2, 16, "random"):
        print(f"    距离 {str(key):<7}{np.mean(dist_sim[key]):.3f}")

    # ---- 跨段端点相似度（4×4 矩阵） ----
    by_video = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)
    pairs = []
    for video, segs in by_video.items():
        segs.sort(key=lambda r: int(r["start_frame"]))
        for a, b in zip(segs, segs[1:]):
            if int(a["stop_frame"]) < int(b["start_frame"]) and \
               int(b["start_frame"]) - int(a["stop_frame"]) <= 30:
                pairs.append((a, b))
        if len(pairs) >= args.n_pairs:
            break
    pairs = pairs[: args.n_pairs]

    matrix = np.zeros((4, 4)); used = 0
    for a, b in pairs:
        pa = os.path.join(args.features, f"{a['video_id']}_{a['uid']}.pt")
        pb = os.path.join(args.features, f"{b['video_id']}_{b['uid']}.pt")
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            continue
        xa = torch.load(pa, map_location="cpu", weights_only=False).float().numpy()
        xb = torch.load(pb, map_location="cpu", weights_only=False).float().numpy()
        if xa.ndim != 3 or xb.ndim != 3:
            continue
        for i in range(min(4, xa.shape[0])):
            for j in range(min(4, xb.shape[0])):
                matrix[i, j] += cos(xa[i], xb[j])
        used += 1

    print("\n【结构假设】（只能说与…相容，不能确认为结论）")
    d1, dr = np.mean(dist_sim[1]), np.mean(dist_sim["random"])
    d16 = np.mean(dist_sim[16])
    if d1 - dr > 0.05:
        print(f"  空间布局：距离 1 比随机高 {d1 - dr:+.3f}，**与局部空间结构相容**；"
              f"256=16² 与该假设相容，但未证实")
    else:
        print(f"  空间布局：距离 1 与随机差 {d1 - dr:+.3f}，**本统计未检测到局部相关性**"
              f"（不构成无空间信息的证据）")
    if d16 - dr > 0.05:
        print(f"  列方向：距离 16 比随机高 {d16 - dr:+.3f}，与按行优先排列的网格相容")

    if used:
        matrix /= used
        print(f"\n  时间顺序：{used} 对相邻段（无重叠、间隔 ≤30 帧）的 4×4 跨段相似度")
        for i in range(4):
            print("    " + "  ".join(f"{matrix[i, j]:.3f}" for j in range(4)))
        last_first = matrix[3, 0]
        others = [matrix[i, j] for i in range(4) for j in range(4) if (i, j) != (3, 0)]
        gap = last_first - np.mean(others)
        print(f"  端点 [-1,0] = {last_first:.3f}｜其余 {np.mean(others):.3f}｜差 {gap:+.3f}")
        print("  → 仅当该差值持续为正时，才与四帧按时间排序相容；"
              "同视频背景相似也会产生正差，故不能替代时间戳")
    else:
        print("  时间顺序：未找到可用的相邻段配对，无法评估")

    print("\n【未知来源】（无法从张量或仓库代码确认）")
    print("  · 提取层（哪一层、是否最后一层）")
    print("  · 四帧的采样时刻（是否均匀、是否含段末帧）")
    print("  · 是否保留位置编码")
    print("  · 是否已做时空融合（若已融合，打乱 token 顺序不等价于消除时序）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
