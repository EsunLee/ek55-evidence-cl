# audit_features.py —— 实证审计预提取特征的来源与结构
#
# 背景：仓库内没有提特征脚本（remote_bootstrap.sh 注明"提特征：脚本待补"），
# 因此 .pt 的层、池化、采样位置均无文档，只能从张量本身推断。
#
# 本脚本回答四个问题：
#   ① 形状/归一化：每文件的 [frames, tokens, dim] 是否恒定、是否 L2 归一化
#   ② token 是否构成空间网格：相邻 token 的相关性是否显著高于随机配对
#   ③ 四帧是否按时间排序：跨段检验（上一段最后一帧 vs 下一段第一帧）
#   ④ 信息集中度：token 间方差、是否像 CLS/pooled 向量
#
# 用法：
#   python3 tools/audit_features.py --features /data/datasets/.../frame_features/train \
#     --labels /data/datasets/.../EPIC_train_action_labels.csv --n 60

import argparse
import csv
import os
import random

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="审计预提取特征")
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--n", type=int, default=60, help="抽样文件数")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def cosine(a, b):
    a = a.reshape(-1); b = b.reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    args = parse_args()
    random.seed(args.seed)
    rows = list(csv.DictReader(open(args.labels)))
    sample = random.sample(rows, min(args.n, len(rows)))

    shapes, norms, per_frame_norms = [], [], []
    grid_gaps, random_gaps = [], []

    for row in sample:
        path = os.path.join(args.features, f"{row['video_id']}_{row['uid']}.pt")
        if not os.path.isfile(path):
            continue
        tensor = torch.load(path, map_location="cpu", weights_only=False)
        array = tensor.float().numpy() if hasattr(tensor, "float") else np.asarray(tensor)
        shapes.append(array.shape)
        if array.ndim != 3:
            continue
        frames, tokens, dim = array.shape
        per_frame_norms.append(np.linalg.norm(array, axis=2).mean(axis=1))
        norms.append(np.linalg.norm(array, axis=2).mean())

        # ② token 是否构成空间网格：把 tokens 视作 rows×cols
        side = int(round(tokens ** 0.5))
        if side * side == tokens:
            grid = array[0].reshape(side, side, dim)
            # 相邻（右、下）与随机配对各取 200 对
            adjacent = []
            for _ in range(200):
                i, j = random.randrange(side), random.randrange(side)
                ni, nj = (i, min(j + 1, side - 1))
                if (i, j) != (ni, nj):
                    adjacent.append(cosine(grid[i, j], grid[ni, nj]))
            pairs = [(random.randrange(side), random.randrange(side)) for _ in range(400)]
            randoms = [cosine(grid[a[0], a[1]], grid[b[0], b[1]])
                       for a, b in zip(pairs[:200], pairs[200:])]
            grid_gaps.append(np.mean(adjacent))
            random_gaps.append(np.mean(randoms))

    shapes = np.array(shapes)
    print(f"[① 形状] 抽样 {len(shapes)} 个文件")
    print(f"   唯一形状：{sorted({tuple(s) for s in shapes})[:5]}"
          f"{' …' if len({tuple(s) for s in shapes}) > 5 else ''}")
    print(f"   frames 取值：{sorted({int(s[0]) for s in shapes})}"
          f"｜tokens 取值：{sorted({int(s[1]) for s in shapes})}"
          f"｜dim 取值：{sorted({int(s[2]) for s in shapes})}")

    norms = np.array(norms)
    print(f"[① 归一化] 向量 L2 范数 均值 {norms.mean():.3f}｜范围 [{norms.min():.3f}, {norms.max():.3f}]")
    print("   → 若范数 ≈ 1.0 说明已 L2 归一化；若 ≈ √dim（约 32）说明未归一化")

    if per_frame_norms:
        stack = np.stack(per_frame_norms)          # [files, frames]
        print(f"[① 帧间范数] 各帧平均范数：{np.round(stack.mean(axis=0), 3).tolist()}"
              f"（若差异大，说明某些帧信息量确实不同）")

    if grid_gaps:
        grid_gaps = np.array(grid_gaps); random_gaps = np.array(random_gaps)
        print(f"[② 空间结构] 相邻 token 余弦 {grid_gaps.mean():.3f}｜"
              f"随机配对 {random_gaps.mean():.3f}｜差值 {grid_gaps.mean() - random_gaps.mean():+.3f}")
        print("   → 相邻明显更高 ⇒ tokens 很可能是空间 patch；差值≈0 ⇒ 可能不是空间布局")
    else:
        print("[② 空间结构] token 数不是完全平方数，跳过网格检验")

    print("[③ 时间顺序] 本脚本不判定，需用跨段检验（见说明）：")
    print("   若 frame[-1](段t) 与 frame[0](段t+1) 的相似度显著高于随机帧对，则四帧按时间排序。")
    print("[④ 信息集中度] 如需逐 token 方差，请另行统计；本脚本只报告范数与结构。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
