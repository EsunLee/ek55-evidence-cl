# diagnose_a1_a2.py —— 路由器与不确定性信号的质量诊断
#
# 两项前提诊断（都需要用真实日志验证，不能靠理论自信）：
#   A1：亲和度 a_m 能否指示"哪个组件可靠"
#   A2：是否存在"亲和度高但常答错"的诱饵组件，且高冲突样本上平滑能否挽回正确率
#
# 用法：
#   python3 diagnose_a1_a2.py --synthetic with_decoys    # 自检：应判 A1 成立、A2 成立
#   python3 diagnose_a1_a2.py --synthetic no_decoys      # 自检：应判 A2 不成立
#   python3 diagnose_a1_a2.py --input gate_logs.npz      # 真实日志（字段见下）
#
# 真实日志需要的字段（在评测时顺手 log 即可，不需要额外算力）：
#   affinity [N, M]  每个样本的组件亲和度 a_m
#   correct  [N]     模型最终是否答对（bool）
#   conflict [N]     证据头的 conflict / dissonance
#   top1     [N]     实际路由选中的组件下标（可选，缺失时用 argmax(affinity) 代替）

import argparse
import numpy as np

from gate_properties import route, vote_weights

# ===== 1. 指标 =====

def auroc(scores, positive):
    """秩平均 AUROC（避免额外依赖；与 metrics.py 的实现一致）。"""
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive).astype(bool)
    n_pos = int(positive.sum())
    n_neg = int(len(positive) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ===== 2. A1：亲和度能否指示组件可靠 =====

def diagnose_a1(affinity, correct):
    affinity = np.asarray(affinity, dtype=np.float64)
    correct = np.asarray(correct).astype(bool)
    top1 = affinity.argmax(axis=1)
    top1_affinity = affinity[np.arange(len(affinity)), top1]

    # 组件可靠性：把该组件选为 top-1 的样本里，最终答对的比例
    reliability = {}
    for m in np.unique(top1):
        mask = top1 == m
        reliability[int(m)] = {"selected": int(mask.sum()), "accuracy": float(correct[mask].mean())}

    auroc_top = auroc(top1_affinity, correct)
    auroc_margin = (auroc(affinity[np.arange(len(affinity)), top1]
                          - np.partition(affinity, -2, axis=1)[:, -2], correct))
    return {
        "auroc_top1_affinity_vs_correct": auroc_top,
        "auroc_affinity_margin_vs_correct": auroc_margin,
        "components_selected": len(reliability),
        "component_reliability": reliability,
        "A1_holds": (auroc_top > 0.6) or (auroc_margin > 0.6),
    }


# ===== 3. A2：诱饵组件 + 平滑能否在高冲突样本上挽回 =====

def diagnose_a2(affinity, correctness_matrix, conflict, kappa=0.4, decoy_accuracy=0.3):
    """A2 的两半：
       ① **样本级**诱饵现象：高冲突样本上，"被选中的那个组件自己是否算对"的比例明显更低
          （不能用"组件全局变坏"的口径——M=100、352 类时组件不会全局坏，只会在个别样本上被错选）
       ② 平滑的增益集中在高冲突组
    """
    affinity = np.asarray(affinity, dtype=np.float64)
    conflict = np.asarray(conflict, dtype=np.float64)
    top1 = affinity.argmax(axis=1)
    own_correct = correctness_matrix[np.arange(len(affinity)), top1]

    quantile = np.quantile(conflict, 0.8)
    high = conflict >= quantile

    def expected(weights, subset_correctness):
        """期望正确率 = Σ_m w_m·r_{s,m}；权重与正确性矩阵必须取同一子集（切片要成对）。"""
        return float((weights * subset_correctness).sum(axis=1).mean())

    baseline_all = vote_weights(route(affinity, 0.0, 0.0, np.zeros(len(affinity)), conflict))
    smooth_all = vote_weights(route(affinity, 0.0, kappa, np.zeros(len(affinity)), conflict))
    gain_high = (expected(smooth_all[high], correctness_matrix[high])
                 - expected(baseline_all[high], correctness_matrix[high]))
    gain_low = (expected(smooth_all[~high], correctness_matrix[~high])
                - expected(baseline_all[~high], correctness_matrix[~high]))

    decoy_share_high = float((own_correct[high] < decoy_accuracy).mean())
    decoy_share_low = float((own_correct[~high] < decoy_accuracy).mean())

    return {
        "conflict_threshold": float(quantile),
        "decoy_share_high_conflict": decoy_share_high,
        "decoy_share_low_conflict": decoy_share_low,
        "smoothing_gain_on_high_conflict": gain_high,
        "smoothing_gain_on_low_conflict": gain_low,
        "A2_holds": (decoy_share_high > decoy_share_low + 0.10)
                    and (gain_high > 0.005)
                    and (gain_high > gain_low),
    }


# ===== 4. 合成场景（自检用）=====

def make_synthetic(kind, rng, n_samples=4000, m_components=50, decoy_strength=2.0, decoy_rate=0.3):
    """两个场景的差别只在"是否有一部分样本被诱饵组件主导"。

    注意：诱饵只能作用于**一部分**样本（decoy_rate），否则基线全错、AUROC 无定义，
    诊断也就无从谈起——这一点在第一版里踩过。
    """
    true_component = rng.integers(0, m_components, n_samples)
    decoy_component = (true_component + rng.integers(1, m_components, n_samples)) % m_components

    affinity = rng.normal(0.0, 0.25, size=(n_samples, m_components))
    affinity[np.arange(n_samples), true_component] += 1.0

    correctness = np.full((n_samples, m_components), 0.35)
    correctness[np.arange(n_samples), true_component] = 0.85

    conflict = np.clip(rng.normal(0.3, 0.1, n_samples), 0.0, 1.0)
    if kind == "with_decoys":
        has_decoy = rng.random(n_samples) < decoy_rate
        rows = np.where(has_decoy)[0]
        affinity[rows, decoy_component[rows]] += decoy_strength       # 诱饵亲和度更高
        correctness[rows, decoy_component[rows]] = 0.05               # 但几乎总是答错
        conflict = np.where(has_decoy,
                            np.clip(rng.normal(0.7, 0.1, n_samples), 0.0, 1.0),
                            np.clip(rng.normal(0.3, 0.1, n_samples), 0.0, 1.0))

    final_weights = vote_weights(affinity)
    correct = (final_weights * correctness).sum(axis=1) > 0.5
    return affinity, correctness, conflict, correct


# ===== 5. 主流程 =====

def main():
    parser = argparse.ArgumentParser(description="A1/A2 前提诊断")
    parser.add_argument("--input", type=str, default=None, help="真实日志 npz")
    parser.add_argument("--synthetic", type=str, default=None, choices=["with_decoys", "no_decoys"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.synthetic:
        affinity, correctness, conflict, correct = make_synthetic(args.synthetic, rng)
        print(f"[合成场景] {args.synthetic}")
    elif args.input:
        data = np.load(args.input, allow_pickle=True)
        affinity = data["affinity"]
        conflict = data["conflict"]
        correct = data["correct"].astype(bool)
        # 真实数据拿不到"每组件是否算对"，用"最终是否答对"近似该样本所有组件的正确性
        correctness = np.repeat(correct[:, None].astype(np.float64), affinity.shape[1], axis=1)
        print(f"[真实日志] {args.input}，样本 {len(correct)}，组件 {affinity.shape[1]}")
    else:
        print("请指定 --synthetic {with_decoys,no_decoys} 或 --input <npz>")
        return 1

    print()
    print("=" * 68)
    print("A1：亲和度能否指示组件可靠")
    print("=" * 68)
    a1 = diagnose_a1(affinity, correct)
    print(f"  AUROC(top-1 亲和度 → 答对) = {a1['auroc_top1_affinity_vs_correct']:.4f}")
    print(f"  AUROC(亲和度间隔 → 答对)   = {a1['auroc_affinity_margin_vs_correct']:.4f}")
    print(f"  被选中的组件数 = {a1['components_selected']}")
    print(f"  → A1 {'成立 ✅（锐化通道有依据）' if a1['A1_holds'] else '不成立 ❌（锐化通道缺乏依据）'}")

    print()
    print("=" * 68)
    print("A2：是否存在自信但常错的诱饵，且高冲突上平滑能挽回")
    print("=" * 68)
    a2 = diagnose_a2(affinity, correctness, conflict)
    print(f"  高冲突组『选中组件自身答错』占比 = {a2['decoy_share_high_conflict']:.3f}")
    print(f"  低冲突组『选中组件自身答错』占比 = {a2['decoy_share_low_conflict']:.3f}"
          f"（前者应明显更高 → 诱饵现象存在）")
    print(f"  高冲突组上平滑增益 = {a2['smoothing_gain_on_high_conflict']:+.4f}")
    print(f"  低冲突组上平滑增益 = {a2['smoothing_gain_on_low_conflict']:+.4f}")
    print(f"  → A2 {'成立 ✅（平滑通道有依据）' if a2['A2_holds'] else '不成立 ❌（平滑通道缺乏依据）'}")

    print()
    print("结论：两个通道各自的前提都被单独检验；")
    print("      真实数据上跑本脚本（评测时顺手 log affinity/conflict/correct 即可），结论直接决定模块去留。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())