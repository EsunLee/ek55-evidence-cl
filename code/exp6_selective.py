# exp6_selective.py —— 二值决策：证据是否更擅长判断"该不该回答"？
#
# 动机（来自 10-28 的结论）：证据（conflict/vacuity）做"该信谁"的路由明显不如校准概率，
# 但它擅长回答另一个问题——"这条我该不该答"。这是一个**二值决策**，与信号能力匹配。
#
# 做法：标准的**选择性预测（rejection）**协议。给定若干"可信度分数"，按分数从高到低
# 保留一部分样本作答，比较不同覆盖率下的准确率与 AURC（风险-覆盖曲线下面积，越小越好）。
#
# 比较的信号（全部越大 = 越可信）：
#   概率侧：maxprob / margin / negentropy
#   证据侧：negentropy(p_dirichlet) = 1 - H(p)  → 用 (1 - conflict) 表示
#           1 - vacuity（证据量）
#           组合：maxprob × (1 - conflict)
#
# 用法：
#   python3 code/exp6_selective.py --alpha /data/ldy/ek55_assets/alpha_verb.npz \
#       --out results/exp6_selective.json

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import dirichlet_conflict, dirichlet_vacuity

EPS = 1e-12


def normalized_entropy(probs):
    safe = np.clip(probs, EPS, None)
    return -(safe * np.log(safe)).sum(axis=1) / np.log(probs.shape[1])


def accuracy_at_coverage(correct, scores, coverage):
    """保留分数最高的 coverage 比例样本，返回其准确率。"""
    n_keep = max(1, int(round(coverage * len(correct))))
    order = np.argsort(-scores)
    return float(correct[order[:n_keep]].mean())


def aurc_curve(correct, scores, steps=50):
    """风险-覆盖曲线下面积：mean(1 − accuracy@coverage)。

    ⚠️ 必须用**风险**（1−准确率）而不是准确率，否则"越小越好"的判读反了（自检抓到过）。
    """
    points = []
    for index in range(steps):
        coverage = (index + 1) / steps
        points.append(1.0 - accuracy_at_coverage(correct, scores, coverage))
    return float(np.mean(points)), points


def main():
    parser = argparse.ArgumentParser(description="选择性预测：证据 vs 概率（二值决策）")
    parser.add_argument("--alpha", required=True, help="exp1 --export-alpha 的 npz")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data = np.load(args.alpha, allow_pickle=True)
    labels = data["y_test"]
    probs_baseline = data["probs"].astype(np.float64)
    probs_evidence = data["evidence_probs"].astype(np.float64)
    alpha = data["alpha"].astype(np.float64)

    conflict = np.asarray(dirichlet_conflict(alpha, k=5), dtype=np.float64)
    vacuity = np.asarray(dirichlet_vacuity(alpha), dtype=np.float64)

    correct_baseline = probs_baseline.argmax(axis=1) == labels
    correct_evidence = probs_evidence.argmax(axis=1) == labels

    signals = {
        # 概率侧（用基线模型）
        "prob:maxprob": (correct_baseline, probs_baseline.max(axis=1)),
        "prob:margin": (correct_baseline, -np.sort(probs_baseline, axis=1)[:, -2]),
        "prob:negentropy": (correct_baseline, 1.0 - normalized_entropy(probs_baseline)),
        # 证据侧（用证据模型）
        "evi:maxprob": (correct_evidence, probs_evidence.max(axis=1)),
        "evi:1-conflict": (correct_evidence, 1.0 - conflict),
        "evi:1-vacuity": (correct_evidence, 1.0 - vacuity),
        "combo:maxprob*(1-conflict)": (correct_evidence, probs_evidence.max(axis=1) * (1.0 - conflict)),
    }

    rows = []
    for name, (correct, score) in signals.items():
        area, _ = aurc_curve(correct, score)
        rows.append({
            "signal": name,
            "accuracy_full": float(correct.mean()),
            "aurc": area,
            "acc_at_50": accuracy_at_coverage(correct, score, 0.5),
            "acc_at_70": accuracy_at_coverage(correct, score, 0.7),
            "acc_at_90": accuracy_at_coverage(correct, score, 0.9),
        })
    rows.sort(key=lambda row: row["aurc"])

    print("=" * 84)
    print("选择性预测（二值决策：该不该回答）—— AURC 越小越好")
    print("=" * 84)
    print(f"  {'信号':<28}{'全样本acc':>10}{'AURC':>9}{'acc@50%':>10}{'acc@70%':>10}{'acc@90%':>10}")
    for row in rows:
        print(f"  {row['signal']:<28}{row['accuracy_full']:>10.4f}{row['aurc']:>9.4f}"
              f"{row['acc_at_50']:>10.4f}{row['acc_at_70']:>10.4f}{row['acc_at_90']:>10.4f}")

    best_prob = min((row for row in rows if row["signal"].startswith("prob:")), key=lambda r: r["aurc"])
    best_evi = min((row for row in rows if row["signal"].startswith("evi:")), key=lambda r: r["aurc"])
    print(f"\n  最好的概率信号：{best_prob['signal']}（AURC {best_prob['aurc']:.4f}）")
    print(f"  最好的证据信号：{best_evi['signal']}（AURC {best_evi['aurc']:.4f}）")
    delta = best_evi["aurc"] - best_prob["aurc"]        # 负数 = 证据的风险更低 = 更好
    verdict = ("证据在二值决策上**优于**概率" if delta < -0.002 else
               "证据在二值决策上**不优于**概率")
    print(f"  [结论] {verdict}（AURC 差 {delta:+.4f}；注意两者用的模型不同，见下方说明）")

    payload = {"rows": rows, "best_probability": best_prob, "best_evidence": best_evi,
               "aurc_delta": float(delta), "verdict": verdict,
               "note": "prob:* 用线性+CE 模型；evi:* 用证据头模型，两者精度略有差异，"
                       "因此对比需同时看 accuracy_full"}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"\n[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
