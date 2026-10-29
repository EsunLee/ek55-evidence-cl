# exp7b_sweep.py —— 强度 × 触发器 的二维扫描
#
# 动机：exp7 发现"修正强度"比"触发器选择"重要约 4 倍（+0.0→+2.0pp vs +0.34→+0.81pp）。
# 所以必须把强度扫开，回答三个问题：
#   ① 强度在哪儿饱和？
#   ② 过强会不会开始**误翻**（把本来对的翻错）而掉精度？
#   ③ 在同一次 stub 抽样下，各触发器的真实上界（oracle）是多少？
#
# 关键实现纪律：
#   - **所有强度共用同一次 stub 抽样**，否则不可比；
#   - oracle 在每个强度下**单独重算**（它依赖该强度下"谁被翻转修对"）。
#
# 用法：
#   python3 code/exp7b_sweep.py --alpha .../alpha_verb.npz --features .../ek55_verb_full.npz \
#       --strengths 0.5,1,2,3,4,6 --triggers confidence,vacuity,random,oracle \
#       --stub-mode class-frequency --out results/exp7b_sweep.json

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp7_llm_evidence import (build_skill_profile, inject_prior, metrics_of,
                               select_by_trigger, simulate_llm)
from metrics import dirichlet_conflict, dirichlet_vacuity

EPS = 1e-12


def main():
    parser = argparse.ArgumentParser(description="强度 × 触发器 扫描")
    parser.add_argument("--alpha", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--strengths", default="0.5,1,2,3,4,6")
    parser.add_argument("--triggers", default="confidence,vacuity,random,oracle")
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--stub-accuracy", type=float, default=0.7)
    parser.add_argument("--stub-temperature", type=float, default=3.0)
    parser.add_argument("--stub-mode", default="class-frequency", choices=["uniform", "class-frequency"])
    parser.add_argument("--skill-rare", type=float, default=0.9)
    parser.add_argument("--skill-common", type=float, default=0.3)
    parser.add_argument("--rare-frac", type=float, default=0.3)
    parser.add_argument("--prior-mode", default="redistribute", choices=["redistribute", "additive"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data = np.load(args.alpha, allow_pickle=True)
    alpha0 = data["alpha"].astype(np.float64)
    labels = data["y_test"]
    n_classes = alpha0.shape[1]
    n_samples = len(labels)

    train_labels = np.load(args.features, allow_pickle=True)["y_train"]
    skill = None
    if args.stub_mode == "class-frequency":
        skill, _ = build_skill_profile(train_labels, n_classes, args.skill_rare,
                                       args.skill_common, args.rare_frac)

    rng = np.random.default_rng(args.seed)
    support = simulate_llm(labels, n_classes, args.stub_accuracy, args.stub_temperature, rng, skill=skill)
    before = metrics_of(alpha0, labels)
    pred_before = before["probs"].argmax(axis=1)
    wrong_before = pred_before != labels

    scores = {
        "vacuity": dirichlet_vacuity(alpha0),
        "conflict": dirichlet_conflict(alpha0, k=5),
        "maxprob": before["probs"].max(axis=1),
    }

    strengths = [float(item) for item in args.strengths.split(",")]
    triggers = [item.strip() for item in args.triggers.split(",")]

    table = {}
    for strength in strengths:
        # oracle 必须在该强度下单独重算
        alpha_all, _ = inject_prior(alpha0, support, np.ones(n_samples, bool), args.prior_mode, strength)
        pred_all = (alpha_all / alpha_all.sum(axis=1, keepdims=True)).argmax(axis=1)
        oracle_benefit = ((pred_before != labels) & (pred_all == labels)).astype(np.float64)

        row = {}
        for trigger in triggers:
            mask = select_by_trigger(trigger, args.budget, scores, np.random.default_rng(args.seed), oracle_benefit)
            alpha1, _ = inject_prior(alpha0, support, mask, args.prior_mode, strength)
            pred1 = (alpha1 / alpha1.sum(axis=1, keepdims=True)).argmax(axis=1)
            delta = float((pred1 == labels).mean() - (pred_before == labels).mean())
            flips_up = int(((pred_before != labels) & (pred1 == labels)).sum())
            flips_down = int(((pred_before == labels) & (pred1 != labels)).sum())
            row[trigger] = {"delta": delta, "up": flips_up, "down": flips_down,
                            "wrong_rate": float(wrong_before[mask].mean())}
        table[str(strength)] = row

    # 打印矩阵
    print("=" * (20 + 14 * len(triggers)))
    print(f"强度 × 触发器｜Δaccuracy（budget={args.budget:.0%}，stub={args.stub_mode}，"
          f"baseline acc={float((pred_before == labels).mean()):.4f}）")
    print("=" * (20 + 14 * len(triggers)))
    header = "  strength".ljust(12) + "".join(trigger.center(14) for trigger in triggers)
    print(header)
    for strength in strengths:
        line = f"  {strength:<10.1f}"
        for trigger in triggers:
            line += f"{table[str(strength)][trigger]['delta']:+.4f}".center(14)
        print(line)

    print("\n  误翻（本来对→被翻错）计数：")
    header2 = "  strength".ljust(12) + "".join(trigger.center(14) for trigger in triggers)
    print(header2)
    for strength in strengths:
        line = f"  {strength:<10.1f}"
        for trigger in triggers:
            line += f"{table[str(strength)][trigger]['down']}".center(14)
        print(line)

    # 结论
    best_strength, best_trigger, best_delta = None, None, -1.0
    for strength in strengths:
        for trigger in triggers:
            value = table[str(strength)][trigger]["delta"]
            if value > best_delta:
                best_strength, best_trigger, best_delta = strength, trigger, value
    print(f"\n  [最优] strength={best_strength} trigger={best_trigger} → Δacc {best_delta:+.4f}")
    line = f"  [判读] "
    if best_trigger == "oracle":
        line += "各可部署触发均未吃满上界，触发器仍有改进空间"
    else:
        line += "可部署触发已达到/接近上界，触发器问题基本解决"
    print(line)

    payload = {"args": vars(args), "baseline_accuracy": float((pred_before == labels).mean()),
               "table": table, "best": {"strength": best_strength, "trigger": best_trigger,
                                        "delta": best_delta}}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"\n[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
