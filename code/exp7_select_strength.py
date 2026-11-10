# exp7_select_strength.py —— 在**验证集**上选注入强度（避免用测试集调参）
#
# 为什么单独一个脚本：exp7 的评估全在测试集上。如果在测试集上挑 strength，
# 等于用答案调参，审稿人会直接质疑。这里用 exp1 --export-alpha-val 导出的 val α，
# 在 val 上扫强度并选最优，再拿到 test 上用（test 只报最终结果）。
#
# 用法：
#   python3 code/exp7_select_strength.py \
#     --alpha-val alpha_verb_val.npz --features ek55_verb_full_text.npz \
#     --candidate-space verb --grid 0.05,0.1,0.2,0.5,1.0,1.5

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp7_llm_evidence import (collect_llm_supports, inject_prior, metrics_of,
                               select_by_trigger)
from metrics import dirichlet_conflict, dirichlet_vacuity


def main():
    parser = argparse.ArgumentParser(description="在验证集上选注入强度")
    parser.add_argument("--alpha-val", required=True, help="exp1 --export-alpha-val 的 npz")
    parser.add_argument("--features", required=True, help="含 past_verb_val/past_noun_val 的 npz")
    parser.add_argument("--candidate-space", default="verb", choices=["verb", "noun"])
    parser.add_argument("--trigger", default="vacuity",
                        choices=["vacuity", "conflict", "confidence", "random"])
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--grid", default="0.05,0.1,0.2,0.5,1.0,1.5")
    parser.add_argument("--prior-mode", default="additive", choices=["redistribute", "additive"])
    parser.add_argument("--llm-rerank-topk", type=int, default=3)
    parser.add_argument("--llm-cache", default="/data/ldy/ek55_assets/llm_cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-argmax", default=None,
                        help="保存每个强度下 val 的逐样本 argmax（验证决策是否饱和）")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    val = np.load(args.alpha_val, allow_pickle=True)
    alpha0 = val["alpha"].astype(np.float64)
    labels = val["y_val"]
    n_classes = alpha0.shape[1]
    print(f"[val] 样本 {len(labels)}｜类别 {n_classes}｜candidate-space={args.candidate_space}")

    # 复用 exp7 的触发与 LLM 调用；这里把 val 的特征路径换成 --features 的 val 字段
    class _Args:
        pass
    llm_args = _Args()
    for key in ("features", "candidate_space", "llm_cache", "llm_rerank_topk",
                "verb_names", "noun_names", "llm_ask_mode", "llm_limit", "llm_mode"):
        setattr(llm_args, key, getattr(args, key, None))
    # exp7 的 collect_llm_supports 默认读 *_test 字段，这里映射到 val
    llm_args.features = args.features
    llm_args.candidate_space = args.candidate_space
    llm_args.llm_cache = args.llm_cache
    llm_args.llm_rerank_topk = args.llm_rerank_topk
    llm_args.verb_names = "/data/datasets/EPIC-KITCHENS/annotations/EPIC_verb_classes.csv"
    llm_args.noun_names = "/data/datasets/EPIC-KITCHENS/annotations/EPIC_noun_classes.csv"
    llm_args.llm_ask_mode = "rerank"
    llm_args.llm_limit = 0
    llm_args.llm_mode = "real"

    before = metrics_of(alpha0, labels)
    probs = before["probs"]
    scores = {
        "vacuity": dirichlet_vacuity(alpha0),
        "conflict": dirichlet_conflict(alpha0, k=5),
        "maxprob": probs.max(axis=1),
    }
    mask = select_by_trigger(args.trigger, args.budget, scores,
                             np.random.default_rng(args.seed), None)
    print(f"[val] 触发 {args.trigger}｜预算 {args.budget:.0%}｜将调用 {int(mask.sum())} 条")

    support, confidence, queried = collect_llm_supports(
        llm_args, mask, n_classes, probs=probs, split="val")

    argmax_store = {}
    print(f"\n{'strength':<10}{'val Δacc':>10}")
    rows = []
    for strength in [float(item) for item in args.grid.split(",")]:
        alpha1, _ = inject_prior(alpha0, support, mask, args.prior_mode, strength)
        after = metrics_of(alpha1, labels)
        argmax_store[str(strength)] = np.asarray(after["probs"]).argmax(axis=1).astype(np.int16)
        delta = after["accuracy"] - before["accuracy"]
        rows.append((strength, delta))
        print(f"{strength:<10}{delta:>+10.4f}")

    best_strength, best_delta = max(rows, key=lambda item: item[1])
    print(f"\n[结论] val 最优强度 = {best_strength}（Δacc {best_delta:+.4f}）")
    if best_strength == max(item[0] for item in rows):
        print("       ⚠️ 最优点在网格上界，建议扩大 --grid")
    print(f"       测试集请固定用：--strength {best_strength}")

    if args.save_argmax and argmax_store:
        np.savez(args.save_argmax, **argmax_store, y=labels.astype(np.int64))
        keys = sorted(argmax_store, key=float)
        base = argmax_store[keys[0]]
        print("\n[饱和检查] 各强度 argmax 与最小强度的逐样本差异：")
        for key in keys:
            diff = int((argmax_store[key] != base).sum())
            print(f"  γ={key:<5} 与 γ={keys[0]} 不同样本数 = {diff}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"rows": rows, "best": {"strength": best_strength, "delta": best_delta},
                       "args": vars(args)}, handle, ensure_ascii=False, indent=2, default=str)
        print(f"[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
