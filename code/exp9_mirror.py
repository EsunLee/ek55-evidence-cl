# exp9_mirror.py —— 镜像实验：LLM 作主预测器，低置信时回退给判别模型
#
# 为什么需要它：ELC 与 ConfTriage 已经做过"不确定度 → 门控 → 另一模型"，
# 与本文的差别只是角色互换。不给出同表对比，审稿人会判为增量。
#
# 本脚本的设定（与本文相反）：
#   本文：判别模型主，vacuity 最高的 x% 交给 LLM 仲裁（候选内）
#   镜像：LLM 主，它自评置信低于 τ 时回退给判别模型
#
# 用法：
#   python3 code/exp9_mirror.py --alpha alpha_verb_enc_s0.npz \
#     --features ek55_verb_full_text.npz --candidate-space verb \
#     --thresholds 0.4,0.5,0.6,0.7,0.8 --strength 0.7

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp7_llm_evidence import collect_llm_supports, inject_prior, metrics_of

DEFAULT_VERB_CSV = "/data/datasets/EPIC-KITCHENS/annotations/EPIC_verb_classes.csv"
DEFAULT_NOUN_CSV = "/data/datasets/EPIC-KITCHENS/annotations/EPIC_noun_classes.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="镜像实验：LLM 主预测 + 低置信回退")
    parser.add_argument("--alpha", required=True, help="exp1/exp8 导出的证据 npz")
    parser.add_argument("--features", required=True, help="含 past_verb_test / past_noun_test 的 npz")
    parser.add_argument("--candidate-space", default="verb", choices=["verb", "noun"])
    parser.add_argument("--llm-rerank-topk", type=int, default=3)
    parser.add_argument("--llm-cache", default="/data/ldy/ek55_assets/llm_cache")
    parser.add_argument("--llm-limit", type=int, default=0)
    parser.add_argument("--strength", type=float, default=0.7, help="注入强度 γ")
    parser.add_argument("--thresholds", default="0.4,0.5,0.6,0.7,0.8",
                        help="LLM 自评置信阈值：低于该值回退给模型")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    data = np.load(args.alpha, allow_pickle=True)
    alpha0 = data["alpha"].astype(np.float64)
    labels = data["y_test"]
    n_classes = alpha0.shape[1]
    print(f"[数据] 样本 {len(labels)}｜类别 {n_classes}｜candidate-space={args.candidate_space}")

    class _LLMArgs:
        pass

    llm_args = _LLMArgs()
    llm_args.features = args.features
    llm_args.candidate_space = args.candidate_space
    llm_args.llm_cache = args.llm_cache
    llm_args.llm_rerank_topk = args.llm_rerank_topk
    llm_args.verb_names = DEFAULT_VERB_CSV
    llm_args.noun_names = DEFAULT_NOUN_CSV
    llm_args.llm_ask_mode = "rerank"
    llm_args.llm_limit = args.llm_limit
    llm_args.llm_mode = "real"

    before = metrics_of(alpha0, labels)
    probs = before["probs"]
    print(f"[基线] accuracy {before['accuracy']:.4f}｜ece {before['ece']:.4f}")

    # 镜像设定需要每个样本都问 LLM（它就是主预测器）
    mask_all = np.ones(len(labels), dtype=bool)
    print(f"[LLM] 全量调用 {len(labels)} 条（缓存命中则秒级）")
    support, confidence, queried = collect_llm_supports(
        llm_args, mask_all, n_classes, probs=probs, split="test")

    confident = np.asarray(confidence, dtype=np.float64)
    valid = queried & ~np.isnan(confident)
    print(f"[LLM] 有效回复 {int(valid.sum())}／{len(labels)}｜自评置信均值 "
          f"{confident[valid].mean():.4f}" if valid.any() else "[LLM] 无有效回复")

    print(f"\n{'阈值 τ':<10}{'回退样本':>10}{'覆盖(LLM)':>12}{'精度':>10}{'Δ精度':>10}")
    rows = []
    for threshold in [float(item) for item in args.thresholds.split(",")]:
        # 镜像逻辑：LLM 置信 ≥ τ 时采纳它的仲裁，否则回退给模型
        keep_llm = valid & (confident >= threshold)
        mask_inject = keep_llm
        alpha1, _ = inject_prior(alpha0, support, mask_inject, "additive", args.strength)
        after = metrics_of(alpha1, labels)
        delta = after["accuracy"] - before["accuracy"]
        rows.append((threshold, int(mask_inject.sum()), delta, after["accuracy"]))
        print(f"{threshold:<10.2f}{int((~keep_llm).sum()):>10}{mask_inject.mean():>12.1%}"
              f"{after['accuracy']:>10.4f}{delta:>+10.4f}")

    best = max(rows, key=lambda item: item[2])
    print(f"\n[结论] 镜像设定最优：τ={best[0]}，Δacc {best[2]:+.4f}"
          f"（本文设定需与此对比）")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"rows": [[r[0], r[1], r[2], r[3]] for r in rows],
                       "baseline": before["accuracy"],
                       "args": vars(args)}, handle, ensure_ascii=False, indent=2)
        print(f"[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
