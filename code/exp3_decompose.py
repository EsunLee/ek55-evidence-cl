# exp3_decompose.py —— verb / noun 联合误差分解
#
# 目的：回答"主要误差来自哪个轴"。做法是训两个头（verb 头、noun 头），
# 在同一个测试集上把它们**组合**成对 (verb, noun) 的预测，再拆成 2×2：
#   (verb 对, noun 对)  (verb 对, noun 错)  (verb 错, noun 对)  (都错)
# 另外做两个"条件分析"：
#   ① P(verb 对 | noun 对) 与 P(noun 对 | verb 对)
#   ② 候选受限准确率：给定真实 noun，把 verb 候选**限制在训练集里与它共现过的 verb 上**
#      —— 如果限制后 verb 准确率大幅上升，说明误差主要来自"noun→verb 的耦合"，而不是 verb 头本身。
#
# 输入：verb 与 noun 两个 npz（样本顺序必须一致；由同一次构建、仅 --label-space 不同产生）
# 用法：
#   python3 code/exp3_decompose.py \
#       --verb /data/ldy/ek55_assets/ek55_verb_full.npz \
#       --noun /data/ldy/ek55_assets/ek55_noun_full.npz \
#       --epochs 150 --out results/exp3_decompose.json

import argparse
import json
import os
import sys
import time

import numpy as np

# ===== 1. 基础：softmax 回归（numpy 实现，冻结特征，便于本地自检）=====

EPS = 1e-12


def softmax_rows(scores):
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def train_softmax(train_x, train_y, n_classes, epochs=150, lr=0.05, batch_size=512, seed=0):
    """小批量 Adam 训练的线性 softmax 回归。返回 (weights, bias)。"""
    rng = np.random.default_rng(seed)
    n_samples, dim = train_x.shape
    weights = rng.normal(0.0, 0.01, size=(dim, n_classes))
    bias = np.zeros(n_classes)
    m_w = np.zeros_like(weights); v_w = np.zeros_like(weights)
    m_b = np.zeros_like(bias); v_b = np.zeros_like(bias)
    beta1, beta2 = 0.9, 0.999
    step = 0
    one_hot = np.eye(n_classes)[train_y]

    for epoch in range(epochs):
        order = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            index = order[start:start + batch_size]
            batch_x = train_x[index]
            batch_y = one_hot[index]
            step += 1

            probs = softmax_rows(batch_x @ weights + bias)
            grad_logits = (probs - batch_y) / len(index)
            grad_w = batch_x.T @ grad_logits + 1e-4 * weights
            grad_b = grad_logits.sum(axis=0)

            for parameter, gradient, m, v in ((weights, grad_w, m_w, v_w),
                                              (bias, grad_b, m_b, v_b)):
                m *= beta1
                m += (1 - beta1) * gradient
                v *= beta2
                v += (1 - beta2) * gradient * gradient
                m_hat = m / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)
                parameter -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return weights, bias


def predict(train_x, train_y, test_x, n_classes, **kwargs):
    weights, bias = train_softmax(train_x, train_y, n_classes, **kwargs)
    return softmax_rows(test_x @ weights + bias)


# ===== 2. 联合分解 =====

def cooccurrence_candidates(train_verb, train_noun):
    """训练集里"与某 noun 共现过的 verb 集合"，用于候选受限分析。"""
    table = {}
    for verb, noun in zip(train_verb, train_noun):
        table.setdefault(int(noun), set()).add(int(verb))
    return table


def candidate_restricted_accuracy(probs, lookup_keys, target_labels, allowed_by_key):
    """候选受限准确率。

    ⚠️ 两个数组分工不同（这里曾经写错）：用 lookup_keys（真实 noun）去**查候选表**，
    用 target_labels（真实 verb）判断对错。混用会得到约 1/K 的无意义结果。
    """
    hits, covered, total = 0, 0, len(target_labels)
    for index in range(len(target_labels)):
        allowed = allowed_by_key.get(int(lookup_keys[index]))
        if not allowed:
            continue
        covered += 1
        mask = np.zeros(probs.shape[1], dtype=bool)
        for candidate in allowed:
            if candidate < probs.shape[1]:
                mask[candidate] = True
        restricted = np.where(mask, probs[index], -np.inf)
        if int(restricted.argmax()) == int(target_labels[index]):
            hits += 1
    return hits / max(covered, 1), covered / total


def decompose(verb_probs, noun_probs, verb_labels, noun_labels, verb_train, noun_train):
    verb_pred = verb_probs.argmax(axis=1)
    noun_pred = noun_probs.argmax(axis=1)
    verb_ok = verb_pred == verb_labels
    noun_ok = noun_pred == noun_labels

    verb_correct = float(verb_ok.mean())
    noun_correct = float(noun_ok.mean())
    both = float((verb_ok & noun_ok).mean())
    verb_only = float((verb_ok & ~noun_ok).mean())
    noun_only = float((~verb_ok & noun_ok).mean())
    neither = float((~verb_ok & ~noun_ok).mean())

    # 条件准确率
    given_noun = float(verb_ok[noun_ok].mean()) if noun_ok.any() else float("nan")
    given_verb = float(noun_ok[verb_ok].mean()) if verb_ok.any() else float("nan")

    # 候选受限：给定真实 noun，限制 verb 候选
    verb_by_noun = cooccurrence_candidates(verb_train, noun_train)
    restricted_verb, coverage = candidate_restricted_accuracy(
        verb_probs, noun_labels, verb_labels, verb_by_noun)
    noun_by_verb = {}
    for verb, noun in zip(verb_train, noun_train):
        noun_by_verb.setdefault(int(verb), set()).add(int(noun))
    restricted_noun, coverage_noun = candidate_restricted_accuracy(
        noun_probs, verb_labels, noun_labels, noun_by_verb)

    return {
        "verb_accuracy": verb_correct,
        "noun_accuracy": noun_correct,
        "joint_both_correct": both,
        "verb_only_correct": verb_only,
        "noun_only_correct": noun_only,
        "neither_correct": neither,
        "verb_acc_given_noun_correct": given_noun,
        "noun_acc_given_verb_correct": given_verb,
        "verb_acc_restricted_by_true_noun": restricted_verb,
        "verb_restriction_coverage": coverage,
        "noun_acc_restricted_by_true_verb": restricted_noun,
        "noun_restriction_coverage": coverage_noun,
        "lift_verb_from_noun": restricted_verb - verb_correct,
        "lift_noun_from_verb": restricted_noun - noun_correct,
    }


# ===== 3. 主流程 =====

def main():
    parser = argparse.ArgumentParser(description="verb/noun 联合误差分解")
    parser.add_argument("--verb", required=True, help="verb 级 npz")
    parser.add_argument("--noun", required=True, help="noun 级 npz")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    verb_data = np.load(args.verb, allow_pickle=True)
    noun_data = np.load(args.noun, allow_pickle=True)

    for split in ("train", "test"):
        n_verb = len(verb_data[f"y_{split}"])
        n_noun = len(noun_data[f"y_{split}"])
        if n_verb != n_noun:
            raise SystemExit(f"[错误] {split} 样本数不一致（{n_verb} vs {n_noun}）——两个 npz 不是同一批样本")

    train_x = verb_data["x_train"].astype(np.float64)
    test_x = verb_data["x_test"].astype(np.float64)
    # 标准化（冻结特征量纲差异大，线性头需要）
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-6
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    verb_train, verb_test = verb_data["y_train"], verb_data["y_test"]
    noun_train, noun_test = noun_data["y_train"], noun_data["y_test"]
    n_verb_classes = int(max(verb_train.max(), verb_test.max())) + 1
    n_noun_classes = int(max(noun_train.max(), noun_test.max())) + 1
    print(f"[数据] train {len(verb_train)} / test {len(verb_test)}；"
          f"verb 类 {n_verb_classes}，noun 类 {n_noun_classes}")

    started = time.time()
    print("[训练] verb 头")
    verb_probs = predict(train_x, verb_train, test_x, n_verb_classes,
                         epochs=args.epochs, lr=args.lr, seed=args.seed)
    print("[训练] noun 头")
    noun_probs = predict(train_x, noun_train, test_x, n_noun_classes,
                         epochs=args.epochs, lr=args.lr, seed=args.seed + 1)

    result = decompose(verb_probs, noun_probs, verb_test, noun_test, verb_train, noun_train)
    result.update({
        "n_test": int(len(verb_test)),
        "n_verb_classes": n_verb_classes,
        "n_noun_classes": n_noun_classes,
        "seconds": round(time.time() - started, 1),
        "verb_npz": os.path.abspath(args.verb),
        "noun_npz": os.path.abspath(args.noun),
    })

    print("\n" + "=" * 62)
    print("联合误差分解（测试集）")
    print("=" * 62)
    print(f"  verb 准确率                     {result['verb_accuracy']:.4f}")
    print(f"  noun 准确率                     {result['noun_accuracy']:.4f}")
    print(f"  两者都对（组合动作正确）        {result['joint_both_correct']:.4f}")
    print(f"  verb 对 / noun 错               {result['verb_only_correct']:.4f}")
    print(f"  verb 错 / noun 对               {result['noun_only_correct']:.4f}")
    print(f"  两者都错                        {result['neither_correct']:.4f}")
    print(f"\n  P(verb 对 | noun 对)            {result['verb_acc_given_noun_correct']:.4f}")
    print(f"  P(noun 对 | verb 对)            {result['noun_acc_given_verb_correct']:.4f}")
    print(f"\n  verb 受限到“真实 noun 的候选”   {result['verb_acc_restricted_by_true_noun']:.4f}"
          f"（提升 {result['lift_verb_from_noun']:+.4f}，覆盖 {result['verb_restriction_coverage']:.2%}）")
    print(f"  noun 受限到“真实 verb 的候选”   {result['noun_acc_restricted_by_true_verb']:.4f}"
          f"（提升 {result['lift_noun_from_verb']:+.4f}，覆盖 {result['noun_restriction_coverage']:.2%}）")

    # 归因：在"组合错误"的样本里，verb 错 vs noun 错 各占多少
    result["verb_wrong_share"] = 0.0
    result["noun_wrong_share"] = 0.0
    error_both = 1.0 - result["joint_both_correct"]
    if error_both > 0:
        # ⚠️ 这里容易写反（自检抓到过）：
        #   verb_only_correct 表示"verb 对、noun 错" → 属于 **noun 错** 的样本
        #   noun_only_correct 表示"verb 错、noun 对" → 属于 **verb 错** 的样本
        #   "两者都错"两边都计入
        verb_wrong_share = (result["neither_correct"] + result["noun_only_correct"]) / error_both
        noun_wrong_share = (result["neither_correct"] + result["verb_only_correct"]) / error_both
        print(f"\n  组合错误的样本中：verb 错占 {verb_wrong_share:.1%}，noun 错占 {noun_wrong_share:.1%}")
        result["verb_wrong_share"] = float(verb_wrong_share)
        result["noun_wrong_share"] = float(noun_wrong_share)

    if result["verb_wrong_share"] > result["noun_wrong_share"] + 0.05:
        verdict = "主要误差在 verb 轴"
    elif result["noun_wrong_share"] > result["verb_wrong_share"] + 0.05:
        verdict = "主要误差在 noun 轴"
    else:
        verdict = "两轴误差相当"
    print(f"\n  [结论] {verdict}")
    result["verdict"] = verdict

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"\n[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())