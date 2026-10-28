# exp5_granularity.py —— B1：不确定度驱动的粒度路由
#
# 问题：把"不确定度"用在决策上（决定答多细），是否比"固定粒度"更优？
#   细粒度 = 直接预测 106 个 verb 类；粗粒度 = 先预测 G 个组（组由类原型聚类得到）。
#   混合 = 一部分样本按粗粒度计分（组对即对），其余按细粒度计分。
#
# 三个对照（**在同一"走粗的比例 α"下比较**，这是关键，否则"答得粗所以更准"无法排除）：
#   ① 固定比例（对照组，不用不确定度，按固定随机顺序取 α 比例走粗）
#   ② 不确定度驱动（按 fine 头的不确定度从高到低取 α 比例走粗）
#   ③ oracle（按"谁对"挑：粗对细错的样本优先走粗）
#
# 判据（写死）：
#   - ② 在多数 α 上 > ①（说明"选择依据"有效，而不是粒度配比带来的）；
#   - ② > 细粒度基线（说明机制净收益为正）；
#   - ② 与 ③ 的差距可量化（说明还有多少空间）。
#
# 用法：
#   python3 code/exp5_granularity.py --features /data/ldy/ek55_assets/ek55_verb_full.npz \
#       --groups 25 --out results/exp5_granularity.json

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp3_decompose import softmax_rows, train_softmax      # 复用线性 softmax 训练
from metrics import fit_temperature                          # 跨粒度校准

EPS = 1e-12


# ===== 1. 读出：lastk 或已池化 =====

def build_readout(data, last_k=5):
    """优先用 x_seq 取最后 k 段的均值（实测最优读出）；否则退回已池化向量。"""
    splits = {}
    for split in ("train", "val", "test"):
        sequence = data.get(f"x_seq_{split}")
        if sequence is not None:
            k = min(last_k, sequence.shape[1])
            features = sequence[:, sequence.shape[1] - k:, :].mean(axis=1)
            mode = f"lastk-{k}"
        else:
            features = data[f"x_{split}"]
            mode = "pooled"
        splits[split] = np.asarray(features, dtype=np.float64)
    return splits, mode


# ===== 2. 粗组：对类原型做 k-means =====

def class_prototypes(features, labels, n_classes):
    prototypes = np.zeros((n_classes, features.shape[1]), dtype=np.float64)
    for class_id in range(n_classes):
        members = features[labels == class_id]
        if len(members) > 0:
            prototypes[class_id] = members.mean(axis=0)
    return prototypes


def kmeans(points, n_groups, seed=0, iterations=50):
    """简单 k-means（numpy），返回每个点的组号。"""
    rng = np.random.default_rng(seed)
    centers = points[rng.choice(len(points), size=n_groups, replace=False)].copy()
    assignments = np.zeros(len(points), dtype=np.int64)
    for _ in range(iterations):
        distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assignments = distances.argmin(axis=1)
        for group in range(n_groups):
            members = points[assignments == group]
            if len(members) > 0:
                centers[group] = members.mean(axis=0)
    return assignments


def class_to_group(prototypes, n_groups, seed=0):
    """把每个类映射到一个粗组（组由类原型聚类得到，与样本无关）。"""
    return kmeans(prototypes, n_groups, seed=seed)


# ===== 3. 评测 =====

def hybrid_accuracy(fine_correct, coarse_correct, go_coarse):
    """混合准确率：走粗的样本按"组对"计分，其余按"类对"计分。"""
    selected = np.where(go_coarse, coarse_correct, fine_correct)
    return float(selected.mean())


def evaluate(fine_probs, coarse_probs, fine_labels, group_labels, alphas,
             uncertainty, seed=0):
    """在若干 α 下比较 固定比例 / 不确定度驱动 / oracle。"""
    fine_correct = fine_probs.argmax(axis=1) == fine_labels
    coarse_correct = coarse_probs.argmax(axis=1) == group_labels
    n_samples = len(fine_labels)
    rng = np.random.default_rng(seed)
    random_order = rng.permutation(n_samples)
    # 不确定度从高到低（越大越不确定 → 越该走粗）
    uncertainty_order = np.argsort(-uncertainty)
    # oracle：优先挑"粗对且细错"的样本
    oracle_priority = np.argsort(-(coarse_correct & ~fine_correct).astype(np.float64))

    rows = []
    for alpha in alphas:
        count = int(round(alpha * n_samples))
        masks = {}
        for name, order in (("fixed", random_order), ("uncertainty", uncertainty_order),
                            ("oracle", oracle_priority)):
            mask = np.zeros(n_samples, dtype=bool)
            mask[order[:count]] = True
            masks[name] = mask
        row = {
            "alpha": float(alpha),
            "fixed": hybrid_accuracy(fine_correct, coarse_correct, masks["fixed"]),
            "uncertainty": hybrid_accuracy(fine_correct, coarse_correct, masks["uncertainty"]),
            "oracle": hybrid_accuracy(fine_correct, coarse_correct, masks["oracle"]),
        }
        row["gain_uncertainty_vs_fixed"] = row["uncertainty"] - row["fixed"]
        row["gap_to_oracle"] = row["oracle"] - row["uncertainty"]
        rows.append(row)
    return {
        "fine_accuracy": float(fine_correct.mean()),
        "coarse_accuracy": float(coarse_correct.mean()),
        "rows": rows,
    }


def evaluate_confidence(fine_probs, coarse_probs, fine_labels, group_labels,
                        uncertainty, seed=0):
    """B1'：按"跨粒度置信度比较"决定走粗 —— 这是决策论上正确的判据。

    规则：若 p_coarse_max > p_fine_max（对"哪个组"比对"哪个类"更有把握），就走粗。
    直觉：细粒度没把握时退一步更划算。该判据不需要扫 α，阈值是自然的。
    为了让结果可比，在**同一个 α_eff** 下与 固定比例 / 不确定度驱动 / oracle 对照。
    """
    fine_correct = fine_probs.argmax(axis=1) == fine_labels
    coarse_correct = coarse_probs.argmax(axis=1) == group_labels
    confidence_mask = coarse_probs.max(axis=1) > fine_probs.max(axis=1)
    alpha_eff = float(confidence_mask.mean())
    n_samples = len(fine_labels)
    count = int(round(alpha_eff * n_samples))

    rng = np.random.default_rng(seed)
    result = {
        "alpha_eff": alpha_eff,
        "confidence": hybrid_accuracy(fine_correct, coarse_correct, confidence_mask),
    }
    orders = {
        "fixed": rng.permutation(n_samples),
        "uncertainty": np.argsort(-uncertainty),
        "oracle": np.argsort(-(coarse_correct & ~fine_correct).astype(np.float64)),
    }
    for name, order in orders.items():
        mask = np.zeros(n_samples, dtype=bool)
        mask[order[:count]] = True
        result[name] = hybrid_accuracy(fine_correct, coarse_correct, mask)

    result["gain_vs_fixed"] = result["confidence"] - result["fixed"]
    result["gain_vs_uncertainty"] = result["confidence"] - result["uncertainty"]
    result["gain_vs_fine"] = result["confidence"] - float(fine_correct.mean())
    result["gap_to_oracle"] = result["oracle"] - result["confidence"]
    return result


def group_level_comparison(fine_probs, coarse_probs, test_y, test_group,
                           group_of_class, seed=0):
    """把两级放到**同一个度量**（组级准确率）上比较。

    为什么必须这样比：粗粒度头直接在"组"上预测，细粒度头在"类"上预测，
    两者的原始准确率不可比（组少 → 天然更容易）。只有当
        「粗粒度头的组准确率」≈「细粒度头映射到组的组准确率」
    时，"在两级之间路由"才是一个有意义的问题；否则只是在两个难度不同的任务间插值。
    """
    fine_mapped = group_of_class[fine_probs.argmax(axis=1)]
    fine_ok = fine_mapped == test_group
    coarse_ok = coarse_probs.argmax(axis=1) == test_group

    fine_accuracy = float(fine_ok.mean())
    coarse_accuracy = float(coarse_ok.mean())
    oracle = float((fine_ok | coarse_ok).mean())          # 上界：每样本都选对的那一级
    headroom = oracle - max(fine_accuracy, coarse_accuracy)

    # 路由规则：逐样本选"校准后置信度更高"的那一级
    choose_coarse = coarse_probs.max(axis=1) > fine_probs.max(axis=1)
    routed = float(np.where(choose_coarse, coarse_ok, fine_ok).mean())

    # 同比例固定混合（对照组：不用置信度，随机挑同样比例走粗）
    n_samples = len(test_y)
    count = int(round(choose_coarse.mean() * n_samples))
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    mask = np.zeros(n_samples, dtype=bool)
    mask[order[:count]] = True
    fixed = float(np.where(mask, coarse_ok, fine_ok).mean())

    return {
        "fine_mapped_group_accuracy": fine_accuracy,
        "coarse_group_accuracy": coarse_accuracy,
        "oracle_group_accuracy": oracle,
        "headroom": headroom,
        "routed_accuracy": routed,
        "routed_coarse_ratio": float(choose_coarse.mean()),
        "fixed_same_ratio_accuracy": fixed,
        "gain_routed_vs_best_single": routed - max(fine_accuracy, coarse_accuracy),
        "gain_routed_vs_fixed": routed - fixed,
        "cells": {
            "both": float((fine_ok & coarse_ok).mean()),
            "fine_only": float((fine_ok & ~coarse_ok).mean()),
            "coarse_only": float((~fine_ok & coarse_ok).mean()),
            "neither": float((~fine_ok & ~coarse_ok).mean()),
        },
    }


# ===== 4. 主流程 =====

def main():
    parser = argparse.ArgumentParser(description="B1：不确定度驱动的粒度路由")
    parser.add_argument("--features", required=True)
    parser.add_argument("--groups", type=int, default=25, help="粗粒度组数 G")
    parser.add_argument("--last-k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--uncertainty", default="margin", choices=["margin", "maxprob"],
                        help="fine 头的不确定度：margin=top1-top2（越大越不确定）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=True)
    splits, readout = build_readout(data, args.last_k)
    train_x, val_x, test_x = splits["train"], splits["val"], splits["test"]
    train_y, val_y, test_y = data["y_train"], data["y_val"], data["y_test"]

    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-6
    train_x, val_x, test_x = (train_x - mean) / std, (val_x - mean) / std, (test_x - mean) / std

    n_classes = int(max(train_y.max(), val_y.max(), test_y.max())) + 1
    prototypes = class_prototypes(train_x, train_y, n_classes)
    group_of_class = class_to_group(prototypes, args.groups, seed=args.seed)

    train_group = group_of_class[train_y]
    test_group = group_of_class[test_y]
    print(f"[数据] 读出={readout}｜train {len(train_y)} / val {len(val_y)} / test {len(test_y)}"
          f"｜{n_classes} 类 → {args.groups} 组")

    print(f"[训练] fine 头（{n_classes} 类）")
    fine_weights, fine_bias = train_softmax(train_x, train_y, n_classes,
                                            epochs=args.epochs, seed=args.seed)
    fine_probs = softmax_rows(test_x @ fine_weights + fine_bias)

    print(f"[训练] coarse 头（{args.groups} 组）")
    coarse_weights, coarse_bias = train_softmax(train_x, train_group, args.groups,
                                                epochs=args.epochs, seed=args.seed + 1)
    coarse_probs = softmax_rows(test_x @ coarse_weights + coarse_bias)

    # ⚠️ 跨粒度校准：类数少的头天然更自信（6 类 vs 24 类的最大概率不可比），
    # 不校准的话 p_coarse > p_fine 会退化成"永远走粗"（自检时 α_eff=0.957）。
    val_fine_z = val_x @ fine_weights + fine_bias
    val_coarse_z = val_x @ coarse_weights + coarse_bias
    temperature_fine, _ = fit_temperature(val_fine_z, val_y)
    temperature_coarse, _ = fit_temperature(val_coarse_z, group_of_class[val_y])
    fine_probs = softmax_rows((test_x @ fine_weights + fine_bias) / temperature_fine)
    coarse_probs = softmax_rows((test_x @ coarse_weights + coarse_bias) / temperature_coarse)
    print(f"[校准] 温度：fine {temperature_fine:.3f} / coarse {temperature_coarse:.3f}")

    if args.uncertainty == "margin":
        top2 = np.sort(fine_probs, axis=1)[:, -2:]
        uncertainty = top2[:, 0] - top2[:, 1]        # 负的间隔：越大越不确定
    else:
        uncertainty = 1.0 - fine_probs.max(axis=1)

    result = evaluate(fine_probs, coarse_probs, test_y, test_group,
                      alphas=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0),
                      uncertainty=uncertainty, seed=args.seed)

    print("\n" + "=" * 78)
    print(f"粒度路由（fine {result['fine_accuracy']:.4f} / coarse {result['coarse_accuracy']:.4f}）")
    print("=" * 78)
    print(f"  {'α(走粗比例)':>12}{'固定比例':>11}{'不确定度驱动':>14}{'比固定':>10}{'oracle':>10}{'距oracle':>10}")
    for row in result["rows"]:
        print(f"  {row['alpha']:>12.1f}{row['fixed']:>11.4f}{row['uncertainty']:>14.4f}"
              f"{row['gain_uncertainty_vs_fixed']:>+10.4f}{row['oracle']:>10.4f}{row['gap_to_oracle']:>10.4f}")

    # 判据：只在 α ∈ (0,1) 的内点上比较
    inner = [row for row in result["rows"] if 0.0 < row["alpha"] < 1.0]
    wins = sum(1 for row in inner if row["gain_uncertainty_vs_fixed"] > 0)
    mean_gain = float(np.mean([row["gain_uncertainty_vs_fixed"] for row in inner]))
    best_uncertainty = max(row["uncertainty"] for row in inner)
    baseline = result["fine_accuracy"]
    verdict = ("不确定度驱动**优于**固定比例（多数 α 上为正）"
               if wins > len(inner) / 2 and mean_gain > 0 else
               "不确定度驱动**未优于**固定比例")
    print(f"\n  内点 {len(inner)} 个中胜出 {wins} 个，平均增益 {mean_gain:+.4f}")
    print(f"  最好混合准确率 {best_uncertainty:.4f} vs 细粒度基线 {baseline:.4f}"
          f"（{best_uncertainty - baseline:+.4f}）")
    print(f"  [结论] {verdict}")

    # B1'：跨粒度置信度判据
    confidence_result = evaluate_confidence(fine_probs, coarse_probs, test_y, test_group,
                                            uncertainty, seed=args.seed)
    print("\n" + "=" * 78)
    print(f"B1'：跨粒度置信度判据（自然阈值 α_eff = {confidence_result['alpha_eff']:.3f}）")
    print("=" * 78)
    print(f"  confidence（p_coarse > p_fine 就走粗） {confidence_result['confidence']:.4f}")
    print(f"  同 α 下 固定比例                     {confidence_result['fixed']:.4f}"
          f"（差 {confidence_result['gain_vs_fixed']:+.4f}）")
    print(f"  同 α 下 不确定度驱动                 {confidence_result['uncertainty']:.4f}"
          f"（差 {confidence_result['gain_vs_uncertainty']:+.4f}）")
    print(f"  同 α 下 oracle                       {confidence_result['oracle']:.4f}"
          f"（距 {confidence_result['gap_to_oracle']:+.4f}）")
    print(f"  相对细粒度基线 {result['fine_accuracy']:.4f}"
          f"（{confidence_result['gain_vs_fine']:+.4f}）")

    # 同度量（组级）比较 —— 只有这个比较才能判断"路由"是否是真问题
    group_result = group_level_comparison(fine_probs, coarse_probs, test_y, test_group,
                                          group_of_class, seed=args.seed)
    print("\n" + "=" * 78)
    print("同一度量（组级准确率）下比较两级")
    print("=" * 78)
    print(f"  粗粒度头（直接预测组）        {group_result['coarse_group_accuracy']:.4f}")
    print(f"  细粒度头映射到组              {group_result['fine_mapped_group_accuracy']:.4f}")
    print(f"  oracle（每样本选对的那一级）  {group_result['oracle_group_accuracy']:.4f}"
          f"  ← 可提升空间 {group_result['headroom']:+.4f}")
    print(f"  置信度路由（{group_result['routed_coarse_ratio']:.1%} 走粗）"
          f"    {group_result['routed_accuracy']:.4f}")
    print(f"  同比例固定混合                {group_result['fixed_same_ratio_accuracy']:.4f}"
          f"（路由 − 固定 = {group_result['gain_routed_vs_fixed']:+.4f}）")
    cells = group_result["cells"]
    print(f"  细分：都对 {cells['both']:.4f}｜只细对 {cells['fine_only']:.4f}｜"
          f"只粗对 {cells['coarse_only']:.4f}｜都错 {cells['neither']:.4f}")
    if group_result["headroom"] < 0.02:
        print("  [结论] 可提升空间 < 2pp → **路由问题不成立**（两级已接近，或错在同一处）")
    elif group_result["gain_routed_vs_fixed"] > 0.005:
        print("  [结论] 路由**优于**固定混合 → 路由是真问题，值得继续")
    else:
        print("  [结论] 可提升空间存在，但当前路由规则没抓到手 → 需要更好的选择器")

    payload = {
        "features": os.path.abspath(args.features),
        "group_level": group_result,
        "temperature_fine": float(temperature_fine),
        "temperature_coarse": float(temperature_coarse),
        "confidence_rule": confidence_result,
        "readout": readout,
        "groups": args.groups,
        "uncertainty": args.uncertainty,
        "fine_accuracy": result["fine_accuracy"],
        "coarse_accuracy": result["coarse_accuracy"],
        "rows": result["rows"],
        "wins": wins,
        "inner_points": len(inner),
        "mean_gain": mean_gain,
        "best_hybrid_accuracy": best_uncertainty,
        "verdict": verdict,
        "seed": args.seed,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"\n[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())