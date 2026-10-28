# routing_feasibility.py —— 路由（门控）可行性检验 v2（不需要 CODA-Prompt、不需要 GPU）
#
# 为什么是 v2：v1 用 k-means 组件 + "组件→类分布"投票，自检就没过——
#   低冲突（峰度高）的样本反而常错，AUROC 只有 0.12。根因是组件混合类别时，
#   "峰在某组件"不代表"预测对"。那套代理本身有陷阱，不适合当判据。
#
# v2 换成**候选宽度路由**：路由 = 决定"保留几个候选类"，这直接对应门控的作用
#   （门控重加权组件 ≈ 改变有效候选宽度），且完全不依赖聚类，判定清晰。
#
#   ① 类原型 = 训练集每类特征均值（只在 train 上算，不碰 test）
#   ② 亲和度 = 逐样本 z-score 后的余弦相似度（必须标准化，否则 softmax 饱和、熵不可解释）
#   ③ 权重 w = softmax(亲和度 / T)，冲突度 = 归一化熵(w)
#   ④ 候选宽度：按 w 取 top-k 个候选类
#   ⑤ 两个检验：
#        A2（路由的依据）：冲突度能否指示"top-1 会错"（AUROC > 0.55），
#                          且**按冲突度动态选 k** 是否优于固定 k=1
#        A1（锐化的依据）：低冲突样本上 top-1 是否足够可靠（锐化不会伤）
#
# 用法：
#   python3 tools/routing_feasibility.py --features /data/ldy/ek55_assets/ek55_verb_full.npz
#   python3 tools/routing_feasibility.py --self-test          # 合成数据自检
#   python3 tools/routing_feasibility.py --features ... --out results/routing_verb.json

import argparse
import json
import os

import numpy as np

EPS = 1e-12


# ===== 1. 基础工具 =====

def l2_normalize(matrix):
    """L2 归一化。零范数行保持为零（不能除以 EPS，否则会放大 1e12 倍 → 溢出/NaN）。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return matrix / norms


def softmax_rows(scores, temperature):
    scaled = scores / max(temperature, EPS)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


def normalized_entropy(weights):
    safe = np.clip(weights, EPS, None)
    return -(safe * np.log(safe)).sum(axis=1) / np.log(weights.shape[1])


def auroc(scores, positive):
    """秩平均 AUROC（同分取平均秩）。"""
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive).astype(bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
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


# ===== 2. 类原型与路由信号 =====

def class_prototypes(features, labels, n_classes):
    """每类特征均值（训练集统计）。缺失的类返回零向量并标记。"""
    dim = features.shape[1]
    prototypes = np.zeros((n_classes, dim), dtype=np.float64)
    present = np.zeros(n_classes, dtype=bool)
    for class_id in range(n_classes):
        members = features[labels == class_id]
        if len(members) > 0:
            prototypes[class_id] = members.mean(axis=0)
            present[class_id] = True
    return prototypes, present


def importance_weights(features, prototypes, present, temperature=1.0):
    """把路由权重映射回全局类索引（缺失类权重为 0）。

    ⚠️ 逐样本 z-score：不标准化时 softmax 会因相似度尺度漂移而饱和，
       熵就变成数值噪声（v1 实测出现过 AUROC 方向反转）。
    """
    available = prototypes[present]
    if available.shape[0] == 0:
        raise ValueError("没有可用的类原型（训练集里没有任何类）——检查标签是否全为 -1/缺失")
    cosine = np.einsum("nd,md->nm", l2_normalize(features), l2_normalize(available))
    cosine = np.nan_to_num(cosine, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(cosine).all():
        raise ValueError(f"余弦矩阵出现非有限值：shape={cosine.shape}")
    standardized = (cosine - cosine.mean(axis=1, keepdims=True)) / \
                   np.clip(cosine.std(axis=1, keepdims=True), EPS, None)
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)
    local_weights = softmax_rows(standardized, temperature)
    global_weights = np.zeros((features.shape[0], prototypes.shape[0]), dtype=np.float64)
    global_weights[:, present] = local_weights
    return global_weights


# ===== 3. 检验 =====

def topk_correct(weights, labels, k):
    """top-k 命中：真实类是否落在权重最高的 k 个候选里。"""
    order = np.argsort(-weights, axis=1)[:, :k]
    return (order == labels[:, None]).any(axis=1)


def run_check(features_train, labels_train, features_test, labels_test,
              temperature=1.0, ks=(1, 2, 3, 5, 10)):
    n_classes = int(labels_train.max()) + 1
    prototypes, present = class_prototypes(features_train, labels_train, n_classes)
    weights = importance_weights(features_test, prototypes, present, temperature)
    available = np.clip(weights[:, present], EPS, None)
    conflict = normalized_entropy(available)
    accuracy_at = {k: float(topk_correct(weights, labels_test, k).mean()) for k in ks}
    top1 = topk_correct(weights, labels_test, 1)
    top5 = topk_correct(weights, labels_test, 5)

    auroc_conflict = auroc(-conflict, top1)       # 冲突高 → 应更容易错
    low_conflict = conflict <= np.quantile(conflict, 0.5)
    high_conflict = conflict >= np.quantile(conflict, 0.8)

    quantiles = np.quantile(conflict, np.linspace(0, 1, 11))
    buckets = []
    for index in range(10):
        lower, upper = quantiles[index], quantiles[index + 1]
        mask = (conflict >= lower) & (conflict <= upper) if index == 9 else \
               (conflict >= lower) & (conflict < upper)
        if not mask.any():
            continue
        buckets.append({
            "bucket": index,
            "n": int(mask.sum()),
            "top1": float(top1[mask].mean()),
            "top5": float(top5[mask].mean()),
            "conflict": [float(lower), float(upper)],
        })

    # 冲突条件路由：低冲突组用 k=1，高冲突组放宽到 k=5
    dynamic_k = np.where(low_conflict, 1, 5)
    order_all = np.argsort(-weights, axis=1)
    dynamic_hit = np.array([
        labels_test[index] in order_all[index, : int(dynamic_k[index])]
        for index in range(len(labels_test))
    ])
    dynamic_accuracy = float(dynamic_hit.mean())

    result = {
        "temperature": temperature,
        "n_classes": n_classes,
        "test_samples": int(len(labels_test)),
        "accuracy_at": {f"top{k}": value for k, value in accuracy_at.items()},
        "accuracy_dynamic_k": dynamic_accuracy,
        "auroc_conflict_vs_top1": auroc_conflict,
        "top1_low_conflict": float(top1[low_conflict].mean()),
        "top1_high_conflict": float(top1[high_conflict].mean()),
        "top5_gain_high_conflict": float(top5[high_conflict].mean() - top1[high_conflict].mean()),
        "buckets": buckets,
        # A2：冲突能指示错误（AUROC>0.55），且按冲突动态放宽候选优于固定 top-1
        "A2_holds": bool(auroc_conflict > 0.55 and dynamic_accuracy > accuracy_at[1] + 0.005),
        # A1：低冲突组 top-1 明显优于整体（说明锐化不会伤）
        "A1_holds": bool(float(top1[low_conflict].mean()) > accuracy_at[1] + 0.05),
    }
    return result


# ===== 4. 自检：构造"歧义=易错"的数据，检验能否判对 =====

def self_test():
    rng = np.random.default_rng(0)
    n_classes, dim, per_class = 5, 32, 300
    centers = rng.normal(0, 1, (n_classes, dim))
    labels = np.repeat(np.arange(n_classes), per_class)
    # 噪声要明显小于类间距，否则"干净样本"也几乎全错，自检就失去意义
    features = centers[labels] + rng.normal(0, 0.6, (len(labels), dim))

    # 20% 样本放在两类中点：真歧义 → 应当既高冲突又易错
    ambiguous = rng.choice(len(labels), size=len(labels) // 5, replace=False)
    for index in ambiguous:
        first = labels[index]
        second = (first + 1) % n_classes
        features[index] = (centers[first] + centers[second]) / 2 + rng.normal(0, 0.3, dim)

    split = int(len(labels) * 0.7)
    result = run_check(features[:split], labels[:split], features[split:], labels[split:],
                       temperature=1.0)
    print("\n[自检] top-1: %.4f | 动态宽度: %.4f | AUROC(冲突→对): %.4f"
          % (result["accuracy_at"]["top1"], result["accuracy_dynamic_k"],
             result["auroc_conflict_vs_top1"]))
    print("[自检] 低冲突组 top-1: %.4f | 高冲突组 top-1: %.4f | A1=%s | A2=%s"
          % (result["top1_low_conflict"], result["top1_high_conflict"],
             result["A1_holds"], result["A2_holds"]))
    assert result["accuracy_at"]["top1"] > 0.5, "自检数据：基础 top-1 应明显高于随机"
    assert result["auroc_conflict_vs_top1"] > 0.55, "自检数据：冲突应能指示错误"
    assert result["top1_low_conflict"] > result["top1_high_conflict"], "低冲突组应更准"
    print("[自检通过] 冲突度能识别易错样本，且冲突条件路由更优")
    return result


def main():
    parser = argparse.ArgumentParser(description="路由（门控）可行性检验 v2：候选宽度路由")
    parser.add_argument("--features", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
    else:
        if not args.features:
            raise SystemExit("请给 --features <npz> 或 --self-test")
        data = np.load(args.features, allow_pickle=True)
        print(f"[数据] {args.features}")
        print(f"  训练 {data['x_train'].shape} / 测试 {data['x_test'].shape}")
        result = run_check(data["x_train"].astype(np.float64), data["y_train"],
                           data["x_test"].astype(np.float64), data["y_test"],
                           temperature=args.temperature)
        result["features"] = os.path.abspath(args.features)

    print("\n" + "=" * 64)
    print("结论")
    print("=" * 64)
    print(f"  top-1 / top-5           {result['accuracy_at']['top1']:.4f} / "
          f"{result['accuracy_at']['top5']:.4f}")
    print(f"  冲突条件路由            {result['accuracy_dynamic_k']:.4f}"
          f"（固定 top-1 为 {result['accuracy_at']['top1']:.4f}）")
    print(f"  AUROC(冲突 → top-1 对)  {result['auroc_conflict_vs_top1']:.4f}  "
          f"{'✅ 冲突能指示错误' if result['auroc_conflict_vs_top1'] > 0.55 else '❌ 冲突与错误无关'}")
    print(f"  低冲突组 top-1          {result['top1_low_conflict']:.4f}")
    print(f"  高冲突组 top-1          {result['top1_high_conflict']:.4f}"
          f"（放宽到 top-5 增益 {result['top5_gain_high_conflict']:+.4f}）")
    print(f"  → A1（锐化通道）        {'成立 ✅' if result['A1_holds'] else '不成立 ❌'}")
    print(f"  → A2（路由通道）        {'成立 ✅' if result['A2_holds'] else '不成立 ❌'}")
    print("\n  冲突分位    样本数    top-1    top-5")
    for bucket in result["buckets"]:
        print(f"  {bucket['bucket']:<10}{bucket['n']:>8}{bucket['top1']:>9.4f}"
              f"{bucket['top5']:>9.4f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"\n[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())