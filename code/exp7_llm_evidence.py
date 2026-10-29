# exp7_llm_evidence.py —— LLM 模块骨架：用"前后证据变化"度量 LLM 的作用
#
# 设计（对应"证据只是证明可信度"的定位）：
#   证据不是方法，而是**测量工具**。流程是
#       ① 冻结模型 + 冻结证据头 → α, p → 记录 accuracy / ECE / vacuity / conflict
#       ② 触发规则选出 k% 样本 → 调用 LLM（此处用**可控的 stub** 模拟）
#       ③ 把 LLM 输出作为 Dirichlet 先验注入：α' = α + u
#       ④ 用**同一个冻结证据头**重算 → 记录 after 指标
#       ⑤ 报告 Δaccuracy / ΔECE / Δvacuity / Δconflict 与分层
#
# 两个必须警惕的陷阱（本脚本会显式检测）：
#   A. 若先验是"纯加性"（u ≥ 0），vacuity = K/α0 **必然下降**——那是算术恒等式，不是证据。
#      → 提供 --prior-mode redistribute（Σu = 0，总证据不变，只有 conflict 能动）
#        与 additive（u ≥ 0，vacuity 机械下降）两种，并打印"注入证据量"以便核对。
#   B. stub 用**真实标签**构造 LLM 输出，这只是**模拟器**；换成真 LLM 时替换 `simulate_llm` 即可。
#
# 用法（先验证度量本身，不必接真 LLM）：
#   python3 code/exp7_llm_evidence.py --alpha /data/ldy/ek55_assets/alpha_verb.npz \
#       --trigger vacuity --budget 0.2 --stub-accuracy 0.7 --prior-mode redistribute \
#       --out results/exp7_stub.json

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import dirichlet_conflict, dirichlet_vacuity, ece_equal_width

EPS = 1e-12


# ===== 1. 模拟 LLM（换成真 LLM 时只改这一个函数）=====

def simulate_llm(labels, n_classes, accuracy, temperature, rng, skill=None):
    """可控 stub：以 accuracy 的概率把质量压到真实类上，否则随机分散。

    返回 π（每个候选类的"支持度"，行归一化）。
    注意：这里用到 labels 是因为它是**模拟器**；真 LLM 只能看到文本与候选列表。
    """
    n_samples = len(labels)
    scores = rng.normal(0.0, 1.0, size=(n_samples, n_classes))
    threshold = accuracy if skill is None else skill[labels]   # 每条样本各自的"帮上概率"
    hit = rng.random(n_samples) < threshold
    # 命中：真实类得分抬高；未命中：随机抬一个错类
    wrong = (labels + rng.integers(1, n_classes, n_samples)) % n_classes
    target = np.where(hit, labels, wrong)
    scores[np.arange(n_samples), target] += temperature
    scores = scores - scores.max(axis=1, keepdims=True)
    support = np.exp(scores)
    return support / support.sum(axis=1, keepdims=True)


def build_skill_profile(train_labels, n_classes, rare_prob=0.9, common_prob=0.3, rare_frac=0.3):
    """按**类频**造"LLM 帮助概率"（越稀有越高）。

    为什么用类频：真实 LLM 靠外部语言知识，对**罕见但语义清晰**的概念更可能有帮助，
    对模型早已见过上千次的高频类无新信息。关键是——**它只依赖数据（类频）**，
    与我们的模型/vacuity/触发规则完全无关，因此检验"vacuity 触发"时**不循环**。
    返回 (每个类的帮助概率, 每类的样本数)。
    """
    counts = np.bincount(train_labels, minlength=n_classes).astype(np.float64)
    order = np.argsort(counts)                       # 出现次数升序 → 最稀有在前
    rank = np.empty(n_classes, dtype=np.float64)
    rank[order] = np.linspace(0.0, 1.0, n_classes)   # 0 = 最稀有，1 = 最常见
    ramp = np.clip((rare_frac - rank) / max(rare_frac, EPS), 0.0, 1.0)
    skill = common_prob + (rare_prob - common_prob) * ramp
    return skill, counts


def fit_logistic(features, labels, steps=400, lr=0.2, l2=1e-3):
    """小逻辑回归（numpy + Adam）。输入已标准化。"""
    n_samples, n_features = features.shape
    weights = np.zeros(n_features)
    bias = 0.0
    m_w, v_w, m_b, v_b = np.zeros(n_features), np.zeros(n_features), 0.0, 0.0
    beta1, beta2 = 0.9, 0.999
    for step in range(1, steps + 1):
        logits = features @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        grad = (probs - labels) / n_samples
        grad_w = features.T @ grad + l2 * weights
        grad_b = grad.sum()
        for parameter, gradient, m, v in ((weights, grad_w, m_w, v_w), (bias, grad_b, m_b, v_b)):
            m *= beta1
            m += (1 - beta1) * gradient
            v *= beta2
            v += (1 - beta2) * gradient * gradient
            m_hat = m / (1 - beta1 ** step)
            v_hat = v / (1 - beta2 ** step)
            if np.ndim(parameter) == 0:
                parameter -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            else:
                parameter -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return weights, bias


def learned_trigger_scores(features, benefit, folds=5, seed=0):
    """交叉拟合：只在**其他折**的实测收益上训练，避免用自己这折的标签（防泄漏）。"""
    n_samples = len(benefit)
    scores = np.zeros(n_samples)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    for fold in range(folds):
        test_index = order[fold::folds]
        train_index = np.setdiff1d(order, test_index)
        weights, bias = fit_logistic(features[train_index], benefit[train_index])
        scores[test_index] = 1.0 / (1.0 + np.exp(-np.clip(features[test_index] @ weights + bias, -30, 30)))
    return scores


# ===== 2. 触发规则 =====

def select_by_trigger(trigger, budget, scores, rng, oracle_benefit=None):
    """返回布尔掩码：哪些样本调用 LLM。"""
    n_samples = len(scores["vacuity"])
    count = max(1, int(round(budget * n_samples)))
    mask = np.zeros(n_samples, dtype=bool)
    if trigger == "random":
        mask[rng.permutation(n_samples)[:count]] = True
    elif trigger == "vacuity":                       # vacuity 高 = 没证据 → 该求助
        mask[np.argsort(-scores["vacuity"])[:count]] = True
    elif trigger == "conflict":                      # conflict 高 = 证据矛盾 → 该求助
        mask[np.argsort(-scores["conflict"])[:count]] = True
    elif trigger == "confidence":                    # 基线：置信度低 → 该求助
        mask[np.argsort(scores["maxprob"])[:count]] = True
    elif trigger == "learned":                    # 学出来的触发器（交叉拟合，防泄漏）
        mask[np.argsort(-scores["learned"])[:count]] = True
    elif trigger == "expected_rarity":            # 部署可用：模型自己说"我要答的类多罕见"
        mask[np.argsort(-scores["expected_rarity"])[:count]] = True
    elif trigger == "rarity_oracle":              # 上界参照：真实类有多罕见（用到标签）
        mask[np.argsort(-scores["true_rarity"])[:count]] = True
    elif trigger == "oracle":                        # 上界：只挑"LLM 真能救"的样本
        if oracle_benefit is None:
            raise ValueError("oracle 触发需要 oracle_benefit")
        mask[np.argsort(-oracle_benefit)[:count]] = True
    else:
        raise ValueError(f"未知 trigger: {trigger}")
    return mask


# ===== 3. 先验注入 =====

def inject_prior(alpha, support, mask, mode, strength):
    """α' = α + u，u 由 LLM 支持度 support 决定。

    redistribute：u 零均值（Σu=0），总证据 α0 不变 → vacuity 不变，只有 conflict 能动。
    additive    ：u ≥ 0，α0 变大 → vacuity **必然**下降（算术效应，须显式承认）。
    """
    # ⚠️ 必须按 **α₀ 的比例**注入：实测 α₀ ≈ 124（106 类 / vacuity 0.856），
    # 而固定量级 ~1 的先验会被完全淹没（Δ 全为 0，自检踩过）。
    # 这样 strength 的语义才清晰：注入量占总证据的比例。
    total = alpha.sum(axis=1, keepdims=True)
    if mode == "redistribute":
        centered = support - support.mean(axis=1, keepdims=True)
        delta = strength * total * centered
    elif mode == "additive":
        delta = strength * total * support
    else:
        raise ValueError(mode)
    delta = delta * mask[:, None]                    # 只对调用样本生效
    return alpha + delta, delta


# ===== 4. 主流程 =====

def metrics_of(alpha, labels):
    strength = alpha.sum(axis=1, keepdims=True)
    probs = alpha / strength
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float((pred == labels).mean()),
        "ece": float(ece_equal_width(probs, labels)),
        "vacuity_mean": float(np.mean(dirichlet_vacuity(alpha))),
        "conflict_mean": float(np.mean(dirichlet_conflict(alpha, k=5))),
        "probs": probs,
    }


def main():
    parser = argparse.ArgumentParser(description="LLM 模块骨架：前后证据变化")
    parser.add_argument("--alpha", required=True, help="exp1 --export-alpha 的 npz")
    parser.add_argument("--trigger", default="vacuity",
                        choices=["vacuity", "conflict", "confidence", "random", "oracle",
                                 "expected_rarity", "rarity_oracle", "learned"])
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--stub-accuracy", type=float, default=0.7, help="模拟 LLM 的命中率")
    parser.add_argument("--stub-temperature", type=float, default=3.0, help="模拟 LLM 的置信程度")
    parser.add_argument("--features", type=str, default=None,
                        help="stub-mode=class-frequency 时必需：含 y_train 的 npz（用于统计类频）")
    parser.add_argument("--stub-mode", default="uniform", choices=["uniform", "class-frequency"],
                        help="uniform=每条样本同样概率；class-frequency=按类频（稀有类更易帮上，不循环）")
    parser.add_argument("--skill-rare", type=float, default=0.9)
    parser.add_argument("--skill-common", type=float, default=0.3)
    parser.add_argument("--rare-frac", type=float, default=0.3, help="把最稀有的这一比例类视为'稀有'")
    parser.add_argument("--prior-mode", default="redistribute", choices=["redistribute", "additive"])
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data = np.load(args.alpha, allow_pickle=True)
    alpha0 = data["alpha"].astype(np.float64)
    labels = data["y_test"]
    n_classes = alpha0.shape[1]
    rng = np.random.default_rng(args.seed)

    skill = None
    if args.stub_mode == "class-frequency":
        if not args.features:
            raise SystemExit("[错误] --stub-mode class-frequency 需要 --features（用 y_train 统计类频）")
        train_labels = np.load(args.features, allow_pickle=True)["y_train"]
        skill, counts = build_skill_profile(train_labels, n_classes, args.skill_rare,
                                            args.skill_common, args.rare_frac)
        print(f"[技能画像] 按类频：帮助概率 {skill.min():.2f}–{skill.max():.2f}"
              f"（稀有 {args.rare_frac:.0%} 的类更高）")

    before = metrics_of(alpha0, labels)

    scores = {"vacuity": dirichlet_vacuity(alpha0), "conflict": dirichlet_conflict(alpha0, k=5),
              "maxprob": before["probs"].max(axis=1)}
    # 稀有度：rank 越大 = 越常见；rarity_c = 1 - rank（越罕见越高）
    if args.features:
        train_labels = np.load(args.features, allow_pickle=True)["y_train"]
        counts_all = np.bincount(train_labels, minlength=n_classes).astype(np.float64)
        order_all = np.argsort(counts_all)
        rank_all = np.empty(n_classes, dtype=np.float64)
        rank_all[order_all] = np.linspace(0.0, 1.0, n_classes)
        rarity_class = 1.0 - rank_all                       # 越罕见越接近 1
        probs_now = before["probs"]
        scores["expected_rarity"] = probs_now @ rarity_class  # E[rarity]：只用概率+类频，不用标签
        scores["true_rarity"] = rarity_class[labels]          # 上界参照
    else:
        if args.trigger in ("expected_rarity", "rarity_oracle"):
            raise SystemExit("[错误] 这两个触发需要 --features（统计类频）")


    # 先算"LLM 真能救"的 oracle 收益：用 stub 的命中情况估计（真 LLM 时改用实测）
    # ⚠️ oracle 必须用**与正式运行同一次**的 stub 抽样（原来用"探针"另抽一次，
    # 导致 oracle 反而低于 confidence，属于实现 bug）
    support = simulate_llm(labels, n_classes, args.stub_accuracy, args.stub_temperature, rng, skill=skill)
    pred_before = (alpha0 / alpha0.sum(axis=1, keepdims=True)).argmax(axis=1)
    alpha_all, _ = inject_prior(alpha0, support, np.ones(len(labels), bool),
                                args.prior_mode, args.strength)
    pred_all = (alpha_all / alpha_all.sum(axis=1, keepdims=True)).argmax(axis=1)
    oracle_benefit = ((pred_before != labels) & (pred_all == labels)).astype(np.float64)

    # 学习式触发器的输入特征：全部可在推理时得到（不含标签）
    sorted_probs = np.sort(before["probs"], axis=1)
    diversity = {
        "maxprob": sorted_probs[:, -1],
        "margin": sorted_probs[:, -1] - sorted_probs[:, -2],
        "negentropy": 1.0 + (before["probs"] * np.log(np.clip(before["probs"], EPS, None))).sum(axis=1),
        "vacuity": scores["vacuity"],
        "conflict": scores["conflict"],
    }
    if "expected_rarity" in scores:
        diversity["expected_rarity"] = scores["expected_rarity"]
    feature_matrix = np.column_stack([value for value in diversity.values()])
    feature_matrix = (feature_matrix - feature_matrix.mean(axis=0)) / (feature_matrix.std(axis=0) + 1e-8)
    scores["learned"] = learned_trigger_scores(feature_matrix, oracle_benefit, seed=args.seed)

    mask = select_by_trigger(args.trigger, args.budget, scores, rng, oracle_benefit)
    alpha1, delta = inject_prior(alpha0, support, mask, args.prior_mode, args.strength)
    after = metrics_of(alpha1, labels)

    print("=" * 78)
    print(f"LLM 骨架｜trigger={args.trigger} budget={args.budget:.0%} "
          f"stub_acc={args.stub_accuracy} prior={args.prior_mode}")
    print("=" * 78)
    print(f"  调用样本 {mask.sum()} / {len(labels)}")
    print(f"  {'指标':<12}{'before':>10}{'after':>10}{'Δ':>10}")
    for key in ("accuracy", "ece", "vacuity_mean", "conflict_mean"):
        delta_value = after[key] - before[key]
        print(f"  {key:<12}{before[key]:>10.4f}{after[key]:>10.4f}{delta_value:>+10.4f}")

    # 诊断：解释触发是否真的挑到了"会错"的样本
    wrong_before = pred_before != labels
    flips = int(((pred_before != labels) & (after["probs"].argmax(axis=1) == labels) & mask).sum())
    print(f"  被选样本中原本答错 {float(wrong_before[mask].mean()):.1%}"
          f"（全体 {float(wrong_before.mean()):.1%}）；其中翻转修对 {flips} 条")

    # 关键检测：vacuity 的下降是否只是"注入证据量"的算术结果
    injected = float(delta.sum(axis=1).mean() / n_classes)
    print(f"\n  注入证据量（Δα0 均值 / K）: {injected:+.4f}")
    if args.prior_mode == "additive" and injected > 0.01:
        print("  ⚠️ 提醒：additive 模式下 vacuity 下降是**算术必然**，不能当作证据")
    print(f"  触发样本上的 Δconflict: "
          f"{float(np.mean(dirichlet_conflict(alpha1, 5)[mask] - dirichlet_conflict(alpha0, 5)[mask])):+.4f}")

    if skill is not None:
        skill_sample = skill[labels]
        corr_vac = float(np.corrcoef(skill_sample, scores["vacuity"])[0, 1])
        corr_con = float(np.corrcoef(skill_sample, scores["conflict"])[0, 1])
        print(f"\n  [诊断] 技能(类频) vs vacuity 相关 = {corr_vac:+.4f}；"
              f"vs conflict 相关 = {corr_con:+.4f}")
        print("         （接近 0 说明：模型的不确定性并不指向 LLM 能帮上的地方）")

    # 分层：按 before 的 vacuity 四分位，看增益集中在哪
    quantiles = np.quantile(scores["vacuity"], [0.25, 0.5, 0.75])
    groups = np.digitize(scores["vacuity"], quantiles)
    pred_before = before["probs"].argmax(axis=1)
    pred_after = after["probs"].argmax(axis=1)
    print("\n  按 vacuity 四分位分层（Q1 最低 → Q4 最高）：")
    print(f"  {'组':<6}{'n':>6}{'before':>10}{'after':>10}{'Δacc':>10}{'调用率':>9}")
    layers = []
    for group in range(4):
        index = groups == group
        if not index.any():
            continue
        before_acc = float((pred_before[index] == labels[index]).mean())
        after_acc = float((pred_after[index] == labels[index]).mean())
        layers.append({"quartile": group + 1, "n": int(index.sum()),
                       "before": before_acc, "after": after_acc,
                       "delta": after_acc - before_acc,
                       "call_rate": float(mask[index].mean())})
        print(f"  Q{group + 1:<5}{index.sum():>6}{before_acc:>10.4f}{after_acc:>10.4f}"
              f"{after_acc - before_acc:>+10.4f}{mask[index].mean():>9.0%}")

    payload = {
        "args": vars(args),
        "before": {k: v for k, v in before.items() if k != "probs"},
        "after": {k: v for k, v in after.items() if k != "probs"},
        "delta": {k: after[k] - before[k] for k in ("accuracy", "ece", "vacuity_mean", "conflict_mean")},
        "called": int(mask.sum()),
        "injected_evidence": injected,
        "quartiles": layers,
        "note": "stub 用真实标签构造 LLM 输出，仅为模拟器；换真 LLM 时改 simulate_llm",
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"\n[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
