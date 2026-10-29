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

def simulate_llm(labels, n_classes, accuracy, temperature, rng):
    """可控 stub：以 accuracy 的概率把质量压到真实类上，否则随机分散。

    返回 π（每个候选类的"支持度"，行归一化）。
    注意：这里用到 labels 是因为它是**模拟器**；真 LLM 只能看到文本与候选列表。
    """
    n_samples = len(labels)
    scores = rng.normal(0.0, 1.0, size=(n_samples, n_classes))
    hit = rng.random(n_samples) < accuracy
    # 命中：真实类得分抬高；未命中：随机抬一个错类
    wrong = (labels + rng.integers(1, n_classes, n_samples)) % n_classes
    target = np.where(hit, labels, wrong)
    scores[np.arange(n_samples), target] += temperature
    scores = scores - scores.max(axis=1, keepdims=True)
    support = np.exp(scores)
    return support / support.sum(axis=1, keepdims=True)


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
    n_classes = alpha.shape[1]
    if mode == "redistribute":
        centered = support - support.mean(axis=1, keepdims=True)
        delta = strength * centered
    elif mode == "additive":
        delta = strength * support * n_classes
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
                        choices=["vacuity", "conflict", "confidence", "random", "oracle"])
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--stub-accuracy", type=float, default=0.7, help="模拟 LLM 的命中率")
    parser.add_argument("--stub-temperature", type=float, default=3.0, help="模拟 LLM 的置信程度")
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

    before = metrics_of(alpha0, labels)
    scores = {"vacuity": dirichlet_vacuity(alpha0), "conflict": dirichlet_conflict(alpha0, k=5),
              "maxprob": before["probs"].max(axis=1)}

    # 先算"LLM 真能救"的 oracle 收益：用 stub 的命中情况估计（真 LLM 时改用实测）
    support_probe = simulate_llm(labels, n_classes, args.stub_accuracy,
                                 args.stub_temperature, np.random.default_rng(args.seed + 1))
    alpha_probe, _ = inject_prior(alpha0, support_probe, np.ones(len(labels), bool),
                                  args.prior_mode, args.strength)
    fixed = (alpha0 / alpha0.sum(axis=1, keepdims=True)).argmax(axis=1) != labels
    helped = (alpha_probe / alpha_probe.sum(axis=1, keepdims=True)).argmax(axis=1) == labels
    oracle_benefit = (fixed & helped).astype(np.float64)

    mask = select_by_trigger(args.trigger, args.budget, scores, rng, oracle_benefit)
    support = simulate_llm(labels, n_classes, args.stub_accuracy, args.stub_temperature, rng)
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

    # 关键检测：vacuity 的下降是否只是"注入证据量"的算术结果
    injected = float(delta.sum(axis=1).mean() / n_classes)
    print(f"\n  注入证据量（Δα0 均值 / K）: {injected:+.4f}")
    if args.prior_mode == "additive" and injected > 0.01:
        print("  ⚠️ 提醒：additive 模式下 vacuity 下降是**算术必然**，不能当作证据")
    print(f"  触发样本上的 Δconflict: "
          f"{float(np.mean(dirichlet_conflict(alpha1, 5)[mask] - dirichlet_conflict(alpha0, 5)[mask])):+.4f}")

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
