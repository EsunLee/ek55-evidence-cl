# exp4_wrapper.py —— 证据包装器 vs 最强事后校准（方案 A2 的决定性对照）
#
# 背景（必须搞清的逻辑）：
#   若包装器让 p = α/α0 与 CE 头**完全相同**，则 ECE 也完全相同，校准不可能改善。
#   所以"证据包装器"的正确形态是：p = softmax(z / τ(x))，即**输入相关的温度**——
#   温度缩放**不改 argmax**，因此精度由构造保证不变。
#   但这样一来，它就是"事后校准"的一种，所以必须先回答：
#       **输入相关温度（最强的输入依赖事后校准）能否追平证据头？**
#
# 本脚本做四组对照（全部在测试集上评估，均只用验证集拟合）：
#   ① CE（原始）                        ② CE + 全局温度缩放
#   ③ CE + 向量缩放（按类）              ④ CE + 输入相关温度 τ(x)（小 MLP，argmax 不变）
# 判据：
#   若 ④ 的 ECE 已 ≤ 证据头（exp1 实测 0.0073–0.0107），则"证据头的校准优势"不成立；
#   若 ④ 仍明显更差，则 A 的主张成立：证据头赢的不是"事后校准能做的事"。
#
# 用法（服务器）：
#   python3 code/exp4_wrapper.py --features /data/ldy/ek55_assets/ek55_verb_full.npz \
#       --out results/exp4_wrapper.json

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp3_decompose import train_softmax, softmax_rows          # 复用线性 softmax 训练
from metrics import (apply_vector_scaling, auroc, brier_score, ece_adaptive,
                     ece_equal_width, fit_temperature, fit_vector_scaling, nll_score)

EPS = 1e-12


# ===== 1. 输入相关温度 τ(x)：小 MLP，输出 softplus 保证为正 =====

def softplus(value):
    return np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0.0)


def train_input_temperature(train_x, train_z, train_y, monitor_x, monitor_z, monitor_y,
                            hidden=64, steps=300, lr=0.01, weight_decay=1e-3, seed=0):
    """学习 τ(x) = softplus(MLP(x))，目标是最小化 NLL。

    为什么 argmax 不变：对所有类别同时除以同一个正数 τ，排序不变 → 精度**由构造保证**不变。
    训练/监控集分离（验证集前 2/3 拟合、后 1/3 早停），避免"自己拟合自己"。
    """
    rng = np.random.default_rng(seed)
    dim = train_x.shape[1]
    n_classes = train_z.shape[1]

    w1 = rng.normal(0.0, 1.0 / np.sqrt(dim), (dim, hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0.0, 1.0 / np.sqrt(hidden), hidden)
    b2 = np.array([0.0])                      # 初始 τ ≈ softplus(0) ≈ 0.69

    params = [w1, b1, w2, b2]
    moments = [np.zeros_like(p) for p in params]
    velocities = [np.zeros_like(p) for p in params]
    beta1, beta2 = 0.9, 0.999
    best = (np.inf, [p.copy() for p in params])
    one_hot_train = np.eye(n_classes)[train_y]

    for step in range(1, steps + 1):
        hidden_raw = train_x @ w1 + b1
        hidden_act = np.tanh(hidden_raw)
        tau_raw = (hidden_act @ w2 + b2[0])[:, None]        # 保持 (N,1)，否则会广播成 (N,N)
        tau = np.clip(softplus(tau_raw), 0.05, 20.0)        # 限幅：τ 太小会让 z/τ 溢出

        probs = softmax_rows(train_z / tau)
        grad_logits = (probs - one_hot_train) / len(train_y)      # dNLL/d(softmax 输入)
        # 链式：dNLL/dτ = Σ_c grad_logits_c · (−z_c/τ²)
        grad_tau = (-(grad_logits * train_z).sum(axis=1) / (tau[:, 0] ** 2))[:, None]
        grad_tau_raw = grad_tau * (1.0 / (1.0 + np.exp(-tau_raw)).clip(1e-6, None))  # softplus 导数

        grad_w2 = hidden_act.T @ grad_tau_raw[:, 0] + weight_decay * w2
        grad_b2 = grad_tau_raw.sum(axis=0)
        grad_hidden = grad_tau_raw * w2[None, :] * (1.0 - hidden_act ** 2)
        grad_w1 = train_x.T @ grad_hidden + weight_decay * w1
        grad_b1 = grad_hidden.sum(axis=0)

        for index, gradient in enumerate((grad_w1, grad_b1, grad_w2, grad_b2)):
            moments[index] *= beta1
            moments[index] += (1 - beta1) * gradient
            velocities[index] *= beta2
            velocities[index] += (1 - beta2) * gradient * gradient
            m_hat = moments[index] / (1 - beta1 ** step)
            v_hat = velocities[index] / (1 - beta2 ** step)
            params[index] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        # 监控集上的 NLL（用于早停）
        monitor_act = np.tanh(monitor_x @ params[0] + params[1])
        monitor_tau = np.clip(softplus((monitor_act @ params[2] + params[3][0])[:, None]), 0.05, 20.0)
        monitor_nll = nll_score(softmax_rows(monitor_z / monitor_tau), monitor_y)
        if monitor_nll < best[0]:
            best = (monitor_nll, [p.copy() for p in params])

    return best[1]


def apply_input_temperature(features, logits, params):
    w1, b1, w2, b2 = params
    hidden_act = np.tanh(features @ w1 + b1)
    tau = np.clip(softplus((hidden_act @ w2 + b2[0])[:, None]), 0.05, 20.0)
    return softmax_rows(logits / tau), tau[:, 0]


# ===== 2. 主流程 =====

def main():
    parser = argparse.ArgumentParser(description="证据包装器 vs 最强事后校准")
    parser.add_argument("--features", required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=True)
    train_x = data["x_train"].astype(np.float64)
    val_x = data["x_val"].astype(np.float64)
    test_x = data["x_test"].astype(np.float64)
    train_y, val_y, test_y = data["y_train"], data["y_val"], data["y_test"]

    # 标准化（用训练集统计量）
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-6
    train_x, val_x, test_x = (train_x - mean) / std, (val_x - mean) / std, (test_x - mean) / std

    n_classes = int(max(train_y.max(), val_y.max(), test_y.max())) + 1
    print(f"[数据] train {len(train_y)} / val {len(val_y)} / test {len(test_y)}；类别 {n_classes}")

    print("[训练] CE 头")
    weights, bias = train_softmax(train_x, train_y, n_classes, epochs=args.epochs, seed=args.seed)
    val_z, test_z = val_x @ weights + bias, test_x @ weights + bias
    ce_test_probs = softmax_rows(test_z)

    print("[拟合] ① 无校准 / ② 全局温度 / ③ 向量缩放 / ④ 输入相关温度")
    temperature, _ = fit_temperature(val_z, val_y)
    scale, offset = fit_vector_scaling(val_z, val_y)

    split = int(len(val_y) * 2 / 3)
    params = train_input_temperature(
        val_x[:split], val_z[:split], val_y[:split],
        val_x[split:], val_z[split:], val_y[split:], seed=args.seed)
    input_temp_probs, tau = apply_input_temperature(test_x, test_z, params)

    rows = {
        "① CE（无校准）": ce_test_probs,
        "② CE + 全局温度": softmax_rows(test_z / temperature),
        "③ CE + 向量缩放": apply_vector_scaling(test_z, scale, offset),
        "④ CE + 输入相关温度": input_temp_probs,
    }

    print("\n" + "=" * 74)
    print("事后校准对照（测试集）｜精度必须一致：只有 ③ 可能改变 argmax")
    print("=" * 74)
    print(f"  {'方法':<24}{'accuracy':>10}{'ECE':>10}{'ECE(自适应)':>12}{'NLL':>10}{'Brier':>9}")
    results = {}
    for name, probs in rows.items():
        record = {
            "accuracy": float((probs.argmax(axis=1) == test_y).mean()),
            "ece_equal_width": float(ece_equal_width(probs, test_y)),
            "ece_adaptive": float(ece_adaptive(probs, test_y)),
            "nll": float(nll_score(probs, test_y)),
            "brier": float(brier_score(probs, test_y)),
            "auroc_confidence": float(auroc(-probs.max(axis=1), probs.argmax(axis=1) == test_y)),
        }
        results[name] = record
        print(f"  {name:<24}{record['accuracy']:>10.4f}{record['ece_equal_width']:>10.4f}"
              f"{record['ece_adaptive']:>12.4f}{record['nll']:>10.3f}{record['brier']:>9.4f}")

    best_posthoc = min(results[name]["ece_equal_width"] for name in ("② CE + 全局温度", "③ CE + 向量缩放", "④ CE + 输入相关温度"))
    evidence_reference = 0.0073            # exp1 实测：证据头在 lastk-5 上的 ECE（最低值）
    print(f"\n  最强事后校准 ECE = {best_posthoc:.4f}；证据头实测 ECE ≈ {evidence_reference:.4f}（exp1, lastk-5）")
    verdict = ("证据头的校准优势**成立**：最强事后校准仍差 "
               f"{(best_posthoc - evidence_reference) * 100:.2f} 个百分点"
               if best_posthoc > evidence_reference * 1.2 else
               "证据头的校准优势**不成立**：事后校准已能追平或超过它")
    print(f"  [结论] {verdict}")
    print(f"  τ 的分布：min {tau.min():.3f} / 中位 {np.median(tau):.3f} / max {tau.max():.3f}")

    payload = {
        "features": os.path.abspath(args.features),
        "results": results,
        "temperature": float(temperature),
        "input_temperature_stats": {"min": float(tau.min()), "median": float(np.median(tau)),
                                    "max": float(tau.max())},
        "best_posthoc_ece": float(best_posthoc),
        "evidence_reference_ece": evidence_reference,
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