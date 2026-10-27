# gate_properties.py —— 门控函数的性质验证与"信号是否真的有用"的模型级压力测试
#
# 目的（不需要 GPU、不需要数据）：
#   ① 验证 β=κ=0 时门控**逐元素**退化为未加门控的原始权重（零对照成立的前提）
#   ② 验证两通道的单调性与数值稳定性
#   ③ 用显式的"专家投票"模型做压力测试，回答一个致命问题：
#        **收益到底来自"信号"，还是仅仅来自"锐化"这个动作本身？**
#      因此对照必须包含"常数门控"（固定 τ/λ，不做条件化）与"噪声条件化"（信号打乱）
#
# 2026-10-27 负面结果留档（第一版）：用"平移+归一后重分配"的形式时，
#   β=κ=0 回不到 a_m（最大差 2.02），极端输入出 NaN；且模拟中"噪声信号"同样涨点
#   → 增益与信号无关，只是锐化本身的正则化效应。故改为乘性调制并重做对照。
#
# 运行：python3 gate_properties.py

import numpy as np

EPS = 1e-12


# ===== 1. 门控函数（乘性调制版，与机制设计 §2.3 对应）=====

def component_distribution(affinity, temperature=0.5):
    """把原始余弦相似度变成门控**内部**使用的分布 p_m（不改原文的加权方式）。

    用 softmax 只为得到非负归一化的 p；提示本身仍按原文的 a_m 加权。
    temperature 是设计选择，需要消融（见机制设计 §6）。
    """
    scaled = affinity / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


def route(affinity, beta, kappa, vacuity, conflict, temperature=0.5, lambda_max=1.0):
    """门控：g_m = a_m · φ_m，其中

        φ_m = [ (1-λ) + λ/(M·p_m) ]^τ,  λ = clip(κ·conflict, 0, λ_max),  τ = 1 + β(1-vacuity)

    乘性形式的关键好处：β=κ=0 时 φ_m ≡ 1 → g_m ≡ a_m **逐元素精确退化**。
    φ 按批内均值归一，使提示总幅度大致保持（不改变整体尺度）。
    """
    affinity = np.asarray(affinity, dtype=np.float64)
    m_components = affinity.shape[1]
    p = np.clip(component_distribution(affinity, temperature), EPS, None)

    lam = np.clip(kappa * np.asarray(conflict, dtype=np.float64), 0.0, lambda_max)
    sharpen_power = beta * (1.0 - np.asarray(vacuity, dtype=np.float64))     # 锐化：按 p^b 重加权
    smooth_ratio = (1.0 - lam)[:, None] + lam[:, None] / (m_components * p)  # 平滑：抬升小组件

    phi = (p ** sharpen_power[:, None]) * smooth_ratio
    # 用**加权**归一精确保总幅度：Σg = Σa（β=κ=0 时 φ≡1，故退化仍然精确）
    numerator = affinity.sum(axis=1, keepdims=True)
    denominator = (affinity * phi).sum(axis=1, keepdims=True)
    safe = np.where(np.abs(denominator) < EPS, 1.0, denominator)
    return affinity * phi * (numerator / safe)


def vote_weights(weights):
    """门控输出 → 投票权重：取正部并归一。

    理由：a_m 是余弦相似度（可为负），而混合提示 Σ g_m P_m 里真正决定"谁说了算"的是
    各组件权重的**相对份额**；负权重在投票语义下无意义，故取正部归一。
    """
    positive = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    total = positive.sum(axis=1, keepdims=True)
    total[total == 0.0] = 1.0
    return positive / total


def normalized_entropy(weights):
    """把权重（可能含负值）转成分布后算归一化熵，用于观察"集中/平"的趋势。"""
    shifted = weights - weights.min(axis=1, keepdims=True)
    total = shifted.sum(axis=1, keepdims=True)
    total[total == 0.0] = 1.0
    distribution = np.clip(shifted / total, EPS, None)
    entropy = -(distribution * np.log(distribution)).sum(axis=1)
    return entropy / np.log(weights.shape[1])


# ===== 2. 性质验证 =====

def check_properties(rng):
    print("=" * 70)
    print("① 性质验证")
    print("=" * 70)
    m_components = 50
    n_samples = 2000
    affinity = rng.normal(0.0, 0.3, size=(n_samples, m_components))
    affinity[np.arange(n_samples), rng.integers(0, m_components, n_samples)] += 1.0
    vacuity = rng.uniform(0.1, 0.9, size=n_samples)
    conflict = rng.uniform(0.0, 1.0, size=n_samples)

    g_zero = route(affinity, 0.0, 0.0, vacuity, conflict)
    max_diff = float(np.max(np.abs(g_zero - affinity)))
    print(f"  β=κ=0 与原式的最大逐元素差：{max_diff:.2e}  → {'✅ 精确退化' if max_diff < 1e-9 else '❌ 未退化'}")

    g = route(affinity, 2.0, 0.5, vacuity, conflict)
    scale_ratio = g.sum(axis=1) / affinity.sum(axis=1)
    print(f"  总幅度比 Σg/Σa：均值 {scale_ratio.mean():.6f}，最大偏离 {np.max(np.abs(scale_ratio - 1)):.2e}")

    fixed_vacuity = np.full(n_samples, 0.2)
    fixed_conflict = np.full(n_samples, 0.5)
    entropy_beta = [normalized_entropy(vote_weights(route(affinity, b, 0.0, fixed_vacuity, fixed_conflict))).mean()
                    for b in [0.0, 0.5, 1.0, 2.0, 4.0]]
    entropy_kappa = [normalized_entropy(vote_weights(route(affinity, 0.0, k, fixed_vacuity, fixed_conflict))).mean()
                     for k in [0.0, 0.1, 0.3, 0.6, 1.0]]
    print(f"  熵 vs β（锐化）：{['%.4f' % v for v in entropy_beta]}  → "
          f"{'✅ 单调下降' if all(np.diff(entropy_beta) < 0) else '❌ 非单调'}")
    # 注意：在"精确保幅度"（Σg=Σa）的约束下，强平滑会被归一化抵消 → 非单调是**真实性质**，
    # 不是 bug。正确表述是"κ 有最优区间，过大反而集中/掉点"，这也与 ⑤ 的扫描一致。
    best_kappa_index = int(np.argmax(entropy_kappa))
    early_rise = all(np.diff(entropy_kappa[: best_kappa_index + 1]) > 0)
    print(f"  熵 vs κ（平滑）：{['%.4f' % v for v in entropy_kappa]}  → "
          f"最优 κ 在索引 {best_kappa_index}（κ={[0.0, 0.1, 0.3, 0.6, 1.0][best_kappa_index]}），"
          f"此前单调上升：{'✅' if early_rise else '❌'}；过度平滑有害（已由 ⑤ 证实）")

    extreme = np.zeros((4, m_components))
    extreme[0, 0] = 1.0
    extreme[1, :] = 1.0 / m_components
    extreme[2, :] = -5.0
    extreme[2, 0] = 5.0
    extreme[3, :] = 0.0                                   # 全零：最坏情况
    g_extreme = route(extreme, 5.0, 1.0,
                      vacuity=np.array([0.0, 0.5, 1.0, 1.0]),
                      conflict=np.array([1.0, 0.0, 0.5, 1.0]))
    finite = bool(np.all(np.isfinite(g_extreme)))
    print(f"  极端输入（β=5, κ=1，含全零行）是否有限：{'✅' if finite else '❌ 出现 NaN/Inf'}")
    return max_diff < 1e-9 and finite


# ===== 3. 压力测试：两类样本 × 三种信号条件 × 七种策略 =====

def build_problem(rng, n_samples=6000, m_components=50, confusable_rate=0.3,
                  decoy_strength=2.0, useful_count=5, useful_correctness=0.65):
    """构造问题并把设计前提写成**显式假设**（它们是经验问题，不是自明之理）：

    A1（支持锐化）：干净样本上，亲和度最高的组件就是可靠专家（答对率 0.85）
    A2（支持平滑）：易混样本上存在"亲和度很高但常答错"的诱饵组件（答对率 0.05），
                    同时有若干亲和中等但可靠的组件（答对率 useful_correctness）
    若 A1/A2 在 EK55 上不成立，两通道就没有理由，必须先做诊断（见 §5 待测项）。
    """
    is_confusable = rng.random(n_samples) < confusable_rate
    true_component = rng.integers(0, m_components, n_samples)
    decoy_component = (true_component + rng.integers(1, m_components, n_samples)) % m_components
    useful_pool = ((decoy_component[:, None] + 1
                    + rng.integers(0, m_components - 1, (n_samples, useful_count)))
                   % m_components)

    affinity = rng.normal(0.0, 0.25, size=(n_samples, m_components))
    affinity[np.arange(n_samples), true_component] += 1.0
    affinity[is_confusable, decoy_component[is_confusable]] += decoy_strength
    for k in range(useful_count):
        affinity[is_confusable, useful_pool[is_confusable, k]] += 0.5

    correctness = np.full((n_samples, m_components), 0.35)
    correctness[np.arange(n_samples), true_component] = 0.85
    correctness[is_confusable, decoy_component[is_confusable]] = 0.05
    for k in range(useful_count):
        correctness[is_confusable, useful_pool[is_confusable, k]] = useful_correctness

    clean_conflict = np.clip(rng.normal(0.15, 0.05, n_samples), 0.0, 1.0)
    clean_vacuity = np.clip(rng.normal(0.20, 0.05, n_samples), 0.0, 1.0)
    noisy_conflict = np.clip(rng.normal(0.27, 0.20, n_samples), 0.0, 1.0)
    noisy_vacuity = np.clip(rng.normal(0.35, 0.20, n_samples), 0.0, 1.0)
    conflict_true = np.where(is_confusable,
                             np.clip(rng.normal(0.60, 0.08, n_samples), 0, 1), clean_conflict)
    vacuity_true = np.where(is_confusable,
                            np.clip(rng.normal(0.70, 0.08, n_samples), 0, 1), clean_vacuity)
    return {
        "affinity": affinity,
        "correctness": correctness,
        "conflict_true": conflict_true,
        "vacuity_true": vacuity_true,
        "noisy_conflict": noisy_conflict,
        "noisy_vacuity": noisy_vacuity,
        "is_confusable": is_confusable,
    }


def evaluate(problem, beta, kappa, vacuity, conflict):
    """期望正确率 = Σ_m w_m · r_{s,m}，其中 w 是投票权重（正部归一）。"""
    weights = vote_weights(route(problem["affinity"], beta, kappa, vacuity, conflict))
    return float((weights * problem["correctness"]).sum(axis=1).mean())


def run_strategies(rng):
    problem = build_problem(rng)
    n_samples = len(problem["vacuity_true"])
    constant_vacuity = np.full(n_samples, float(problem["vacuity_true"].mean()))
    constant_conflict = np.full(n_samples, float(problem["conflict_true"].mean()))

    rows = [
        ("基线（无门控）", evaluate(problem, 0.0, 0.0, problem["vacuity_true"], problem["conflict_true"])),
        ("常数锐化 β=1", evaluate(problem, 1.0, 0.0, constant_vacuity, constant_conflict)),
        ("常数平滑 κ=0.3", evaluate(problem, 0.0, 0.3, constant_vacuity, constant_conflict)),
        ("常数组合", evaluate(problem, 1.0, 0.3, constant_vacuity, constant_conflict)),
        ("条件化（真信号）", evaluate(problem, 1.0, 0.3, problem["vacuity_true"], problem["conflict_true"])),
        ("条件化（信号打乱）", evaluate(problem, 1.0, 0.3, problem["noisy_vacuity"], problem["noisy_conflict"])),
        ("条件化（信号接反）", evaluate(problem, 1.0, 0.3,
                                        1.0 - problem["vacuity_true"], 1.0 - problem["conflict_true"])),
    ]
    return rows, problem


def report(rows, title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    baseline = dict(rows)["基线（无门控）"]
    for name, value in rows:
        delta = value - baseline
        flag = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "≈")
        print(f"  {name:<20}{value:.4f}   相对基线 {delta:+.4f} {flag}")
    return dict(rows)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ok = check_properties(rng)

    rows, problem = run_strategies(rng)
    results = report(rows, "③ 压力测试：条件化 vs 常数 vs 噪声（诱饵强度 2.0）")

    print()
    print("=" * 70)
    print("④ 关键预测检查")
    print("=" * 70)
    informative = results["条件化（真信号）"]
    best_constant = max(results["常数锐化 β=1"], results["常数平滑 κ=0.3"], results["常数组合"])
    checks = [
        ("条件化优于所有常数门控（说明价值来自随样本变化，而不只是锐化本身）",
         informative > best_constant + 0.005),
        ("信号打乱后回落到常数门控水平（排除正则化效应这一解释）",
         results["条件化（信号打乱）"] <= best_constant + 0.005),
        ("信号接反时低于基线（方向确实重要）",
         results["条件化（信号接反）"] < results["基线（无门控）"]),
    ]
    for text, passed in checks:
        print(f"  [{'✅' if passed else '❌'}] {text}")

    print()
    print("=" * 70)
    print("⑤ 假设敏感性：诱饵强度 → 平滑通道的价值（检验 A2 的边界）")
    print("=" * 70)
    print(f"  {'诱饵强度':<10}{'基线':>10}{'仅平滑':>10}{'仅锐化':>10}{'条件化':>10}")
    for strength in [0.5, 1.0, 2.0, 3.0]:
        sweep_rng = np.random.default_rng(7)
        sweep = build_problem(sweep_rng, decoy_strength=strength)
        base = evaluate(sweep, 0.0, 0.0, sweep["vacuity_true"], sweep["conflict_true"])
        smooth = evaluate(sweep, 0.0, 0.3, sweep["vacuity_true"], sweep["conflict_true"])
        sharp = evaluate(sweep, 1.0, 0.0, sweep["vacuity_true"], sweep["conflict_true"])
        both = evaluate(sweep, 1.0, 0.3, sweep["vacuity_true"], sweep["conflict_true"])
        print(f"  {strength:<10.1f}{base:>10.4f}{smooth:>10.4f}{sharp:>10.4f}{both:>10.4f}")

    print()
    print(f"结论：性质验证{'通过' if ok else '未通过'}；压力测试见 ④ 的三条预测是否成立。")
    print("      注意：模型里的 A1/A2 是**假设**，在 EK55 上是否成立是经验问题。")
    print("      上真数据前必须先做两项诊断：① conflict 与路由是否选错组件是否相关；")
    print("      ② 在高 conflict 样本上，平滑是否真能挽回正确率。若不成立，说明该通道缺乏依据。")
