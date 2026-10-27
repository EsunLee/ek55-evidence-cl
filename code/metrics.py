# 可信度指标库（numpy 实现，不依赖 torch —— 便于在 macOS 上自检）
#
# 为什么单独一个文件：所有实验共用同一份实现，避免各脚本各写一份 ECE 导致口径不一致。
#
# 用法：
#   from metrics import ece_equal_width, auroc, aurc, bucket_report
#   python3 metrics.py          # 跑自检（合成数据，验证实现正确）
#
# 约定：
#   probs  [N, K] 每行和为 1；labels [N] 为 0..K-1；logits [N, K] 未归一化
#   多分类 Brier 用 Σ_k (p_k - y_k)^2 的平均值（y 为 one-hot）

import numpy as np

# 兼容性：np.trapezoid 是 numpy 2.0 才有的名字，旧环境（如 conda 里的 numpy 1.x）用 np.trapz
_trapezoid = getattr(np, "trapezoid", None)
if _trapezoid is None:
    _trapezoid = np.trapz


# ===== 1. 输入整理 =====

def prepare(probs, labels):
    """统一 dtype 并算出置信度 / 预测 / 是否正确，供后续指标复用。"""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = (prediction == labels).astype(np.float64)
    return probs, labels, confidence, prediction, correct


def softmax(logits, temperature=1.0):
    """数值稳定的 softmax，带温度（温度缩放对照用）。"""
    logits = np.asarray(logits, dtype=np.float64) / float(temperature)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / exp_shifted.sum(axis=1, keepdims=True)


# ===== 2. 校准指标 =====

def ece_equal_width(probs, labels, n_bins=15):
    """等宽分箱 ECE：|平均置信度 - 平均准确率| 按样本数加权。"""
    probs, labels, confidence, _, correct = prepare(probs, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(labels)
    ece = 0.0
    for i in range(n_bins):
        lower = edges[i]
        upper = edges[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(confidence[mask].mean() - correct[mask].mean())
        ece += (count / total) * gap
    return float(ece)


def ece_adaptive(probs, labels, n_bins=15):
    """等质量分箱 ECE（每个箱样本数相近）——类不平衡时比等宽更稳。"""
    probs, labels, confidence, _, correct = prepare(probs, labels)
    order = np.argsort(confidence, kind="mergesort")
    total = len(labels)
    ece = 0.0
    for index in np.array_split(order, n_bins):
        if len(index) == 0:
            continue
        gap = abs(confidence[index].mean() - correct[index].mean())
        ece += (len(index) / total) * gap
    return float(ece)


def brier_score(probs, labels):
    probs, labels, _, _, _ = prepare(probs, labels)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def nll_score(probs, labels, eps=1e-12):
    probs, labels, _, _, _ = prepare(probs, labels)
    true_prob = np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)
    return float(-np.mean(np.log(true_prob)))


def reliability_table(probs, labels, n_bins=15):
    """可靠性表：画 reliability diagram 与自查用。"""
    probs, labels, confidence, _, correct = prepare(probs, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lower = edges[i]
        upper = edges[i + 1]
        mask = (confidence >= lower) & (confidence <= upper) if i == n_bins - 1 \
            else (confidence >= lower) & (confidence < upper)
        count = int(mask.sum())
        rows.append({
            "bin": i,
            "range": (float(lower), float(upper)),
            "count": count,
            "avg_confidence": float(confidence[mask].mean()) if count else None,
            "avg_accuracy": float(correct[mask].mean()) if count else None,
        })
    return rows


# ===== 3. 误分类检测（排序类指标）=====

def auroc(scores, positive):
    """用 score 给"正类"排序的 AUROC（Mann-Whitney U，含并列平均秩）。

    典型用法：auroc(-vacuity, ~correct) 或 auroc(-confidence, ~correct)
    """
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive).astype(bool)
    n_pos = int(positive.sum())
    n_neg = int(len(positive) - n_pos)
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
        average_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = average_rank
        i = j + 1
    u_statistic = ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u_statistic / (n_pos * n_neg))


def auprc(scores, positive):
    """平均精度（AP）：类不平衡时比 AUROC 更敏感。"""
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive).astype(bool)
    if positive.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_positive = positive[order]
    cumulative_true = np.cumsum(sorted_positive)
    cumulative_count = np.arange(1, len(sorted_positive) + 1)
    precision = cumulative_true / cumulative_count
    recall = cumulative_true / positive.sum()
    delta_recall = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(precision * delta_recall))


# ===== 4. 选择性预测 =====

def risk_coverage_curve(probs, labels):
    """按置信度从高到低保留样本：返回 (coverage, accuracy, risk) 三条曲线。"""
    probs, labels, confidence, _, correct = prepare(probs, labels)
    order = np.argsort(-confidence, kind="mergesort")
    sorted_correct = correct[order]
    coverage = np.arange(1, len(labels) + 1) / len(labels)
    accuracy = np.cumsum(sorted_correct) / np.arange(1, len(labels) + 1)
    return coverage, accuracy, 1.0 - accuracy


def aurc(probs, labels):
    """风险—覆盖曲线下面积（梯形积分，覆盖率为 0→1）。越小越好。"""
    coverage, _, risk = risk_coverage_curve(probs, labels)
    return float(_trapezoid(risk, coverage))


def accuracy_at_coverage(probs, labels, coverage):
    """只保留置信度最高的 coverage 比例样本时的准确率。"""
    probs, labels, confidence, _, correct = prepare(probs, labels)
    keep = max(1, int(round(coverage * len(labels))))
    order = np.argsort(-confidence, kind="mergesort")
    return float(correct[order[:keep]].mean())


def coverage_for_accuracy(probs, labels, target_accuracy, min_samples=30):
    """达到目标精度所需的最小覆盖率——"同等精度比较"的核心函数。

    两方法各自求 coverage_for_accuracy，再在同一覆盖率/精度水平上比 ECE 与 AURC。

    min_samples 的作用：覆盖率极小时（例如只保留 1 个样本），经验准确率非 0 即 1，
    会立刻"达到"任何目标精度，使该函数退化。因此要求保留样本数不少于 min_samples。
    """
    coverage, accuracy, _ = risk_coverage_curve(probs, labels)
    retained = np.arange(1, len(labels) + 1)
    valid = retained >= int(min_samples)
    reached = np.where(valid & (accuracy >= target_accuracy))[0]
    if len(reached) == 0:
        return float("nan")
    return float(coverage[reached[0]])


# ===== 5. 温度缩放（事后校准对照）=====

def fit_temperature(logits, labels, grid=None):
    """在给定集合上最小化 NLL 选温度（网格搜索，避免梯度实现差异）。"""
    if grid is None:
        grid = np.concatenate([
            np.linspace(0.5, 3.0, 26),
            np.linspace(3.2, 20.0, 43),
        ])
    best_temperature = 1.0
    best_nll = np.inf
    for temperature in grid:
        candidate = softmax(logits, temperature)
        value = nll_score(candidate, labels)
        if value < best_nll:
            best_nll = value
            best_temperature = float(temperature)
    if grid is not None and best_temperature >= float(np.max(grid)):
        # 最优点落在网格上界：说明真实温度更大，调用方需要放宽 grid
        print("[警告] 温度最优点位于网格上界，建议放宽 fit_temperature 的 grid")
    return best_temperature, float(best_nll)


# ===== 6. Dirichlet 证据读出 =====

def dirichlet_mean(alpha):
    alpha = np.asarray(alpha, dtype=np.float64)
    return alpha / alpha.sum(axis=1, keepdims=True)


def dirichlet_vacuity(alpha):
    """空虚度（认知不确定性）：K / Σα。证据越少越接近 1。"""
    alpha = np.asarray(alpha, dtype=np.float64)
    n_classes = alpha.shape[1]
    return n_classes / alpha.sum(axis=1)


def dirichlet_dissonance(alpha, prior=1.0):
    """Dissonance：基于 belief mass 的成对平衡度（标准定义）。

        b = e / S,  Bal(b_i,b_c) = 1 - |b_i-b_c|/(b_i+b_c)（分母为 0 时取 0）
        Diss = Σ_c b_c · [ Σ_{i≠c} b_i·Bal(b_i,b_c) / Σ_{i≠c} b_i ]
    参考文献：Aguilar et al., Continual Evidential Deep Learning for OOD Detection,
    ICCVW 2023（式 5）。两个类别 belief 各 0.5 时取最大值 1；证据集中在单类时趋近 0。

    prior：先验证据量。单位先验（α = e + 1）时默认 prior=1；若模型使用可学先验 u，则传 u。
    ⚠️ 注意：b = e/S 而 S 含先验质量，故**先验越大、同一冲突下 dissonance 越小**；
    跨方法比较时必须统一 prior，否则数值不可比（这是我实测发现的，不是原文说法）。
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    n_samples, n_classes = alpha.shape
    evidence = np.clip(alpha - prior, 0.0, None)
    strength = np.clip(alpha.sum(axis=1, keepdims=True), 1e-12, None)
    belief = evidence / strength
    result = np.zeros(n_samples, dtype=np.float64)
    for c in range(n_classes):
        b_c = belief[:, c:c + 1]
        others = belief.copy()
        others[:, c] = 0.0
        denominator = others.sum(axis=1, keepdims=True)
        both_positive = (others > 0.0) & (b_c > 0.0)
        balance = np.where(
            both_positive,
            1.0 - np.abs(others - b_c) / np.clip(others + b_c, 1e-12, None),
            0.0,
        )
        contribution = (others * balance).sum(axis=1, keepdims=True) / np.clip(denominator, 1e-12, None)
        result += (b_c * contribution).squeeze(axis=1)
    return result


def dirichlet_conflict(alpha, k=5):
    """冲突度（本文定义）：top-k 证据占比的归一化熵。越接近 1 越"平"、越易混。

    注意：这不是标准 dissonance（见 dirichlet_dissonance），而是 top-k 证据占比的
    归一化熵，属于本文自定义量，引用时须写清定义。
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    proportions = dirichlet_mean(alpha)
    top_k = np.sort(proportions, axis=1)[:, -k:]
    top_k = top_k / top_k.sum(axis=1, keepdims=True)
    entropy = -(top_k * np.log(np.clip(top_k, 1e-12, None))).sum(axis=1)
    return entropy / np.log(k)


# ===== 7. 分桶报告（低证据区间是老师最关心的那一档）=====

def bucket_report(probs, labels, score, n_buckets=10):
    """按 score 分桶后逐桶报告准确率/ECE——score 用 vacuity 即可得到"证据水平"分桶。"""
    probs, labels, confidence, _, correct = prepare(probs, labels)
    score = np.asarray(score, dtype=np.float64)
    edges = np.quantile(score, np.linspace(0.0, 1.0, n_buckets + 1))
    rows = []
    for i in range(n_buckets):
        lower = edges[i]
        upper = edges[i + 1]
        mask = (score >= lower) & (score <= upper) if i == n_buckets - 1 \
            else (score >= lower) & (score < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append({
            "bucket": i,
            "score_range": (float(lower), float(upper)),
            "count": count,
            "accuracy": float(correct[mask].mean()),
            "ece": ece_equal_width(probs[mask], labels[mask]),
            "mean_confidence": float(confidence[mask].mean()),
        })
    return rows


# ===== 8. 自检 =====

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # --- 8.1 校准：构造"置信度 = 准确率"的数据，ECE 应接近 0 ---
    n = 4000
    picked_confidence = rng.choice([0.6, 0.8, 0.9], size=n)
    labels = np.zeros(n, dtype=np.int64)
    probs = np.full((n, 5), 0.0)
    for i in range(n):
        rest = (1.0 - picked_confidence[i]) / 4.0
        is_correct = rng.random() < picked_confidence[i]
        if is_correct:
            probs[i, 0] = picked_confidence[i]      # 置信度落在真实类上
        else:
            probs[i, 1] = picked_confidence[i]      # 置信度落在错误类上
            probs[i, 0] = rest
        for k in range(2, 5):
            probs[i, k] = rest
    print("[自检] 已校准数据 ECE(等宽) =", round(ece_equal_width(probs, labels), 4))
    print("[自检] 已校准数据 ECE(等质量) =", round(ece_adaptive(probs, labels), 4))

    # --- 8.2 过度自信：置信度 0.99 但只有 60% 正确，ECE 应接近 0.39 ---
    n = 2000
    over_probs = np.full((n, 5), 0.01 / 4.0)
    over_probs[:, 0] = 0.99
    over_labels = (rng.random(n) > 0.6).astype(np.int64)
    print("[自检] 过度自信 ECE =", round(ece_equal_width(over_probs, over_labels), 4),
          "（期望≈0.39）")

    # --- 8.3 AUROC：完美分离应 = 1.0，随机应 ≈ 0.5 ---
    perfect_scores = np.concatenate([rng.uniform(1.0, 2.0, 500), rng.uniform(0.0, 1.0, 500)])
    positive = np.concatenate([np.ones(500), np.zeros(500)]).astype(bool)
    print("[自检] 完全不重叠 AUROC =", round(auroc(perfect_scores, positive), 4), "（期望=1.0）")
    overlapping = np.concatenate([rng.normal(2.0, 1.0, 500), rng.normal(0.0, 1.0, 500)])
    print("[自检] 重叠分布 AUROC =", round(auroc(overlapping, positive), 4), "（期望≈0.92）")
    random_scores = rng.normal(0.0, 1.0, 1000)
    print("[自检] 随机打分 AUROC =", round(auroc(random_scores, positive), 4), "（期望≈0.5）")

    # --- 8.4 AURC：按置信度排序有效时应显著低于随机 ---
    n = 3000
    good_probs = np.zeros((n, 5))
    good_confidence = rng.uniform(0.5, 0.75, size=n)     # 限制区间，使覆盖率有信息量
    good_probs[:, 0] = good_confidence
    good_probs[:, 1] = 1.0 - good_confidence
    # 预测类固定为 0，故 label=0 表示答对；答对概率 = 0.5 + 0.45 * 置信度（随置信度上升）
    correct_probability = 0.5 + 0.45 * good_confidence
    good_labels = (rng.random(n) > correct_probability).astype(np.int64)
    print("[自检] 置信度有效 AURC =", round(aurc(good_probs, good_labels), 4))
    print("[自检] 达到 0.80 精度所需覆盖率 =",
          round(coverage_for_accuracy(good_probs, good_labels, 0.80), 4),
          "（满覆盖精度约 0.73–0.84，故该值应落在 (0,1) 内）")

    # --- 8.5 温度缩放应降低 NLL ---
    logit_labels = rng.integers(0, 5, size=2000)
    one_hot = np.eye(5)[logit_labels]
    # 有信息的 logits：真实类略高 + 噪声 → 最优温度应落在网格内部
    logits = one_hot * 1.2 + rng.normal(0.0, 1.0, size=(2000, 5))
    temperature, nll_after = fit_temperature(logits, logit_labels)
    nll_before = nll_score(softmax(logits, 1.0), logit_labels)
    print("[自检] 温度 =", round(temperature, 3),
          "NLL", round(nll_before, 4), "→", round(nll_after, 4), "（应下降）")

    # --- 8.6 证据读出：α 全为 1 → vacuity = 1；α 很大 → vacuity → 0 ---
    alpha_low = np.ones((1, 10))
    alpha_high = np.full((1, 10), 100.0)
    print("[自检] vacuity(α=1) =", round(float(dirichlet_vacuity(alpha_low)[0]), 4),
          "vacuity(α=100) =", round(float(dirichlet_vacuity(alpha_high)[0]), 4))
    # 解析可验证的构造：prior=0 时，两类等 belief → 1.0；单类集中 → 0
    diss_two = dirichlet_dissonance(np.array([[100.0, 100.0]]), prior=0.0)
    diss_one = dirichlet_dissonance(np.array([[100.0, 1.0]]), prior=0.0)
    print("[自检] dissonance（prior=0）：两类各半 =", round(float(diss_two[0]), 4),
          "（期望=1）  单类集中 =", round(float(diss_one[0]), 4), "（期望≈0）")
    # 单位先验会稀释 belief → 同一冲突下数值更小，跨方法比较时不可混用
    diss_unit = dirichlet_dissonance(np.array([[100.0, 100.0] + [1.0] * 8]))
    print("[自检] dissonance（prior=1，会被先验质量稀释）=", round(float(diss_unit[0]), 4))
    alpha_flat = np.full((1, 10), 5.0)
    alpha_spiky = np.array([[50.0] + [0.1] * 9])
    print("[自检] conflict(平) =", round(float(dirichlet_conflict(alpha_flat)[0]), 4),
          "conflict(尖) =", round(float(dirichlet_conflict(alpha_spiky)[0]), 4))

    # --- 8.7 分桶：低证据桶的准确率应低于高证据桶 ---
    alpha = np.concatenate([np.full((500, 5), 1.0), np.full((500, 5), 20.0)])
    bucket_probs = dirichlet_mean(alpha)
    bucket_labels = (rng.random(1000) > dirichlet_vacuity(alpha) / 2.0).astype(np.int64)
    rows = bucket_report(bucket_probs, bucket_labels, dirichlet_vacuity(alpha), n_buckets=5)
    print("[自检] 分桶准确率（低→高证据） =",
          [round(r["accuracy"], 3) for r in rows])