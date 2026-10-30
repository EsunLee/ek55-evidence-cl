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
from llm_client import ask as llm_ask
from llm_prompts import build_prompt, load_class_names, parse_response, ranking_to_support
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


def collect_llm_supports(args, mask, n_classes, probs=None):
    """对选中的样本调用真 LLM，返回 (support [N,K], confidence [N])。

    只对 mask=True 的样本花钱；其余行留空（后续只用到 mask 行）。
    """
    feats = np.load(args.features, allow_pickle=True)
    meta = json.loads(str(feats["meta"]))
    class_ids = meta["classes"]
    verb_names = load_class_names(args.verb_names)
    noun_names = load_class_names(args.noun_names)
    candidate_names = [verb_names.get(int(cid), f"verb{cid}") for cid in class_ids]

    past_verb = feats["past_verb_test"]
    past_noun = feats["past_noun_test"]
    support = np.zeros((len(mask), n_classes), dtype=np.float64)
    confidence = np.ones(len(mask), dtype=np.float64)

    if args.llm_ask_mode == "rerank" and probs is None:
        raise SystemExit("[错误] rerank 模式需要模型概率")

    indices = np.flatnonzero(mask)
    if args.llm_limit > 0:
        indices = indices[:args.llm_limit]
    queried = np.zeros(len(mask), dtype=bool)
    queried[indices] = True
    print(f"[LLM] 将调用 {len(indices)} 条（缓存目录 {args.llm_cache}）", flush=True)
    import time as _time
    started = _time.time()
    for done, index in enumerate(indices, start=1):
        if args.llm_ask_mode == "rerank":
            # 只把模型 top-k 交给 LLM 仲裁：支持度也只允许落在这些类上
            top_k = min(args.llm_rerank_topk, n_classes)
            local = np.argsort(-probs[index])[:top_k]
            local_names = [candidate_names[int(c)] for c in local]
            prompt = build_prompt(past_verb[index], past_noun[index], local_names,
                                  verb_names, noun_names)
        else:
            local = np.arange(n_classes)
            local_names = candidate_names
            prompt = build_prompt(past_verb[index], past_noun[index], candidate_names,
                                  verb_names, noun_names)
        call_started = _time.time()
        text = llm_ask(prompt, cache_dir=args.llm_cache)
        elapsed = _time.time() - call_started
        ranking, conf = parse_response(text)
        if args.llm_ask_mode == "rerank":
            local_support = ranking_to_support(ranking, local_names, conf)
            support[index] = 0.0
            support[index, local] = local_support
        else:
            support[index] = ranking_to_support(ranking, candidate_names, conf)
        confidence[index] = 0.7 if conf is None else float(np.clip(conf, 0.05, 1.0))
        print(f"[LLM] {done}/{len(indices)} 本条 {elapsed:.1f}s  "
              f"累计 {_time.time() - started:.0f}s", flush=True)
    return support, confidence, queried


def fit_logistic(features, labels, steps=400, lr=0.2, l2=1e-3):
    """小逻辑回归（numpy + Adam）。输入已标准化。

    注意：全部用逐元素求和而非 matmul（BLAS 在本数据上会报浮点告警），
    且 bias 用显式局部变量更新（早先版本对 float 用 -= 只是重新绑定，bias 从未被训练）。
    """
    n_samples, n_features = features.shape
    weights = np.zeros(n_features)
    bias = 0.0
    m_w, v_w = np.zeros(n_features), np.zeros(n_features)
    m_b = v_b = 0.0
    beta1, beta2 = 0.9, 0.999
    for step in range(1, steps + 1):
        logits = (features * weights[None, :]).sum(axis=1) + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        residual = (probs - labels) / n_samples
        grad_w = (features * residual[:, None]).sum(axis=0) + l2 * weights
        grad_b = float(residual.sum())

        m_w = beta1 * m_w + (1 - beta1) * grad_w
        v_w = beta2 * v_w + (1 - beta2) * grad_w * grad_w
        step_w = lr * (m_w / (1 - beta1 ** step)) / (np.sqrt(v_w / (1 - beta2 ** step)) + 1e-8)
        weights = np.clip(np.nan_to_num(weights - step_w), -50.0, 50.0)

        m_b = beta1 * m_b + (1 - beta1) * grad_b
        v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
        step_b = lr * (m_b / (1 - beta1 ** step)) / (np.sqrt(v_b / (1 - beta2 ** step)) + 1e-8)
        bias = float(np.clip(np.nan_to_num(bias - step_b), -50.0, 50.0))
    return weights, bias


def ridge_fit(features, target, l2=1.0):
    """闭式岭回归：w = (XᵀX + λI)⁻¹ Xᵀy。用于从特征预测"这条样本有多稀有"。"""
    n_samples, n_features = features.shape
    gram = features.T @ features + l2 * np.eye(n_features)
    return np.linalg.solve(gram, features.T @ target)


def prototype_distance_rarity(train_x, train_labels, test_x, n_classes, top_k=3):
    """原型距离：到最近类原型的距离越大 → 越可能是没见过的稀有类（经典 OOD 信号）。

    比线性岭回归更贴合"稀有 = 偏离"的结构：线性模型只能沿全局方向加权，
    而原型距离是**逐类的局部**判据。
    """
    prototypes = np.zeros((n_classes, train_x.shape[1]), dtype=np.float64)
    for class_id in range(n_classes):
        members = train_x[train_labels == class_id]
        if len(members) > 0:
            prototypes[class_id] = members.mean(axis=0)
    # 分块算距离，避免 (N_test, K, D) 爆内存
    scores = np.empty(len(test_x), dtype=np.float64)
    chunk = 256
    for start in range(0, len(test_x), chunk):
        block = test_x[start:start + chunk]
        distances = ((block[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
        scores[start:start + chunk] = np.partition(distances, top_k - 1, axis=1)[:, :top_k].mean(axis=1)
    return scores


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
        raw = (features[test_index] * weights[None, :]).sum(axis=1) + bias
        scores[test_index] = 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))
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
    elif trigger == "knn_rarity":                 # 原型距离（可部署）
        mask[np.argsort(-scores["knn_rarity"])[:count]] = True
    elif trigger == "learned_rarity":             # 特征预测的稀有度（可部署）
        mask[np.argsort(-scores["learned_rarity"])[:count]] = True
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
    strength_vector = np.asarray(strength, dtype=np.float64)
    if strength_vector.ndim == 0:
        strength_vector = np.full(len(alpha), float(strength_vector))
    strength_column = strength_vector[:, None]          # 支持逐样本强度（自适应修正力度）
    if mode == "redistribute":
        centered = support - support.mean(axis=1, keepdims=True)
        delta = strength_column * total * centered
    elif mode == "additive":
        delta = strength_column * total * support
    else:
        raise ValueError(mode)
    delta = delta * mask[:, None]                    # 只对调用样本生效
    updated = alpha + delta
    # ⚠️ Dirichlet 参数必须 > 0：零均值先验在高强度下会把未入选的类压成负数，
    # 导致概率含负值、置信度 >1，ECE/conflict 全部失真（曾假报 ECE=0.0000）。
    # 这里裁剪到 ≥1.0（合法且温和），并把裁剪量如实记录。
    floor = 1.0
    clipped = np.maximum(updated, floor)
    return clipped, delta


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
    parser.add_argument("--preset", default=None, choices=["verb", "group"],
                        help="快捷预设：verb=EK55 动词级；group=25 组（自动填 --alpha/--features）")
    parser.add_argument("--alpha", default=None, help="exp1 --export-alpha 的 npz")
    parser.add_argument("--trigger", default="vacuity",
                        choices=["vacuity", "conflict", "confidence", "random", "oracle",
                                 "expected_rarity", "rarity_oracle", "learned", "learned_rarity", "knn_rarity"])
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--stub-accuracy", type=float, default=0.7, help="模拟 LLM 的命中率")
    parser.add_argument("--stub-temperature", type=float, default=3.0, help="模拟 LLM 的置信程度")
    parser.add_argument("--feature-readout", default="pooled", choices=["pooled", "lastk"],
                        help="稀有度预测器用的特征：pooled=已池化 x_*（快）；lastk=读 x_seq 取末 k 段（慢，850MB）")
    parser.add_argument("--features", type=str, default=None,
                        help="stub-mode=class-frequency 时必需：含 y_train 的 npz（用于统计类频）")
    parser.add_argument("--stub-mode", default="uniform", choices=["uniform", "class-frequency"],
                        help="uniform=每条样本同样概率；class-frequency=按类频（稀有类更易帮上，不循环）")
    parser.add_argument("--skill-rare", type=float, default=0.9)
    parser.add_argument("--skill-common", type=float, default=0.3)
    parser.add_argument("--rare-frac", type=float, default=0.3, help="把最稀有的这一比例类视为'稀有'")
    parser.add_argument("--llm-mode", default="stub", choices=["stub", "real"],
                        help="stub=模拟器（默认）；real=真调 LLM（需 LLM_API_KEY）")
    parser.add_argument("--verb-names", default="/data/datasets/EPIC-KITCHENS/annotations/EPIC_verb_classes.csv")
    parser.add_argument("--noun-names", default="/data/datasets/EPIC-KITCHENS/annotations/EPIC_noun_classes.csv")
    parser.add_argument("--llm-cache", default="/data/ldy/ek55_assets/llm_cache")
    parser.add_argument("--llm-ask-mode", default="full", choices=["full", "rerank"],
                        help="full=在全部候选里问；rerank=只在模型 top-k 里仲裁（任务更小，更不容易带偏）")
    parser.add_argument("--llm-rerank-topk", type=int, default=3)
    parser.add_argument("--llm-limit", type=int, default=0, help="最多调用多少条（0=全部选中的）")
    parser.add_argument("--llm-confidence-weight", action="store_true",
                        help="用 LLM 的 confidence 调节注入强度（即'证据增加量 c'这一路）")
    parser.add_argument("--prior-mode", default="redistribute", choices=["redistribute", "additive"])
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--strength-mode", default="fixed",
                        choices=["fixed", "adaptive-maxprob", "adaptive-vacuity"],
                        help="修正力度：fixed=统一；adaptive-* = 按该样本的置信/证据调节（均值对齐）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    presets = {
        "verb": {"alpha": "/data/ldy/ek55_assets/alpha_verb.npz",
                 "features": "/data/ldy/ek55_assets/ek55_verb_full.npz"},
        "group": {"alpha": "/data/ldy/ek55_assets/alpha_group25.npz",
                  "features": "/data/ldy/ek55_assets/ek55_group25.npz"},
    }
    if args.preset:
        for key, value in presets[args.preset].items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    if not args.alpha:
        raise SystemExit("[错误] 需要 --alpha，或用 --preset verb / --preset group")

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
    # 只在触发/诊断真正需要时才计算（岭回归 1024 维 + 原型距离要 1–2 分钟）
    needs_rarity = args.trigger in {"expected_rarity", "rarity_oracle",
                                    "learned_rarity", "knn_rarity"}
    if args.features and needs_rarity:
        train_labels = np.load(args.features, allow_pickle=True)["y_train"]
        counts_all = np.bincount(train_labels, minlength=n_classes).astype(np.float64)
        order_all = np.argsort(counts_all)
        rank_all = np.empty(n_classes, dtype=np.float64)
        rank_all[order_all] = np.linspace(0.0, 1.0, n_classes)
        rarity_class = 1.0 - rank_all                       # 越罕见越接近 1
        probs_now = before["probs"]
        scores["expected_rarity"] = (probs_now * rarity_class[None, :]).sum(axis=1)  # E[rarity]：只用概率+类频，不用标签
        scores["true_rarity"] = rarity_class[labels]          # 上界参照
        # 可部署版：只用**训练集**标签训练"稀有度预测器"，在测试集上打分
        # 可部署版：只用**训练集**标签训练"稀有度预测器"，在测试集上打分
        # （NpzFile 不是 dict，必须按 .files 判断字段）
        feats = np.load(args.features, allow_pickle=True)

        def readout(container, split, last_k=5):
            key = f"x_seq_{split}"
            if args.feature_readout == "lastk" and key in container.files:
                print(f"[稀有度] 读取 {key}（较大，约数秒）…", flush=True)
                sequence = container[key]
                k = min(last_k, sequence.shape[1])
                return np.asarray(sequence[:, sequence.shape[1] - k:, :].mean(axis=1), dtype=np.float64)
            return np.asarray(container[f"x_{split}"], dtype=np.float64)

        print("[稀有度] 构建特征…", flush=True)
        train_x = readout(feats, "train")
        test_x = readout(feats, "test")
        mean_x = train_x.mean(axis=0, keepdims=True)
        std_x = train_x.std(axis=0, keepdims=True) + 1e-6
        print("[稀有度] 岭回归求解（1024 维，数秒）…", flush=True)
        weights_rarity = ridge_fit((train_x - mean_x) / std_x, rarity_class[feats["y_train"]])
        scores["learned_rarity"] = ((test_x - mean_x) / std_x) @ weights_rarity
        print("[稀有度] 原型距离…", flush=True)
        scores["knn_rarity"] = prototype_distance_rarity(
            (train_x - mean_x) / std_x, feats["y_train"], (test_x - mean_x) / std_x, n_classes)
        print("[稀有度] 完成", flush=True)
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

    llm_confidence = None
    llm_queried = None
    if args.llm_mode == "real":
        if "past_verb_test" not in np.load(args.features, allow_pickle=True).files:
            raise SystemExit("[错误] 需要 --with-text 重建的 npz（含 past_verb_test）")
        llm_support, llm_confidence, llm_queried = collect_llm_supports(
            args, mask, n_classes, probs=before["probs"])
        support = np.where(mask[:, None], llm_support, support)
    # 逐样本修正力度（均值对齐，保证与 fixed 的"总预算"可比）
    if args.strength_mode == "fixed":
        strength_vector = np.full(len(labels), args.strength)
    elif args.strength_mode == "adaptive-maxprob":
        weight = 1.0 - scores["maxprob"]                 # 越不自信 → 修得越狠
        strength_vector = args.strength * weight / max(weight.mean(), EPS)
    else:                                                # adaptive-vacuity
        weight = scores["vacuity"]                       # 越空虚 → 修得越狠
        strength_vector = args.strength * weight / max(weight.mean(), EPS)
    if args.llm_mode == "real" and args.llm_confidence_weight and llm_confidence is not None:
        strength_vector = strength_vector * np.where(mask, llm_confidence, 1.0)
    strength_vector = strength_vector * mask             # 未调用样本强度为 0
    alpha1, delta = inject_prior(alpha0, support, mask, args.prior_mode, strength_vector)
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

    # 诊断（真 LLM 模式）：LLM 的自我置信是否对应它实际的帮助？
    # 这是"证据作为度量"能否成立的核心检验——若 confidence 高的样本并没有更常被修对，
    # 那么用 LLM 的置信去驱动 vacuity 注入就是无依据的。
    if args.llm_mode == "real" and llm_confidence is not None:
        queried_mask = llm_queried if llm_queried is not None else mask     # 只统计真正调用过 LLM 的样本
        pred_after_now = after["probs"].argmax(axis=1)
        helped = queried_mask & (pred_before != labels) & (pred_after_now == labels)
        hurt = queried_mask & (pred_before == labels) & (pred_after_now != labels)
        called = int(queried_mask.sum())
        if called > 5:
            correlation = float(np.corrcoef(llm_confidence[queried_mask], helped[queried_mask].astype(np.float64))[0, 1])
            mean_help = float(llm_confidence[helped].mean()) if helped.any() else float("nan")
            mean_hurt = float(llm_confidence[hurt].mean()) if hurt.any() else float("nan")
            mean_all = float(llm_confidence[queried_mask].mean())
            print("\n  [诊断·LLM 置信] 调用 %d 条｜自评置信均值 %.3f" % (called, mean_all))
            print(f"         帮上 {int(helped.sum())} 条（均值 %.3f）｜帮倒忙 {int(hurt.sum())} 条（均值 %.3f）"
                  % (mean_help, mean_hurt))
            print(f"         置信 ↔ 帮上 的相关系数 {correlation:+.4f}"
                  f"（>0.15 才算自评可信）")

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
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        print(f"\n[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
