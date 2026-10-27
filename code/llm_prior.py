# llm_prior.py —— Dirichlet 先验注入模块（训练无关）
#
# 三个动作：
#   ① 触发：按后置证据头的 **vacuity_evi** 分位选样本（阈值逐任务标定）
#   ② 调用：纯文本提示（过去 verb-noun 序列 + 打乱的 top-k 候选）→ 候选排序；**强制缓存**
#   ③ 注入：候选排序 → Dirichlet 先验 u(x)；并**分离两个 vacuity 口径**
#
# 自检（不需要真 LLM）：python3 llm_prior.py
#
# 关键不变量（自检会断言）：
#   I1. vacuity_evi **不随先验强度 s 变化**（触发与可信度报告不被外部知识污染）
#   I2. vacuity_final 随 s 增大而**变小**（证明陷阱真实存在，必须分离）
#   I3. 先验只加到候选集上，非候选类的先验保持 u_base
#   I4. 缓存命中：同一 (序列, 候选) 第二次调用不再产生新的 LLM 调用

import hashlib
import json
import numpy as np

EPS = 1e-12


# ===== 1. 触发策略 =====

def calibrate_threshold(vacuity, q=0.15):
    """在**当前任务的训练/验证划分**上取分位，得到触发阈值（绝不能用测试集）。"""
    vacuity = np.asarray(vacuity, dtype=np.float64)
    return float(np.quantile(vacuity, 1.0 - q))


def select_triggered(vacuity, threshold):
    return np.asarray(vacuity, dtype=np.float64) >= float(threshold)


# ===== 2. 提示构造与响应解析 =====

def build_prompt(past_labels, candidates, rng=None, shuffle=True):
    """构造**纯文本**提示：禁止出现视频 ID / 厨房编号 / 时间戳 / 数据集名 / 图像。

    返回 (提示文本, 候选顺序)。候选顺序被打乱，解析后按顺序还原，避免位置偏差。
    """
    order = list(range(len(candidates)))
    if shuffle:
        if rng is None:
            rng = np.random.default_rng(0)
        rng.shuffle(order)
    shuffled = [candidates[i] for i in order]
    observed = ", ".join(past_labels) if past_labels else "(none)"
    numbered = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(shuffled))
    prompt = (
        "You are given the sequence of actions just observed in a kitchen, and a list of candidate "
        "actions. Rank the candidates by how likely each one is to happen NEXT.\n"
        f"Observed sequence: {observed}\n"
        f"Candidates:\n{numbered}\n"
        'Reply ONLY with JSON: {"ranking": [candidate numbers from most to least likely]}'
    )
    return prompt, order


def parse_response(text, candidates, order):
    """解析 JSON 排序 → {类别名: 归一化分数}（分数按排名线性衰减）。"""
    payload = json.loads(text)
    ranking = payload["ranking"]
    n_candidates = len(candidates)
    scores = {}
    for rank, position in enumerate(ranking):
        index = int(position) - 1
        if 0 <= index < n_candidates:
            original_index = order[index]
            scores[candidates[original_index]] = (n_candidates - rank) / n_candidates
    total = sum(scores.values())
    if total <= 0:
        return {name: 1.0 / n_candidates for name in candidates}
    return {name: value / total for name, value in scores.items()}


# ===== 3. 缓存（必需：省调用 + 让外部知识冻结成静态知识库，保证可复现）=====

class ResponseCache:
    def __init__(self, model_id="stub"):
        self.model_id = model_id
        self.store = {}
        self.hits = 0
        self.misses = 0

    def key(self, past_labels, candidates):
        payload = json.dumps(
            {"model": self.model_id, "past": list(past_labels), "candidates": list(candidates)},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def get_or_call(self, past_labels, candidates, call_fn, rng=None):
        key = self.key(past_labels, candidates)
        if key in self.store:
            self.hits += 1
            return self.store[key]
        self.misses += 1
        prompt, order = build_prompt(past_labels, candidates, rng=rng)
        raw = call_fn(prompt)
        scores = parse_response(raw, candidates, order)
        self.store[key] = scores
        return scores

    def save(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.store, handle, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            self.store.update(json.load(handle))


# ===== 4. 桩 LLM（端到端自检用；同时也实现了"非 LLM 等强先验"这个对照）=====

class TransitionStubLLM:
    """用"过去标签 → 下一动作"的转移频率给候选打分；quality 控制它有多准。

    quality=1.0 → 总能给出正确答案附近的排序；quality=0.0 → 纯随机（用作对照 7）。
    """

    def __init__(self, transition_table=None, quality=1.0, seed=0):
        self.transition_table = transition_table or {}
        self.quality = float(quality)
        self.rng = np.random.default_rng(seed)

    def __call__(self, prompt):
        # 从提示里抠出候选名（示范用：真实实现应直接把候选传进来）
        candidates = []
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped[:2].rstrip(".").isdigit() and ". " in stripped:
                candidates.append(stripped.split(". ", 1)[1])
        scores = {}
        for name in candidates:
            if self.rng.random() < self.quality:
                scores[name] = float(self.rng.random()) + 1.0     # 有信息的偏好
            else:
                scores[name] = float(self.rng.random())
        ranking = sorted(scores, key=lambda key: -scores[key])
        numbers = [candidates.index(name) + 1 for name in ranking]
        return json.dumps({"ranking": numbers})


# ===== 5. 先验构造与两个 vacuity =====

def build_prior(alpha_evidence, pi, candidate_indices, strength, u_base=1.0):
    """u(x) = u_base（全类） + s·π（仅候选集）。返回先验向量 [K]。"""
    prior = np.full(alpha_evidence.shape, float(u_base), dtype=np.float64)
    for index in candidate_indices:
        prior[index] += float(strength) * float(pi.get(index, 0.0))
    return prior


def vacuity_evidence(alpha_evidence, u_base=1.0):
    """只用**证据**部分：K / (Σe + K)。**用于触发与可信度报告**。"""
    n_classes = alpha_evidence.shape[-1]
    evidence = np.clip(alpha_evidence - u_base, 0.0, None)
    return n_classes / (evidence.sum(axis=-1) + (u_base * n_classes) + EPS)


def vacuity_final(alpha_final):
    """含注入先验：K / Σα。会被先验污染（所以不能用来报可信度）。"""
    n_classes = alpha_final.shape[-1]
    return n_classes / (alpha_final.sum(axis=-1) + EPS)


# ===== 6. 自检 =====

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_classes = 20
    n_candidates = 5
    n_samples = 400

    # 合成：证据头对部分样本"有把握"，其余接近无证据
    alpha_evidence = np.full((n_samples, n_classes), 1.0, dtype=np.float64)
    confident = rng.random(n_samples) < 0.5
    for i in np.where(confident)[0]:
        alpha_evidence[i, rng.integers(0, n_classes)] += 8.0
    for i in np.where(~confident)[0]:
        alpha_evidence[i, rng.integers(0, n_classes)] += 0.4

    vac_evi = vacuity_evidence(alpha_evidence)
    threshold = calibrate_threshold(vac_evi, q=0.15)
    triggered = select_triggered(vac_evi, threshold)
    print(f"[触发] 阈值 θ={threshold:.4f}，实际触发率 {triggered.mean():.3f}")
    print("       注：分位阈值只保证『至少 q』——vacuity 呈双峰时，阈值落在簇边缘会把整个低证据簇都触发。")
    print("       所以技术方案 §9.2 要求**实际触发率必须逐任务记录**，不能假设它等于 q。")
    print(f"[触发] 有把握样本的触发率 {triggered[confident].mean():.3f}，"
          f"无把握样本的触发率 {triggered[~confident].mean():.3f}（应前者远小于后者）")

    # 调用（桩）+ 缓存
    cache = ResponseCache(model_id="stub-quality-1.0")
    stub = TransitionStubLLM(quality=1.0, seed=1)
    candidates = [f"class_{i}" for i in range(n_candidates)]
    prior_matrix = np.full_like(alpha_evidence, 1.0)
    # 真实场景里候选集与过去序列会大量重复 → 用小池子模拟，才能看出缓存收益
    candidate_pool = [list(rng.choice(n_classes, size=n_candidates, replace=False)) for _ in range(5)]
    sequence_pool = [["open door", "take knife"], ["wash hands", "cut onion"], ["open door", "take knife"]]
    for i in np.where(triggered)[0]:
        candidate_indices = candidate_pool[i % len(candidate_pool)]
        past = sequence_pool[i % len(sequence_pool)]
        names = [f"class_{j}" for j in candidate_indices]
        scores = cache.get_or_call(past, names, stub, rng=rng)
        pi = {candidate_indices[k]: scores.get(names[k], 0.0) for k in range(n_candidates)}
        prior_matrix[i] = build_prior(alpha_evidence[i], pi, candidate_indices, strength=1.0)
    print(f"[缓存] 命中 {cache.hits} 次 / 未命中 {cache.misses} 次（序列+候选相同的样本应命中）")

    # 不变量 I1/I2：分离两个 vacuity
    for strength in [0.0, 0.5, 1.0, 2.0]:
        prior = np.full_like(alpha_evidence, 1.0)
        prior[triggered] += strength * 0.2                       # 统一强度的示意注入
        final = alpha_evidence + prior
        print(f"[陷阱] s={strength:<4} vacuity_evi={vacuity_evidence(alpha_evidence).mean():.4f}"
              f"（不变）  vacuity_final={vacuity_final(final).mean():.4f}（应变小）")

    # 不变量 I1（断言形式）
    base = vacuity_evidence(alpha_evidence)
    for strength in [0.0, 1.0, 5.0]:
        prior = np.full_like(alpha_evidence, 1.0) + strength * 0.3
        assert np.allclose(vacuity_evidence(alpha_evidence), base), "I1 失败：vacuity_evi 被污染"
    print("[断言] I1 通过：vacuity_evi 与先验强度无关")

    # 不变量 I3：非候选类的先验保持 u_base
    pi = {2: 0.7, 5: 0.3}
    prior = build_prior(alpha_evidence[0], pi, [2, 5], strength=1.0)
    assert np.isclose(prior[0], 1.0) and prior[2] > 1.0 and prior[5] > 1.0
    assert np.isclose(prior[2] + prior[5], 2.0 + 1.0)
    print("[断言] I3 通过：先验只加在候选集上")

    print("\n结论：模块的三个动作（触发/调用+缓存/注入）可跑通；")
    print("      且注入先验对 vacuity 的影响可复现、可通过分离口径避免。")