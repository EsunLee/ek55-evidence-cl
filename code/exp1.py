# exp1.py —— 静态实验：在冻结特征上比较「线性头」与「证据头」的可信度指标
#
#   ① 基线：线性头 + 交叉熵（用 softmax 置信度）
#   ② 证据头：EDL（e = ReLU(FC)，u = exp(u_log)，α = e + u）+ L_rec + L_inf [+ L_dist]
# 输出：精度（控制变量）与 ECE / Brier / NLL / AUROC / AURC / 低证据桶。
#
# 运行：
#   python exp1.py --synthetic                            # 先验证流程（合成特征，无需数据）
#   python exp1.py --features data\ek55_past30.npz        # 真实特征（格式见 data/README.md）
#
# 代码风格遵循工作区约定：一个参数一行、显式中间变量、中文注释、# ===== 分节。
# 所有可信度指标一律来自 metrics.py，禁止在本文件重复实现。

# ===== 1. 依赖与命令行 =====

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from metrics import (
    accuracy_at_coverage,
    aurc,
    auroc,
    auprc,
    brier_score,
    bucket_report,
    coverage_for_accuracy,
    dirichlet_conflict,
    dirichlet_mean,
    dirichlet_vacuity,
    ece_adaptive,
    ece_equal_width,
    fit_temperature,
    fit_vector_scaling,
    apply_vector_scaling,
    nll_score,
    softmax,
)

try:
    from env_report import collect_env          # 记录 GPU/驱动/torch/CUDA 与代码版本
except ImportError:                             # 缺失时不能让实验脚本挂掉
    def collect_env():
        return {"note": "env_report 不可用（tools/ 未同步？）"}


def parse_args():
    parser = argparse.ArgumentParser(description="EK55 动作预测的可信度：证据头 vs 线性基线")
    parser.add_argument("--features", type=str, default=None, help="特征 .npz 路径（见 data/README.md）")
    parser.add_argument("--synthetic", action="store_true", help="用合成特征自检流程")
    parser.add_argument("--n-classes", type=int, default=352, help="合成模式的类别数（EK55 为 352）")
    parser.add_argument("--dim", type=int, default=1024, help="合成模式的特征维度")
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-inf", type=float, default=0.0,
                        help="L_inf（trigamma 信息平衡）权重；默认 0，先跑纯 EDL 再逐步加")
    parser.add_argument("--lambda-fit", type=float, default=0.1,
                        help="ce_evi 模式下纯拟合项的权重（λ1）")
    parser.add_argument("--lambda-var", type=float, default=0.0,
                        help="ce_evi 模式下方差项的权重（λ2）；0 表示关闭")
    parser.add_argument("--lambda-conc", type=float, default=0.0,
                        help="浓度正则权重：把 α0 拉回 K 附近（推荐 0.01–1.0；0=关闭）")
    parser.add_argument("--var-topk", type=int, default=0,
                        help="方差项只压前 k 个候选类（0=全部；建议 5–10，避免压平整个分布）")
    parser.add_argument("--soft-label-beta", type=float, default=1.0,
                        help="平滑预测标签 β（1=硬 one-hot；0.7 表示 0.7·onehot + 0.3·预测）")
    parser.add_argument("--evidence-param", type=str, default="mean", choices=["mean", "alpha"],
                        help="证据头参数化：mean=均值(线性 logits)+浓度，精度与基线同构（推荐）；alpha=原 α=ReLU(Wf)+u")
    parser.add_argument("--loss", type=str, default="ce_evi",
                        choices=["ce_evi", "edl", "fisher"],
                        help="ce_evi=CE 主导 + 可分离证据项（推荐）；edl/fisher 为旧实现，保留以便复现")
    parser.add_argument("--label-smoothing", type=float, default=0.01,
                        help="标签平滑；DUDE 用 0.7，但在 K=100+ 时会把目标压成近均匀")
    parser.add_argument("--select-by", type=str, default="nll", choices=["nll", "ece", "accuracy"],
                        help="早停选模型用的验证指标（EDL 的 val NLL 偏向欠自信模型，可改 accuracy/ece）")
    parser.add_argument("--ce-weight", type=float, default=0.0,
                        help="证据头损失里 CE 的权重（>0 时 EDL + w·CE，用来保住判别力）")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Fisher 形式的 σ；默认 1.0 即去掉 1/(2σ²) 的 50 倍放大")
    parser.add_argument("--lambda-dist", type=float, default=0.0, help="L_dist 权重（有未来特征时才可用）")
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--readout", type=str, default="mean", choices=["mean", "lastk", "attention"],
                        help="时序读出：mean=用已池化向量（旧行为）；lastk=最后 k 段均值；attention=注意力池化")
    parser.add_argument("--last-k", type=int, default=5, help="lastk 读出的 k")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=None)
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===== 2. 数据 =====

def load_features(path):
    """读取 .npz 特征文件；字段约定见 data/README.md。"""
    data = np.load(path, allow_pickle=True)
    bundle = {
        "x_train": data["x_train"].astype(np.float32),
        "y_train": data["y_train"].astype(np.int64),
        "x_val": data["x_val"].astype(np.float32),
        "y_val": data["y_val"].astype(np.int64),
        "x_test": data["x_test"].astype(np.float32),
        "y_test": data["y_test"].astype(np.int64),
        "seq_train": data["x_seq_train"].astype(np.float32) if "x_seq_train" in data else None,
        "seq_val": data["x_seq_val"].astype(np.float32) if "x_seq_val" in data else None,
        "seq_test": data["x_seq_test"].astype(np.float32) if "x_seq_test" in data else None,
        "x_future_train": data["x_future_train"].astype(np.float32) if "x_future_train" in data else None,
        "novel_mask_test": data["novel_mask_test"].astype(bool) if "novel_mask_test" in data else None,
    }
    return bundle


def make_synthetic(args):
    """合成可学的特征：每类一个均值向量 + 高斯噪声。

    目的只是验证流程与指标能跑通；合成数据上的胜负**不能**当作实验结论。
    """
    rng = np.random.default_rng(args.seed)
    centers = rng.normal(0.0, 1.0, size=(args.n_classes, args.dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    def sample(n):
        labels = rng.integers(0, args.n_classes, size=n)
        noise = rng.normal(0.0, 1.0, size=(n, args.dim)).astype(np.float32)
        features = centers[labels] * 3.0 + noise
        return features, labels.astype(np.int64)

    x_train, y_train = sample(args.n_train)
    x_val, y_val = sample(args.n_val)
    x_test, y_test = sample(args.n_test)
    novel_mask_test = np.zeros(len(y_test), dtype=bool)
    novel_mask_test[: len(y_test) // 10] = True       # 伪造 10% "新类"，用于自检分解指标
    return {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
        "x_future_train": None,
        "novel_mask_test": novel_mask_test,
    }


def to_tensor(array, device):
    return torch.from_numpy(np.asarray(array)).to(device)


# ===== 3. 模型（forward 一律返回 (features, 输出)）=====

class LinearHead(nn.Module):
    """基线：线性分类头 + 交叉熵。"""

    def __init__(self, dim, n_classes):
        super().__init__()
        self.classifier = nn.Linear(
            in_features=dim,
            out_features=n_classes,
        )

    def forward(self, features):
        logits = self.classifier(features)
        return features, logits


class EvidenceHead(nn.Module):
    """证据头：e = ReLU(FC(f))，u = exp(u_log)，α = e + u。"""

    def __init__(self, dim, n_classes):
        super().__init__()
        self.evidence_fc = nn.Linear(
            in_features=dim,
            out_features=n_classes,
        )
        # 先验参数化：u = 1 + softplus(u_log) **保证 u >= 1**
        # 为什么不能用 exp(u_log)：它没有下界，训练中学到的先验可能掉到 1 以下，
        # 使 α0 < K、vacuity = K/α0 > 1，失去"空虚度"的意义（实测踩过：mean_vacuity = 1.45）。
        self.u_log = nn.Parameter(torch.full((n_classes,), -4.0))   # softplus(-4)≈0.018 → u≈1.018

    def forward(self, features):
        evidence = F.relu(self.evidence_fc(features))
        prior = 1.0 + F.softplus(self.u_log)
        alpha = evidence + prior
        return features, alpha


class FutureEvidence(nn.Module):
    """未来分支的证据头：训练时可见未来片段，用它的 α_g 当 teacher（DUDE 的 L_dist）。

    只在 --lambda-dist > 0 且数据带 x_future_train 时启用；推理时**不存在**这一支。
    """

    def __init__(self, dim_future, n_classes):
        super().__init__()
        self.evidence_fc = nn.Linear(
            in_features=dim_future,
            out_features=n_classes,
        )
        self.u_log = nn.Parameter(torch.zeros(n_classes))

    def forward(self, future_features):
        evidence = F.relu(self.evidence_fc(future_features))
        prior = torch.exp(self.u_log)
        alpha = evidence + prior
        return alpha


# ===== 4. 损失 =====

def edl_bayes_risk_loss(alpha, targets, n_classes, label_smoothing=0.01):
    """EDL 标准 Bayes risk（Sensoy et al. 2018）：拟合项 + Dirichlet 方差项。

    为什么加上它：全量 verb 上原始实现（Fisher 加权 + smoothing=0.7 + 1/(2σ²)=50 倍）
    把目标压平了 —— accuracy 掉 3.5pp、ECE 翻倍、但排序 AUROC 反而更好，
    这是典型的「被训练成不敢表态」。标准形式先用温和的标签平滑（0.01）作对照。
    """
    strength = alpha.sum(dim=1, keepdim=True)
    probs = alpha / strength
    one_hot = F.one_hot(targets, num_classes=n_classes).float()
    smoothed = one_hot * (1.0 - label_smoothing) + label_smoothing / n_classes
    error_term = (smoothed - probs) ** 2
    variance_term = probs * (1.0 - probs) / (strength + 1.0)
    return (error_term + variance_term).sum(dim=1).mean()


def soft_targets(alpha, targets, n_classes, beta=1.0):
    """平滑预测标签：ŷ = β·onehot + (1−β)·p̂（p̂ 取当前预测并 detach）。

    对应 DUDE 的 "smoothed prediction modeling"。β=1 时退化为硬 one-hot。
    detach 是必须的：否则等价于给自己加一个自洽约束，会退化成常数解。
    """
    one_hot = nn.functional.one_hot(targets, num_classes=n_classes).float()
    if beta >= 1.0:
        return one_hot
    with torch.no_grad():
        predicted = alpha / alpha.sum(dim=1, keepdim=True)
    return beta * one_hot + (1.0 - beta) * predicted


def pure_fit_loss(alpha, targets, n_classes, sigma=1.0, soft_beta=1.0):
    """① 纯拟合项：Fisher 加权的平方误差，**不含方差项**。

    为什么必须与方差项分开：消融第 1 步（CE + L_fit）与第 2 步（+ L_var）要能独立测量。
    edl_bayes_risk_loss 把两者焊在一起，所以它不能当"纯拟合项"用。
    """
    strength = alpha.sum(dim=1, keepdim=True)
    probs = alpha / strength
    targets_soft = soft_targets(alpha, targets, n_classes, soft_beta)
    fisher_weight = torch.polygamma(1, alpha)          # ψ⁽¹⁾(α_i)
    squared_error = (targets_soft - probs) ** 2
    return (squared_error * fisher_weight).sum(dim=1).div(2.0 * sigma ** 2).mean()


def concentration_regularizer(alpha, n_classes):
    """把总浓度 α0 拉回 K 附近，使 vacuity = K/α0 落在 ~1 的量级。

    为什么必须有：步1 实测 L_fit 会把 α0 顶到极大（mean_vacuity 0.59 → 0.039），
    后果是 (a) 方差项 p(1−p)/(α0+1) 恒为 0（步2 因此完全无效）；
    (b) 模型过度自信，原始 ECE 反而比基线差（0.0620 vs 0.0513）。
    这个正则只作用在浓度上，不碰均值的相对大小，因此不影响判别方向。
    """
    strength = alpha.sum(dim=1, keepdim=True)
    log_strength = torch.log(strength.clamp_min(1e-6))
    return (log_strength - float(np.log(n_classes))).pow(2).mean()


def variance_penalty(alpha, topk=0):
    """② 纯方差项：Dirichlet 方差 p(1−p)/(α0+1) 的和，**不含拟合项**。

    topk>0 时只压"候选类"（按 α 排序取前 k）——对应 DUDE 的"只在候选上收紧分布"，
    避免把整个预测分布压平（那正是掉精度的主因）。
    """
    strength = alpha.sum(dim=1, keepdim=True)
    probs = alpha / strength
    variance = probs * (1.0 - probs) / (strength + 1.0)
    if topk and topk > 0:
        index = torch.topk(alpha, k=min(topk, alpha.shape[1]), dim=1).indices
        mask = torch.zeros_like(alpha).scatter_(1, index, 1.0)
        variance = variance * mask
    return variance.sum(dim=1).mean()


def fisher_weighted_reconstruction_loss(alpha, targets, n_classes, sigma=0.1, label_smoothing=0.7):
    """L_rec（DUDE 形式）：Fisher 信息加权的重构项。

    原文事实（2026-10-27 由翻译稿核实）：L_rec = (1/2σ²) Σ_i (ŷ_i - p̂_i)² ψ^(1)(α_i) + ...
    其中 ψ^(1) 为 trigamma，ŷ 为平滑标签。

    待核实：翻译稿把方差项截断成省略号，这里用 EDL 标准的 Dirichlet 方差项占位；
    以及 label_smoothing=0.7 的确切作用方向（是"真实类保留 1-0.7"还是别的写法）需回原文。
    """
    strength = alpha.sum(dim=1, keepdim=True)
    probs = alpha / strength
    one_hot = F.one_hot(targets, num_classes=n_classes).float()
    smoothed = one_hot * (1.0 - label_smoothing) + label_smoothing / n_classes
    trigamma_alpha = torch.polygamma(1, alpha)                 # ψ^(1)(α_i)：Fisher 信息
    weighted_error = ((smoothed - probs) ** 2 * trigamma_alpha).sum(dim=1, keepdim=True)
    variance_term = (probs * (1.0 - probs) / (strength + 1.0)).sum(dim=1, keepdim=True)
    return (weighted_error / (2.0 * sigma ** 2) + variance_term).mean()


def fisher_information_balance_loss(alpha):
    """L_inf（DUDE 原文形式）：基于 trigamma 的类间信息平衡项。

    原文事实（2026-10-27 由翻译稿核实）：
        L_inf = -(1/2) Σ_i ln ψ^(1)(α_i) - (1/2) ln[ 1 - ψ^(1)(α_0) / Σ_j ψ^(1)(α_j) ]
    第一项惩罚单类信息不足；第二项度量局部与全局信息失衡（某类占优时趋近 0，ln 急剧下降）。

    待核实：第二项里 ψ^(1)(α_0) 的聚合方式（是否为 α_0 = Σ_i α_i）需回原文核对。
    """
    trigamma_alpha = torch.polygamma(1, alpha)                 # ψ^(1)(α_i)
    first_term = -0.5 * torch.log(trigamma_alpha).sum(dim=1)
    alpha_0 = alpha.sum(dim=1, keepdim=True)
    trigamma_sum = trigamma_alpha.sum(dim=1, keepdim=True)
    ratio = torch.polygamma(1, alpha_0) / trigamma_sum
    second_term = -0.5 * torch.log(torch.clamp(1.0 - ratio, min=1e-6)).squeeze(dim=1)
    return (first_term + second_term).mean()


def kl_dirichlet(alpha_a, alpha_b):
    """KL(Dir(a) || Dir(b))，用于 L_dist（需要未来特征当 teacher）。"""
    sum_a = alpha_a.sum(dim=1, keepdim=True)
    sum_b = alpha_b.sum(dim=1, keepdim=True)
    log_beta_a = torch.lgamma(alpha_a).sum(dim=1, keepdim=True) - torch.lgamma(sum_a)
    log_beta_b = torch.lgamma(alpha_b).sum(dim=1, keepdim=True) - torch.lgamma(sum_b)
    digamma_diff = torch.digamma(alpha_a) - torch.digamma(sum_a)
    digamma_b_diff = torch.digamma(alpha_b) - torch.digamma(sum_b)
    return (log_beta_b - log_beta_a + ((alpha_a - alpha_b) * (digamma_diff - digamma_b_diff)).sum(dim=1, keepdim=True)).mean()


def validation_metric(kind, probs, labels):
    """早停用的验证指标。EDL 常因 val NLL 偏向欠自信模型，故允许按 ECE / accuracy 选。"""
    if kind == "accuracy":
        return -float((probs.argmax(axis=1) == labels).mean())     # 取负，仍是"越小越好"
    if kind == "ece":
        return float(ece_equal_width(probs, labels, 15))
    return float(nll_score(probs, labels))


# ===== 5. 训练与推理 =====

def iterate_batches(x, y, batch_size, device, shuffle, generator=None, extra=None):
    """按批产出 (特征, 标签[, 额外输入])；extra 用于预测任务的未来片段特征（L_dist）。"""
    n = len(y)
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        index = order[start:start + batch_size].numpy()
        batch = (to_tensor(x[index], device), to_tensor(y[index], device))
        if extra is not None:
            batch = batch + (to_tensor(extra[index], device),)
        yield batch


class EvidenceHeadMean(nn.Module):
    """均值 + 浓度 的解耦参数化：α = α0(x) · p(x)，其中 p = softmax(线性 logits)。

    为什么需要它：原 EvidenceHead 用 α = ReLU(Wf)+u，α 只能非负相加，
    表达不了任意 logit → 实测即使损失退化成纯 CE 也掉约 6pp（0.2066 vs 0.2665）。
    这里让均值走与线性基线**完全同构**的 logit 路径，因此
        log α − log Σα = log p
    精度由构造与基线一致；浓度 α0 单独学，用来给出 vacuity 与证据项的作用对象。
    """

    def __init__(self, dim, n_classes):
        super().__init__()
        self.logits = nn.Linear(dim, n_classes)          # 与 LinearHead 同构
        self.concentration = nn.Linear(dim, 1)           # α0 的来源

    def forward(self, features):
        logits = self.logits(features)
        probs = torch.softmax(logits, dim=1)
        # α0 = K · (1 + softplus(g)) → vacuity = K/α0 ≤ 1
        strength = self.logits.out_features * (1.0 + torch.nn.functional.softplus(self.concentration(features)))
        alpha = probs * strength
        return features, alpha


class LastKPool(nn.Module):
    """取最后 k 段做均值：最朴素的"重近轻远"读出，无参数。"""

    def __init__(self, k=5):
        super().__init__()
        self.k = k

    def forward(self, sequence):
        k = min(self.k, sequence.shape[1])
        return sequence[:, sequence.shape[1] - k:, :].mean(dim=1)


class AttentionPool(nn.Module):
    """注意力池化：[B,S,D] → [B,D]。

    参数很少（一个投影 + 一个查询向量），配合冻结骨干正好：只学"该看哪几段"。
    """

    def __init__(self, dim, hidden=128):
        super().__init__()
        self.project = nn.Linear(dim, hidden)
        self.query = nn.Parameter(torch.zeros(hidden))
        self.scale = hidden ** -0.5

    def forward(self, sequence):
        projected = torch.tanh(self.project(sequence))        # [B,S,H]
        scores = projected @ self.query * self.scale          # [B,S]
        weights = torch.softmax(scores, dim=1)
        pooled = (weights.unsqueeze(-1) * sequence).sum(dim=1)
        return pooled


class ReadoutWrapper(nn.Module):
    """先做时序读出，再交给原来的头；返回值形状与原头一致（features, logits/alpha）。"""

    def __init__(self, pool, head):
        super().__init__()
        self.pool = pool
        self.head = head

    def forward(self, sequence):
        pooled = self.pool(sequence)
        return self.head(pooled)


def wrap_readout(head, args, dim):
    """按 --readout 决定是否给头套一层时序读出。mean 时原样返回（保持旧行为完全不变）。"""
    if args.readout == "mean":
        return head
    if args.readout == "lastk":
        return ReadoutWrapper(LastKPool(args.last_k), head)
    return ReadoutWrapper(AttentionPool(dim), head)


def train_baseline(model, bundle, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_val_nll = float("inf")
    for epoch in range(args.epochs):
        model.train()
        for features, targets in iterate_batches(bundle["x_train"], bundle["y_train"], args.batch_size, device, True):
            optimizer.zero_grad()
            _, logits = model(features)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
        # 用验证集 NLL 早停（可信度任务里 NLL 比精度更适合做选择标准）
        val_probs = predict_baseline(model, bundle["x_val"], device)
        val_nll = validation_metric(args.select_by, val_probs, bundle["y_val"])
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_nll


def train_evidence(model, bundle, args, device):
    has_future = bundle["x_future_train"] is not None and args.lambda_dist > 0.0
    if args.lambda_dist > 0.0 and not has_future:
        print("[提示] 指定了 --lambda-dist 但没有 x_future_train，L_dist 将被跳过")
    future_model = None
    if has_future:
        future_dim = int(bundle["x_future_train"].shape[1])
        future_model = FutureEvidence(future_dim, args.n_classes).to(device)
        print(f"[提示] 启用 L_dist（未来 teacher，权重 {args.lambda_dist}），未来特征维度 {future_dim}")
    parameters = list(model.parameters())
    if future_model is not None:
        parameters += list(future_model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_val_nll = float("inf")
    for epoch in range(args.epochs):
        model.train()
        if future_model is not None:
            future_model.train()
        batches = iterate_batches(
            bundle["x_train"],
            bundle["y_train"],
            args.batch_size,
            device,
            True,
            extra=bundle["x_future_train"] if has_future else None,
        )
        for batch in batches:
            if has_future:
                features, targets, future_features = batch
            else:
                features, targets = batch
            optimizer.zero_grad()
            _, alpha = model(features)
            if args.loss == "ce_evi":
                # L = L_CE + λ1·L_fit + λ2·L_var + λ3·L_inf —— 四项独立可关
                # λ1=λ2=λ3=0 时严格退化为纯 CE（可用它验证没有改坏基线）
                log_probs = torch.log(alpha) - torch.log(alpha.sum(dim=1, keepdim=True))
                loss = F.nll_loss(log_probs, targets)
                if args.lambda_fit > 0.0:
                    loss = loss + args.lambda_fit * pure_fit_loss(
                        alpha, targets, args.n_classes,
                        sigma=args.sigma, soft_beta=args.soft_label_beta)
                if args.lambda_conc > 0.0:
                    loss = loss + args.lambda_conc * concentration_regularizer(alpha, args.n_classes)
                if args.lambda_var > 0.0:
                    loss = loss + args.lambda_var * variance_penalty(alpha, topk=args.var_topk)
                if args.lambda_inf > 0.0:
                    loss = loss + args.lambda_inf * fisher_information_balance_loss(alpha)
            # 旧实现：EDL Bayes risk 替代 CE（会掉精度，仅用于复现历史结果）
            elif args.loss == "edl":
                loss_rec = edl_bayes_risk_loss(alpha, targets, args.n_classes,
                                               label_smoothing=args.label_smoothing)
            else:
                loss_rec = fisher_weighted_reconstruction_loss(alpha, targets, args.n_classes,
                                                               sigma=args.sigma,
                                                               label_smoothing=args.label_smoothing)
            if args.loss != "ce_evi":
                loss_inf = fisher_information_balance_loss(alpha)
                loss = loss_rec + args.lambda_inf * loss_inf
            if args.ce_weight > 0.0:
                # 用 Dirichlet 均值算 CE：log p = log α − log Σα
                log_probs = torch.log(alpha) - torch.log(alpha.sum(dim=1, keepdim=True))
                loss = loss + args.ce_weight * F.nll_loss(log_probs, targets)
            if has_future:
                # KL 方向按原文写成 KL(Dir_g || Dir_p)，仍需回原文核对
                alpha_global = future_model(future_features)
                loss = loss + args.lambda_dist * kl_dirichlet(alpha_global, alpha)
            loss.backward()
            optimizer.step()
        val_alpha = predict_evidence(model, bundle["x_val"], device)
        val_probs = dirichlet_mean(val_alpha)
        val_nll = validation_metric(args.select_by, val_probs, bundle["y_val"])
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_nll


@torch.no_grad()
def predict_baseline(model, x, device):
    model.eval()
    logits = model(to_tensor(x, device))[1]
    return softmax(logits.detach().cpu().numpy(), 1.0)


@torch.no_grad()
def predict_evidence(model, x, device):
    model.eval()
    alpha = model(to_tensor(x, device))[1]
    return alpha.detach().cpu().numpy()


# ===== 6. 可信度评测（全部走 metrics.py）=====

def credibility_report(probs, labels, n_bins, confidence_score=None, evidence_alpha=None):
    """一份完整的可信度报告：校准 + 误分类检测 + 选择性预测 + 低证据桶。"""
    confidence = probs.max(axis=1) if confidence_score is None else np.asarray(confidence_score)
    correct = (probs.argmax(axis=1) == labels)
    report = {
        "accuracy": float(correct.mean()),
        "ece_equal_width": ece_equal_width(probs, labels, n_bins),
        "ece_adaptive": ece_adaptive(probs, labels, n_bins),
        "brier": brier_score(probs, labels),
        "nll": nll_score(probs, labels),
        "auroc_confidence": auroc(confidence, correct),
        "auprc_confidence": auprc(confidence, correct),
        "aurc": aurc(probs, labels),
        "accuracy_at_80pct_coverage": accuracy_at_coverage(probs, labels, 0.8),
    }
    if evidence_alpha is not None:
        vacuity = dirichlet_vacuity(evidence_alpha)
        conflict = dirichlet_conflict(evidence_alpha)
        report["auroc_vacuity"] = auroc(-vacuity, correct)
        report["auroc_conflict"] = auroc(conflict, ~correct)     # 冲突高 → 更容易错
        report["mean_vacuity"] = float(vacuity.mean())
        report["mean_conflict"] = float(conflict.mean())
    return report


def matched_comparison(baseline_probs, evidence_probs, labels, n_bins, min_samples=30):
    """同等精度比较：以基线精度为目标，各自保留最少样本达到该精度，再比选择性 ECE。

    可信度与精度是耦合的，因此不能只报整体 ECE，必须在同一精度水平上比较。
    """
    target_accuracy = float((baseline_probs.argmax(axis=1) == labels).mean())
    baseline_coverage = coverage_for_accuracy(baseline_probs, labels, target_accuracy, min_samples)
    evidence_coverage = coverage_for_accuracy(evidence_probs, labels, target_accuracy, min_samples)

    def selective_ece(probs, coverage):
        if not np.isfinite(coverage):
            return float("nan")
        keep = max(int(round(coverage * len(labels))), min_samples)
        order = np.argsort(-probs.max(axis=1), kind="mergesort")[:keep]
        return ece_equal_width(probs[order], labels[order], n_bins)

    return {
        "target_accuracy": target_accuracy,
        "baseline_coverage": baseline_coverage,
        "evidence_coverage": evidence_coverage,
        "baseline_selective_ece": selective_ece(baseline_probs, baseline_coverage),
        "evidence_selective_ece": selective_ece(evidence_probs, evidence_coverage),
    }


# ===== 7. 主流程 =====

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.features is None and not args.synthetic:
        print("请指定 --features <path.npz> 或 --synthetic")
        return 1
    bundle = make_synthetic(args) if args.synthetic else load_features(args.features)
    if not args.synthetic:
        observed_max = max(bundle["y_train"].max(), bundle["y_val"].max(), bundle["y_test"].max())
        args.n_classes = int(observed_max + 1)
    dim = int(bundle["x_train"].shape[1])
    if args.readout != "mean":
        for split_name in ("train", "val", "test"):
            sequence = bundle.get(f"seq_{split_name}")
            if sequence is None:
                raise SystemExit(f"[错误] --readout {args.readout} 需要 x_seq_{split_name}，当前 npz 缺该字段")
            bundle[f"x_{split_name}"] = sequence
        print(f"[读出] {args.readout}：输入形状 {tuple(bundle['x_train'].shape)}")
    print(f"[数据] 训练 {len(bundle['y_train'])} / 验证 {len(bundle['y_val'])} / 测试 {len(bundle['y_test'])}"
          f"，维度 {dim}，类别 {args.n_classes}，设备 {device}")

    # --- 7.1 训练两个头 ---
    baseline_model = wrap_readout(LinearHead(dim=dim, n_classes=args.n_classes), args, dim).to(device)
    started = time.time()
    baseline_model, baseline_val_nll = train_baseline(baseline_model, bundle, args, device)
    baseline_seconds = time.time() - started

    evidence_class = EvidenceHeadMean if args.evidence_param == "mean" else EvidenceHead
    evidence_model = wrap_readout(evidence_class(dim=dim, n_classes=args.n_classes), args, dim).to(device)
    started = time.time()
    evidence_model, evidence_val_nll = train_evidence(evidence_model, bundle, args, device)
    evidence_seconds = time.time() - started

    # --- 7.2 测试集预测 ---
    baseline_probs = predict_baseline(baseline_model, bundle["x_test"], device)
    evidence_alpha = predict_evidence(evidence_model, bundle["x_test"], device)
    evidence_probs = dirichlet_mean(evidence_alpha)
    labels = bundle["y_test"]

    baseline_report = credibility_report(baseline_probs, labels, args.n_bins)
    evidence_report = credibility_report(evidence_probs, labels, args.n_bins, evidence_alpha=evidence_alpha)

    # --- 7.3 事后校准对照（温度缩放）---
    baseline_val_probs = predict_baseline(baseline_model, bundle["x_val"], device)
    evidence_val_probs = dirichlet_mean(predict_evidence(evidence_model, bundle["x_val"], device))
    # 温度缩放是"事后校准"对照：每个方法各自在验证集上选温度，再在测试集上评 ECE
    baseline_temperature, _ = fit_temperature(np.log(np.clip(baseline_val_probs, 1e-12, None)), bundle["y_val"])
    evidence_temperature, _ = fit_temperature(np.log(np.clip(evidence_val_probs, 1e-12, None)), bundle["y_val"])
    baseline_report["temperature"] = float(baseline_temperature)
    evidence_report["temperature"] = float(evidence_temperature)
    # A2 对照补强：向量缩放（按类缩放+偏置，比温度缩放更强的事后基线）
    baseline_scale, baseline_bias = fit_vector_scaling(
        np.log(np.clip(baseline_val_probs, 1e-12, None)), bundle["y_val"])
    evidence_scale, evidence_bias = fit_vector_scaling(
        np.log(np.clip(evidence_val_probs, 1e-12, None)), bundle["y_val"])
    baseline_report["ece_after_vector_scaling"] = ece_equal_width(
        apply_vector_scaling(np.log(np.clip(baseline_probs, 1e-12, None)), baseline_scale, baseline_bias),
        labels, args.n_bins)
    evidence_report["ece_after_vector_scaling"] = ece_equal_width(
        apply_vector_scaling(np.log(np.clip(evidence_probs, 1e-12, None)), evidence_scale, evidence_bias),
        labels, args.n_bins)

    baseline_report["ece_after_temperature"] = ece_equal_width(
        softmax(np.log(np.clip(baseline_probs, 1e-12, None)), baseline_temperature), labels, args.n_bins)
    evidence_report["ece_after_temperature"] = ece_equal_width(
        softmax(np.log(np.clip(evidence_probs, 1e-12, None)), evidence_temperature), labels, args.n_bins)

    # --- 7.4 同等精度比较 + 低证据桶 ---
    matched = matched_comparison(baseline_probs, evidence_probs, labels, args.n_bins)
    baseline_buckets = bucket_report(baseline_probs, labels, 1.0 - baseline_probs.max(axis=1), n_buckets=10)
    evidence_buckets = bucket_report(evidence_probs, labels, dirichlet_vacuity(evidence_alpha), n_buckets=10)
    low_bucket_baseline = baseline_buckets[-1] if baseline_buckets else {}
    low_bucket_evidence = evidence_buckets[-1] if evidence_buckets else {}

    # --- 7.5 不确定性分解有效性（有 novel_mask 时才做）---
    decomposition = {}
    novel_mask = bundle.get("novel_mask_test")
    if novel_mask is not None and novel_mask.any():
        vacuity = dirichlet_vacuity(evidence_alpha)
        conflict = dirichlet_conflict(evidence_alpha)
        decomposition = {
            "auroc_vacuity_novel_class": auroc(vacuity, novel_mask),
            "auroc_conflict_novel_class": auroc(conflict, novel_mask),
            "note": "期望：vacuity 对新类显著 > 0.5；若两个信号给出相同排序则分解无效",
        }

    # --- 7.6 打印对照表 ---
    keys = ["accuracy", "ece_equal_width", "ece_adaptive", "brier", "nll",
            "auroc_confidence", "aurc", "accuracy_at_80pct_coverage",
            "ece_after_temperature", "ece_after_vector_scaling"]
    print("\n================ 可信度对照（测试集）================")
    print(f"{'指标':<28}{'线性基线':>14}{'证据头':>14}")
    for key in keys:
        left = baseline_report.get(key, float("nan"))
        right = evidence_report.get(key, float("nan"))
        print(f"{key:<28}{left:>14.4f}{right:>14.4f}")
    for key in ["auroc_vacuity", "auroc_conflict", "mean_vacuity", "mean_conflict"]:
        if key in evidence_report:
            print(f"{key:<28}{'—':>14}{evidence_report[key]:>14.4f}")
    print("\n---- 低证据桶（按不确定性最高的一档）----")
    print(f"基线：准确率 {low_bucket_baseline.get('accuracy', float('nan')):.4f}，"
          f"ECE {low_bucket_baseline.get('ece', float('nan')):.4f}")
    print(f"证据头：准确率 {low_bucket_evidence.get('accuracy', float('nan')):.4f}，"
          f"ECE {low_bucket_evidence.get('ece', float('nan')):.4f}")
    print("\n---- 同等精度比较 ----")
    print(f"目标精度 {matched['target_accuracy']:.4f}：基线需覆盖率 {matched['baseline_coverage']:.4f}，"
          f"证据头需覆盖率 {matched['evidence_coverage']:.4f}")
    if decomposition:
        print("\n---- 不确定性分解有效性 ----")
        for key, value in decomposition.items():
            print(f"{key}: {value}")

    # --- 7.7 一句话结论（自动生成，人工复核后再写进研究记录）---
    ece_gain = baseline_report["ece_equal_width"] - evidence_report["ece_equal_width"]
    auroc_gain = evidence_report["auroc_confidence"] - baseline_report["auroc_confidence"]
    accuracy_gap = evidence_report["accuracy"] - baseline_report["accuracy"]
    if ece_gain > 0.02 and auroc_gain >= 0.0 and accuracy_gap > -0.01:
        conclusion = "证据头在可信度上有实质改善且精度未掉（可进入 Stage 2）"
    elif accuracy_gap <= -0.01:
        conclusion = "证据头精度掉超过 1 个百分点：先修训练，不解读可信度"
    else:
        conclusion = "证据头未显示实质可信度改善：需要检查损失（如 L_inf 形式）或特征质量"
    print(f"\n[一句话结论] {conclusion}")

    # --- 7.8 落盘 ---
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    payload = {
        "script": os.path.basename(__file__),
        "timestamp": stamp,
        "seed": args.seed,
        "readout": args.readout,
        "evidence_param": args.evidence_param,
        "lambda_conc": args.lambda_conc,
        "args": vars(args),
        "env": collect_env(),                   # 三种机器（Windows/macOS/AutoDL）都要能追溯
        "data": {
            "source": "synthetic" if args.synthetic else args.features,
            "sha1": None if args.synthetic else hashlib.sha1(open(args.features, "rb").read()).hexdigest()[:12],
            "n_train": int(len(bundle["y_train"])),
            "n_test": int(len(bundle["y_test"])),
            "dim": dim,
            "n_classes": args.n_classes,
        },
        "baseline": baseline_report,
        "evidence": evidence_report,
        "matched_comparison": matched,
        "low_evidence_bucket": {"baseline": low_bucket_baseline, "evidence": low_bucket_evidence},
        "decomposition": decomposition,
        "val_nll": {"baseline": baseline_val_nll, "evidence": evidence_val_nll},
        "seconds": {"baseline": baseline_seconds, "evidence": evidence_seconds},
        "conclusion": conclusion,
    }
    out_path = os.path.join(out_dir, f"exp1_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[落盘] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())