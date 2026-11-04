# exp8_encoder.py —— 强基线：时序 Transformer 编码器（检验当前特征的上限）
#
# 为什么需要它：现有主结果的基线是"最后 5 段均值 + 线性头"（verb 27.7%），
# 视觉会议的审稿人一定会问"收益能否外推到更强的模型"。本脚本把读出换成
# 时序 Transformer，看同一组冻结特征能到多高——这是从"工作坊可发"走到
# "CCF-B 可投"的硬门槛。
#
# 用法：
#   python3 code/exp8_encoder.py --features ek55_verb_full_text.npz \
#     --layers 2 --d-model 256 --epochs 40 --seed 0 \
#     --export-alpha alpha_verb_enc_s0.npz --export-alpha-val alpha_verb_enc_val_s0.npz
#
# 输出 npz 与 exp1 完全兼容，可直接喂给 exp7_llm_evidence.py。

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import (accuracy_at_coverage, aurc, brier_score, dirichlet_mean,
                     dirichlet_vacuity, ece_adaptive, ece_equal_width,
                     fit_temperature, nll_score)
from exp1 import iterate_batches, set_seed, to_tensor


def parse_args():
    parser = argparse.ArgumentParser(description="时序 Transformer 强基线")
    parser.add_argument("--features", required=True, help="build_ek55_npz.py 产出的 npz")
    parser.add_argument("--layers", type=int, default=2, help="Transformer 层数")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--last-k", type=int, default=5, help="池化时取的末尾段数")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--loss", default="ce", choices=["ce", "ce_evi"],
                        help="ce=纯交叉熵；ce_evi=额外鼓励浓度贴近 K（用于导出 vacuity）")
    parser.add_argument("--lambda-conc", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--export-alpha", type=str, default=None)
    parser.add_argument("--export-alpha-val", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


class TemporalEncoder(nn.Module):
    """[B, 30, 1024] → 时序 Transformer → 池化特征 [B, 2d] → 输出 logits 或 α。"""

    def __init__(self, dim, n_classes, d_model, layers, heads, dropout, last_k,
                 evidence=False):
        super().__init__()
        self.last_k = last_k
        self.evidence = evidence
        self.project = nn.Linear(in_features=dim, out_features=d_model)
        self.position = nn.Parameter(torch.zeros(1, 30, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        pooled_dim = d_model * 2                     # 末尾 k 段均值 ⊕ 全序列均值
        self.classifier = nn.Linear(in_features=pooled_dim, out_features=n_classes)
        if evidence:
            self.concentration = nn.Linear(in_features=pooled_dim, out_features=1)

    def forward(self, sequence):
        hidden = self.project(sequence) + self.position[:, : sequence.shape[1]]
        hidden = self.norm(self.encoder(hidden))
        tail = hidden[:, -self.last_k :].mean(dim=1)   # 最近 k 段
        whole = hidden.mean(dim=1)                     # 全序列
        pooled = torch.cat([tail, whole], dim=1)
        logits = self.classifier(pooled)
        if not self.evidence:
            return pooled, logits
        probs = torch.softmax(logits, dim=1)
        strength = logits.shape[1] * (1.0 + F.softplus(self.concentration(pooled)))
        return pooled, probs * strength                # α = α0 · p


def predict(model, x, device, batch_size=512):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = to_tensor(x[start : start + batch_size], device)
            outputs.append(model(batch)[1].cpu())
    stacked = torch.cat(outputs, dim=0).numpy().astype(np.float64)
    return stacked


def evaluate(probs, labels):
    return {
        "accuracy": float((probs.argmax(axis=1) == labels).mean()),
        "ece_equal_width": float(ece_equal_width(probs, labels)),
        "ece_adaptive": float(ece_adaptive(probs, labels)),
        "brier": float(brier_score(probs, labels)),
        "nll": float(nll_score(probs, labels)),
        "aurc": float(aurc(probs, labels)),
        "accuracy_at_80pct_coverage": float(accuracy_at_coverage(probs, labels, 0.8)),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = np.load(args.features, allow_pickle=True)
    bundle = {
        "x_train": data["x_seq_train"].astype(np.float32),
        "y_train": data["y_train"].astype(np.int64),
        "x_val": data["x_seq_val"].astype(np.float32),
        "y_val": data["y_val"].astype(np.int64),
        "x_test": data["x_seq_test"].astype(np.float32),
        "y_test": data["y_test"].astype(np.int64),
    }
    dim = bundle["x_train"].shape[-1]
    n_classes = int(max(bundle["y_train"].max(), bundle["y_test"].max()) + 1)
    print(f"[数据] 训练 {len(bundle['y_train'])} / 验证 {len(bundle['y_val'])} / "
          f"测试 {len(bundle['y_test'])}｜维度 {dim}｜类别 {n_classes}｜设备 {device}")

    evidence = args.loss == "ce_evi"
    model = TemporalEncoder(dim, n_classes, args.d_model, args.layers, args.heads,
                            args.dropout, args.last_k, evidence=evidence).to(device)
    parameter_count = sum(item.numel() for item in model.parameters())
    print(f"[模型] 层数 {args.layers}｜d_model {args.d_model}｜参数量 {parameter_count / 1e3:.0f}K")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_state, best_val_nll, best_epoch = None, float("inf"), 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for features, targets in iterate_batches(
                bundle["x_train"], bundle["y_train"], args.batch_size, device, True):
            optimizer.zero_grad()
            _, output = model(features)
            if evidence:
                probs = output / output.sum(dim=1, keepdim=True)
                loss = F.cross_entropy(torch.log(probs.clamp_min(1e-12)), targets)
                concentration = output.sum(dim=1)
                target_concentration = torch.full_like(concentration, float(n_classes))
                loss = loss + args.lambda_conc * (torch.log(concentration) -
                                                  torch.log(target_concentration)).pow(2).mean()
            else:
                loss = F.cross_entropy(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_output = predict(model, bundle["x_val"], device)
        val_probs = val_output / val_output.sum(axis=1, keepdims=True) if evidence \
            else torch.softmax(torch.from_numpy(val_output), dim=1).numpy()
        val_nll = nll_score(val_probs, bundle["y_val"])
        val_accuracy = (val_probs.argmax(axis=1) == bundle["y_val"]).mean()
        if val_nll < best_val_nll:
            best_val_nll, best_epoch = val_nll, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}｜val nll {val_nll:.4f}｜val acc {val_accuracy:.4f}"
                  f"｜{time.time() - started:.0f}s")

    model.load_state_dict(best_state)
    print(f"[训练] 最优 epoch {best_epoch}｜val nll {best_val_nll:.4f}"
          f"｜总耗时 {time.time() - started:.0f}s")

    test_output = predict(model, bundle["x_test"], device)
    test_probs = test_output / test_output.sum(axis=1, keepdims=True) if evidence \
        else torch.softmax(torch.from_numpy(test_output), dim=1).numpy()
    report = evaluate(test_probs, bundle["y_test"])
    print("\n================ 测试集 ================")
    for key, value in report.items():
        print(f"  {key:<28}{value:.4f}")

    if args.export_alpha:
        payload = {"probs": test_probs.astype(np.float32), "y_test": bundle["y_test"]}
        if evidence:
            payload["alpha"] = test_output.astype(np.float32)
            payload["evidence_probs"] = dirichlet_mean(test_output).astype(np.float32)
        np.savez(args.export_alpha, **payload)
        print(f"[导出] {args.export_alpha}")
    if args.export_alpha_val and evidence:
        val_alpha = predict(model, bundle["x_val"], device)
        np.savez(args.export_alpha_val, alpha=val_alpha.astype(np.float32),
                 probs=dirichlet_mean(val_alpha).astype(np.float32),
                 evidence_probs=dirichlet_mean(val_alpha).astype(np.float32),
                 y_val=bundle["y_val"])
        print(f"[导出] {args.export_alpha_val}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"report": report, "args": vars(args),
                       "best_epoch": best_epoch, "parameters": parameter_count},
                      handle, ensure_ascii=False, indent=2)
        print(f"[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
