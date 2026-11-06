# exp10_past_predict.py —— 用"预测的历史动作名"替代"标注真值"，检验信息泄漏
#
# 为什么必须做：现在 prompt 里的 30 段动作名来自数据集标注（past_verb_* / past_noun_*），
# 而视觉基线只能从池化特征推断。LLM 因此拿到了一条真实推理时拿不到的通道。
# 本脚本训练"逐段动作识别器"（输入单段 1024 维特征 → 该段的 verb/noun），
# 再把 30 段全部预测一遍，产出 pred_past_verb_* / pred_past_noun_*，
# 供 exp7 以 --predicted-history 使用。
#
# 用法：
#   python3 code/exp10_past_predict.py --features ek55_verb_full_text.npz \
#     --out /data/ldy/ek55_assets/ek55_verb_predhist.npz

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="逐段动作识别器 → 预测的历史动作名")
    parser.add_argument("--features", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(args.features, allow_pickle=True)

    # 逐段样本：把每个 split 的 (30 段, 1024) 展平成 (N*30, 1024)，标签取该段自己的 verb/noun
    def build(split):
        seq = data[f"x_seq_{split}"].astype(np.float32)          # (N, 30, 1024)
        verb = data[f"past_verb_{split}"].astype(np.int64)       # (N, 30)
        noun = data[f"past_noun_{split}"].astype(np.int64)       # (N, 30)
        n_samples, window, dim = seq.shape
        return (seq.reshape(n_samples * window, dim),
                verb.reshape(-1), noun.reshape(-1), n_samples, window)

    x_train, v_train, n_train, _, _ = build("train")
    x_val, v_val, n_val, _, _ = build("val")
    x_test, v_test, n_test, n_samples, window = build("test")

    n_verb = int(max(v_train.max(), v_test.max()) + 1)
    n_noun = int(max(n_train.max(), n_test.max()) + 1)
    print(f"[数据] 逐段样本 训练 {len(v_train)} / 验证 {len(v_val)} / 测试 {len(v_test)}"
          f"｜verb {n_verb} 类｜noun {n_noun} 类")

    def train_head(x, y, n_classes, name):
        model = nn.Sequential(
            nn.Linear(x.shape[1], 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        tensor_x = torch.from_numpy(x)
        tensor_y = torch.from_numpy(y)
        for epoch in range(args.epochs):
            model.train()
            order = torch.randperm(len(tensor_y))
            for start in range(0, len(order), args.batch_size):
                index = order[start : start + args.batch_size]
                batch_x = tensor_x[index].to(device)
                batch_y = tensor_y[index].to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            val_acc = (model(torch.from_numpy(x_val).to(device)).argmax(1).cpu()
                       == torch.from_numpy(v_val if name == "verb" else n_val)).float().mean().item()
        print(f"[{name}] 验证集逐段准确率 {val_acc:.4f}")
        return model, val_acc

    verb_model, verb_acc = train_head(x_train, v_train, n_verb, "verb")
    noun_model, noun_acc = train_head(x_train, n_train, n_noun, "noun")

    # 对 test 的 30 段逐段预测
    def predict(model):
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(x_test), 4096):
                batch = torch.from_numpy(x_test[start : start + 4096]).to(device)
                outputs.append(model(batch).argmax(1).cpu())
        return torch.cat(outputs).numpy().reshape(n_samples, window)

    pred_verb = predict(verb_model)
    pred_noun = predict(noun_model)
    seq_acc = (pred_verb == data["past_verb_test"]).mean()
    print(f"[序列] 30 段整体逐段命中率 {seq_acc:.4f}")

    payload = {key: data[key] for key in data.files}
    payload["past_verb_test"] = pred_verb
    payload["past_noun_test"] = pred_noun
    np.savez(args.out, **payload)
    print(f"[落盘] {args.out}（past_verb_test 已替换为预测值）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
