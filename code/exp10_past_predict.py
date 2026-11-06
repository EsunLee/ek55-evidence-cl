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



class CausalTagger(nn.Module):
    """因果逐段标注器：第 t 段的输出只依赖第 1..t 段特征。

    为什么必须因果：第 t 段的动作名是"已发生"的信息，真实系统在 t 时刻只能看到 1..t。
    用双向上下文或整段平均都会把未来信息带进来。
    """

    def __init__(self, dim, n_classes, hidden=256, layers=2, dropout=0.1):
        super().__init__()
        self.project = nn.Linear(in_features=dim, out_features=hidden)
        self.position = nn.Parameter(torch.zeros(1, 30, hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.classifier = nn.Linear(in_features=hidden, out_features=n_classes)

    @staticmethod
    def causal_mask(length, device):
        """上三角为 -inf：位置 t 只能看 <= t。"""
        mask = torch.triu(torch.ones(length, length, device=device), diagonal=1).bool()
        return mask

    def forward(self, sequence):
        hidden = self.project(sequence) + self.position[:, : sequence.shape[1]]
        mask = self.causal_mask(sequence.shape[1], sequence.device)
        hidden = self.encoder(hidden, mask=mask)
        return self.classifier(hidden)


def train_causal_taggers(args, device, data):
    """训练因果逐段标注器（verb/noun 各一个），返回 test 上的预测。"""
    import numpy as _np
    seq_train = data["x_seq_train"].astype(_np.float32)
    seq_val = data["x_seq_val"].astype(_np.float32)
    seq_test = data["x_seq_test"].astype(_np.float32)
    v_train = data["past_verb_train"].astype(_np.int64)
    v_val = data["past_verb_val"].astype(_np.int64)
    n_train = data["past_noun_train"].astype(_np.int64)
    n_val = data["past_noun_val"].astype(_np.int64)
    n_verb = int(max(v_train.max(), data["past_verb_test"].max()) + 1)
    n_noun = int(max(n_train.max(), data["past_noun_test"].max()) + 1)

    results = {}
    for name, y_train, y_val, n_classes in (("verb", v_train, v_val, n_verb),
                                            ("noun", n_train, n_val, n_noun)):
        model = CausalTagger(seq_train.shape[-1], n_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        best_state, best_val = None, -1.0
        for epoch in range(args.epochs):
            model.train()
            order = torch.randperm(len(seq_train))
            for start in range(0, len(order), max(args.batch_size // 8, 8)):
                index = order[start : start + max(args.batch_size // 8, 8)]
                batch_x = torch.from_numpy(seq_train[index.numpy()]).to(device)
                batch_y = torch.from_numpy(y_train[index.numpy()]).to(device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = F.cross_entropy(logits.reshape(-1, n_classes), batch_y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_logits = model(torch.from_numpy(seq_val).to(device))
                accuracy = (val_logits.argmax(-1).cpu().numpy() == y_val).mean()
            if accuracy > best_val:
                best_val = accuracy
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 5 == 0:
                print(f"[causal {name}] epoch {epoch + 1} 验证逐段准确率 {accuracy:.4f}")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_logits = model(torch.from_numpy(seq_test).to(device))
            results[name] = test_logits.argmax(-1).cpu().numpy()
            val_logits = model(torch.from_numpy(seq_val).to(device))
            results[name + "_val"] = val_logits.argmax(-1).cpu().numpy()
        print(f"[causal {name}] 验证集最优逐段准确率 {best_val:.4f}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="逐段动作识别器 → 预测的历史动作名")
    parser.add_argument("--features", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temporal", action="store_true",
                        help="因果时序：第 t 段只用第 1..t 段特征（严格因果，不偷看未来）")
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

    if args.temporal:
        predictions = train_causal_taggers(args, device, data)
        pred_verb = predictions["verb"]
        pred_noun = predictions["noun"]
        seq_acc = (pred_verb == data["past_verb_test"]).mean()
        print(f"[因果序列] 30 段整体逐段命中率 {seq_acc:.4f}")
        payload = {key: data[key] for key in data.files}
        payload["past_verb_test"] = pred_verb
        payload["past_noun_test"] = pred_noun
        # 关键：val 也必须用同一种（预测的）历史，否则强度选择与最终评估条件不一致
        payload["past_verb_val"] = predictions["verb_val"]
        payload["past_noun_val"] = predictions["noun_val"]
        np.savez(args.out, **payload)
        print(f"[落盘] {args.out}（因果预测的历史动作名）")
        return 0

    raise SystemExit(
        "[错误] 非因果分支不导出 val 预测历史，会造成 val/test 输入不一致；\n"
        "       请加 --temporal 使用因果逐段标注器（推荐），或自行保证 val 一致性后再改代码。")

    verb_model, verb_acc = train_head(x_train, v_train, n_verb, "verb")
    noun_model, noun_acc = train_head(x_train, n_train, n_noun, "noun")

    # 对 test 的 30 段逐段预测
    def predict(model, seq=x_test):
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(x_test), 4096):
                batch = torch.from_numpy(seq[start : start + 4096]).to(device)
                outputs.append(model(batch).argmax(1).cpu())
        return torch.cat(outputs).numpy().reshape(n_samples, window)

    pred_verb = predict(verb_model)
    pred_noun = predict(noun_model)
    seq_acc = (pred_verb == data["past_verb_test"]).mean()
    print(f"[序列] 30 段整体逐段命中率 {seq_acc:.4f}")

    pred_verb_val = predict(verb_model, x_val.reshape(-1, window, x_val.shape[-1])
                            if x_val.ndim == 2 else None) if False else None
    payload = {key: data[key] for key in data.files}
    payload["past_verb_test"] = pred_verb
    payload["past_noun_test"] = pred_noun
    np.savez(args.out, **payload)
    print(f"[落盘] {args.out}（past_verb_test 已替换为预测值）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
