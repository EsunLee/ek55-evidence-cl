# ek55-evidence-cl

EK55（EPIC-KITCHENS-55）动作预测的实验代码与工具：把视频帧整理成可搬运的数据资产、提取特征，并在评测阶段计算**可信度指标**（校准、误分类检测、选择性预测）。

> 状态：进行中的研究代码，接口与文件格式可能变动。
> 研究方案、文献笔记与实验纪律保存在私有工作区，**不在本仓库**。

## 目录

| 路径 | 内容 |
|---|---|
| `code/metrics.py` | 可信度指标库（ECE/Brier/NLL/AUROC/AUPR/AURC/分桶；numpy，可独立自检） |
| `code/exp1.py` | 静态实验：在冻结特征上比较线性头与证据头（PyTorch） |
| `code/gate_properties.py` | 门控函数的性质验证与压力测试（numpy） |
| `code/llm_prior.py` | Dirichlet 先验注入模块 + 响应缓存（numpy，自带桩模型） |
| `code/diagnose_a1_a2.py` | 路由器与不确定性信号的质量诊断（numpy） |
| `tools/inventory_data.py` | 清点 EK55 帧与标注（有什么、缺什么） |
| `tools/build_ek55_assets.py` | 生成采样帧资产（`frames/` + `segments.csv` + `manifest.json`） |
| `tools/env_report.py` | 记录运行环境（GPU/torch/commit）到 `results/` |
| `tools/remote_bootstrap.sh` | 远程一键盘点（环境 + 数据 + 资产干跑） |
| `tools/run_remote.sh` | 远程运行：日志落盘 + 断线保护 |
| `tools/sync_features.sh` | 与对象存储同步（rclone/ossutil 模板，默认预演） |
| `data/` `runs/` `results/` | 各自的文件格式与纪律，见目录内 README |

## 运行

```bash
# ---- 不需要 GPU（纯 numpy，自带自检）----
python3 code/metrics.py
python3 code/gate_properties.py
python3 code/llm_prior.py
python3 code/diagnose_a1_a2.py --synthetic with_decoys     # 也可 --input <日志 npz>

# ---- 数据清点与资产生成（在数据所在机器上跑）----
python3 tools/inventory_data.py --data-root <帧根目录> --annotations <标注目录>
python3 tools/build_ek55_assets.py --annotations <标注CSV> \
    --frames-root <帧根目录> --out-dir <产物目录> \
    --frames-per-segment 8 --image-size 224 --workers 16 --dry-run

# ---- 静态实验（需要 torch）----
python3 code/exp1.py --synthetic                # 先验证流程
python3 code/exp1.py --features <features.npz>  # 真实特征
```

## 约定

- **指标只有一份实现**（`code/metrics.py`），不在各脚本里重复写 ECE/AUROC。
- 每个实验脚本结尾打印并落盘：数据文件 sha1、关键指标、一句话结论；结果写 `results/*.json`。
- 中文注释；模型 `forward` 返回 `(features, alpha)` 或 `(features, logits)`，便于在 feature 上接后续模块。
- 数据集、帧、模型权重、特征 npz **不入库**。

## 许可

未附许可协议（默认保留所有权利）。代码处于研究阶段，请勿在未经同意的情况下转载或引用其结果。
