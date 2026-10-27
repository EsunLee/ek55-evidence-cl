# data

> ⚠️ **共享数据集只读**：`/data/datasets/EPIC-KITCHENS/` 由多人共用（帧、标注、他人特征），
> **本项目只读它、绝不写入**。所有产物（采样帧、npz、清单、日志、检查点）一律写自己的目录，
> 约定为 `/data/ldy/ek55_assets/`（脚本内置护栏：指向数据集目录的写入会被直接拒绝）。

**本目录不放数据。** EK55 的原始视频与预提特征在 Windows 训练机、远程实例或 OSS 上，不入库。

## 数据现状与清点（**数据已在本地**）

> **2026-10-26 确认（来源：INSIGHT 项目工作日志 + 使用者本人）**：EK55 **已下载并在组内**，形态是 **JPEG 帧**（不是原始视频）。
>
> | 事实 | 数值 |
> |---|---|
> | 位置 | amax 服务器 `/data/datasets/EPIC-KITCHENS/`（帧在 `rgb_frames/`，全部帧在同一层目录） |
> | 规模 | **1,251,563 张 JPEG、228 GB、432/434 个 tar**；P22/P24/P31 有缺帧（2 个 tar 永久失败） |
> | 标注 | 已由 INSIGHT 转换为 JSON：**28,472 个 action、272 个 clip、10% val**；标签为 **125 verbs / 352 nouns** |
> | 已有管线 | `feature_extraction/extract_features.py`（CLIP ViT-L/14，每个 action 采 4 帧取平均 → 1024 维；15,480 个 `.pt`，约 2 分钟跑完） |
>
> 官方事实供对照：EK55 sequences 官方为 **1 TB zipped**（含 RGB 帧与光流帧，[官方页](https://epic-kitchens.github.io/2020-55)）——所以**原始数据绝不跨机器搬运**，我们只需要在 amax 上就地提特征。

### 第 0 步：清点（提特征之前必须做）

```bash
# 在 amax 服务器上跑（只看文件名与体积，不解码图像，很快）
python tools/inventory_data.py --data-root /data/datasets/EPIC-KITCHENS/rgb_frames \
                               --annotations /data/datasets/EPIC-KITCHENS/insight_annotations
# 若帧目录层级与预期不同，先用 ls / find 确认真实路径再跑
```

输出并落盘 `data/inventory_<时间戳>.json`：视频文件数与总体积、覆盖多少 participant、预提帧目录与张数、标注 CSV 的行数与列是否完整，以及体检提示（是否是子集、标注是否齐）。

**清点结果要提交进仓库**（JSON 很小），并在下表登记：

| 日期 | 覆盖范围 | 形态 | 标注 | inventory 文件 |
|---|---|---|---|---|
| 待填 | 全量 32 kitchen / 子集（写清哪几个） | 原始视频 / 预提帧 / 两者都有 | train+val+test 是否齐 | `data/inventory_xxx.json` |

### 勘察清单（现状已变，留档）

| # | 检查 | 状态 |
|---|---|---|
| 1 | 组内是否已有数据 | ✅ **已完成**（2026-10-26：组内已有，本人下载） |
| 2 | 官方渠道实测速度与体积 | ⏸️ 不再需要（不下载） |
| 3 | 是否有公开特征包 | ⏸️ 不再需要（自己有数据） |
| 4 | 实例网络与磁盘布局 | ⏳ Stage 2 前再验证（`hf-mirror.com`、数据盘大小；只上传特征，不上传视频） |

**重要**：Stage 1 可以先用子集（甚至 1–2 个 kitchen 的特征）跑通——它的目标是回答"证据头在我们的特征上动不动可信度"，不是出最终数字。

### 合规提醒

- EPIC-KITCHENS 是 **CC BY-NC 4.0**，官方渠道是 Data.Bris 与 Academic Torrents。第三方网盘/HF 上的转载未必有再分发许可。
- 若最终使用了非官方镜像，**必须在论文与实验记录里如实写明数据获取方式**，不要写成"从官方下载"。

## 可搬运资产（amax → 对象存储 → 云实例）

**为什么要这样切**：CODA-Prompt 的提示插在 ViT layer 1–5，**Stage 2 必须让图像过 ViT 前向**——预提特征只够 Stage 1。所以可搬运的是"三件套"：

| 产物 | 体积（估算） | 谁用 |
|---|---|---|
| `frames/<narration_id>/000.jpg … 007.jpg` | ≈ 6–10 GB | Stage 2（提示微调）；已 resize 到固定尺寸，云上不必解码原帧 |
| `segments.csv` | 几 MB | 两边组装"过去 30 段 / 未来 8 段"窗口 |
| `manifest.json` | 几 KB | 配置、跳过清单、sha1——可复现与对账 |
| （可选）`features.npz` | 0.2–1 GB | Stage 1 静态 go/no-go |

生成命令（**在 amax 上跑**；采帧是 CPU/IO 活，不抢 GPU）：

```bash
python tools/build_ek55_assets.py \
    --annotations /data/datasets/EPIC-KITCHENS/annotations/EPIC_55_train.csv \
    --frames-root /data/datasets/EPIC-KITCHENS/rgb_frames \
    --out-dir     /data/ldy/ek55_assets \
    --frames-per-segment 8 --image-size 224 --workers 16

# 先干跑看会跳过多少段：
python tools/build_ek55_assets.py ... --dry-run
```

**采样规则固定为"在 `[start_frame, stop_frame]` 上等距取 N 帧（含两端）"**，写进 manifest，保证可复现。缺帧的段不猜不补，记入 `manifest.json` 的 `skipped` 列表，由上层决定丢弃还是重下（已知 P22/P24/P31 有缺帧）。

## exp1.py 期望的特征文件格式（`.npz`）

```text
x_train [N, D]   过去 30 段聚合后的特征（或 30×D 展平；D 由骨干决定）
y_train [N]      未来第 1 段的动作类别（0..351）
x_val   [N, D]   验证集（用于早停与温度缩放）
y_val   [N]
x_test  [N, D]   测试集
y_test  [N]
x_seq_train [N, T, D]    可选：过去 T 段的**序列**（DUDE 用 30 段），保留时序供后续阶段用
x_seq_val / x_seq_test    同上
x_future_train [N, Df]   可选：未来片段特征，用于 L_dist（有则启用，无则跳过）
novel_mask_test [N]      可选：bool，标记"测试时的新类样本"，用于不确定性分解有效性检验
meta                     可选：dict，记录骨干、采样策略、生成时间、脚本 sha1
```

## 生成方式（待补）

- 首选：复用 DUDE 的预提特征口径（TSM 骨干 + 过去 30 段 / 未来 8 段），保证与文献可比。
- 备选：ViT-B/16 帧级特征 + 时间池化（与 CODA-Prompt 的骨干一致）。
- **无论哪种，都要在 `meta` 里写清骨干、预训练权重、采样与传统方式**，否则后续无法归因。