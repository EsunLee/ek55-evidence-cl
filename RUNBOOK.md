# 运行手册：在哪跑、怎么跑、数字怎么回来

> 2026-10-26 定。结论先说：**Stage 1 不需要租卡；数据准备与 Stage 2–4 建议放 AutoDL；仓库永远是唯一真相**。
> 运行环境（conda、CUDA、已知坑）与实验索引统一维护在**研究工作区**（不在本仓库）；本文件只讲"哪个环节放哪台机器、数据怎么流"。

## 1. 决策表：每个环节放哪

| 环节 | 计算/存储量 | 建议位置 | 理由 |
|---|---|---|---|
| Stage 0：指标与代码 | 无 GPU 需求 | **macOS 本机**（已完成） | 只有 numpy 指标与脚本骨架 |
| 数据准备：解帧 / 采样 / 骨干前向 | 小（帧已在服务器上，一次前向约几分钟） | **amax 服务器**（`/data/datasets/EPIC-KITCHENS/`） | 帧已在服务器，**不搬家**；复用 INSIGHT 的 `extract_features.py` |
| **Stage 1：静态 go/no-go** | **极小** | **实验室 Windows 即可，甚至 CPU** | 冻结特征 + 线性/证据头，分钟级；**不必为此租卡** |
| Stage 2–4：持续学习训练与消融 | 大（GPU 小时级） | **AutoDL 单卡 24 GB 级** | 提示微调必须跑 ViT 前向；消融多、要反复跑 |
| 结果、日志、文档 | 小（MB 级） | **回写仓库**（`results/`、`runs/`） | 实例可能被回收，数字必须落在仓库里 |

## 2. 数据流

```text
[amax 服务器 /data/datasets/EPIC-KITCHENS/] JPEG 帧（1.25M 帧、228 GB；已在机器上）
        │  ① 清点  ② 采帧+resize（CPU/IO，不抢卡）  ③（可选）提特征（几分钟 GPU）
        ▼
   可搬运资产（≈6–10 GB）：frames/ + segments.csv + manifest.json（+ features.npz）
        │  ④ 一次性上传
        ▼
   [对象存储 / OSS]  ← 云实例随时来取，实例被回收也不用重传
        │  ⑤ 拉取
        ▼
   [云 GPU 实例] Stage 1（特征，分钟级）→ Stage 2+（采样帧，提示微调）
        │  ⑥ 结果与日志（很小）回传
        ▼
   [本地仓库 论文/] 代码 + 文档 + results + runs 索引   ← 唯一真相
（amax 作为备用算力：GPU 0,1,3、torch ≤2.5.1+cu118）
```

## 3. 存储与下载（**官方数字，2026-10-26 核实**）

| 内容 | 体积 | 出处 / 说明 |
|---|---|---|
| **EK55 sequences**（含 RGB 帧 + 光流帧） | **1 TB zipped** | [EK55 官方页](https://epic-kitchens.github.io/2020-55) → Data.Bris / Academic Torrents |
| EK100 extended sequences（+RGB/光流帧、陀螺仪） | **740 GB zipped** | [EK100 官方页](https://epic-kitchens.github.io/2026) |
| EK100 original sequences（+RGB 与光流帧） | **1.1 TB zipped** | 同上 |
| EK100 自动标注（掩码 / 手 / 物体） | 10 GB | 同上 |
| 只下视频（不含预提帧与光流） | **官方页未给，必须实测** | 官方提供分部分下载脚本 |
| DUDE 口径特征（每段 1 个向量） | ≈ 0.16 GB/模态（估算） | 约 4 万段 × 1024 维 × 4 B；三模态 ≈ 0.5 GB |
| ViT 帧级特征（8 帧/段） | ≈ 1 GB（估算） | 约 4 万 × 8 × 768 × 4 B |
| 采样帧图片（JPEG） | ≈ 10 GB（估算） | 只有需要反复跑 ViT 前向时才值得存 |
| Stage 1 训练 | CPU 分钟级 | 不必上 GPU |
| Stage 2+ 完整协议 | 单卡 8–20 GPU 小时 | 10 任务 × 50 epoch 量级；需实测后回填 |

> ⚠️ 教训留档（2026-10-26）：本文件初稿把原始数据写成"数十 GB"——那是按 55 小时片长臆测的，**与官方 1 TB 差一个数量级**。凡体积、类数、协议数字，一律先查官方页或数据文件再写。

### 3.1 数据流转策略（**已有本地拷贝，2026-10-26 确认**）

1. **先清点**：在数据所在机器跑 `tools/inventory_data.py`，把 `data/inventory_*.json` 提交进仓库，并在 [data/README.md](data/README.md) 登记覆盖范围（全量 32 kitchen 还是子集）。
2. **在数据所在的机器提特征**：谁有数据就在谁那里解码与跑骨干；**原始视频不跨机器搬运**（官方 1 TB 级别）。
3. **只有特征参与流转**：特征（GB 级）与可选采样帧（≈10 GB）才进 OSS 或实例。
4. **仓库永不收**：视频、帧、特征都不进 `论文/`；`data/` 只放说明与 inventory JSON。
5. **若必须用实例**：只上传特征/采样帧，不上传视频；HF 缓存指到数据盘：

   ```bash
   export HF_HOME=/root/autodl-tmp/cache/
   ```

6. **实例网络现实（AutoDL 官方[学术加速文档](https://docs.coreshub.cn/console/compute_platform/faq/speed_up/)，2026-02-04 更新）**：官方只加速 GitHub 代理（`gh-proxy.com`）与 HuggingFace 镜像（`hf-mirror.com`），**未提 BT/种子**；"在实例上走 torrent"未经验证，不要写进计划。

### 3.2 类数口径（同样已核实）

- 官方页写：EK55 = **125 verb / 331 noun**、39,594 个动作片段、55 小时、11.5M 帧、32 个厨房。
- 但 **INSIGHT 项目基于实际使用的 annotation 转换后得到 125 verbs / 352 nouns**（28,472 actions、272 clips，见其工作日志）——说明 **noun 数随 annotation 版本/口径而变**。
- 结论：**以我们实际使用的 annotation 文件为准，并在论文与实验记录里注明来源与版本**；引用"331 / 352"时都要带出处，不要当成绝对常数。
- **动作类 = verb×noun 对**，其数量取决于我们的划分（持续学习协议要自己定），必须从实际 annotation 统计。

### 3.3 第一步：**清点**（不是下载）

不凭印象决定"手上有什么"。在数据所在机器跑一次：

```bash
python tools/inventory_data.py --data-root <数据根目录> --annotations <标注目录>
# 例（amax 服务器）：python tools/inventory_data.py --data-root /data/datasets/EPIC-KITCHENS/rgb_frames \
#                        --annotations /data/datasets/EPIC-KITCHENS/insight_annotations
```

它只读文件名与体积（不解码视频，很快），输出 `data/inventory_<时间戳>.json`：视频数与总体积、覆盖多少 kitchen、预提帧目录与张数、标注 CSV 的行数/列是否完整，并给出体检提示。

**清点结果要回答三个问题**，答案决定下一步：

| 问题 | 影响 |
|---|---|
| 覆盖全部 32 个 kitchen，还是子集？ | 子集 → 在协议里写明范围；Stage 1 可以直接开工 |
| 有预提帧吗，还是只有原始视频？ | 只有视频 → Stage 2 需要先解帧（采样 8 帧/段）；有帧 → 省一大步 |
| 标注（train/val/test，含 verb/noun 与时间段）齐吗？ | 缺 → 先从 epic-kitchens/annotations 补齐，否则无法构造过去/未来片段 |

### 3.4 剩余的待验证项（Stage 2 之前做）

数据获取已解决（组内已有拷贝），**唯一还需实测的是实例侧**：`hf-mirror.com` 是否可用、数据盘大小、以及只上传特征（GB 级）的实际耗时。这些等真要上 AutoDL 时用最小实例验证即可，**不必现在租**。

## 4. 服务器操作清单（amax 优先；AutoDL 为备选）

### 4.1 amax 服务器（**首选**，据 2026-10-26 核实的 INSIGHT 日志）

| 事项 | 做法 |
|---|---|
| GPU | **只用 physical `0,1,3`**（GPU 2 有 Xid 79/48 硬件故障）；ms-swift 不认 UUID，要写数字 |
| 框架版本 | **torch ≤ 2.5.1+cu118**（驱动 495.29.05 / CUDA 11.5；2.6+/vllm 不可用） |
| 环境 | 复用 `/data/conda_envs/insight_env`，或自建新环境但同样锁 torch 版本 |
| 写文件 | **产物写 `/data` 或 `/tmp`**；`/home` 只有 244G 且曾 100% 满 |
| 数据 | `/data/datasets/EPIC-KITCHENS/`（JPEG 帧 + 已转换标注 + 既有特征） |
| 共享礼仪 | 只占需要的卡（`CUDA_VISIBLE_DEVICES=0` 起步），跑完释放显存；先 `nvidia-smi` 看谁在用 |
| 日志 | `tmux` + `tee`；产物与结果回传后登记进研究工作区的实验索引 |

### 4.2 AutoDL（**Stage 2+ 首选**：需要新驱动 / 不想等共享机器时）

1. **实例**：单卡 24 GB 级即可（无需多卡）；镜像选 PyTorch。**先开"无卡模式"做完环境安装与资产拉取**，再开机跑训练，省钱。数据只用"三件套"（≈10 GB），**不上传原始帧**。
2. **环境**：自建 conda/venv，不要依赖镜像预装版本；跑 `python tools/env_report.py` 把版本与 GPU 记下来（要进结果 JSON）。
3. **数据**：原始视频放数据盘或 OSS，**不要放 `~/`**（系统盘小）；用 `tools/sync_features.sh` 里的 rclone/ossutil 模板同步。
4. **运行**：一律 `tmux` + `tee` 日志（`tools/run_remote.sh`）；每个 epoch 存 checkpoint，保证实例被抢占后能续跑。
5. **回传**：任务一结束就 `rclone copy` 结果与日志回 OSS/仓库；把摘要登记进研究工作区的实验索引。
6. **关机**：任务完成即 stop 实例；OSS 只长期保留特征、帧与结果。

## 5. 可复现纪律（2026-10-27 已补齐）

- ✅ **版本库已就位**（2026-10-27）：**仓库根 = 项目目录本身**，远端 GitHub 私有仓库 `EsunLee/ek55-evidence-cl`（历史经 `git subtree` 从旧仓库切出，未丢失）。服务器上 `git clone git@github.com:EsunLee/ek55-evidence-cl.git` 即得项目本身。
- 每个结果 JSON 由 `tools/env_report.py` 自动写入 **commit / branch / dirty** ⇒ 数字可追到确切代码版本（已验证）。
- **远程运行纪律**：服务器上改过脚本必须 `git commit`（或在结果里记脚本 sha1）；结果 JSON 与日志回传后提交进仓库。
- 数据集、帧、权重、特征 npz、服务器上的大产物**一律不提交**。
- 每个结果 JSON 必须含：脚本 sha1、数据文件 sha1、环境（GPU/驱动/torch/CUDA）、随机种子、耗时。
- 未运行的实验不写成结果；`L_inf` 的具体形式、EK55 切分细节等**待核实**项要在结果里显式标注。

## 6. 与 exp-runner 的关系

- 实验室 Windows 机上的 exp-runner **用于纯特征实验**；其索引在研究工作区（不在本仓库）。
- AutoDL 上 exp-runner 不适用（Windows 专用路径与 conda 布局）。远程用 `tools/run_remote.sh` 这套轻量替代：
  日志落 `runs/<exp>_<时间戳>.log`，结果落 `results/<exp>_<时间戳>.json`，回传后**手动**登记到研究工作区的实验索引。
