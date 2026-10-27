# results

本目录放**每次运行的机器可读记录**，并**纳入版本控制**（很小，是实验留痕的一部分）。

## 文件约定

| 文件 | 由谁写 | 内容 |
|---|---|---|
| `env_<时间戳>.json` | `tools/env_report.py` | 运行环境：Python/torch/CUDA、GPU 名称与显存、**代码 commit / branch / dirty**、主机与平台 |
| `exp1_<时间戳>.json` | `code/exp1.py` | 静态 go/no-go：args、数据文件 sha1、基线 vs 证据头两组指标、同等精度比较、低证据桶、一句话结论、耗时 |
| `bootstrap_<时间戳>.txt` | `tools/remote_bootstrap.sh` | 远程盘点报告（环境 + 数据清点 + 资产干跑），供回传核对 |

## 纪律

- **只放 JSON/TXT，不放日志正文**（日志走 `runs/` 或服务器的 `runs/`）。
- 每个文件必须在内容里带上**能追溯的标识**：commit（或脚本 sha1）、数据文件 sha1、随机种子。
- 实验做完就提交：`git add ek55-evidence-cl/results && git commit`。
- 未运行的实验**不**在这里留下文件；手工编造的结果 JSON 一律不允许。