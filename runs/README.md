# runs

**本目录放实验日志与运行说明。** 每次运行的记录应含：脚本名、sha1、关键指标、用时、状态、checkpoint 与日志路径；远程运行由 `tools/run_remote.sh` 落到本目录。

本目录保留给"运行说明"：

- Windows 上的运行命令（含 conda 环境与工作目录要求）写进 [../README.md](../README.md)。
- 每次运行前先确认：数据文件路径、随机种子、`--synthetic` 自检是否通过。
- 运行后把 `results/expN_<timestamp>.json` 里的一句话结论抄进当天的研究记录（Problem → Hypothesis → Method → Experiment → Observation → Conclusion）。