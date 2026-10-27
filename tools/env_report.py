# env_report.py —— 记录运行环境，写进结果 JSON（可复现性要求）
#
# 为什么需要：实验可能在实验室 Windows、macOS、AutoDL 三种机器上跑，
# 只记 args 与种子不够；必须同时记 GPU/驱动/torch/CUDA 与代码版本。
#
# 用法：
#   python tools/env_report.py                # 打印并写 results/env_<时间戳>.json
#   python tools/env_report.py --quiet        # 只写文件
# 在实验脚本里：
#   from env_report import collect_env        # 然后把 collect_env() 塞进结果 JSON
#
# 注意：本文件必须在"没有 torch、没有 git"的环境下也能跑（macOS 本机就是这种情况）。

import argparse
import json
import os
import platform
import subprocess
import sys
import time


# ===== 1. 代码版本 =====

def collect_code_version(repo_root=None):
    """尽量拿到 git 版本信息；不是仓库时返回 None 并说明原因（这本身是要记录的事实）。"""
    # 仓库根 = **项目目录本身**（2026-10-27 起仓库根即 ek55-evidence-cl/，不再是上层「论文/」）
    root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    info = {"repo_root": root, "is_git_repo": False, "commit": None, "branch": None, "dirty": None}

    def run_git(args):
        try:
            completed = subprocess.run(
                ["git", "-C", root] + args,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return completed.stdout.strip() if completed.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    top_level = run_git(["rev-parse", "--show-toplevel"])
    if top_level is None:
        info["note"] = "不是 git 仓库——只能靠脚本 sha1 追溯"
        return info
    info["is_git_repo"] = True
    info["commit"] = run_git(["rev-parse", "--short", "HEAD"])
    info["branch"] = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = run_git(["status", "--porcelain"])
    info["dirty"] = bool(status) if status is not None else None
    return info


# ===== 2. 运行环境 =====

def collect_env(repo_root=None):
    env = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "code": collect_code_version(repo_root),
    }

    # --- torch / CUDA / GPU（缺失时要能优雅退化）---
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            env["gpu_memory_gb"] = [
                round(torch.cuda.get_device_properties(i).total_memory / 1024 ** 3, 1)
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        env["torch"] = None
        env["note_torch"] = "未安装 torch（例如 macOS 本机）——该环境下不能跑训练"

    # --- numpy / 其他关键库 ---
    for name in ["numpy", "scipy", "sklearn"]:
        try:
            module = __import__(name)
            env[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            env[name] = None

    # --- 驱动（Linux 上读 nvidia-smi，失败就算了）---
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.returncode == 0:
            env["nvidia_smi"] = completed.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        pass

    return env


# ===== 3. 命令行入口 =====

def main():
    parser = argparse.ArgumentParser(description="记录运行环境，写入 results/")
    parser.add_argument("--out-dir", type=str, default=None, help="默认写到项目 results/")
    parser.add_argument("--quiet", action="store_true", help="只写文件，不打印")
    args = parser.parse_args()

    env = collect_env()
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"env_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(env, handle, ensure_ascii=False, indent=2)

    if not args.quiet:
        print(json.dumps(env, ensure_ascii=False, indent=2))
    print(f"[落盘] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())