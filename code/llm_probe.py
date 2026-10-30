# llm_probe.py —— 探测"如何关掉推理型模型的思考"
#
# 背景：deepseek-v4.1-flash 把 max_tokens 全花在 reasoning_tokens 上，text_tokens=0，
# 导致空回复且极慢。不同网关关闭推理的字段不同，本脚本用**同一个真实 prompt**
# 逐个试，报告 finish_reason / reasoning_tokens / text，选最快且非空的那个。
#
# 用法： python3 code/llm_probe.py [--npz ...] [--max-tokens 512]

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_prompts import build_prompt, load_class_names

DEFAULT_NPZ = "/data/ldy/ek55_assets/ek55_verb_full_text.npz"
VERB_CSV = "/data/datasets/EPIC-KITCHENS/annotations/EPIC_verb_classes.csv"
NOUN_CSV = "/data/datasets/EPIC-KITCHENS/annotations/EPIC_noun_classes.csv"

VARIANTS = [
    ("baseline", {}),
    ("enable_thinking=False", {"enable_thinking": False}),
    ("chat_template_kwargs", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("thinking=disabled", {"thinking": {"type": "disabled"}}),
    ("reasoning_effort=minimal", {"reasoning_effort": "minimal"}),
]


def load_one_prompt(npz_path):
    import numpy as np
    data = np.load(npz_path, allow_pickle=True)
    classes = json.loads(str(data["meta"]))["classes"]
    verb_names = load_class_names(VERB_CSV)
    noun_names = load_class_names(NOUN_CSV)
    candidates = [verb_names.get(int(cid), f"verb{cid}") for cid in classes]
    return build_prompt(data["past_verb_test"][0], data["past_noun_test"][0],
                        candidates, verb_names, noun_names)


def load_one_prompt_truncated(npz_path, top_n):
    """截断候选表：用来验证"任务越简单，推理越短"这一假设。"""
    import numpy as np
    data = np.load(npz_path, allow_pickle=True)
    classes = json.loads(str(data["meta"]))["classes"]
    verb_names = load_class_names(VERB_CSV)
    noun_names = load_class_names(NOUN_CSV)
    candidates = [verb_names.get(int(cid), f"verb{cid}") for cid in classes][:top_n]
    return build_prompt(data["past_verb_test"][0], data["past_noun_test"][0],
                        candidates, verb_names, noun_names)


def call(prompt, extra, max_tokens, api_base, api_key, model, timeout=90):
    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    body.update(extra)
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST")
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    elapsed = time.time() - started
    choice = (payload.get("choices") or [{}])[0]
    usage = payload.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    text = (choice.get("message") or {}).get("content") or ""
    return {
        "elapsed": elapsed,
        "finish": choice.get("finish_reason"),
        "reasoning_tokens": details.get("reasoning_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "text": text.strip(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=DEFAULT_NPZ)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--top-candidates", type=int, default=0,
                        help="只保留前 N 个候选（测试'任务更简单→推理更短'的假设）")
    parser.add_argument("--models", default="",
                        help="逗号分隔的模型 id 列表：逐个用同一个真实 prompt 测速与是否空回复")
    args = parser.parse_args()

    api_base = os.environ.get("LLM_API_BASE", "")
    api_key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "")
    if not (api_base and api_key and model):
        raise SystemExit("[错误] 需要 LLM_API_BASE / LLM_API_KEY / LLM_MODEL")

    prompt = (load_one_prompt_truncated(args.npz, args.top_candidates)
              if args.top_candidates else load_one_prompt(args.npz))

    if args.models:
        # 换模型扫描：统一用"关推理"配置，比较耗时/是否空回复/回答是否含上下文
        print(f"[模型扫描] prompt={len(prompt)} 字符 max_tokens={args.max_tokens}\n")
        no_thinking = {"chat_template_kwargs": {"enable_thinking": False}}
        for candidate in [item.strip() for item in args.models.split(",") if item.strip()]:
            try:
                result = call(prompt, no_thinking, args.max_tokens, api_base, api_key, candidate)
                ok = bool(result["text"])
                print(f"  {candidate:<40} {result['elapsed']:>6.1f}s  "
                      f"reasoning={result['reasoning_tokens']:<5} {'✅' if ok else '❌ 空'}  "
                      f"{result['text'][:110]}")
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", "ignore")[:120]
                print(f"  {candidate:<40} HTTP {error.code}  {detail}")
            except Exception as error:
                print(f"  {candidate:<40} 失败：{error}")
        return 0
    print(f"[探针] model={model} prompt={len(prompt)} 字符 max_tokens={args.max_tokens}\n")

    rows = []
    for name, extra in VARIANTS:
        try:
            result = call(prompt, extra, args.max_tokens, api_base, api_key, model)
            ok = bool(result["text"])
            rows.append((name, result, ok))
            preview = result["text"][:120].replace("\n", " ")
            print(f"  {name:<26} {result['elapsed']:>6.1f}s  finish={result['finish']:<8} "
                  f"reasoning={result['reasoning_tokens']:<5} {'✅' if ok else '❌ 空'}  {preview}")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "ignore")[:200]
            print(f"  {name:<26} HTTP {error.code}  {detail}")
        except Exception as error:
            print(f"  {name:<26} 失败：{error}")

    usable = [(row[0], row[1]) for row in rows if row[2]]
    if not usable:
        print("\n[结论] 五种写法都拿不到答案：建议换非推理模型，或把 max_tokens 提到 8192+ 并接受慢")
        return 1
    best_name, best = min(usable, key=lambda item: item[1]["elapsed"])
    print(f"\n[结论] 可用写法 {len(usable)} 种；最快：{best_name}（{best['elapsed']:.1f}s）")
    print(f"       建议：export LLM_EXTRA_BODY='{json.dumps(dict(VARIANTS)[best_name])}'")
    print(f"       回复预览：{best['text'][:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
