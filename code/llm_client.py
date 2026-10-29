# llm_client.py —— 最小依赖的 LLM 客户端（OpenAI 兼容接口）+ 磁盘缓存
#
# 为什么自己写而不用 SDK：只依赖标准库（urllib），服务器上零安装风险；
# 缓存到磁盘，保证同一 prompt 只花钱一次（反复调参时不重复计费）。
#
# 环境变量：
#   LLM_API_BASE   默认 https://api.openai.com/v1
#   LLM_API_KEY    必需（offline 模式除外）
#   LLM_MODEL      默认 gpt-4o-mini
#   LLM_CACHE_DIR  默认 /data/ldy/ek55_assets/llm_cache
#
# 用法：
#   python3 code/llm_client.py --self-test        # 不联网，验证缓存/解析/重试
#   python3 code/llm_client.py --ping             # 花 1 次调用，确认接口可用

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_CACHE = "/data/ldy/ek55_assets/llm_cache"


def cache_key(prompt, model, temperature):
    digest = hashlib.sha1(f"{model}|{temperature}|{prompt}".encode("utf-8")).hexdigest()
    return digest


def read_cache(cache_dir, key):
    path = os.path.join(cache_dir, f"{key}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    return None


def write_cache(cache_dir, key, payload):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)


def call_api(prompt, model, temperature, api_base, api_key, timeout=60, retries=3,
             transport=None):
    """调用 /chat/completions。transport 可注入（自检时替换成假函数）。"""
    url = api_base.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error = None
    for attempt in range(retries):
        try:
            if transport is not None:
                return transport(url, headers, body, timeout)
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
            last_error = error
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"LLM 调用失败（重试 {retries} 次）：{last_error}")


def extract_text(payload):
    """从 OpenAI 兼容响应里取出文本。"""
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return message.get("content") or ""


def ask(prompt, model=None, temperature=0.0, api_base=None, api_key=None,
        cache_dir=None, offline=False, transport=None):
    """带缓存的一次问答。offline=True 时返回固定桩（用于打通流程，不花钱）。"""
    model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    api_base = api_base or os.environ.get("LLM_API_BASE", DEFAULT_BASE)
    api_key = api_key or os.environ.get("LLM_API_KEY", "")
    cache_dir = cache_dir or os.environ.get("LLM_CACHE_DIR", DEFAULT_CACHE)

    if offline:
        return '{"ranking": [], "confidence": 0.5, "_offline": true}'

    key = cache_key(prompt, model, temperature)
    cached = read_cache(cache_dir, key)
    if cached is not None:
        return cached["text"]

    payload = call_api(prompt, model, temperature, api_base, api_key, transport=transport)
    text = extract_text(payload)
    write_cache(cache_dir, key, {"text": text, "model": model, "temperature": temperature,
                                 "usage": payload.get("usage")})
    return text


def self_test():
    workdir = os.path.join("/tmp", "llm_client_test")
    os.makedirs(workdir, exist_ok=True)
    calls = {"n": 0}

    def fake_transport(url, headers, body, timeout):
        calls["n"] += 1
        request = json.loads(body.decode("utf-8"))
        assert request["model"] == "fake-model"
        assert "Authorization" in headers
        return {"choices": [{"message": {"content": '{"ranking": ["take egg"], "confidence": 0.9}'}}],
                "usage": {"total_tokens": 42}}

    first = ask("hello", model="fake-model", api_key="k", cache_dir=workdir,
                transport=fake_transport)
    second = ask("hello", model="fake-model", api_key="k", cache_dir=workdir,
                 transport=fake_transport)
    assert first == second, "缓存未生效"
    assert calls["n"] == 1, f"应只调用 1 次，实际 {calls['n']}"

    other = ask("hello", model="fake-model", temperature=0.7, api_key="k", cache_dir=workdir,
                transport=fake_transport)
    assert calls["n"] == 2, "不同 temperature 应重新调用（不同 key）"

    assert extract_text({"choices": [{"message": {"content": "x"}}]}) == "x"
    assert extract_text({}) == ""
    assert ask("p", offline=True).find("_offline") > 0
    print(f"自检通过：缓存命中（1 次调用 / 2 次请求）、离线桩、响应解析；输出示例 {other[:30]}…")


def ping():
    api_key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    if not api_key:
        print("[错误] 未设置 LLM_API_KEY")
        return 1
    text = ask("Reply with exactly: pong", model=model, temperature=0.0)
    print(f"[接口] model={model} 回复={text!r}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--ping", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.ping:
        sys.exit(ping())
    else:
        print(__doc__ or "见文件头注释")
