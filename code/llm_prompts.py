# llm_prompts.py —— 把"过去动作序列 + 候选类"拼成给 LLM 的 prompt
#
# 依赖：build_ek55_npz.py --with-text 产出的 past_verb_* / past_noun_*
#       + EPIC 的 verb/noun 类名表（CSV）
#
# 输出格式约定（便于解析）：
#   要求模型返回 JSON：{"ranking": [候选名，最多 5 个], "confidence": 0.0-1.0}
#   ranking 第 1 位 = 模型最看好的候选；confidence = 它对这个判断的把握
#
# 自检（不需要网络、不需要数据）：
#   python3 code/llm_prompts.py

import csv
import json
import os
import sys

DEFAULT_VERB_CSV = "/data/datasets/EPIC-KITCHENS/annotations/EPIC_verb_classes.csv"
DEFAULT_NOUN_CSV = "/data/datasets/EPIC-KITCHENS/annotations/EPIC_noun_classes.csv"


def load_class_names(csv_path):
    """读取 EPIC 类名表 → {id: 名字}。列名不固定，按"第一列是数字"判断。"""
    names = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                class_id = int(row[0])
            except ValueError:
                continue                      # 表头或非数字行
            name = None
            for cell in row[1:]:
                text = cell.strip()
                if text and not text.isdigit():
                    name = text
                    break
            names[class_id] = name or f"class_{class_id}"
    return names


def build_prompt(past_verb_ids, past_noun_ids, candidate_names, verb_names, noun_names):
    """拼一条 prompt。past_* 是长度 30 的 id 序列（旧→新），candidate_names 是候选动作名。"""
    lines = ["You are reviewing a cooking video. Recent actions, oldest to newest:"]
    for index, (verb_id, noun_id) in enumerate(zip(past_verb_ids, past_noun_ids), start=1):
        verb = verb_names.get(int(verb_id), f"verb{verb_id}")
        noun = noun_names.get(int(noun_id), f"noun{noun_id}")
        lines.append(f"{index}. {verb} {noun}")

    lines.append("")
    lines.append("The next action is one of these candidates:")
    for name in candidate_names:
        lines.append(f"- {name}")

    lines.append("")
    lines.append("Reply with JSON only, in this exact form:")
    lines.append('{"ranking": ["<candidate name>", "..."], "confidence": <0.0-1.0>}')
    lines.append("ranking lists at most 5 candidates, most likely first. "
                 "confidence is your own certainty about the top candidate.")
    return "\n".join(lines)


def parse_response(text):
    """从模型回复里抠出 (ranking, confidence)。容错：允许 ```json 包裹、前后有废话。"""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return [], None
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return [], None
    ranking = payload.get("ranking") or payload.get("candidates") or []
    if not isinstance(ranking, list):
        ranking = []
    ranking = [str(item) for item in ranking][:5]
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        confidence = None
    return ranking, confidence


def ranking_to_support(ranking, candidate_names, confidence=None, temperature=8.0):
    """把 LLM 的排名转成候选上的"支持度"π（行归一化）。

    第 1 名权重最高，依次衰减；confidence 缺省时按 0.7。
    真 LLM 接入后，这里就是 π 的来源；c 用 confidence（对应 vacuity 的注入量）。
    """
    import numpy as np
    n_candidates = len(candidate_names)
    support = np.zeros(n_candidates, dtype=np.float64)
    position = {name: index for index, name in enumerate(candidate_names)}
    certainty = 0.7 if confidence is None else max(0.05, min(1.0, confidence))
    for rank, name in enumerate(ranking):
        if name in position:
            support[position[name]] += np.exp(-rank / max(temperature / 8.0, 1e-6)) * certainty
    if support.sum() <= 0:
        support[:] = 1.0
    return support / support.sum()


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as workdir:
        verb_csv = os.path.join(workdir, "verb.csv")
        noun_csv = os.path.join(workdir, "noun.csv")
        with open(verb_csv, "w", encoding="utf-8") as handle:
            handle.write("id,key\n0,take\n1,close\n")
        with open(noun_csv, "w", encoding="utf-8") as handle:
            handle.write("id,key,type\n0,egg,noun\n1,fridge,noun\n")
        verb_names = load_class_names(verb_csv)
        noun_names = load_class_names(noun_csv)
        assert verb_names == {0: "take", 1: "close"}, verb_names
        assert noun_names == {0: "egg", 1: "fridge"}, noun_names

        prompt = build_prompt([0, 1], [0, 1], ["take egg", "close fridge"], verb_names, noun_names)
        assert "1. take egg" in prompt and "2. close fridge" in prompt
        assert prompt.rstrip().endswith("top candidate.")

        ranking, confidence = parse_response('```json\n{"ranking": ["take egg"], "confidence": 0.8}\n```')
        assert ranking == ["take egg"] and abs(confidence - 0.8) < 1e-9, (ranking, confidence)
        assert parse_response("no json here") == ([], None)

        support = ranking_to_support(["take egg"], ["take egg", "close fridge"], confidence=0.8)
        assert abs(support.sum() - 1.0) < 1e-9
        assert support[0] > support[1], support
        print("自检通过：类名表 / prompt / 解析 / 支持度")
        print("---- prompt 预览 ----")
        print(prompt[-320:])


if __name__ == "__main__":
    self_test()
    sys.exit(0)
