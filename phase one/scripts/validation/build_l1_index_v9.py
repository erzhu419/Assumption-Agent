"""
v9: build L1-index for trigger library — GenericAgent-style compression.

For each category, group 161-library triggers into 5-7 "signal classes":
  {"label": "沉没成本自察" (6-10 字),
   "hint": "反复投入后如何识别自欺" (15-25 字)}

In EXECUTE prompt, show only these labels + hints (L1 index, ~200 chars/category).
LLM sees the EXISTENCE of signal types, not the unpacked content.

Paper §2.3.2: "The always-on layer can stay minimal because it needs to encode only
the existence of each knowledge category; the LLM itself serves as both compressor
and decoder."

Writes trigger_library_v9_l1index.json.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "phase zero" / "scripts"))
from llm_client import parse_json_from_llm
from gpt5_client import GPT5Client


CACHE = Path("/home/erzhu419/mine_code/Asumption Agent/phase two/analysis/cache")
IN_PATH = CACHE / "trigger_library.json"
OUT_PATH = CACHE / "trigger_library_v9_l1index.json"


INDEX_PROMPT = """你是 attention scaffold 的 L1-index 设计师。任务：把一类 trigger 抽象为 5-7 条 "signal label" ——记录"这类信号存在"而非"信号的内容"。

## 设计原则（来自 GenericAgent §2.3.2）
L1 index 是 LLM 的认知目录——不展开细节，只让 LLM 知道"在这个类别下，存在这些类型的警觉"。LLM 自己会把相关的展开，不相关的会自然忽略。

## 输入 trigger 集合（类别 {cat}，共 {n} 条）
{triggers_json}

## 输出要求
- 把 {n} 条 trigger 抽象为 **5-7 条** signal label
- 每条 label 两字段：
  - `label`: 6-10 中文字符，名词性，概括信号类别（例："沉没成本自察"、"反例驱动检验"、"尺度分解断裂"）
  - `hint`: 15-25 中文字符，说明何时该激活（例："反复投入后，辨认是否在自欺"）

## 反例（不要这样做）
- ❌ 动词开头："检查你是否..." (label 是信号类，不是动作)
- ❌ 过长："当你发现系统性能在..." (label 不是 trigger 本身)
- ❌ 过于抽象："认知偏差类" (要具体到可以激活一类反应)

## 好例子
```
{{"label": "沉没成本自察", "hint": "反复投入后辨认自我欺骗"}}
{{"label": "反例驱动检验", "hint": "方案看似稳时主动找能击穿它的情境"}}
{{"label": "尺度分解断裂", "hint": "微观和宏观切换时检查连接变量"}}
```

## 输出 JSON（不要代码块）
{{
  "labels": [
    {{"label": "...", "hint": "..."}},
    ... (5-7 条)
  ]
}}
"""


def build_index_for_category(client, cat, triggers, max_retries=3):
    for attempt in range(max_retries):
        try:
            prompt = INDEX_PROMPT.format(
                cat=cat,
                n=len(triggers),
                triggers_json=json.dumps(triggers, ensure_ascii=False, indent=2),
            )
            resp = client.generate(prompt, max_tokens=1500, temperature=0.3)
            parsed = parse_json_from_llm(resp["text"])
            labels = parsed.get("labels", [])
            valid = []
            for item in labels:
                if not isinstance(item, dict):
                    continue
                L = item.get("label", "").strip()
                H = item.get("hint", "").strip()
                if not (4 <= len(L) <= 14) or not (12 <= len(H) <= 30):
                    continue
                valid.append({"label": L, "hint": H})
            if 4 <= len(valid) <= 8:
                return valid
            raise ValueError(f"got {len(valid)} valid labels (need 4-8)")
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"  [retry {attempt+1}] {e}")
            time.sleep(5)


def main():
    orig = json.loads(IN_PATH.read_text(encoding="utf-8"))
    print(f"Input library: {sum(len(v) for v in orig.values())} triggers "
          f"across {len(orig)} categories")

    client = GPT5Client()
    out = {}
    t0 = time.time()

    for cat, triggers in orig.items():
        if not triggers:
            out[cat] = []
            continue
        print(f"\n[{cat}] {len(triggers)} triggers -> index")
        labels = build_index_for_category(client, cat, triggers)
        out[cat] = labels
        print(f"  -> {len(labels)} labels")
        for item in labels:
            print(f"    • [{item['label']}] {item['hint']}")

    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    total = sum(len(v) for v in out.values())
    print(f"\n=== Summary ===")
    print(f"  {len(orig)} categories -> {total} L1 labels total ({time.time()-t0:.0f}s)")
    print(f"  avg per category: {total/len(orig):.1f}")
    print(f"  Saved to {OUT_PATH.name}")


if __name__ == "__main__":
    main()
