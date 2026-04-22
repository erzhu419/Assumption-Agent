"""
v11: compress each 40-80 char trigger to 20-30 char aphorism form via GPT-5.4.

Hypothesis (from GenericAgent paper §2.1): conciseness > completeness when
scaffolding already provides enough breadth. We're packing higher-density
info into a smaller slot.

Key: must preserve SIGNAL+ACTION pair. "场景触发 + 具体自问" 压到
"XXXX问YYY" 的四五字雏形——要保证 LLM 还能 decode (naturalness constraint).

Writes trigger_library_v11.json (same keys, compressed values).
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
OUT_PATH = CACHE / "trigger_library_v11.json"


COMPRESS_PROMPT = """你是 scaffold 信息密度优化专家。任务：把每条 40-80 字的 trigger 压缩为 22-30 字的"高密度警句 + 自问"形式。

## 输入 trigger 格式
"当 [情境] 时，反问：[具体自问]" 或类似

## 压缩目标
- **22-30 中文字符**（不能超过 30，不能少于 20）
- 保留"场景 → 自问"的双结构
- 语言自然（LLM 能 decode）、非电报式缩写
- 不要像标语口号，要像脑中自然浮现的警觉

## 压缩例子（你要学的风格）
- 原文："当发现自己反复朝同一个方向优化时，反问：优化目标本身是不是定错了？"
  压缩："单向优化撞墙时，先问目标本身有没有定错"（23 字）

- 原文："当团队在某个决策上反复拉锯，且每次都有新理由时，反问：大家真正的担忧是什么？"
  压缩："反复拉锯而理由常新，背后的真担忧是什么？"（21 字）

- 原文："在给出建议前，反问：对方真正想要的是答案，还是能让他自己做决定的框架？"
  压缩："给建议前：对方要答案，还是想要一套决策框架？"（23 字）

## 输入
以下是 {n} 条 trigger：
{triggers_json}

## 输出
JSON 格式，不要代码块：
{{"compressed": [
  "第 1 条压缩后 (22-30 字)",
  "第 2 条压缩后",
  ...
]}}

注意：条数必须与输入完全一致，顺序保持不变。
"""


def compress_batch(client, triggers, max_retries=3):
    for attempt in range(max_retries):
        try:
            prompt = COMPRESS_PROMPT.format(
                n=len(triggers),
                triggers_json=json.dumps(triggers, ensure_ascii=False, indent=2),
            )
            resp = client.generate(prompt, max_tokens=3000, temperature=0.3)
            parsed = parse_json_from_llm(resp["text"])
            compressed = parsed.get("compressed", [])
            if len(compressed) != len(triggers):
                raise ValueError(f"count mismatch: {len(compressed)} vs {len(triggers)}")
            return compressed
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"  [retry {attempt+1}] {e}")
            time.sleep(5)


def main():
    orig = json.loads(IN_PATH.read_text(encoding="utf-8"))
    total = sum(len(v) for v in orig.values())
    print(f"Original library: {total} triggers across {len(orig)} categories")

    client = GPT5Client()
    out = {}
    t0 = time.time()
    processed = 0
    rejected = 0

    for cat, triggers in orig.items():
        if not triggers:
            out[cat] = []
            continue
        print(f"\n[{cat}] {len(triggers)} triggers")
        compressed = compress_batch(client, triggers)

        # Validation: length must be in [20, 32]
        kept = []
        for orig_t, comp_t in zip(triggers, compressed):
            if not isinstance(comp_t, str):
                rejected += 1
                continue
            L = len(comp_t.strip())
            if L < 18 or L > 34:
                print(f"  [reject len={L}] {comp_t}")
                rejected += 1
                continue
            kept.append(comp_t.strip())
        out[cat] = kept
        processed += len(kept)
        print(f"  kept {len(kept)}/{len(triggers)}")

    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    new_total = sum(len(v) for v in out.values())
    print(f"\n=== Summary ===")
    print(f"  {total} original -> {new_total} compressed ({rejected} rejected) in {time.time()-t0:.0f}s")
    print(f"  Saved to {OUT_PATH.name}")

    # Show length stats
    lens = [len(t) for v in out.values() for t in v]
    if lens:
        print(f"  length: min={min(lens)} avg={sum(lens)/len(lens):.1f} max={max(lens)}")
    # Sample output
    print("\n=== Sample compressed ===")
    for cat in list(out.keys())[:3]:
        print(f"\n  [{cat}]")
        for t in out[cat][:3]:
            print(f"    - ({len(t)}字) {t}")


if __name__ == "__main__":
    main()
