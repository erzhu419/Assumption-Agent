"""
v12: build the L1∩L3 data files.

L1: triggers that are BOTH verified (v10) AND compressed (v11).
L3: wisdom entries with unpacked_for_llm compressed 60-120 -> 30-50 chars via GPT-5.4.

Writes:
  trigger_library_v12.json       (compressed + verified triggers, ~72 per category)
  wisdom_library_v12.json        (original entries with shortened unpacked_for_llm)
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "phase zero" / "scripts"))
from llm_client import parse_json_from_llm
from gpt5_client import GPT5Client


CACHE = Path("/home/erzhu419/mine_code/Asumption Agent/phase two/analysis/cache")
ORIG_TRIGGERS = CACHE / "trigger_library.json"
V10_VERIFIED = CACHE / "trigger_library_v10_verified.json"
V11_COMPRESSED = CACHE / "trigger_library_v11.json"
OUT_TRIGGERS = CACHE / "trigger_library_v12.json"

ORIG_WISDOM = CACHE / "wisdom_library.json"
OUT_WISDOM = CACHE / "wisdom_library_v12.json"


def build_trigger_intersection():
    """For each category, keep compressed form of only the verified-subset triggers."""
    orig = json.loads(ORIG_TRIGGERS.read_text(encoding="utf-8"))
    verified = json.loads(V10_VERIFIED.read_text(encoding="utf-8"))
    compressed = json.loads(V11_COMPRESSED.read_text(encoding="utf-8"))

    out = {}
    for cat in orig:
        orig_list = orig.get(cat, [])
        ver_set = set(verified.get(cat, []))
        comp_list = compressed.get(cat, [])

        # Match verified triggers back to original positions, then pull compressed form
        combo = []
        for i, t_orig in enumerate(orig_list):
            if t_orig in ver_set:
                # Length mismatch means position won't align directly
                if i < len(comp_list):
                    combo.append(comp_list[i])
        out[cat] = combo

    OUT_TRIGGERS.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    total = sum(len(v) for v in out.values())
    print(f"  [triggers] {sum(len(v) for v in verified.values())} verified ∩ compressed "
          f"= {total} kept")
    # Length stats
    lens = [len(t) for v in out.values() for t in v]
    if lens:
        print(f"  avg length: {sum(lens)/len(lens):.1f} chars")


COMPRESS_WISDOM_PROMPT = """你是 attention scaffold 压缩专家。请把下面 {n} 条 wisdom 条目的 `unpacked_for_llm` 字段（原 60-120 字的场景自问）压缩到 **30-50 中文字符**。

## 压缩原则
- 保留核心的 "scenario + self-question" 结构
- 保留 aphorism 原有的独特语感（避免沦为 generic 劝诫）
- 语言自然、LLM 可一眼 decode

## 例子
原 unpacked（80 字）："当发现自己反复朝同一方向优化某个指标时，先暂停并自问：我的优化目标本身定错了吗？这个方向是在解决真问题，还是在证明决定正确？"
压缩（37 字）："反复朝一方向优化时先问：目标定错没？是解决问题，还是证明决定？"

## 输入
{entries_json}

## 输出 JSON（不要代码块）
{{"compressed": [
  "第 1 条压缩 unpacked (30-50 字)",
  ...（条数与输入一致，顺序保持）
]}}
"""


def compress_wisdom(batch_size=15):
    wisdom = json.loads(ORIG_WISDOM.read_text(encoding="utf-8"))
    print(f"  [wisdom] {len(wisdom)} entries, compressing in batches of {batch_size}")

    client = GPT5Client()
    compressed_entries = []
    t0 = time.time()

    for i in range(0, len(wisdom), batch_size):
        batch = wisdom[i:i+batch_size]
        # Only show aphorism + unpacked to GPT-5.4 (enough context)
        show = [{"idx": j, "aphorism": e["aphorism"], "unpacked_for_llm": e["unpacked_for_llm"]}
                for j, e in enumerate(batch)]
        prompt = COMPRESS_WISDOM_PROMPT.format(
            n=len(batch),
            entries_json=json.dumps(show, ensure_ascii=False, indent=2),
        )
        for attempt in range(3):
            try:
                resp = client.generate(prompt, max_tokens=3000, temperature=0.3)
                parsed = parse_json_from_llm(resp["text"])
                comp_list = parsed.get("compressed", [])
                if len(comp_list) != len(batch):
                    raise ValueError(f"count mismatch: got {len(comp_list)}")
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"    [retry {attempt+1}] {e}")
                time.sleep(5)

        kept = 0
        for orig_e, comp_unp in zip(batch, comp_list):
            if not isinstance(comp_unp, str):
                continue
            L = len(comp_unp.strip())
            if L < 25 or L > 60:
                # Fall back to original if length out of range
                new_entry = dict(orig_e)
            else:
                new_entry = dict(orig_e)
                new_entry["unpacked_for_llm"] = comp_unp.strip()
                new_entry["unpacked_original"] = orig_e["unpacked_for_llm"]
                kept += 1
            compressed_entries.append(new_entry)
        print(f"  [batch {i//batch_size+1}] compressed {kept}/{len(batch)}, "
              f"total so far: {len(compressed_entries)}, {time.time()-t0:.0f}s")

    OUT_WISDOM.write_text(json.dumps(compressed_entries, ensure_ascii=False, indent=2))
    # Length stats
    lens = [len(e["unpacked_for_llm"]) for e in compressed_entries]
    print(f"  [wisdom] saved {len(compressed_entries)} entries. "
          f"unpacked len: avg={sum(lens)/len(lens):.1f} min={min(lens)} max={max(lens)}")


def main():
    print("=== Building v12 combo data ===")
    build_trigger_intersection()
    compress_wisdom()
    print("\n=== Done ===")
    print(f"  triggers: {OUT_TRIGGERS.name}")
    print(f"  wisdom:   {OUT_WISDOM.name}")


if __name__ == "__main__":
    main()
