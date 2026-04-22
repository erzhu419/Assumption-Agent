"""
Extend wisdom_library.json with 25-30 entries targeting gaps:
  - Kahneman/Tversky (cognitive bias) — the cluster that 402'd in original build
  - Drucker/Porter (management + strategy) — we have 0 business-oriented aphorisms
  - Brooks/Knuth/Dijkstra (software engineering heuristics) — v5 losses show sw_eng gaps
  - Hanson/Carroll/Polya meta (scientific method pitfalls)

Writes wisdom_library_v2.json (merged: 75 original + new).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import parse_json_from_llm
from gpt5_client import GPT5Client
from build_wisdom_library import BATCH_PROMPT, validate_entry

OUT_BASE = Path("/home/erzhu419/mine_code/Asumption Agent/phase two/analysis/cache/wisdom_library.json")
OUT_V2 = Path("/home/erzhu419/mine_code/Asumption Agent/phase two/analysis/cache/wisdom_library_v2.json")


EXTENSION_CLUSTERS = [
    {
        "name": "kahneman_tversky_bias",
        "description": "Kahneman、Tversky、Ariely 等的认知偏差与决策警觉（原 build 因 402 漏掉的簇）",
        "examples_to_include": [
            "我们高估眼前，低估远方（双曲贴现）",
            "让你舒服的答案通常是错的（confirmation bias）",
            "事后看起来理所当然，事前完全猜不到（hindsight bias）",
            "对等概率事件，人更怕损失（loss aversion）",
        ],
        "target_count": 8,
    },
    {
        "name": "drucker_porter_management",
        "description": "Drucker、Porter、Christensen 的管理与战略洞察",
        "examples_to_include": [
            "正确地做事 vs 做正确的事（Drucker）",
            "战略的本质是选择不做什么（Porter）",
            "卓越的执行杀死优质战略（Christensen）",
            "衡量什么，就得到什么（Drucker）",
        ],
        "target_count": 8,
    },
    {
        "name": "brooks_knuth_software",
        "description": "Brooks、Knuth、Dijkstra、Hoare 的软件工程深层警觉（必须抽象化，不能出现'代码/bug'等具体词）",
        "examples_to_include": [
            "添加人力反而延迟项目（Brooks 法则）",
            "过早优化是万恶之源（Knuth）",
            "简单比复杂难（Dijkstra）",
            "让错误无处可藏（Hoare）",
        ],
        "target_count": 8,
    },
    {
        "name": "scientific_method_pitfalls",
        "description": "科学方法中的认知陷阱（Hanson、Popper 反例、Feynman cargo cult、数据 snooping）",
        "examples_to_include": [
            "观察本身就被理论渗透（Hanson, theory-laden）",
            "数据挖得够久，任何 pattern 都会出现（data snooping）",
            "Cargo cult science: 形式对了但缺核心（Feynman）",
            "实验设计决定了你能看到什么",
        ],
        "target_count": 6,
    },
]


def main():
    client = GPT5Client()
    original = json.loads(OUT_BASE.read_text(encoding="utf-8"))
    existing_aphorisms = {e["aphorism"] for e in original}
    print(f"Original library: {len(original)} entries")

    new_entries = []
    t0 = time.time()

    for cluster in EXTENSION_CLUSTERS:
        print(f"\n[{cluster['name']}] targeting {cluster['target_count']} entries")
        examples_text = "\n".join(f"  - {ex}" for ex in cluster["examples_to_include"])
        prompt = BATCH_PROMPT.format(
            source_name=cluster["name"],
            description=cluster["description"],
            examples_to_include=examples_text,
            count=cluster["target_count"],
        )
        try:
            resp = client.generate(prompt, max_tokens=5000, temperature=0.4)
            parsed = parse_json_from_llm(resp["text"])
            entries = parsed.get("entries", [])
        except Exception as e:
            print(f"  [error] {e}")
            continue

        kept = 0
        for entry in entries:
            ok, reason = validate_entry(entry)
            if not ok:
                print(f"  [reject] {entry.get('aphorism','?')[:30]} — {reason}")
                continue
            aph = entry.get("aphorism", "").strip()
            if aph in existing_aphorisms:
                print(f"  [dup] {aph}")
                continue
            existing_aphorisms.add(aph)
            entry["cluster"] = cluster["name"]
            entry["id"] = f"W{len(original) + len(new_entries) + 1:03d}"
            new_entries.append(entry)
            kept += 1
        print(f"  [ok] kept {kept}/{len(entries)}")

    merged = original + new_entries
    OUT_V2.write_text(json.dumps(merged, ensure_ascii=False, indent=2))

    print(f"\n=== Summary ===")
    print(f"  Original: {len(original)}  New: {len(new_entries)}  Total: {len(merged)}")
    print(f"  Saved to {OUT_V2.name}  ({time.time()-t0:.0f}s)")

    print("\n=== New entries by cluster ===")
    from collections import Counter
    c = Counter(e.get("cluster", "?") for e in new_entries)
    for k, v in c.most_common():
        print(f"  {k}: {v}")

    print("\n=== Sample new aphorisms ===")
    for e in new_entries[:10]:
        print(f"  [{e['id']}] {e['aphorism']} ({e['source']})")


if __name__ == "__main__":
    main()
