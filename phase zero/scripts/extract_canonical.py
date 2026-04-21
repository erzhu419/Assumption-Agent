"""
Extract canonical heuristic questions from Polya/Popper source texts via LLM.

For each of 12 strategies, ask Gemini to reproduce (as faithfully as possible)
the original heuristic questions or orientation phrasing from:
  - Polya, "How to Solve It" (1945) — for 10 strategies
  - Popper, "The Logic of Scientific Discovery" (1959) — for S08 S13

Output: phase zero/kb/strategies_canonical/S*.json with:
  - heuristic_questions_en (EXACT quotes from source if LLM knows them)
  - heuristic_questions_zh (faithful translation)
  - source with author/work/year/section
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import create_client, parse_json_from_llm

KB_DIR = Path("/home/erzhu419/mine_code/Asumption Agent/phase zero/kb/strategies")
OUT_DIR = Path("/home/erzhu419/mine_code/Asumption Agent/phase zero/kb/strategies_canonical")
OUT_DIR.mkdir(parents=True, exist_ok=True)


STRATEGY_SOURCE_MAP = {
    "S02": ("Polya, George", "How to Solve It", 1945, "Second Main Part (heuristic dictionary): 'Decomposing and recombining'"),
    "S03": ("Polya, George", "How to Solve It", 1945, "First Main Part: 'Have you seen it before?' / 'Analogy'"),
    "S04": ("Polya, George", "How to Solve It", 1945, "Heuristic dictionary: 'Reductio ad absurdum and indirect proof'"),
    "S06": ("Polya, George", "How to Solve It", 1945, "Heuristic dictionary: 'Specialization' / 'Working backwards'"),
    "S08": ("Popper, Karl R.", "The Logic of Scientific Discovery", 1959, "Chapters on conjecture and refutation"),
    "S09": ("Polya, George", "How to Solve It", 1945, "Heuristic dictionary: 'Auxiliary problem' / 'Simplification'"),
    "S10": ("Polya, George", "How to Solve It", 1945, "Heuristic dictionary: 'Symmetry' / 'Invariants'"),
    "S13": ("Popper, Karl R.", "The Logic of Scientific Discovery", 1959, "§6 Falsifiability as criterion of demarcation"),
    "S14": ("Polya, George", "How to Solve It", 1945, "Heuristic dictionary: 'Auxiliary elements' / 'Limiting cases'"),
    "S18": ("Polya, George", "How to Solve It", 1945, "Heuristic dictionary: 'Generalization'"),
    "S19": ("Polya, George", "How to Solve It", 1945, "Heuristic dictionary: 'Condition' / 'Could you drop a condition?'"),
    "S22": ("Polya, George", "How to Solve It", 1945, "First Main Part: 'Restate the problem' / 'Variation of the problem'"),
}


EXTRACT_PROMPT = """你要为一条方法论策略提取**原典中的 heuristic questions / orientation phrasing**。

## 策略
ID: {sid}
中文名: {name_zh}
英文名: {name_en}
一句话描述: {one_sentence}

## 来源
作者: {author}
作品: {work} ({year})
相关章节: {section}

## 你的任务
你记忆里该来源对这条方法论的**原文启发式问句 / 原始表达**是什么？

要求：
1. 给出 3-6 条**尽可能还原原文**的 heuristic questions（如 Polya 的 "Have you seen it before? Or have you seen the same problem in a slightly different form?"）
2. 每条都是**问句或祈使句**（orientation form），不是陈述或步骤
3. 英文版**忠于原文**，中文版**准确翻译**（不要意译发挥）
4. 如果是 Popper，找的是他针对 falsification / conjecture / bold hypothesis 的典型用语

不确定时可以标注（"approximate"），但要尽量还原。

输出 JSON（不要代码块）：
{{
  "heuristic_questions_en": ["...", "..."],
  "heuristic_questions_zh": ["...", "..."],
  "fidelity_note": "几条是原文 verbatim / 几条是 approximate reconstruction"
}}
"""


def extract_for_strategy(client, sid: str):
    src_path = list(KB_DIR.glob(f"{sid}_*.json"))[0]
    d = json.loads(src_path.read_text(encoding="utf-8"))

    author, work, year, section = STRATEGY_SOURCE_MAP[sid]

    prompt = EXTRACT_PROMPT.format(
        sid=sid,
        name_zh=d["name"]["zh"],
        name_en=d["name"].get("en", ""),
        one_sentence=d["description"]["one_sentence"],
        author=author, work=work, year=year, section=section,
    )

    for attempt in range(4):
        try:
            resp = client.generate(prompt, max_tokens=800, temperature=0.1)
            parsed = parse_json_from_llm(resp["text"])
            break
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(5 * (2 ** attempt))
            print(f"    retry {attempt+1}...")

    # Build canonical JSON
    canonical = {
        "id": sid,
        "name": d["name"],
        "aliases": d.get("aliases", []),
        "category": d.get("category", ""),
        "form": "canonical_orientation",
        "source": {
            "author": author, "work": work, "year": year, "section": section,
        },
        "description": {
            "one_sentence": d["description"]["one_sentence"],
            "background": d["description"].get("detailed", "")[:400],
        },
        "heuristic_questions_en": parsed.get("heuristic_questions_en", []),
        "heuristic_questions_zh": parsed.get("heuristic_questions_zh", []),
        "fidelity_note": parsed.get("fidelity_note", ""),
        "original_source_references": d.get("source_references", []),
        "applicability_conditions": d.get("applicability_conditions", {}),
        "metadata": {
            **(d.get("metadata", {})),
            "extracted_at": "2026-04-22",
            "form_version": "canonical_orientation_v1",
        },
    }

    out_path = OUT_DIR / src_path.name
    out_path.write_text(json.dumps(canonical, ensure_ascii=False, indent=2))
    return out_path, canonical


def main():
    client = create_client()
    print(f"Extracting canonical heuristic questions for {len(STRATEGY_SOURCE_MAP)} strategies...")
    t0 = time.time()
    for sid in sorted(STRATEGY_SOURCE_MAP.keys()):
        out_path, canonical = extract_for_strategy(client, sid)
        n_en = len(canonical["heuristic_questions_en"])
        n_zh = len(canonical["heuristic_questions_zh"])
        print(f"  ✓ {sid}: en={n_en} zh={n_zh} — {canonical.get('fidelity_note', '')[:80]}")
        if n_en > 0:
            print(f"    first EN: {canonical['heuristic_questions_en'][0]}")
    print(f"\nDone in {time.time()-t0:.0f}s. Written to {OUT_DIR}")


if __name__ == "__main__":
    main()
