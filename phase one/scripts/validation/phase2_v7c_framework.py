"""
Phase 2 v7c — expanded wisdom library (75 → 100 entries).

Only difference vs v5: wisdom pool is wisdom_library_v2.json (75 + 25 from
extension clusters: Kahneman bias, Drucker mgmt, Brooks/Knuth sw-eng heuristics,
scientific method pitfalls).

Re-runs Stage-A (per-problem selection) over the enlarged pool. Reuses v5's
Stage-1 priors + 161 triggers. Clean ablation: wisdom breadth only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))

import _config as cfg  # noqa: E402
from llm_client import create_client, parse_json_from_llm  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from cached_framework import _generate_with_retry
from phase2_framework import select_triggers_for_category
from phase2_v3_framework import SELECT_PROMPT, build_brief_library


CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
ANSWERS_DIR = CACHE / "answers"
STRUCTURES_DIR = CACHE / "structures"
WISDOM_V2_PATH = CACHE / "wisdom_library_v2.json"
SELECTIONS_V2_PATH = CACHE / "phase2_v7c_selections.json"
TRIGGERS_PATH = CACHE / "trigger_library.json"


EXECUTE_V7C = """# 你要解决下面的问题。

## 思维背景 1：本类别的通用 attention priors
{priors_block}

## 思维背景 2：从历史失败中沉淀的 category-specific 警觉
{triggers_block}

## 思维背景 3：可能适用的跨文明 wisdom（已 pre-unpacked，非强制引用）
{wisdom_block}

## 问题
{problem}

## 要求
- **不要**用 Step 1/2 格式
- 带这三层觉知直接答题；引用 aphorism 仅当真 fire
- 语言精炼，不超过 500 字

开始：
"""


def cache_load(p: Path):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def cache_save(p: Path, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def format_priors(priors: List[str]) -> str:
    return "\n".join(f"  {i+1}. {p}" for i, p in enumerate(priors)) or "  (无)"


def format_triggers(trs: List[str]) -> str:
    return "\n".join(f"  - {t}" for t in trs) or "  (无)"


def format_wisdom(selected: List[Dict]) -> str:
    if not selected:
        return "  (本题没有特别适用的 wisdom)"
    parts = []
    for e in selected:
        parts.append(
            f"  • 【{e['id']} — {e['source']}】《{e['aphorism']}》\n"
            f"      {e['unpacked_for_llm']}"
        )
    return "\n".join(parts)


def select_entries(client, problem: str, library: List[Dict]) -> List[str]:
    brief = build_brief_library(library)
    resp = _generate_with_retry(client, SELECT_PROMPT.format(
        problem=problem[:600], library_brief=brief, n=len(library)),
        max_tokens=400, temperature=0.2)
    try:
        parsed = parse_json_from_llm(resp["text"])
        ids = parsed.get("selected_ids", [])
        valid_ids = {e["id"] for e in library}
        return [s for s in ids if isinstance(s, str) and s in valid_ids][:5]
    except Exception:
        import re
        found = re.findall(r"W\d{3}", resp.get("text", ""))
        valid_ids = {e["id"] for e in library}
        return [s for s in found if s in valid_ids][:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="phase2_v7c")
    ap.add_argument("--base", default="orient_hybrid")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    answers_path = ANSWERS_DIR / f"{args.variant}_answers.json"
    struct_path = STRUCTURES_DIR / f"{args.variant}_structures.json"
    answers = cache_load(answers_path)
    selections = cache_load(SELECTIONS_V2_PATH)

    if not struct_path.exists():
        base_struct = cache_load(STRUCTURES_DIR / f"{args.base}_structures.json")
        if base_struct:
            cache_save(struct_path, base_struct)
    structures = cache_load(struct_path)

    library = json.loads(WISDOM_V2_PATH.read_text(encoding="utf-8"))
    lib_by_id = {e["id"]: e for e in library}
    triggers_db = cache_load(TRIGGERS_PATH)
    print(f"  wisdom_v2: {len(library)} entries, {len(selections)} pre-selected")
    print(f"  triggers: {sum(len(v) for v in triggers_db.values())} (v5 library)")

    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))[: args.n]

    client = create_client()
    t0 = time.time()
    new_count = 0
    hit_count = 0

    for i, p in enumerate(sample):
        pid = p["problem_id"]
        if pid in answers:
            hit_count += 1
            continue
        dom = p.get("domain", "?")
        diff = p.get("difficulty", "?")
        key = f"{dom}__{diff}"
        struct = structures.get(key)
        if struct is None:
            print(f"  [skip] {pid}: no structure for {key}")
            continue

        priors = struct.get("attention_priors", [])
        priors_block = format_priors(priors)
        triggers = select_triggers_for_category(triggers_db, dom, diff)
        triggers_block = format_triggers(triggers)

        # Stage-A: select from v2 library (re-run; 75 original selections are not valid for 100-entry pool)
        if pid in selections:
            sel_ids = selections[pid]
        else:
            try:
                sel_ids = select_entries(client, p.get("description", ""), library)
            except Exception as e:
                print(f"  [select-error] {pid}: {e}")
                continue
            selections[pid] = sel_ids
            if len(selections) % 10 == 0:
                cache_save(SELECTIONS_V2_PATH, selections)

        wisdom_entries = [lib_by_id[sid] for sid in sel_ids if sid in lib_by_id]
        wisdom_block = format_wisdom(wisdom_entries)

        prompt = EXECUTE_V7C.format(
            priors_block=priors_block,
            triggers_block=triggers_block,
            wisdom_block=wisdom_block,
            problem=p.get("description", ""),
        )

        try:
            resp = _generate_with_retry(client, prompt, max_tokens=900, temperature=0.3)
            answers[pid] = resp["text"].strip()
        except Exception as e:
            print(f"  [error] {pid}: {e}")
            continue

        new_count += 1
        if new_count % 10 == 0:
            cache_save(answers_path, answers)
            cache_save(SELECTIONS_V2_PATH, selections)
            print(f"  [{args.variant}] {i+1}/{len(sample)} {time.time()-t0:.0f}s")

    cache_save(answers_path, answers)
    cache_save(SELECTIONS_V2_PATH, selections)
    print(f"\n  [{args.variant}] done: {new_count} new, {hit_count} cached ({time.time()-t0:.0f}s)")

    # Which new entries (W076+) got picked
    from collections import Counter
    picks = Counter()
    for ids in selections.values():
        for sid in ids:
            picks[sid] += 1
    print("\n  Top 15 picked entries:")
    for sid, n in picks.most_common(15):
        if sid in lib_by_id:
            new_marker = "[NEW]" if int(sid[1:]) > 75 else "     "
            print(f"    {new_marker} {n:>3}x  [{sid}] {lib_by_id[sid]['aphorism']}")


if __name__ == "__main__":
    main()
