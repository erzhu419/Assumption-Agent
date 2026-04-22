"""
Phase 2 v6 — iterate on v5 via v5-loss-derived triggers.

Same 3-layer stack as v5 (Stage-1 priors + category triggers + wisdom), but:
  - trigger_library_v6.json  (161 original + 140 mined from v5 losses = 301)
  - max_per_category = 6     (v5 used 4; widen so new targeted triggers surface)

Rationale: v5's remaining 48 losses were re-mined via mine_v5_losses.py. The new
triggers are more targeted (aimed at specific v5-residual failure modes). With
max=4, new triggers would be pushed out by the older 161 in each category. We
widen the retrieval slot to give them a chance to fire.
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
from llm_client import create_client  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from cached_framework import _generate_with_retry
from phase2_framework import select_triggers_for_category


CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
ANSWERS_DIR = CACHE / "answers"
STRUCTURES_DIR = CACHE / "structures"
WISDOM_PATH = CACHE / "wisdom_library.json"
SELECTIONS_PATH = CACHE / "phase2_v3_selections.json"
TRIGGERS_V6_PATH = CACHE / "trigger_library_v6.json"
TRIGGERS_V5_PATH = CACHE / "trigger_library.json"


EXECUTE_V6 = """# 你要解决下面的问题。

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


def select_v6_triggers(v6_db: Dict[str, List[str]], v5_db: Dict[str, List[str]],
                       domain: str, difficulty: str, max_total: int = 6) -> List[str]:
    """Interleave: half from NEW (v6 \ v5), half from ORIGINAL (v5).

    Ensures new targeted triggers actually surface without discarding proven ones.
    """
    cat = f"{domain}__{difficulty}"
    v5_set = set(v5_db.get(cat, []))
    new_triggers = [t for t in v6_db.get(cat, []) if t not in v5_set]
    old_triggers = [t for t in v6_db.get(cat, []) if t in v5_set]

    # Target: up to half new, half old. If new < quota, fill with old.
    half = max_total // 2
    picked_new = new_triggers[:half]
    remaining = max_total - len(picked_new)
    picked_old = old_triggers[:remaining]

    result = picked_new + picked_old

    # If still short (rare small categories), fallback to global v6 retrieval
    if len(result) < max_total:
        fallback = select_triggers_for_category(v6_db, domain, difficulty,
                                                max_per_category=max_total)
        seen = {t[:30] for t in result}
        for t in fallback:
            if len(result) >= max_total:
                break
            if t[:30] not in seen:
                result.append(t)
                seen.add(t[:30])
    return result[:max_total]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="phase2_v6")
    ap.add_argument("--base", default="orient_hybrid",
                    help="which variant's Stage-1 structures to reuse")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-triggers", type=int, default=6)
    args = ap.parse_args()

    answers_path = ANSWERS_DIR / f"{args.variant}_answers.json"
    struct_path = STRUCTURES_DIR / f"{args.variant}_structures.json"
    answers = cache_load(answers_path)

    if not struct_path.exists():
        base_struct = cache_load(STRUCTURES_DIR / f"{args.base}_structures.json")
        if base_struct:
            cache_save(struct_path, base_struct)
            print(f"  [reused] {args.base} structures ({len(base_struct)} categories)")
    structures = cache_load(struct_path)

    library = json.loads(WISDOM_PATH.read_text(encoding="utf-8"))
    lib_by_id = {e["id"]: e for e in library}
    selections = cache_load(SELECTIONS_PATH)
    print(f"  wisdom library: {len(library)} entries, "
          f"{len(selections)} problems pre-selected")

    v6_db = json.loads(TRIGGERS_V6_PATH.read_text(encoding="utf-8"))
    v5_db = json.loads(TRIGGERS_V5_PATH.read_text(encoding="utf-8"))
    v6_total = sum(len(v) for v in v6_db.values())
    v5_total = sum(len(v) for v in v5_db.values())
    print(f"  trigger library: {v5_total} (v5) -> {v6_total} (v6), +{v6_total-v5_total} new")
    print(f"  max triggers per problem: {args.max_triggers} (half new, half proven)")

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

        triggers = select_v6_triggers(v6_db, v5_db, dom, diff,
                                       max_total=args.max_triggers)
        triggers_block = format_triggers(triggers)

        sel_ids = selections.get(pid, [])
        wisdom_entries = [lib_by_id[sid] for sid in sel_ids if sid in lib_by_id]
        wisdom_block = format_wisdom(wisdom_entries)

        prompt = EXECUTE_V6.format(
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
            print(f"  [{args.variant}] {i+1}/{len(sample)} "
                  f"(new={new_count} hit={hit_count}) {time.time()-t0:.0f}s")

    cache_save(answers_path, answers)
    print(f"\n  [{args.variant}] total: new={new_count} hit={hit_count} "
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
