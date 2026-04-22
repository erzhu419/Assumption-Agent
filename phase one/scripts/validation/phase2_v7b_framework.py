"""
Phase 2 v7b — domain-gated architecture.

Motivation: v6 data showed trigger/wisdom breadth helps soft domains (business,
daily_life, sw_eng) but HURTS hard-science domains (math, science) — likely
because concept-specific triggers become noise when they don't match.

Design:
  - domain ∈ {mathematics, science}: THIN = Stage-1 priors only
  - else:                             FULL = v5 stack (priors + 161 triggers + wisdom)

Keeps v5's 161-trigger library (not v6's 301) for FULL path — cleanest ablation
isolating "domain gating" as the only change vs v5.
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
TRIGGERS_PATH = CACHE / "trigger_library.json"

THIN_DOMAINS = {"mathematics", "science"}


EXECUTE_THIN = """# 你要解决下面的问题。

## 思维背景：本类别的通用 attention priors
{priors_block}

## 问题
{problem}

## 要求
- **不要**用 Step 1/2 格式，带上面 priors 直接答题
- 语言精炼，不超过 500 字

开始：
"""


EXECUTE_FULL = """# 你要解决下面的问题。

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="phase2_v7b")
    ap.add_argument("--base", default="orient_hybrid")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    answers_path = ANSWERS_DIR / f"{args.variant}_answers.json"
    struct_path = STRUCTURES_DIR / f"{args.variant}_structures.json"
    answers = cache_load(answers_path)

    if not struct_path.exists():
        base_struct = cache_load(STRUCTURES_DIR / f"{args.base}_structures.json")
        if base_struct:
            cache_save(struct_path, base_struct)
    structures = cache_load(struct_path)

    library = json.loads(WISDOM_PATH.read_text(encoding="utf-8"))
    lib_by_id = {e["id"]: e for e in library}
    selections = cache_load(SELECTIONS_PATH)
    triggers_db = cache_load(TRIGGERS_PATH)
    print(f"  wisdom: {len(library)}, triggers: {sum(len(v) for v in triggers_db.values())} (v5 library)")
    print(f"  THIN domains: {THIN_DOMAINS}")

    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))[: args.n]

    client = create_client()
    t0 = time.time()
    new_count = 0
    hit_count = 0
    thin_count = full_count = 0

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

        if dom in THIN_DOMAINS:
            prompt = EXECUTE_THIN.format(
                priors_block=priors_block, problem=p.get("description", ""))
            thin_count += 1
        else:
            triggers = select_triggers_for_category(triggers_db, dom, diff)
            triggers_block = format_triggers(triggers)
            sel_ids = selections.get(pid, [])
            wisdom_entries = [lib_by_id[sid] for sid in sel_ids if sid in lib_by_id]
            wisdom_block = format_wisdom(wisdom_entries)
            prompt = EXECUTE_FULL.format(
                priors_block=priors_block,
                triggers_block=triggers_block,
                wisdom_block=wisdom_block,
                problem=p.get("description", ""),
            )
            full_count += 1

        try:
            resp = _generate_with_retry(client, prompt, max_tokens=900, temperature=0.3)
            answers[pid] = resp["text"].strip()
        except Exception as e:
            print(f"  [error] {pid}: {e}")
            continue

        new_count += 1
        if new_count % 10 == 0:
            cache_save(answers_path, answers)
            print(f"  [{args.variant}] {i+1}/{len(sample)} thin={thin_count} full={full_count} {time.time()-t0:.0f}s")

    cache_save(answers_path, answers)
    print(f"\n  [{args.variant}] done: thin={thin_count} full={full_count} {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
