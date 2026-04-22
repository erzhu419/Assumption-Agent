"""
Phase 2 v9 — L1-index scaffold (GenericAgent paper §2.3.2).

Replaces 161-trigger full unpack with ~6-label compact index per category.
Each label = {"label": 6-10 char name, "hint": 15-25 char activation cue}.

Hypothesis: LLM is both compressor AND decoder. Showing only "这类信号存在"
lets the LLM fire relevant patterns without token-budget dilution.

Rest of stack identical to v5 (Stage-1 priors + wisdom unchanged).
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


CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
ANSWERS_DIR = CACHE / "answers"
STRUCTURES_DIR = CACHE / "structures"
WISDOM_PATH = CACHE / "wisdom_library.json"
SELECTIONS_PATH = CACHE / "phase2_v3_selections.json"
L1_INDEX_PATH = CACHE / "trigger_library_v9_l1index.json"


EXECUTE_V9 = """# 你要解决下面的问题。

## 思维背景 1：本类别的通用 attention priors
{priors_block}

## 思维背景 2：本类别常出现的信号类型（L1 index，非穷举）
{index_block}
（这些是"这里有一类警觉"的提示，不是具体指令——如果某条和当前问题真的共鸣，你会自然用到；否则忽略。）

## 思维背景 3：可能适用的跨文明 wisdom
{wisdom_block}

## 问题
{problem}

## 要求
- **不要**用 Step 1/2 格式
- 带这三层觉知直接答题
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


def format_l1_index(labels: List[Dict]) -> str:
    if not labels:
        return "  (无)"
    return "\n".join(f"  • [{l['label']}] — {l['hint']}" for l in labels)


def format_wisdom(selected: List[Dict]) -> str:
    if not selected:
        return "  (本题没有特别适用的 wisdom)"
    return "\n".join(
        f"  • 【{e['id']} — {e['source']}】《{e['aphorism']}》\n"
        f"      {e['unpacked_for_llm']}"
        for e in selected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="phase2_v9")
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
    l1_index = json.loads(L1_INDEX_PATH.read_text(encoding="utf-8"))
    total_labels = sum(len(v) for v in l1_index.values())
    print(f"  wisdom: {len(library)}  L1-index: {total_labels} labels across {len(l1_index)} cats")

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
        labels = l1_index.get(key, [])
        sel_ids = selections.get(pid, [])
        wisdom_entries = [lib_by_id[sid] for sid in sel_ids if sid in lib_by_id]

        prompt = EXECUTE_V9.format(
            priors_block=format_priors(priors),
            index_block=format_l1_index(labels),
            wisdom_block=format_wisdom(wisdom_entries),
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
            print(f"  [{args.variant}] {i+1}/{len(sample)} {time.time()-t0:.0f}s")

    cache_save(answers_path, answers)
    print(f"\n  [{args.variant}] done: {new_count} new, {hit_count} cached ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
