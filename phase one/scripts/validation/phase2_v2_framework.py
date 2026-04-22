"""
Phase 2 v2: mining v2 aphorism triggers, applied globally (no category-keying).

All 6 universal triggers are shown to every problem. LLM self-selects which
one(s) apply while generating the answer.

Reuses orient_hybrid's Stage-1 structures (same seed, same pipeline).
Variant name: phase2_v2_triggers
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
TRIGGERS_V2_PATH = CACHE / "trigger_library_v2.json"
ANSWERS_DIR = CACHE / "answers"
STRUCTURES_DIR = CACHE / "structures"


EXECUTE_V2 = """# 你要解决下面的问题。

## 跨领域警句（在任何问题上都可能适用，读完要自己判断哪几条真正触发）

{triggers_block}

## 任务类别的 attention priors

{priors_block}

## 问题

{problem}

## 要求

- **不要**用 "Step 1、Step 2" 格式
- 读问题时，让上面警句在脑中浮现。如果某条真的 fire 了（不只是"感觉沾边"），在答案里**明确提及它如何塑形了你的思考**（举例："'大数不吓人，占比才吓人' 让我先算出 X%..."）
- 不适用就不引用，别硬套
- 带着觉知直接回答，语言精炼，不超过 500 字

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="phase2_v2_triggers")
    ap.add_argument("--base", default="orient_hybrid",
                    help="variant whose structures to reuse")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    answers_path = ANSWERS_DIR / f"{args.variant}_answers.json"
    struct_path = STRUCTURES_DIR / f"{args.variant}_structures.json"
    answers = cache_load(answers_path)

    # Reuse base structures
    if not struct_path.exists():
        base_struct = cache_load(STRUCTURES_DIR / f"{args.base}_structures.json")
        if base_struct:
            cache_save(struct_path, base_struct)
            print(f"  [reused] {args.base} structures ({len(base_struct)} categories)")
    structures = cache_load(struct_path)

    if not TRIGGERS_V2_PATH.exists():
        print("ERROR: trigger_library_v2.json not found. Run mine_triggers_v2.py first.")
        return

    triggers_v2 = json.loads(TRIGGERS_V2_PATH.read_text(encoding="utf-8"))
    print(f"  loaded {len(triggers_v2)} aphorism triggers:")
    for t in triggers_v2:
        print(f"    - {t['trigger']}  (insight: {t['core_insight'][:60]})")

    triggers_block = "\n".join(
        f"  • 【警句 {i+1}】{t['trigger']}\n"
        f"      （含义：{t['core_insight']}）"
        for i, t in enumerate(triggers_v2)
    )

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
        key = f"{p.get('domain', '?')}__{p.get('difficulty', '?')}"
        struct = structures.get(key)
        if struct is None:
            print(f"  [skip] {pid}: no structure for {key}")
            continue
        priors = struct.get("attention_priors", [])
        priors_block = "\n".join(f"  {j+1}. {pr}" for j, pr in enumerate(priors))
        try:
            resp = _generate_with_retry(
                client,
                EXECUTE_V2.format(
                    triggers_block=triggers_block,
                    priors_block=priors_block,
                    problem=p.get("description", ""),
                ),
                max_tokens=900, temperature=0.3,
            )
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
