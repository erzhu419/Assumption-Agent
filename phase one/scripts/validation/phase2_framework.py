"""
Phase 2 改造 — variant `phase2_triggers`.

Extends orientation_framework by injecting failure-mined awareness triggers
(from trigger_library.json) into the EXECUTE prompt. Per-category triggers
act as pre-attention priors — reminders of "what we historically missed here".

Uses the SAME Stage 0/1 orientation structures cached from orient_hybrid.
Only EXECUTE prompt changes.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))

import _config as cfg  # noqa: E402
from llm_client import create_client, parse_json_from_llm  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from orientation_framework import (
    discover_structure_orient, load_orient_modules
)
from cached_framework import _generate_with_retry

CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
TRIGGERS_PATH = CACHE / "trigger_library.json"
ANSWERS_DIR = CACHE / "answers"
STRUCTURES_DIR = CACHE / "structures"


EXECUTE_WITH_TRIGGERS = """# 你要解决下面的问题。

## 在开始前请保持以下"觉知"（不是步骤，是注意力方向）：

### 从历史失败案例中积累的警觉（**这些是前面类似问题上我们错过的信号**）：
{triggers_block}

### 本任务类别的通用注意力先验：
{priors_block}

## 问题
{problem}

## 要求
- **不要**用 "Step 1、Step 2" 格式
- 带着上面的觉知直接回答
- 如果某个警觉让你发现了答案里的盲点，在结尾用一两句点出
- 语言精炼，不超过 500 字

请直接开始你的答案：
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


def select_triggers_for_category(all_triggers: Dict[str, List[str]],
                                  domain: str, difficulty: str,
                                  max_per_category: int = 4) -> List[str]:
    """Pick up to N diverse triggers for this category.

    Priority: exact category match → same domain → same difficulty → global.
    Deduplicate by first 30 chars.
    """
    seen = set()
    result: List[str] = []

    def take_from(key: str):
        for t in all_triggers.get(key, []):
            if len(result) >= max_per_category:
                return
            prefix = t[:30]
            if prefix in seen:
                continue
            seen.add(prefix)
            result.append(t)

    # Exact
    take_from(f"{domain}__{difficulty}")
    # Same domain, other difficulties
    for diff in ("easy", "medium", "hard"):
        if diff == difficulty:
            continue
        take_from(f"{domain}__{diff}")
    # Same difficulty, other domains
    for d in ("business", "daily_life", "engineering", "mathematics", "science", "software_engineering"):
        if d == domain:
            continue
        take_from(f"{d}__{difficulty}")
    return result


def execute_with_triggers(client, problem: str, structure: Dict,
                           triggers: List[str]) -> str:
    priors = structure.get("attention_priors", [])
    priors_block = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(priors))
    triggers_block = "\n".join(f"  - {t}" for t in triggers) if triggers else "  (无历史警觉)"
    r = _generate_with_retry(client,
        EXECUTE_WITH_TRIGGERS.format(
            triggers_block=triggers_block,
            priors_block=priors_block,
            problem=problem,
        ), max_tokens=900, temperature=0.3)
    return r["text"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="phase2_triggers")
    ap.add_argument("--base", default="orient_hybrid",
                    help="which cached variant's structures to reuse (default orient_hybrid)")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    answers_path = ANSWERS_DIR / f"{args.variant}_answers.json"
    struct_path = STRUCTURES_DIR / f"{args.variant}_structures.json"
    answers = cache_load(answers_path)
    # Reuse base variant structures (same Stage 0/1) unless already cached for this variant
    if not struct_path.exists():
        base_struct = cache_load(STRUCTURES_DIR / f"{args.base}_structures.json")
        if base_struct:
            cache_save(struct_path, base_struct)
            print(f"  [reused] copied {args.base} structures ({len(base_struct)} categories)")
    structures = cache_load(struct_path)

    if not TRIGGERS_PATH.exists():
        print("ERROR: trigger_library.json not found. Run mine_triggers.py first.")
        return
    triggers_db = json.loads(TRIGGERS_PATH.read_text(encoding="utf-8"))
    print(f"  triggers loaded: {sum(len(v) for v in triggers_db.values())} across {len(triggers_db)} categories")

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
            print(f"  [skip-answer] {pid}: no structure for {key}")
            continue
        triggers = select_triggers_for_category(
            triggers_db, p.get("domain", "?"), p.get("difficulty", "?"))
        answers[pid] = execute_with_triggers(
            client, p.get("description", ""), struct, triggers)
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
