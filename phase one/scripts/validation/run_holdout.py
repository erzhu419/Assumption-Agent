"""
Phase A: Run baseline / v5 / v11 / v12c on held-out 50 problems, judge, report.

Uses sample_holdout_50.json as problem pool (50 problems disjoint from sample_100).
Writes to _holdout50 suffixed files to preserve original 100-problem artifacts:
  answers/{variant}_holdout50_answers.json
  judgments/{a}_vs_{b}_holdout50.json

Goal: does v12c's +14pp advantage over v11 replicate on unseen problems?
"""

from __future__ import annotations

import argparse
import json
import random
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
from cached_framework import (_generate_with_retry, BASELINE_PROMPT, judge_pair,
                              CACHE_ROOT)
from phase2_framework import select_triggers_for_category
from phase2_v12_framework import (EXECUTE_FULL, EXECUTE_MATH, EXECUTE_SCIENCE,
                                   format_priors, format_triggers, format_wisdom,
                                   get_variant_config, MATH_SCI)
from phase2_v45_framework import EXECUTE_V5
from phase2_v11_framework import EXECUTE_V11

ANSWERS_DIR = CACHE_ROOT / "answers"
JUDGMENTS_DIR = CACHE_ROOT / "judgments"
STRUCTURES_DIR = CACHE_ROOT / "structures"
HOLDOUT_PATH = CACHE_ROOT / "sample_holdout_50.json"

WISDOM_PATH = CACHE_ROOT / "wisdom_library.json"
SELECTIONS_PATH = CACHE_ROOT / "phase2_v3_selections.json"
TRIGGERS_V5 = CACHE_ROOT / "trigger_library.json"
TRIGGERS_V11 = CACHE_ROOT / "trigger_library_v11.json"


def cache_load(p):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def cache_save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def gen_answers(variant: str, problems: List[Dict], client):
    """Generate answers for one variant on the held-out problems."""
    answers_path = ANSWERS_DIR / f"{variant}_holdout50_answers.json"
    answers = cache_load(answers_path)
    t0 = time.time()
    new = hit = 0

    # Load per-variant config
    if variant == "baseline":
        pass  # baseline just uses BASELINE_PROMPT
    elif variant in ("phase2_v5", "phase2_v11", "phase2_v12c"):
        structures = cache_load(STRUCTURES_DIR / "phase2_v5_structures.json") or \
                     cache_load(STRUCTURES_DIR / "orient_hybrid_structures.json")
        library = json.loads(WISDOM_PATH.read_text(encoding="utf-8"))
        lib_by_id = {e["id"]: e for e in library}

        # For held-out problems, selections don't exist — would need stage-A.
        # Pragmatic simplification: pick 3 wisdom entries by DOMAIN match (reuse
        # existing selections as pool, sample by domain frequency). This is a
        # best-effort since full re-selection costs 50 extra LLM calls.
        selections = cache_load(SELECTIONS_PATH)

        if variant == "phase2_v5":
            triggers_db = json.loads(TRIGGERS_V5.read_text(encoding="utf-8"))
        else:
            triggers_db = json.loads(TRIGGERS_V11.read_text(encoding="utf-8"))

    for i, p in enumerate(problems):
        pid = p["problem_id"]
        if pid in answers:
            hit += 1
            continue
        dom = p.get("domain", "?")
        diff = p.get("difficulty", "?")
        problem = p.get("description", "")

        if variant == "baseline":
            prompt = BASELINE_PROMPT.format(problem=problem)
            max_tok = 800
        elif variant == "phase2_v12c" and dom in MATH_SCI:
            if dom == "mathematics":
                prompt = EXECUTE_MATH.format(problem=problem)
                max_tok = 1100
            else:
                prompt = EXECUTE_SCIENCE.format(problem=problem)
                max_tok = 900
        else:
            # Full stack (phase2_v5 / phase2_v11 / phase2_v12c non-math)
            key = f"{dom}__{diff}"
            struct = structures.get(key, {"attention_priors": []})
            priors = struct.get("attention_priors", [])
            triggers = select_triggers_for_category(triggers_db, dom, diff)
            # wisdom: for held-out problems, pick 3 most commonly-used wisdom entries
            # in that domain's existing selections (cheap heuristic; avoids extra API calls)
            from collections import Counter
            dom_pids = [q["problem_id"] for q in problems if q.get("domain") == dom]
            # Look at all existing selections
            wisdom_pool = Counter()
            for qid, ids in selections.items():
                for wid in ids:
                    wisdom_pool[wid] += 1
            top_wisdom_ids = [wid for wid, _ in wisdom_pool.most_common(3)]
            wisdom_entries = [lib_by_id[w] for w in top_wisdom_ids if w in lib_by_id]

            if variant == "phase2_v5":
                prompt = EXECUTE_V5.format(
                    priors_block=format_priors(priors),
                    triggers_block=format_triggers(triggers),
                    wisdom_block=format_wisdom(wisdom_entries),
                    problem=problem,
                )
            elif variant == "phase2_v11":
                prompt = EXECUTE_V11.format(
                    priors_block=format_priors(priors),
                    triggers_block=format_triggers(triggers),
                    wisdom_block=format_wisdom(wisdom_entries),
                    problem=problem,
                )
            elif variant == "phase2_v12c":
                prompt = EXECUTE_FULL.format(
                    priors_block=format_priors(priors),
                    triggers_block=format_triggers(triggers),
                    wisdom_block=format_wisdom(wisdom_entries),
                    problem=problem,
                )
            max_tok = 900

        try:
            resp = _generate_with_retry(client, prompt, max_tokens=max_tok, temperature=0.3)
            answers[pid] = resp["text"].strip()
        except Exception as e:
            print(f"  [err] {pid}: {e}")
            continue

        new += 1
        if new % 10 == 0:
            cache_save(answers_path, answers)
            print(f"  [{variant}] {i+1}/{len(problems)} {time.time()-t0:.0f}s")

    cache_save(answers_path, answers)
    print(f"  [{variant}] done: new={new} hit={hit} ({time.time()-t0:.0f}s)")


def judge(variant_a: str, variant_b: str, problems: List[Dict], client):
    """Judge variant_a vs variant_b on held-out problems."""
    ans_a = cache_load(ANSWERS_DIR / f"{variant_a}_holdout50_answers.json")
    ans_b = cache_load(ANSWERS_DIR / f"{variant_b}_holdout50_answers.json")
    j_path = JUDGMENTS_DIR / f"{variant_a}_vs_{variant_b}_holdout50.json"
    judgments = cache_load(j_path)
    rng = random.Random(42)
    t0 = time.time()
    new = hit = 0

    for i, p in enumerate(problems):
        pid = p["problem_id"]
        if pid in judgments:
            hit += 1
            continue
        a = ans_a.get(pid)
        b = ans_b.get(pid)
        if not a or not b:
            continue
        if rng.random() < 0.5:
            left, right, a_was = a, b, "A"
        else:
            left, right, a_was = b, a, "B"
        v = judge_pair(client, p.get("description", ""), left, right)
        winner_raw = v.get("winner", "tie")
        if winner_raw == "tie":
            winner = "tie"
        elif winner_raw == a_was:
            winner = variant_a
        else:
            winner = variant_b
        judgments[pid] = {
            "winner": winner,
            "score_a": int(v.get("score_a", 5)),
            "score_b": int(v.get("score_b", 5)),
            "reasoning": v.get("reasoning", ""),
            "a_was": a_was,
            "domain": p.get("domain", "?"),
            "difficulty": p.get("difficulty", "?"),
        }
        new += 1
        if new % 10 == 0:
            cache_save(j_path, judgments)
            print(f"  [judge {variant_a} vs {variant_b}] {i+1}/{len(problems)} {time.time()-t0:.0f}s")

    cache_save(j_path, judgments)
    print(f"  [judge {variant_a} vs {variant_b}] done ({time.time()-t0:.0f}s)")
    return judgments


def report(variant_a: str, variant_b: str, judgments: Dict):
    results = list(judgments.values())
    a_wins = sum(1 for r in results if r["winner"] == variant_a)
    b_wins = sum(1 for r in results if r["winner"] == variant_b)
    ties = sum(1 for r in results if r["winner"] == "tie")
    total = a_wins + b_wins
    wr = a_wins / total if total else 0.5
    print(f"\n  {variant_a} vs {variant_b} (n={len(results)}): "
          f"{variant_a}={a_wins} {variant_b}={b_wins} tie={ties} wr={wr:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+",
                    default=["baseline", "phase2_v11", "phase2_v12c"])
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args()

    problems = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))
    # Strip metadata
    problems = [p for p in problems if "description" in p]
    print(f"Held-out: {len(problems)} problems")

    client = create_client()

    if not args.skip_gen:
        for v in args.variants:
            print(f"\n=== Generating {v} ===")
            gen_answers(v, problems, client)

    if not args.skip_judge:
        # Judge pairs: v12c vs baseline, v12c vs v11, v11 vs baseline
        pairs = [
            ("phase2_v12c", "baseline"),
            ("phase2_v12c", "phase2_v11"),
            ("phase2_v11", "baseline"),
        ]
        for a, b in pairs:
            if a in args.variants and b in args.variants:
                print(f"\n=== Judging {a} vs {b} ===")
                j = judge(a, b, problems, client)
                report(a, b, j)


if __name__ == "__main__":
    main()
