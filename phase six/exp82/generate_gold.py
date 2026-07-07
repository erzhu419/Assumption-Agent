"""Exp 82 v2 step 0: generate gold answers for the 50 holdout problems.

Uses expensive-tier (claude-opus-4-6) to produce a reference answer for each
problem. We later use these gold answers as the correctness anchor: every
candidate answer (BASE / GENERIC / EXT-via-Hypothesis) is graded against the
gold via a cheap-tier judge.

This decouples the ablation from v1's pairwise judge-wr metric.

Output: phase six/exp82/gold_answers.json
  {pid: {"description": "...", "domain": "...", "gold": "..."}}
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from model_router import expensive  # noqa: E402

OUT = Path(__file__).parent / "gold_answers.json"
HOLDOUT = PROJECT / "phase two" / "analysis" / "cache" / "sample_holdout_50.json"


GOLD_PROMPT_TEMPLATE = """You are an expert problem solver writing a reference solution.

PROBLEM:
{description}

Write a concise, correct, complete reference answer to this problem. Your answer should:
- Solve the problem fully
- Be correct enough that future answers can be graded against it
- Cover the key reasoning steps so a grader can identify what a candidate answer must contain
- Be in {language}

Do NOT include meta-commentary like "Here is my answer". Just produce the reference solution.
"""


def detect_language(desc: str) -> str:
    """Return 'Chinese' or 'English' based on description."""
    if any('一' <= ch <= '鿿' for ch in desc[:200]):
        return "Chinese"
    return "English"


def generate_one(client, pid: str, desc: str) -> dict:
    """Call expensive-tier to write a reference answer."""
    lang = detect_language(desc)
    prompt = GOLD_PROMPT_TEMPLATE.format(description=desc, language=lang)
    t0 = time.time()
    for attempt in range(3):
        try:
            r = client.generate(prompt, max_tokens=2000, temperature=0.2)
            text = (r.get("text") or "").strip()
            if text:
                return {"pid": pid, "gold": text, "elapsed": time.time() - t0,
                        "model": r.get("model", "")}
            raise RuntimeError("empty response")
        except Exception as e:
            if attempt == 2:
                return {"pid": pid, "gold": "", "elapsed": time.time() - t0,
                        "model": "", "error": str(e)[:200]}
            time.sleep(2 ** attempt)


def main():
    print(f"Loading {HOLDOUT.relative_to(PROJECT)}...", flush=True)
    problems = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    print(f"  {len(problems)} problems", flush=True)

    # Resume support: skip already-generated pids
    out: dict = {}
    if OUT.exists():
        out = json.loads(OUT.read_text(encoding="utf-8"))
        print(f"  resume: {len(out)} already done, skipping", flush=True)

    todo = [p for p in problems if p["problem_id"] not in out]
    print(f"  to generate: {len(todo)}", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    client = expensive("claude_opus")
    print(f"  using {client.model}", flush=True)

    n_done = len(out)
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(generate_one, client, p["problem_id"], p["description"]): p
                   for p in todo}
        for fut in as_completed(futures):
            p = futures[fut]
            r = fut.result()
            n_done += 1
            if r.get("gold"):
                out[p["problem_id"]] = {
                    "description": p["description"],
                    "domain": p.get("domain", ""),
                    "difficulty": p.get("difficulty", ""),
                    "gold": r["gold"],
                    "model": r.get("model", ""),
                }
                # incremental save
                OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
                print(f"  [{n_done}/{len(problems)}] {p['problem_id']:30s} OK ({r['elapsed']:.1f}s, {len(r['gold'])} chars)", flush=True)
            else:
                print(f"  [{n_done}/{len(problems)}] {p['problem_id']:30s} FAIL: {r.get('error','')[:80]}", flush=True)

    print(f"\nFinal: {len(out)}/{len(problems)} gold answers saved → {OUT.relative_to(PROJECT)}", flush=True)


if __name__ == "__main__":
    main()
