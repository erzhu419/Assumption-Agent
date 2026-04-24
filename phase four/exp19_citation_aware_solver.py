"""Exp 19 — Citation-aware solver (architectural L3).

Prospective version of Exp 16: instead of retrospectively asking a
judge 'is W cited in this answer?', we modify the solver's EXECUTE
prompt to require the solver ITSELF to declare which wisdom(s) it
used and to quote the step where each wisdom applies. Output schema:

  {
    "answer": "<main answer text>",
    "used_wisdoms": [
      {"id": "W0XX", "applied_to_step": "quoted step", "reason": "..."}
    ]
  }

We run this on 20 held-out pids for 1 KEEP (W076) + base library,
then measure:
  1. Citation rate: fraction of pids where solver declares any wisdom.
  2. Target-citation rate: fraction of pids where solver declares W076.
  3. Citation agreement with Exp 16's retrospective judge on the same
     pids (different source: solver-declared vs judge-attributed).

If citation-aware solving is feasible at all, (1) should be > 50%.
If W076 is retrievable + useful, (2) should exceed its background rate.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase one" / "scripts" / "validation"))
sys.path.insert(0, str(PROJECT / "phase four"))

from llm_client import create_client, parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp19_citation_aware_log.json"

PARALLEL = 6
N_SAMPLE_PIDS = 20


CITATION_AWARE_PROMPT = """你要解决下面的问题。Solver 必须**显式 cite** 它用到的 wisdom。

## 问题
{problem}

## 可用 wisdom (最多 2 条可以用)
{wisdom_block}

## 要求
1. 直接给答卷（200-400 字），风格同通常的解答。
2. 答卷末尾，**如果**你在答题过程中用到了任何 wisdom，必须列出：
   - "wisdom_id": "W0XX"
   - "applied_at": 一段原文 quote（30 字内，来自你答卷里的某个具体段落）
   - "reason": "30-50 字说明为什么这里用它"
3. 如果没用，`used_wisdoms: []`。不要虚报。

## 输出 JSON（不要代码块，且严格遵守）
{{"answer": "你的答卷正文",
  "used_wisdoms": [
    {{"wisdom_id": "W0XX", "applied_at": "quoted step", "reason": "..."}}
  ]}}
"""


def cache_load(p, default=None):
    if Path(p).exists():
        try: return json.loads(Path(p).read_text(encoding="utf-8"))
        except: return default
    return default


def load_candidate_record(aphorism):
    for src in ("success_distilled_candidates.json", "cross_llm_candidates.json"):
        data = cache_load(AUTO_DIR / src, default=[])
        for c in data:
            if c.get("aphorism", "").strip() == aphorism.strip():
                return c
    return None


def format_wisdom(w):
    return (f"- id: {w['id']}\n"
            f"  aphorism: {w['aphorism']}\n"
            f"  signal: {w.get('signal','')}\n"
            f"  unpacked: {w.get('unpacked_for_llm','')[:200]}")


def solve_citing(client, problem, wisdoms):
    """Run the citation-aware solver on a single problem."""
    wisdom_block = "\n\n".join(format_wisdom(w) for w in wisdoms)
    prompt = CITATION_AWARE_PROMPT.format(problem=problem, wisdom_block=wisdom_block)
    try:
        r = client.generate(prompt, max_tokens=1200, temperature=0.3)
        parsed = parse_json_from_llm(r["text"])
        return parsed.get("answer", ""), parsed.get("used_wisdoms", [])
    except Exception as e:
        return f"[err: {e}]", []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-wid", default="W076",
                    help="the target wisdom whose citation rate we track")
    ap.add_argument("--target-aphorism", default="凡益之道，与时偕行")
    ap.add_argument("--n", type=int, default=N_SAMPLE_PIDS)
    args = ap.parse_args()

    problems = json.loads((CACHE / "sample_holdout_50.json").read_text(encoding="utf-8"))
    problems = [p for p in problems if "description" in p]

    # Pick stratified N pids deterministically
    rng = random.Random(42)
    sample = rng.sample(problems, min(args.n, len(problems)))

    # Library: base 75 + target candidate
    lib = json.loads((CACHE / "wisdom_library.json").read_text(encoding="utf-8"))
    target = load_candidate_record(args.target_aphorism)
    if not target:
        print(f"Target wisdom record not found: {args.target_aphorism}"); return
    target_entry = {k: v for k, v in target.items()
                     if k not in {"_cluster_id", "_cluster_size", "_source",
                                   "novelty_sim", "covers_batch_pids",
                                   "rationale", "_evidence_pids", "tentative_id"}}
    target_entry["id"] = args.target_wid
    lib_ext = lib + [target_entry]
    print(f"Library: {len(lib)} base + 1 target ({args.target_wid} {args.target_aphorism})")

    # Retrieval: use phase2_v3_selections if available, else top-2 by keyword overlap
    selections = cache_load(CACHE / "phase2_v3_selections.json", default={})
    lib_by_id = {w["id"]: w for w in lib_ext}

    solver = create_client()
    print(f"Solver: {solver.provider} / {solver.model}\n")

    def task(p):
        pid = p["problem_id"]
        # retrieval: top-2 from selections + always include target
        sel = selections.get(pid, [])[:2]
        retrieved = [lib_by_id[wid] for wid in sel if wid in lib_by_id]
        # Inject target wisdom into retrieval (simulating it would have been retrieved)
        if target_entry["id"] not in {w["id"] for w in retrieved}:
            retrieved = retrieved[:1] + [target_entry]
        retrieved = retrieved[:2]
        answer, used = solve_citing(solver, p["description"], retrieved)
        return pid, retrieved, answer, used

    results = []
    print(f"[solving {len(sample)} problems with citation requirement]")
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(task, p) for p in sample]
        for f in as_completed(futs):
            try:
                pid, retrieved, answer, used = f.result()
            except Exception as e:
                print(f"  [err] {e}"); continue
            used_ids = [u.get("wisdom_id") for u in used]
            retrieved_ids = [w["id"] for w in retrieved]
            target_shown = args.target_wid in retrieved_ids
            target_cited = args.target_wid in used_ids
            any_cited = len(used) > 0
            results.append({
                "pid": pid,
                "retrieved_ids": retrieved_ids,
                "target_shown": target_shown,
                "used_wisdoms": used,
                "target_cited": target_cited,
                "any_cited": any_cited,
                "answer_length": len(answer or ""),
            })

    # Summary
    total = len(results)
    target_shown_n = sum(1 for r in results if r["target_shown"])
    any_cited_n = sum(1 for r in results if r["any_cited"])
    target_cited_n = sum(1 for r in results if r["target_cited"])
    target_cited_given_shown = sum(1 for r in results
                                    if r["target_shown"] and r["target_cited"])

    print(f"\n=== SUMMARY: citation-aware solver on {args.target_wid} ===")
    print(f"  Problems solved:           {total}")
    print(f"  Target shown in retrieval: {target_shown_n}/{total} "
          f"({target_shown_n/total*100:.0f}%)")
    print(f"  Any wisdom cited:          {any_cited_n}/{total} "
          f"({any_cited_n/total*100:.0f}%)")
    print(f"  Target cited:              {target_cited_n}/{total} "
          f"({target_cited_n/total*100:.0f}%)")
    if target_shown_n:
        rate = target_cited_given_shown / target_shown_n
        print(f"  Target cited | shown:      {target_cited_given_shown}/{target_shown_n} "
              f"({rate*100:.0f}%)")

    # Sample 2 cited quotes
    cited_examples = [r for r in results if r["target_cited"]][:3]
    for r in cited_examples:
        w = next((w for w in r["used_wisdoms"]
                   if w.get("wisdom_id") == args.target_wid), {})
        applied = w.get("applied_at", "")[:50]
        reason = w.get("reason", "")[:60]
        print(f"    [{r['pid']}] applied_at: {applied}")
        print(f"                reason: {reason}")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "solver": solver.model, "target_wid": args.target_wid,
           "target_aphorism": args.target_aphorism,
           "n_sampled": len(sample),
           "target_shown_n": target_shown_n,
           "any_cited_n": any_cited_n,
           "target_cited_n": target_cited_n,
           "target_cited_given_shown": target_cited_given_shown,
           "results": results}
    prev = cache_load(OUT_LOG, default=[]) or []
    prev.append(out)
    OUT_LOG.write_text(json.dumps(prev, ensure_ascii=False, indent=2))
    print(f"Saved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
