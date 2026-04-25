"""Exp 43 — Clean inner-loop rerun without exemplar held-out leak.

Closes review weakness #1 (held-out contamination): in the original
inner-loop A/B gate, the 3 cross-domain exemplars per candidate were
mined from the same 50-problem `sample_holdout_50` set used for
evaluation. We re-mine exemplars from `sample_extend_50` (a disjoint
50-problem pool) and re-run the gate on `sample_holdout_50`.

If the same 3 KEEPs (W076/W077/W078) pass under the clean (no-leak)
mining, the original null verdict's audit-stack interpretation
strengthens: the loop's accept decisions reproduce on a clean held-
out, and the audit-stack rejection is independent of the exemplar
leak.

If different KEEPs pass, the original "3/12" inner-loop result was
partially driven by exemplar leakage; we report the clean result
honestly and note that the audit-stack analysis was on contaminated
ext answers.

Cost: 12 candidates x 50 pids x 1 (ext only; base reused from
Exp 10) generations + 12 x 50 judge calls = ~1200 calls.
~$15-20, ~1.5h at 6-worker parallelism.
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))


def _load_api_keys():
    """Source ~/.api_keys (shell-export format) into os.environ."""
    if os.environ.get("RUOLI_GPT_KEY") and os.environ.get("RUOLI_BASE_URL"):
        return
    keyfile = Path.home() / ".api_keys"
    if not keyfile.exists():
        return
    pat = re.compile(r'^\s*export\s+(\w+)=("([^"]*)"|\'([^\']*)\'|(\S+))')
    for line in keyfile.read_text().splitlines():
        m = pat.match(line)
        if not m: continue
        name = m.group(1)
        val = m.group(3) if m.group(3) is not None else (
              m.group(4) if m.group(4) is not None else m.group(5))
        os.environ.setdefault(name, val)
        # also set legacy env names for model_router compatibility
        if name == "RUOLI_BASE_URL":
            os.environ.setdefault("CLAUDE_PROXY_BASE_URL", val + "/v1" if not val.endswith("/v1") else val)
            os.environ.setdefault("GPT5_BASE_URL", val + "/v1" if not val.endswith("/v1") else val)
            os.environ.setdefault("GEMINI_PROXY_BASE_URL", val + "/v1" if not val.endswith("/v1") else val)
        if name == "RUOLI_GEMINI_KEY":
            os.environ.setdefault("GEMINI_PROXY_API_KEY", val)
        if name == "RUOLI_GPT_KEY":
            os.environ.setdefault("GPT5_API_KEY", val)
        if name == "RUOLI_CLAUDE_KEY":
            os.environ.setdefault("CLAUDE_PROXY_API_KEY", val)

_load_api_keys()

from model_router import cheap
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
ANS = CACHE / "answers"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp43_clean_innerloop_rerun_log.json"

PARALLEL = 6


# 12 candidates from validation_log_parallel.json
def load_candidates():
    val = json.loads((AUTO_DIR / "validation_log_parallel.json").read_text())
    candidates = []
    for entry in val:
        for r in entry["results"]:
            candidates.append({
                "tid": r["tid"],
                "aphorism": r["candidate"],
                "source": r.get("source", ""),
                "cluster_size": r.get("cluster_size", 0),
                # We need signal/unpacked from cross_llm_candidates.json or success_distilled
            })
    return candidates


EXEMPLAR_PROMPT = """For the methodological wisdom below, select 3
cross-domain example problems from the candidate list that best
illustrate when this wisdom applies. Pick problems from at least 2
different domains. Each must be a problem where the wisdom would
plausibly improve solving.

## Wisdom
- aphorism: {aphorism}
- source: {source}
- signal: {signal}
- unpacking: {unpacked}

## Candidate problems
{problems_brief}

## Output JSON only:
{{"selected": [
   {{"pid": "<problem_id>", "why_applies": "20-40字"}},
   {{"pid": "<problem_id>", "why_applies": "20-40字"}},
   {{"pid": "<problem_id>", "why_applies": "20-40字"}}
]}}
"""


FRAME_PROMPT = """对下面问题产生 frame + 重写。

## 原题
{problem}

## 输出 JSON
{{"frame": "object_level/paradigm/hybrid",
  "critical_reframe": "30-80字",
  "rewritten_problem": "120-250字"}}
"""

EXECUTE_PROMPT = """# 解决问题

## PRIMARY FRAME
- frame: {frame}
- critical reframe: {critical_reframe}

## 问题（重写）
{rewritten_problem}

## 次要参考 wisdom
{wisdom_block}

## 要求：≤ 500 字
开始："""

JUDGE_PROMPT = """方法论评审.

## 问题
{problem}

## 解答 A
{answer_a}

## 解答 B
{answer_b}

Output JSON: {{"winner": "A"/"B"/"tie", "score_a": 1-10, "score_b": 1-10,
  "reasoning": "80 chars max"}}
"""


def cache_load(p, default=None):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except: return default


def load_problems():
    """Load all 1768 problems by pid."""
    pid_to_prob = {}
    pid_to_meta = {}
    for f in (PROJECT / "phase zero" / "benchmark" / "problems").glob("*.json"):
        for q in json.loads(f.read_text()):
            pid_to_prob[q["problem_id"]] = q.get("description") or q.get("problem") or ""
            pid_to_meta[q["problem_id"]] = q
    return pid_to_prob, pid_to_meta


def mine_exemplars_clean(client, cand, mining_pool, pid_to_meta):
    """Mine 3 exemplars from mining_pool (disjoint from holdout)."""
    problems_brief = "\n".join(
        f"[{p['problem_id']}] [{p.get('domain','?')}] {(p.get('description') or '')[:80]}"
        for p in mining_pool
    )
    # Use whatever signal/unpacking we have
    signal = cand.get("signal", "")
    unpacked = cand.get("unpacked_for_llm", "")
    # If empty, derive a placeholder from aphorism
    if not signal:
        signal = f"在 {cand['aphorism'][:20]} 适用的情境"
    if not unpacked:
        unpacked = f"应用 {cand['aphorism']} 的方法论指导"

    try:
        r = client.generate(EXEMPLAR_PROMPT.format(
            aphorism=cand["aphorism"], source=cand["source"],
            signal=signal, unpacked=unpacked,
            problems_brief=problems_brief),
            max_tokens=500, temperature=0.3)
        parsed = parse_json_from_llm(r["text"])
        selected = parsed.get("selected", [])
    except Exception as e:
        print(f"    [exemplar err for {cand['tid']}] {e}")
        return []

    pid_to_info = {p["problem_id"]: p for p in mining_pool}
    result = []
    for item in selected[:3]:
        pid = item.get("pid", "").strip()
        if pid not in pid_to_info:
            continue
        info = pid_to_info[pid]
        result.append({
            "pid": pid, "domain": info.get("domain", "?"),
            "problem_sketch": (info.get("description") or "")[:300],
            "why_applies": item.get("why_applies", ""),
        })
    return result


def solve(client, problem, wisdoms):
    try:
        r = client.generate(FRAME_PROMPT.format(problem=problem),
                            max_tokens=500, temperature=0.2)
        m = parse_json_from_llm(r["text"])
    except Exception as e:
        return f"[turn0 err: {e}]"
    if wisdoms:
        wb_parts = []
        for w in wisdoms:
            ex_text = ""
            for e in w.get("cross_domain_examples", [])[:2]:
                ex_text += f"\n  - 例 ({e.get('domain', '?')}): {e.get('why_applies', '')[:100]}"
            wb_parts.append(f"• {w['aphorism']}: {w.get('unpacked_for_llm', w.get('aphorism',''))[:180]}{ex_text}")
        wb = "\n".join(wb_parts)
    else:
        wb = "(无)"
    try:
        r = client.generate(EXECUTE_PROMPT.format(
            frame=m.get("frame", "object_level"),
            critical_reframe=m.get("critical_reframe", ""),
            rewritten_problem=m.get("rewritten_problem", problem),
            wisdom_block=wb), max_tokens=900, temperature=0.2)
        return r["text"].strip()
    except Exception as e:
        return f"[turn1 err: {e}]"


def judge_one(client, problem, a, b):
    try:
        r = client.generate(JUDGE_PROMPT.format(problem=problem, answer_a=a, answer_b=b),
                            max_tokens=300, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        return v.get("winner", "tie")
    except Exception:
        return "err"


def main():
    pid_to_prob, pid_to_meta = load_problems()

    # Load disjoint pools
    holdout = json.loads((CACHE / "sample_holdout_50.json").read_text())
    mining_pool_raw = json.loads((CACHE / "sample_extend_50.json").read_text())
    mining_pool = [p for p in mining_pool_raw if "description" in p]
    print(f"Holdout (eval): {len(holdout)} pids")
    print(f"Mining pool (exemplars): {len(mining_pool)} pids; disjoint from holdout? "
          f"{not bool(set(p['problem_id'] for p in holdout) & set(p['problem_id'] for p in mining_pool))}")

    # Load 12 candidates with their signal/unpacked from original cross-llm and success-distilled records
    candidates = load_candidates()
    print(f"\nCandidates to re-evaluate: {len(candidates)}")

    # Need to merge in signal/unpacked from success_distilled / cross_llm logs
    sd = cache_load(AUTO_DIR / "success_distilled_candidates.json", default=[])
    cl = cache_load(AUTO_DIR / "cross_llm_candidates.json", default=[])
    aphorism_to_record = {}
    for r in sd + cl:
        if "aphorism" in r:
            aphorism_to_record[r["aphorism"]] = r
    enriched_count = 0
    for c in candidates:
        rec = aphorism_to_record.get(c["aphorism"], {})
        c["signal"] = rec.get("signal", c.get("signal", ""))
        c["unpacked_for_llm"] = rec.get("unpacked_for_llm", c.get("unpacked_for_llm", ""))
        if c["signal"] or c["unpacked_for_llm"]:
            enriched_count += 1
    print(f"  enriched with signal/unpacking: {enriched_count}/{len(candidates)}")

    # Step 1: mine clean exemplars (per candidate, from extend_50)
    print(f"\n[1/3] Mining clean exemplars from extend_50 for {len(candidates)} candidates...")
    gpt_client = cheap("gpt_mini")  # use cheap GPT for mining
    t0 = time.time()
    for i, c in enumerate(candidates):
        ex = mine_exemplars_clean(gpt_client, c, mining_pool, pid_to_meta)
        c["clean_exemplars"] = ex
        if i % 4 == 0:
            print(f"  mined {i+1}/{len(candidates)} ({time.time()-t0:.0f}s)")
    print(f"  done in {time.time()-t0:.0f}s, "
          f"{sum(1 for c in candidates if len(c.get('clean_exemplars',[]))==3)}/{len(candidates)} "
          f"have 3 exemplars")

    # Step 2: regenerate ext answers using clean wisdom records
    # base answers reused from Exp 10 (no leak — base library was authored, not mined)
    print(f"\n[2/3] Generating clean ext answers ({len(candidates)} x {len(holdout)} = "
          f"{len(candidates)*len(holdout)} calls)...")
    solver = cheap("gemini")

    def gen_ext(cand, p):
        # Build wisdom record with clean exemplars
        w = {
            "aphorism": cand["aphorism"],
            "source": cand["source"],
            "signal": cand.get("signal", ""),
            "unpacked_for_llm": cand.get("unpacked_for_llm", cand["aphorism"]),
            "cross_domain_examples": cand.get("clean_exemplars", []),
        }
        return cand["tid"], p["problem_id"], solve(solver, p["description"], [w])

    ext_answers = {c["tid"]: {} for c in candidates}
    tasks = [(c, p) for c in candidates for p in holdout]
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(gen_ext, c, p) for c, p in tasks]
        for f in as_completed(futs):
            tid, pid, ans = f.result()
            ext_answers[tid][pid] = ans
            done += 1
            if done % 50 == 0:
                print(f"  gen {done}/{len(tasks)} ({time.time()-t0:.0f}s)")

    # Step 3: judge ext vs base (reuse cached base from Exp 10)
    print(f"\n[3/3] Judging clean ext vs base (cached)...")
    base_ans = json.loads((ANS / "_exp10_v20_base_answers.json").read_text())
    common_pids = sorted(set(p["problem_id"] for p in holdout) & set(base_ans))
    print(f"  base coverage: {len(common_pids)}/{len(holdout)} pids")

    def judge_pair(c, pid):
        b = base_ans.get(pid, "")
        e = ext_answers[c["tid"]].get(pid, "")
        if not b or not e or b.startswith("[") or e.startswith("["):
            return c["tid"], pid, "missing"
        rng = random.Random((hash(pid) ^ hash(c["tid"])) % (2**32))
        if rng.random() < 0.5:
            left, right, ext_was = e, b, "A"
        else:
            left, right, ext_was = b, e, "B"
        prob = next((p for p in holdout if p["problem_id"] == pid), None)
        if prob is None: return c["tid"], pid, "missing"
        w = judge_one(solver, prob["description"], left, right)
        if w == "tie": v = "tie"
        elif w in ("A", "B"): v = "ext" if w == ext_was else "base"
        else: v = "err"
        return c["tid"], pid, v

    verdicts = {c["tid"]: {} for c in candidates}
    jtasks = [(c, pid) for c in candidates for pid in common_pids]
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(judge_pair, c, pid) for c, pid in jtasks]
        for f in as_completed(futs):
            tid, pid, v = f.result()
            verdicts[tid][pid] = v
            done += 1
            if done % 50 == 0:
                print(f"  judge {done}/{len(jtasks)} ({time.time()-t0:.0f}s)")

    # Aggregate + compare to original (leaky) inner-loop wr
    print(f"\n=== Clean rerun: wr_ext vs original (leaky) inner-loop ===")
    print(f"{'tid':12s} {'aphorism':25s}  {'orig':>6s}  {'clean':>6s}  {'delta':>+8s}  decision")
    print("-" * 80)
    original = {"WCAND05": 0.64, "WCAND10": 0.60, "WCROSSL01": 0.60,
                "WCAND01": 0.58, "WCAND02": 0.46, "WCAND03": 0.56,
                "WCAND04": 0.48, "WCAND06": 0.58, "WCAND07": 0.56,
                "WCAND08": 0.51, "WCAND09": 0.55, "WCAND11": 0.54}

    summary = []
    for c in candidates:
        v = verdicts[c["tid"]]
        ne = sum(1 for x in v.values() if x == "ext")
        nb = sum(1 for x in v.values() if x == "base")
        nt = sum(1 for x in v.values() if x == "tie")
        n_eff = ne + nb
        wr_clean = ne / n_eff if n_eff else 0.5
        wr_orig = original.get(c["tid"], None)
        delta = (wr_clean - wr_orig) if wr_orig is not None else None
        decision = "KEEP" if wr_clean >= 0.60 else "REVERT"
        summary.append({"tid": c["tid"], "aphorism": c["aphorism"],
                          "wr_clean": wr_clean, "n_eff": n_eff,
                          "ext": ne, "base": nb, "tie": nt,
                          "wr_original_leaky": wr_orig,
                          "delta_clean_vs_leaky": delta,
                          "clean_decision": decision})
        delta_str = f"{delta:+.2f}" if delta is not None else "  ?  "
        print(f"{c['tid']:12s} {c['aphorism'][:23]:25s}  "
              f"{wr_orig if wr_orig else '?':>6}  {wr_clean:>6.2f}  {delta_str:>8s}  {decision}")

    n_clean_keep = sum(1 for s in summary if s["clean_decision"] == "KEEP")
    print(f"\n  CLEAN inner-loop verdict: {n_clean_keep}/{len(candidates)} candidates pass +10pp gate")
    n_orig_keep = 3
    print(f"  Original (leaky) inner-loop: {n_orig_keep}/{len(candidates)} (W076, W077, W078)")
    clean_keeps = [s["tid"] for s in summary if s["clean_decision"] == "KEEP"]
    orig_keeps = ["WCAND05", "WCAND10", "WCROSSL01"]
    overlap = set(clean_keeps) & set(orig_keeps)
    print(f"  Overlap of clean and original KEEPs: {len(overlap)} ({list(overlap)})")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n_holdout": len(holdout),
           "mining_pool": "sample_extend_50 (disjoint)",
           "summary": summary,
           "verdicts": verdicts,
           "clean_exemplars_per_candidate": {c["tid"]: c.get("clean_exemplars", []) for c in candidates}}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
