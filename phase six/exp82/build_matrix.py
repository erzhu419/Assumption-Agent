"""Exp 82 step 1: build the (W, P, J) verdict matrix from cached data.

We assemble:
  - 12 wisdom candidates (WCAND01-09 + WCAND10/W077 + WCROSSL01/W078;
    note WCAND05 has wid W076)
  - 50 hold-out problems (sample_holdout_50.json)
  - Multiple judge families:
      * gemini-3-flash via judge_content_cache (the ORIGINAL gate's judge)
      * exp36 cheap-tier panel (gemini, claude_haiku, gpt_mini) for
        W076/W077/W078 only
      * exp1 claude_opus rejudgment for W076/W077/W078

For each (cid, pid): load base_answer + ext_answer from
phase two/analysis/cache/answers/, compute SHA-256 hash of
(problem_description, ext_or_base_left, ext_or_base_right) using the
same side-randomization seed as exp17 (rng = Random(hash(pid))), look
up in judge_content_cache. Output the resolved 'ext'/'base'/'tie'
verdict.

Output: phase six/exp82/verdict_matrix.json
  {pid: {cid: {judge: 'ext'|'base'|'tie'|'missing'}}}
"""
import hashlib
import json
import random
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO = PROJECT / "phase four" / "autonomous"
OUT = Path(__file__).parent / "verdict_matrix.json"


def content_hash(problem: str, a: str, b: str) -> str:
    h = hashlib.sha256()
    h.update(problem.encode("utf-8"))
    h.update(b"\x00")
    h.update(a.encode("utf-8"))
    h.update(b"\x00")
    h.update(b.encode("utf-8"))
    return h.hexdigest()


def load_candidates():
    cands = json.loads((AUTO / "success_distilled_candidates.json").read_text(encoding="utf-8"))
    # Add WCROSSL01 (cross-LLM) which is W078
    crossl = json.loads((AUTO / "cross_llm_candidates.json").read_text(encoding="utf-8"))
    if isinstance(crossl, list) and crossl:
        crossl[0]["cid"] = "WCROSSL01"
        cands = list(cands) + [crossl[0]]
    # ensure cid field is set for all
    for i, c in enumerate(cands[:11]):
        c["cid"] = c.get("cid") or f"WCAND{i+1:02d}"
    return cands


def load_problems():
    return json.loads((CACHE / "sample_holdout_50.json").read_text(encoding="utf-8"))


def load_answer_cache(cid: str):
    """Load base + ext answer dict for a given candidate (matches exp17 logic)."""
    base_defaults = {"WCROSSL01": "_valp_v20_base"}
    default_base = "_valp_v20p1_base"
    base_stem = base_defaults.get(cid, default_base)
    base_path = CACHE / "answers" / f"{base_stem}_answers.json"
    ext_path = CACHE / "answers" / f"_valp_v20_ext_{cid}_answers.json"
    if not base_path.exists() or not ext_path.exists():
        return None, None
    base = json.loads(base_path.read_text(encoding="utf-8"))
    ext = json.loads(ext_path.read_text(encoding="utf-8"))
    return base, ext


def reconstruct_gemini_verdicts(cands, problems, judge_cache):
    """For each (cid, pid): compute the side-randomized hash and look up
    in judge_content_cache. Returns {cid: {pid: 'ext'|'base'|'tie'|'missing'}}.

    Side-randomization seed matches exp17_trigger_conditioned_gate.py line 209:
      rng = random.Random(hash(pid) % (2**32))
      if rng.random() < 0.5: left=ext, right=base, ext_was='A'
      else: left=base, right=ext, ext_was='B'
    """
    out = {}
    for c in cands:
        cid = c.get("cid")
        if not cid: continue
        base, ext = load_answer_cache(cid)
        if base is None:
            out[cid] = {p["problem_id"]: "missing_answers" for p in problems}
            continue
        verdicts = {}
        for p in problems:
            pid = p["problem_id"]
            ba = base.get(pid)
            ea = ext.get(pid)
            prob_text = p.get("description", "")
            if not ba or not ea or not prob_text:
                verdicts[pid] = "missing"
                continue
            rng = random.Random(hash(pid) % (2**32))
            if rng.random() < 0.5:
                left, right, ext_was = ea, ba, "A"
            else:
                left, right, ext_was = ba, ea, "B"
            key = content_hash(prob_text, left, right)
            cached = judge_cache.get(key)
            if not cached:
                verdicts[pid] = "missing"
                continue
            w = cached.get("winner", "tie")
            if w == "tie":
                verdicts[pid] = "tie"
            elif w == ext_was:
                verdicts[pid] = "ext"
            else:
                verdicts[pid] = "base"
        out[cid] = verdicts
    return out


def add_exp36_verdicts(matrix):
    """exp36 has W076 W077 W078 across (gemini, claude_haiku, gpt_mini).
    cand_id is the wid (W076 etc), need to map to cid.
    """
    log = json.loads((AUTO / "exp36_cheap_verdicts_log.json").read_text(encoding="utf-8"))
    # wid -> cid mapping (per the paper: WCAND05/W076, WCAND10/W077, WCROSSL01/W078)
    wid_to_cid = {"W076": "WCAND05", "W077": "WCAND10", "W078": "WCROSSL01"}
    for r in log["results"]:
        wid = r["cand_id"]
        cid = wid_to_cid.get(wid, wid)
        for judge_short, verdicts in r["verdicts"].items():
            judge_full = {"gemini": "gemini-3-flash",
                            "claude_haiku": "claude-haiku-4-5",
                            "gpt_mini": "gpt-5.4-mini"}.get(judge_short, judge_short)
            for pid, v in verdicts.items():
                matrix.setdefault(pid, {}).setdefault(cid, {})[judge_full] = v if v in ("ext","base","tie") else "missing"


def add_exp1_verdicts(matrix):
    """exp1 has W076 W077 W078 from claude-opus-4-6."""
    log = json.loads((AUTO / "exp1_cross_judge_log.json").read_text(encoding="utf-8"))
    wid_to_cid = {"W076": "WCAND05", "W077": "WCAND10", "W078": "WCROSSL01"}
    for run in log:
        judge = run.get("judge_model", "claude-opus-4-6")
        for r in run.get("results", []):
            wid = r["wid"]
            cid = wid_to_cid.get(wid, wid)
            verdicts = r.get("verdicts", {})
            for pid, v in verdicts.items():
                w = v.get("winner", "tie") if isinstance(v, dict) else v
                if w not in ("ext","base","tie","A","B"):
                    matrix.setdefault(pid, {}).setdefault(cid, {})[judge] = "missing"
                    continue
                # exp1 verdicts already encoded as 'ext'/'base' (winner), check
                if w in ("ext","base","tie"):
                    matrix.setdefault(pid, {}).setdefault(cid, {})[judge] = w
                else:
                    matrix.setdefault(pid, {}).setdefault(cid, {})[judge] = "missing"


def main():
    print(f"Loading caches...", flush=True)
    cands = load_candidates()
    problems = load_problems()
    print(f"  {len(cands)} candidates, {len(problems)} problems", flush=True)
    judge_cache = json.loads((CACHE / "judge_content_cache.json").read_text(encoding="utf-8"))
    print(f"  judge_content_cache: {len(judge_cache)} entries", flush=True)

    print(f"\nReconstructing original gemini-3-flash verdicts...", flush=True)
    gem = reconstruct_gemini_verdicts(cands, problems, judge_cache)

    # Build {pid: {cid: {judge: verdict}}}
    matrix = {}
    n_hit = n_miss = 0
    for cid, vd in gem.items():
        for pid, v in vd.items():
            matrix.setdefault(pid, {}).setdefault(cid, {})["gemini-3-flash"] = v
            if v in ("ext","base","tie"): n_hit += 1
            else: n_miss += 1
    print(f"  gemini cache hit: {n_hit}, miss: {n_miss}", flush=True)

    # Per-cid hit count
    print(f"\n  Per-cid gemini hit count:", flush=True)
    for cid in sorted(gem.keys()):
        hits = sum(1 for v in gem[cid].values() if v in ("ext","base","tie"))
        print(f"    {cid}: {hits}/{len(gem[cid])}", flush=True)

    print(f"\nMerging exp36 cheap-tier panel verdicts (W076/W077/W078)...", flush=True)
    add_exp36_verdicts(matrix)

    print(f"Merging exp1 claude-opus rejudgment verdicts (W076/W077/W078)...", flush=True)
    add_exp1_verdicts(matrix)

    # Final stats
    cells = sum(1 for pid in matrix for cid in matrix[pid] for j in matrix[pid][cid])
    valid = sum(1 for pid in matrix for cid in matrix[pid] for j, v in matrix[pid][cid].items()
                  if v in ("ext","base","tie"))
    print(f"\nFinal matrix:", flush=True)
    print(f"  Total (pid, cid, judge) cells: {cells}", flush=True)
    print(f"  Valid (ext/base/tie): {valid}  invalid (missing): {cells-valid}", flush=True)

    # Per-judge counts
    judge_counts = {}
    for pid in matrix:
        for cid in matrix[pid]:
            for j, v in matrix[pid][cid].items():
                judge_counts.setdefault(j, {"valid":0, "missing":0})
                if v in ("ext","base","tie"): judge_counts[j]["valid"] += 1
                else: judge_counts[j]["missing"] += 1
    print(f"\n  Per judge:", flush=True)
    for j, c in sorted(judge_counts.items()):
        print(f"    {j}: valid={c['valid']}, missing={c['missing']}", flush=True)

    OUT.write_text(json.dumps({"problems": [{"pid": p["problem_id"],
                                                  "description": p["description"],
                                                  "domain": p.get("domain", ""),
                                                  "difficulty": p.get("difficulty", "")}
                                                 for p in problems],
                                 "candidates": [{"cid": c.get("cid"),
                                                  "aphorism": c.get("aphorism", ""),
                                                  "source": c.get("source", ""),
                                                  "signal": c.get("signal", ""),
                                                  "unpacked": c.get("unpacked_for_llm", "")}
                                                 for c in cands if c.get("cid")],
                                 "matrix": matrix},
                                ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT.relative_to(PROJECT)}", flush=True)


if __name__ == "__main__":
    main()
