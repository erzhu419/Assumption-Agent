"""Staged validation gate — retrospective analysis on the 12 existing
candidates from the original plan
(manual/phase_four_dev_doc.md § 4.3-4.5):

  Stage 1 (Pilot):        n=10 domain-stratified pre-filter
  Stage 2 (Comparative):  n=50 full A/B (the current +10pp gate)
  Stage 3 (Generalization):  per-domain wr — no catastrophe AND
                           majority-domain uplift
  Stage 4 (Cross-family): Claude Opus re-judge
  Stage 5 (Recursion):    covered by auto_recurse.py (belief_state.json)

For Stages 1-3, all verdicts come from the cached gemini judgments
(content-hash cache hits; no new LLM calls).
For Stage 4, we run Claude on any candidate whose cross-judge score
is missing (~ 9 of 12 rows).

Output:
  phase four/autonomous/staged_gate_log.json
    per-candidate stage-by-stage pass/fail, final verdict, funnel stats.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase one" / "scripts" / "validation"))
sys.path.insert(0, str(PROJECT / "phase four"))

from llm_client import create_client, parse_json_from_llm
from claude_proxy_client import ClaudeProxyClient
from cached_framework import judge_pair, _save_content_cache

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "staged_gate_log.json"
HOLDOUT_SAMPLE = "sample_holdout_50.json"

PARALLEL_JUDGES = 6

# ---------- gate thresholds ----------
STAGE1_N = 10
STAGE1_PASS_WR = 0.60
STAGE2_PASS_WR = 0.60
STAGE3_NO_CATASTROPHE = 0.40       # no domain may fall below this
STAGE3_UPLIFT_WR = 0.55             # threshold for a domain to count as uplift
STAGE3_MIN_UPLIFT_DOMAINS = 3       # require ≥3 of 6 uplift domains
STAGE4_CLAUDE_WR = 0.55

# ---------- candidate table (also used by auto_recurse) ----------
CANDIDATES = {
    "WCAND01": {"ext": "_valp_v20_ext_WCAND01_answers.json",
                "base": "_valp_v20p1_base_answers.json",
                "aphorism": "上工治未病，不治已病"},
    "WCAND02": {"ext": "_valp_v20_ext_WCAND02_answers.json",
                "base": "_valp_v20p1_base_answers.json",
                "aphorism": "别高效解决一个被看错的问题"},
    "WCAND03": {"ext": "_valp_v20_ext_WCAND03_answers.json",
                "base": "_valp_v20p1_base_answers.json",
                "aphorism": "凡事预则立，不预则废"},
    "WCAND04": {"ext": "_valp_v20_ext_WCAND04_answers.json",
                "base": "_valp_v20p1_base_answers.json",
                "aphorism": "急则治其标，缓则治其本"},
    "WCAND05": {"ext": "_valp_v20_ext_WCAND05_answers.json",
                "base": "_valp_v20p1_base_answers.json",
                "aphorism": "凡益之道，与时偕行",
                "committed_id": "W076"},
    "WCAND06": {"ext": "_valp_v20_ext_WCAND06_answers.json",
                "base": "_valp_v20p1_base_answers.json",
                "aphorism": "覆水难收，向前算账"},
    "WCAND07": {"ext": "_valp_v20_ext_WCAND07_answers.json",
                "base": "_valp_v20p1_base_answers.json",
                "aphorism": "亲兄弟，明算账"},
    "WCAND08": {"ext": "_valp_v20_ext_WCAND08_answers.json",
                "base": "_valp_v20p1_base_answers.json",
                "aphorism": "想理解行为，先看激励"},
    "WCAND09": {"ext": "_valp_v20_ext_WCAND09_answers.json",
                "base": "_valp_v20p1_base_answers.json",
                "aphorism": "不谋全局者，不足谋一域"},
    "WCAND10": {"ext": "_valp_v20_ext_WCAND10_answers.json",
                "base": "_valp_v20p1_base_answers.json",
                "aphorism": "没有调查，就没有发言权",
                "committed_id": "W077"},
    "WCAND11": {"ext": "_valp_v20_ext_WCAND11_answers.json",
                "base": "_valp_v20p1_base_answers.json",
                "aphorism": "若不是品牌，你就只是商品。"},
    "WCROSSL01": {"ext": "_valp_v20_ext_WCROSSL01_answers.json",
                   "base": "_valp_v20_base_answers.json",
                   "aphorism": "是骡子是马，拉出来遛遛",
                   "committed_id": "W078"},
}


def cache_load(p, default=None):
    if Path(p).exists():
        try:
            return json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def cache_save(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def stratified_pilot_pids(problems, n=STAGE1_N):
    """Pick n pids stratified by domain; deterministic by seed."""
    by_dom = defaultdict(list)
    for p in problems:
        by_dom[p["domain"]].append(p)
    rng = random.Random(42)
    out = []
    doms = sorted(by_dom.keys())
    # Round-robin pick until we have n, take in stratified order
    while len(out) < n:
        for d in doms:
            if len(out) >= n: break
            if by_dom[d]:
                # Pick deterministically
                chosen = by_dom[d].pop(rng.randrange(len(by_dom[d])))
                out.append(chosen)
    return sorted(out, key=lambda p: p["problem_id"])


def judge_batch_gemini_cached(problems, ans_base, ans_ext, client):
    """Per-PID A/B using cached judge. Returns {pid: 'ext'|'base'|'tie'}."""
    def one(p):
        pid = p["problem_id"]
        ba, ea = ans_base.get(pid), ans_ext.get(pid)
        if not ba or not ea:
            return pid, "missing"
        rng = random.Random(hash(pid) % (2**32))
        if rng.random() < 0.5:
            left, right, ext_was = ea, ba, "A"
        else:
            left, right, ext_was = ba, ea, "B"
        try:
            v = judge_pair(client, p["description"], left, right)
        except Exception:
            return pid, "error"
        w = v.get("winner", "tie")
        if w == "tie":
            return pid, "tie"
        return pid, ("ext" if w == ext_was else "base")

    res = {}
    with ThreadPoolExecutor(max_workers=PARALLEL_JUDGES) as ex:
        futs = [ex.submit(one, p) for p in problems]
        for f in as_completed(futs):
            pid, v = f.result()
            res[pid] = v
    return res


def wr_of(verdicts):
    e = sum(1 for v in verdicts.values() if v == "ext")
    b = sum(1 for v in verdicts.values() if v == "base")
    t = sum(1 for v in verdicts.values() if v == "tie")
    tot_decided = e + b
    wr = e / tot_decided if tot_decided else 0.5
    return {"ext": e, "base": b, "tie": t, "wr_ext": wr}


def stage3_domain_breakdown(verdicts, problems):
    pid_dom = {p["problem_id"]: p["domain"] for p in problems}
    by_dom = defaultdict(list)
    for pid, v in verdicts.items():
        dom = pid_dom.get(pid, "?")
        by_dom[dom].append(v)
    out = {}
    for d, vs in by_dom.items():
        e = sum(1 for v in vs if v == "ext")
        b = sum(1 for v in vs if v == "base")
        t = sum(1 for v in vs if v == "tie")
        wr = e / (e + b) if (e + b) else 0.5
        out[d] = {"n": len(vs), "wr": wr, "ext": e, "base": b, "tie": t}
    return out


def claude_judge_batch(problems, ans_base, ans_ext, client):
    from auto_recurse import JUDGE_PROMPT
    def one(p):
        pid = p["problem_id"]
        ba, ea = ans_base.get(pid), ans_ext.get(pid)
        if not ba or not ea:
            return pid, "missing"
        rng = random.Random(hash(pid) % (2**32))
        if rng.random() < 0.5:
            left, right, ext_was = ea, ba, "A"
        else:
            left, right, ext_was = ba, ea, "B"
        prompt = JUDGE_PROMPT.format(problem=p["description"], answer_a=left,
                                      answer_b=right)
        try:
            r = client.generate(prompt, max_tokens=400, temperature=0.0)
            v = parse_json_from_llm(r["text"])
        except Exception:
            return pid, "error"
        w = v.get("winner", "tie")
        if w == "tie":
            return pid, "tie"
        return pid, ("ext" if w == ext_was else "base")
    res = {}
    with ThreadPoolExecutor(max_workers=PARALLEL_JUDGES) as ex:
        futs = [ex.submit(one, p) for p in problems]
        for f in as_completed(futs):
            pid, v = f.result()
            res[pid] = v
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-claude", action="store_true",
                    help="skip stage 4 (for dry-run)")
    args = ap.parse_args()

    problems = json.loads((CACHE / HOLDOUT_SAMPLE).read_text(encoding="utf-8"))
    problems = [p for p in problems if "description" in p]

    # Pilot subset is deterministic
    pilot = stratified_pilot_pids(problems, STAGE1_N)
    pilot_pids = {p["problem_id"] for p in pilot}
    print(f"Pilot set: {len(pilot)} problems, domains = "
          f"{sorted(set(p['domain'] for p in pilot))}")

    # Seed Claude scores from existing Exp1 log
    exp1_log = cache_load(AUTO_DIR / "exp1_cross_judge_log.json", default=[])
    claude_from_exp1 = {}  # {committed_id: wr}
    if exp1_log:
        for r in exp1_log[-1].get("results", []):
            claude_from_exp1[r["wid"]] = r["cross_wr_ext"]

    # Also seed from belief_state if present
    belief = cache_load(AUTO_DIR / "belief_state.json")
    if belief:
        for cid, b in belief.get("beliefs", {}).items():
            if b.get("claude_wr") is not None:
                CANDIDATES[cid]["claude_wr_cached"] = b["claude_wr"]

    gemini = create_client()
    claude = None if args.skip_claude else ClaudeProxyClient()

    results = []
    for cid, info in CANDIDATES.items():
        print(f"\n=== [{cid}] {info['aphorism']} ===")
        base_path = CACHE / "answers" / info["base"]
        ext_path = CACHE / "answers" / info["ext"]
        if not base_path.exists() or not ext_path.exists():
            print(f"  [SKIP] missing answers")
            continue
        ans_base = json.loads(base_path.read_text(encoding="utf-8"))
        ans_ext = json.loads(ext_path.read_text(encoding="utf-8"))

        # === Stage 2 first (has all data; cheapest cached) ===
        verdicts_50 = judge_batch_gemini_cached(problems, ans_base, ans_ext, gemini)
        s2 = wr_of(verdicts_50)
        s2_pass = s2["wr_ext"] >= STAGE2_PASS_WR
        print(f"  Stage 2 (n=50, gemini): ext={s2['ext']} base={s2['base']} "
              f"wr={s2['wr_ext']:.2f}  {'PASS' if s2_pass else 'FAIL'}")

        # Stage 1 = subset of stage 2 verdicts
        verdicts_pilot = {pid: v for pid, v in verdicts_50.items() if pid in pilot_pids}
        s1 = wr_of(verdicts_pilot)
        s1_pass = s1["wr_ext"] >= STAGE1_PASS_WR
        print(f"  Stage 1 (n=10, pilot):  ext={s1['ext']} base={s1['base']} "
              f"wr={s1['wr_ext']:.2f}  {'PASS' if s1_pass else 'FAIL'}")

        # Stage 3 — per domain breakdown from stage-2 verdicts
        dom_bd = stage3_domain_breakdown(verdicts_50, problems)
        n_uplift = sum(1 for d in dom_bd.values() if d["wr"] >= STAGE3_UPLIFT_WR)
        min_dom_wr = min((d["wr"] for d in dom_bd.values()), default=0.5)
        s3_pass = (min_dom_wr >= STAGE3_NO_CATASTROPHE and
                   n_uplift >= STAGE3_MIN_UPLIFT_DOMAINS)
        print(f"  Stage 3 (per-domain):    {n_uplift}/6 uplift, worst={min_dom_wr:.2f}  "
              f"{'PASS' if s3_pass else 'FAIL'}")
        for d, b in sorted(dom_bd.items()):
            print(f"      {d:20s} n={b['n']:2d}  wr={b['wr']:.2f} "
                  f"({b['ext']}:{b['base']}:{b['tie']}t)")

        # Stage 4 — Claude re-judge
        claude_wr = None; s4_pass = None
        committed_id = info.get("committed_id")
        if committed_id and committed_id in claude_from_exp1:
            claude_wr = claude_from_exp1[committed_id]
        elif "claude_wr_cached" in info:
            claude_wr = info["claude_wr_cached"]

        if claude_wr is None and claude is not None:
            # only run Claude on candidates that passed stages 1+2+3
            if s1_pass and s2_pass and s3_pass:
                print(f"  Stage 4 (claude): scoring (passed stages 1-3)...")
                t0 = time.time()
                v_claude = claude_judge_batch(problems, ans_base, ans_ext, claude)
                sc = wr_of(v_claude)
                claude_wr = sc["wr_ext"]
                print(f"    wr_claude={claude_wr:.2f} ({time.time()-t0:.0f}s)")
            else:
                print(f"  Stage 4 skipped (did not pass stages 1-3)")
        if claude_wr is not None:
            s4_pass = claude_wr >= STAGE4_CLAUDE_WR
            print(f"  Stage 4 (claude):       wr={claude_wr:.2f}  "
                  f"{'PASS' if s4_pass else 'FAIL'}")

        # Final verdict
        stages_passed = [s1_pass, s2_pass, s3_pass,
                          s4_pass if claude_wr is not None else None]
        # Where did it fall out?
        first_fail = None
        for i, p in enumerate(stages_passed, start=1):
            if p is False:
                first_fail = i
                break
        if first_fail is None:
            if None in stages_passed:
                final = "STAGE4_UNTESTED"
            else:
                final = "PASS_ALL"
        else:
            final = f"FAIL_STAGE_{first_fail}"

        _save_content_cache()
        results.append({
            "cid": cid,
            "committed_id": committed_id,
            "aphorism": info["aphorism"],
            "stage1": {"wr": s1["wr_ext"], "pass": s1_pass},
            "stage2": {"wr": s2["wr_ext"], "pass": s2_pass},
            "stage3": {"pass": s3_pass, "n_uplift": n_uplift,
                       "worst_domain_wr": min_dom_wr, "domains": dom_bd},
            "stage4": {"wr": claude_wr, "pass": s4_pass},
            "final": final,
        })

    # ---------- summary ----------
    print(f"\n{'='*60}\nFUNNEL SUMMARY\n{'='*60}")
    for stage_name, count in [
        ("Total candidates", len(results)),
        ("Stage 1 PASS (n=10 pilot)", sum(1 for r in results if r["stage1"]["pass"])),
        ("Stage 2 PASS (n=50 A/B)",   sum(1 for r in results if r["stage2"]["pass"])),
        ("Stage 3 PASS (cross-domain)", sum(1 for r in results if r["stage3"]["pass"])),
        ("Stage 4 PASS (claude)",    sum(1 for r in results
                                           if r["stage4"]["pass"] is True)),
        ("All stages PASS",           sum(1 for r in results if r["final"] == "PASS_ALL")),
    ]:
        print(f"  {stage_name:40s} {count}")

    print(f"\n{'cid':10s} {'wid':5s} {'S1':5s} {'S2':5s} {'S3':5s} {'S4':5s} final")
    print("-" * 60)
    for r in results:
        s1m = "P" if r["stage1"]["pass"] else "F"
        s2m = "P" if r["stage2"]["pass"] else "F"
        s3m = "P" if r["stage3"]["pass"] else "F"
        s4m = "—" if r["stage4"]["pass"] is None else ("P" if r["stage4"]["pass"] else "F")
        wid = r["committed_id"] or "----"
        print(f"  {r['cid']:9s} {wid:5s} {s1m:5s} {s2m:5s} {s3m:5s} {s4m:5s} "
              f"{r['final']}")

    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pilot_pids": [p["problem_id"] for p in pilot],
        "thresholds": {
            "stage1_pilot_wr": STAGE1_PASS_WR,
            "stage2_comparative_wr": STAGE2_PASS_WR,
            "stage3_no_catastrophe": STAGE3_NO_CATASTROPHE,
            "stage3_uplift_wr": STAGE3_UPLIFT_WR,
            "stage3_min_uplift_domains": STAGE3_MIN_UPLIFT_DOMAINS,
            "stage4_claude_wr": STAGE4_CLAUDE_WR,
        },
        "results": results,
    }
    log = cache_load(OUT_LOG, default=[])
    log.append(entry)
    cache_save(OUT_LOG, log)
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
