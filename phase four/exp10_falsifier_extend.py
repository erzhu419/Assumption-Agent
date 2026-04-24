"""Exp 10: Falsifier of Exp 3's meta-wisdom.

Exp 3 proposed (paraphrased): for candidates with wr in [0.53, 0.63],
extend to n=100 or run a second independent A/B. If gray-zone
candidates stabilise wr < 0.55 at n>=100, the current gate is too
strict AND the old wr values were noise-dominated. If any gray-zone
candidate stabilises above 0.60 under extension, the gate is the
problem, not the candidate.

We test this by running all SIX candidates in the gray zone
([0.53, 0.63]) — including the three original KEEPs — on a FRESH
50-problem held-out extension sampled disjoint from train and the
original holdout. We generate v20 base + ext answers and judge
them with the gemini-3-flash + Claude Opus combination already in
use.

Result interpretation:
  new_wr < 0.55 on the extension → consistent with old gate's decision
  new_wr in [0.55, 0.60)  → gray on extension too; meta-wisdom
                             would say "needs more n"
  new_wr >= 0.60         → candidate would pass new gate on the
                             extension (different verdict from
                             original gate decision)
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
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
OUT_LOG = AUTO_DIR / "exp10_falsifier_log.json"

V20_SCRIPT = PROJECT / "phase one" / "scripts" / "validation" / "phase2_v20_framework.py"
EXTEND_SAMPLE = "sample_extend_50.json"
EXEMPLARS_PATH = CACHE / "wisdom_diverse_exemplars.json"

PARALLEL_V20 = 6
PARALLEL_JUDGES = 6

# Six gray-zone candidates (original gemini wr ∈ [0.53, 0.64], including
# the three KEEPs which are at wr=0.60, 0.60, 0.64).
GRAY_ZONE = [
    {"cid": "WCAND05", "wid": "W076", "wr": 0.64, "aphorism": "凡益之道，与时偕行",
     "tentative_id": "WCAND05"},
    {"cid": "WCAND10", "wid": "W077", "wr": 0.60, "aphorism": "没有调查，就没有发言权",
     "tentative_id": "WCAND10"},
    {"cid": "WCROSSL01", "wid": "W078", "wr": 0.60, "aphorism": "是骡子是马，拉出来遛遛",
     "tentative_id": "WCROSSL01"},
    {"cid": "WCAND01", "wid": None, "wr": 0.58, "aphorism": "上工治未病，不治已病",
     "tentative_id": "WCAND01"},
    {"cid": "WCAND03", "wid": None, "wr": 0.56, "aphorism": "凡事预则立，不预则废",
     "tentative_id": "WCAND03"},
    {"cid": "WCAND11", "wid": None, "wr": 0.54, "aphorism": "若不是品牌，你就只是商品。",
     "tentative_id": "WCAND11"},
]


def cache_load(p, default=None):
    if Path(p).exists():
        try: return json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception: return default
    return default


def cache_save(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def run_v20(variant, sample_file, wisdom_file, n):
    cmd = ["python", "-u", str(V20_SCRIPT),
           "--variant", variant, "--n", str(n),
           "--sample", sample_file, "--wisdom", wisdom_file]
    start = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - start
    if res.returncode != 0:
        print(f"    [{variant} FAILED] {res.stderr[-200:]}")
        return None, elapsed
    ans = cache_load(CACHE / "answers" / f"{variant}_answers.json")
    return ans, elapsed


def run_v20_parallel(tasks):
    out = {}
    with ThreadPoolExecutor(max_workers=PARALLEL_V20) as ex:
        futs = {ex.submit(run_v20, *t): t[0] for t in tasks}
        for f in as_completed(futs):
            name = futs[f]
            try:
                ans, dt = f.result()
                out[name] = (ans, dt)
                n_ans = len(ans) if ans else 0
                print(f"  [DONE] {name}: {n_ans} answers in {dt:.0f}s")
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")
                out[name] = (None, 0)
    return out


def judge_batch(problems, ans_base, ans_ext, client, judge_name="gemini"):
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
            if judge_name == "gemini":
                v = judge_pair(client, p["description"], left, right)
                w = v.get("winner", "tie")
            else:
                from exp9_claude_on_reverts import JUDGE_PROMPT
                r = client.generate(
                    JUDGE_PROMPT.format(problem=p["description"],
                                        answer_a=left, answer_b=right),
                    max_tokens=400, temperature=0.0)
                v = parse_json_from_llm(r["text"])
                w = v.get("winner", "tie")
        except Exception as e:
            return pid, f"err:{e}"[:30]
        if w == "tie": return pid, "tie"
        if w in ("A", "B"):
            return pid, ("ext" if w == ext_was else "base")
        return pid, "err"
    c = {"ext": 0, "base": 0, "tie": 0, "missing": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=PARALLEL_JUDGES) as ex:
        futs = [ex.submit(one, p) for p in problems]
        for f in as_completed(futs):
            _pid, res = f.result()
            if res.startswith("err"):
                c["error"] += 1
            else:
                c[res] = c.get(res, 0) + 1
    if judge_name == "gemini":
        _save_content_cache()
    tot = c["ext"] + c["base"]
    return {**c, "wr_ext": c["ext"] / tot if tot else 0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-claude", action="store_true")
    args = ap.parse_args()

    problems = json.loads((CACHE / EXTEND_SAMPLE).read_text(encoding="utf-8"))
    problems = [p for p in problems if "description" in p]
    print(f"Extension sample: {len(problems)} problems (disjoint from train & holdout)\n")

    # Build ext library JSON files (base + each candidate)
    # Base lib is the 75-wisdom original (same as v20.1 base library)
    base_lib_src = "_valp_v20p1_base_library.json"
    if not (CACHE / base_lib_src).exists():
        print(f"[FATAL] need {base_lib_src} (pre-prune 75-wisdom library)")
        return
    base_lib = json.loads((CACHE / base_lib_src).read_text(encoding="utf-8"))

    exemplars_all = cache_load(EXEMPLARS_PATH, default={})
    ext_tasks = []
    for cand in GRAY_ZONE:
        tid = cand["tentative_id"]
        # Find the original candidate record (to get aphorism, source, etc.)
        src_candidate = None
        for src_file in ("success_distilled_candidates.json",
                          "cross_llm_candidates.json"):
            src_data = cache_load(AUTO_DIR / src_file, default=[])
            for c in src_data:
                if c.get("aphorism") == cand["aphorism"]:
                    src_candidate = c; break
            if src_candidate: break
        if not src_candidate:
            print(f"  [{cand['cid']}] source record not found; skipping")
            continue

        # Exemplars: use committed_id if KEEP, else the tentative id from earlier
        if cand["wid"] and cand["wid"] in exemplars_all:
            # already exists
            pass
        elif tid in exemplars_all:
            pass
        else:
            print(f"  [{cand['cid']}] no exemplars cached; skipping")
            continue

        # Build ext library
        META = {"_cluster_id", "_cluster_size", "_source", "novelty_sim",
                "covers_batch_pids", "rationale", "tentative_id",
                "_evidence_pids"}
        cand_entry = {k: v for k, v in src_candidate.items() if k not in META}
        cand_entry["id"] = cand["wid"] or tid
        ext_lib = base_lib + [cand_entry]
        lib_filename = f"_exp10_ext_{tid}.json"
        cache_save(CACHE / lib_filename, ext_lib)
        variant = f"_exp10_v20_ext_{tid}"
        ext_tasks.append((variant, EXTEND_SAMPLE, lib_filename, len(problems)))

    # Base run
    base_variant = "_exp10_v20_base"
    tasks = [(base_variant, EXTEND_SAMPLE, base_lib_src, len(problems))] + ext_tasks
    print(f"[1/3] Running {len(tasks)} v20 subprocesses in parallel "
          f"(up to {PARALLEL_V20})...")
    t0 = time.time()
    v20_out = run_v20_parallel(tasks)
    print(f"  All v20 done in {time.time()-t0:.0f}s\n")

    ans_base = v20_out.get(base_variant, (None, 0))[0]
    if not ans_base:
        print("[FATAL] base failed"); return

    # Judge
    gemini = create_client()
    claude = None if args.skip_claude else ClaudeProxyClient()
    print(f"[2/3] Gemini judging extension pairs...")
    results = []
    for cand in GRAY_ZONE:
        tid = cand["tentative_id"]
        variant = f"_exp10_v20_ext_{tid}"
        ans_ext = v20_out.get(variant, (None, 0))[0]
        if not ans_ext:
            print(f"  [{cand['cid']}] ext missing"); continue
        # Gemini extension
        g = judge_batch(problems, ans_base, ans_ext, gemini, "gemini")
        print(f"  [{cand['cid']:9s} {cand['aphorism'][:18]:18s}]  gemini_ext_wr={g['wr_ext']:.2f}  "
              f"({g['ext']}:{g['base']}:{g['tie']}t)")
        # Claude extension
        c_res = None
        if claude is not None:
            c_res = judge_batch(problems, ans_base, ans_ext, claude, "claude")
            print(f"     claude_ext_wr={c_res['wr_ext']:.2f}  "
                  f"({c_res['ext']}:{c_res['base']}:{c_res['tie']}t)")

        # Combined n=100: add original 50 verdicts + these 50
        combined_gemini = {
            "ext": cand["wr"] * 50 if cand["wr"] else 0,  # approximate
            # For cleaner math: use the reported ext wins from original
        }
        # Actually let's use exact counts
        wr_orig = cand["wr"]
        ext_orig = int(round(wr_orig * 50))  # 0.64 -> 32
        base_orig = 50 - ext_orig
        ext_new = g["ext"]; base_new = g["base"]
        wr_combined_gemini = (ext_orig + ext_new) / (ext_orig + base_orig + ext_new + base_new)
        print(f"     combined n=100 gemini_wr={wr_combined_gemini:.2f}")

        results.append({
            "cid": cand["cid"], "wid": cand["wid"],
            "aphorism": cand["aphorism"],
            "original_wr_gemini": cand["wr"],
            "extension_wr_gemini": g["wr_ext"],
            "extension_wr_claude": c_res["wr_ext"] if c_res else None,
            "combined_n100_wr_gemini": wr_combined_gemini,
            "extension_gemini_summary": g,
            "extension_claude_summary": c_res,
        })

    log = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "extension_sample": EXTEND_SAMPLE, "n_extension": len(problems),
           "results": results}
    prev = json.loads(OUT_LOG.read_text()) if OUT_LOG.exists() else []
    prev.append(log)
    OUT_LOG.write_text(json.dumps(prev, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}\n")

    # Summary + falsifier check
    print(f"[3/3] Falsifier check (Exp 3 meta-wisdom):")
    print(f"  {'cid':10s} {'wid':5s} {'orig':6s} {'ext_g':6s} {'ext_c':6s} "
          f"{'n100_g':6s}  fate")
    print("-" * 72)
    stable_below_055 = 0; stable_above_060 = 0; still_gray = 0
    for r in results:
        orig = r["original_wr_gemini"]
        ext_g = r["extension_wr_gemini"]
        ext_c = r["extension_wr_claude"]
        n100 = r["combined_n100_wr_gemini"]
        if n100 < 0.55:
            fate = "STABLE_<0.55 (candidate truly weak)"
            stable_below_055 += 1
        elif n100 >= 0.60:
            fate = "STABLE_>=0.60 (gate too strict)"
            stable_above_060 += 1
        else:
            fate = "STILL_GRAY"
            still_gray += 1
        ec_s = f"{ext_c:.2f}" if ext_c is not None else "  —"
        print(f"  {r['cid']:9s} {(r['wid'] or '----'):4s} {orig:<6.2f} "
              f"{ext_g:<6.2f} {ec_s:6s} {n100:<6.2f}  {fate}")

    n = len(results)
    print(f"\n  Exp 3 predicted: gray-zone candidates stabilise <0.55 at n>=100.")
    print(f"    stable_<0.55:  {stable_below_055}/{n}  (supports Exp 3 meta-wisdom)")
    print(f"    stable_>=0.60: {stable_above_060}/{n}  (falsifies: gate IS too strict)")
    print(f"    still_gray:    {still_gray}/{n}  (inconclusive; need more n)")


if __name__ == "__main__":
    main()
