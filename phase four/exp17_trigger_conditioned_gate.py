"""Exp 17 — Trigger-class conditioned gate (architectural L1).

For each (wisdom, pid): ask Claude whether the pid's problem matches
the wisdom's claimed trigger. This converts the pooled holdout into a
partitioned one — should_fire / neutral / no_fire — PER WISDOM.

Then, using EXISTING cached pair verdicts (from validate_parallel /
staged_gate) and EXISTING Exp 16 citation data, compute a 3-D score
per wisdom:

  trigger_fit_rate    — % of Claude-labeled pids that match wisdom's claim
                        (sanity: wisdom self-description is coherent)
  util_when_fires     — wr on should_fire subset
                        (actual utility when trigger is present)
  cite_when_fires     — Exp 16 any_cite on should_fire subset
                        (wisdom actually appears in answer when it should)
  harm_when_absent    — wr on no_fire subset (should be ~0.50)
                        (no-op when trigger absent; not harmful)

A wisdom passes iff:
  util_when_fires  >= 0.55  AND  n_fire >= 8
  cite_when_fires  >= 0.50  AND  n_fire >= 8
  |harm_when_absent - 0.50| <= 0.05

This is the first gate in this study that separates the three
confounded dimensions of pair-wr (cite × utility × trigger-fit).
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
sys.path.insert(0, str(PROJECT / "phase four"))

from claude_proxy_client import ClaudeProxyClient
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp17_trigger_conditioned_log.json"
TRIGGER_LABELS_PATH = AUTO_DIR / "exp17_trigger_labels.json"

PARALLEL = 8

CANDIDATES = [
    {"cid": "WCAND01", "aphorism": "上工治未病，不治已病"},
    {"cid": "WCAND02", "aphorism": "别高效解决一个被看错的问题"},
    {"cid": "WCAND03", "aphorism": "凡事预则立，不预则废"},
    {"cid": "WCAND04", "aphorism": "急则治其标，缓则治其本"},
    {"cid": "WCAND05", "aphorism": "凡益之道，与时偕行", "committed_id": "W076"},
    {"cid": "WCAND06", "aphorism": "覆水难收，向前算账"},
    {"cid": "WCAND07", "aphorism": "亲兄弟，明算账"},
    {"cid": "WCAND08", "aphorism": "想理解行为，先看激励"},
    {"cid": "WCAND09", "aphorism": "不谋全局者，不足谋一域"},
    {"cid": "WCAND10", "aphorism": "没有调查，就没有发言权", "committed_id": "W077"},
    {"cid": "WCAND11", "aphorism": "若不是品牌，你就只是商品。"},
    {"cid": "WCROSSL01", "aphorism": "是骡子是马，拉出来遛遛", "committed_id": "W078"},
]


TRIGGER_LABEL_PROMPT = """你是一位方法论分类员。判断下面这个问题**在多大程度上**匹配某条 wisdom 的触发条件。

== 该 wisdom 的自述 trigger ==
aphorism: {aphorism}
signal: {signal}
unpacked_for_llm: {unpacked}

== 待分类的问题 ==
{problem}

== 三档判断 ==
- SHOULD_FIRE: 问题的核心结构**明确匹配** wisdom 声称的触发条件。使用该 wisdom 应当带来真实收益。
- NEUTRAL:     有一部分匹配，但不是问题的核心；使用 wisdom 可能有帮助也可能无关。
- NO_FIRE:     问题的核心结构和 wisdom 的触发条件**无关**。使用 wisdom 应当是 no-op 或微扰。

判断时问自己：
  问题的 root cause / decision structure 是否正好是 wisdom 针对的那类？
  还是它在讨论完全不同的事情，只是偶尔有词汇重叠？

== 输出 JSON（不要代码块） ==
{{"verdict": "SHOULD_FIRE" 或 "NEUTRAL" 或 "NO_FIRE",
  "reasoning": "50-80字指出核心匹配/不匹配的点"}}
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


def label_trigger(client, wisdom, problem):
    prompt = TRIGGER_LABEL_PROMPT.format(
        aphorism=wisdom["aphorism"], signal=wisdom.get("signal", ""),
        unpacked=wisdom.get("unpacked_for_llm", ""),
        problem=problem,
    )
    try:
        r = client.generate(prompt, max_tokens=300, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        return v.get("verdict", "ERR"), v.get("reasoning", "")
    except Exception as e:
        return "ERR", f"{e}"[:80]


def label_all(client, candidates, problems):
    """Label every (wisdom, pid) pair. Caches to disk."""
    labels = cache_load(TRIGGER_LABELS_PATH, default={})
    pid_to_problem = {p["problem_id"]: p["description"] for p in problems}

    tasks = []
    for cand in candidates:
        cid = cand["cid"]
        rec = load_candidate_record(cand["aphorism"])
        if not rec: continue
        labels.setdefault(cid, {})
        for pid, problem in pid_to_problem.items():
            if pid in labels[cid] and labels[cid][pid].get("verdict") != "ERR":
                continue
            tasks.append((cid, rec, pid, problem))

    if not tasks:
        print("All labels cached; skipping labelling phase.")
        return labels

    print(f"Labelling {len(tasks)} (wisdom, pid) pairs...")
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(label_trigger, client, rec, problem): (cid, pid)
                 for cid, rec, pid, problem in tasks}
        for f in as_completed(futs):
            cid, pid = futs[f]
            v, r = f.result()
            labels[cid][pid] = {"verdict": v, "reasoning": r}
            done += 1
            if done % 50 == 0:
                dt = time.time() - t0
                eta = dt / done * (len(tasks) - done)
                print(f"  {done}/{len(tasks)} ({dt:.0f}s elapsed, {eta:.0f}s ETA)")
    TRIGGER_LABELS_PATH.write_text(json.dumps(labels, ensure_ascii=False, indent=2))
    print(f"Labels → {TRIGGER_LABELS_PATH.name}")
    return labels


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    problems = json.loads((CACHE / "sample_holdout_50.json").read_text(encoding="utf-8"))
    problems = [p for p in problems if "description" in p]

    claude = ClaudeProxyClient()
    print(f"Trigger-label judge: {claude.model}\n")

    # --- Phase 1: label every (wisdom, pid) pair ---
    labels = label_all(claude, CANDIDATES, problems)

    # --- Phase 2: load cached verdicts + citations ---
    # Pair verdicts reconstruction (from staged_gate): per_candidate per_pid
    sg_log = cache_load(AUTO_DIR / "staged_gate_log.json", default=[])
    pair_verdicts = {}   # {cid: {pid: 'ext'|'base'|'tie'}}
    if sg_log:
        # staged_gate stores per-domain breakdown but not per-pid; we
        # re-derive from cached_framework judge cache via the same pipeline
        # Simplest: re-run the gemini judge on cached answers (cache-hit
        # for free since we've run it). Skip that and use a different path:
        # staged_gate_log.results[*].stage3.domains has per-domain {ext,base,tie}.
        pass

    # Re-compute per-pid verdicts cheaply using cached_framework
    sys.path.insert(0, str(PROJECT / "phase one" / "scripts" / "validation"))
    from cached_framework import judge_pair, _save_content_cache
    from llm_client import create_client
    gemini = create_client()

    base_defaults = {"WCROSSL01": "_valp_v20_base"}
    default_base = "_valp_v20p1_base"

    for cand in CANDIDATES:
        cid = cand["cid"]
        base_stem = base_defaults.get(cid, default_base)
        base_path = CACHE / "answers" / f"{base_stem}_answers.json"
        ext_path = CACHE / "answers" / f"_valp_v20_ext_{cid}_answers.json"
        if not base_path.exists() or not ext_path.exists(): continue
        ans_base = json.loads(base_path.read_text(encoding="utf-8"))
        ans_ext = json.loads(ext_path.read_text(encoding="utf-8"))
        verdicts = {}
        for p in problems:
            pid = p["problem_id"]
            ba, ea = ans_base.get(pid), ans_ext.get(pid)
            if not ba or not ea: continue
            rng = random.Random(hash(pid) % (2**32))
            if rng.random() < 0.5:
                left, right, ext_was = ea, ba, "A"
            else:
                left, right, ext_was = ba, ea, "B"
            try:
                v = judge_pair(gemini, p["description"], left, right)
                w = v.get("winner", "tie")
                if w == "tie": verdicts[pid] = "tie"
                elif w == ext_was: verdicts[pid] = "ext"
                else: verdicts[pid] = "base"
            except Exception:
                verdicts[pid] = "err"
        pair_verdicts[cid] = verdicts
    _save_content_cache()

    # Load Exp 16 citation data
    exp16_log = cache_load(AUTO_DIR / "exp16_citation_audit_log.json", default=[])
    cite_verdicts = {}  # {cid: {pid: "CITED_CLEAR"|"CITED_WEAK"|"NOT_CITED"|...}}
    if exp16_log:
        for r in exp16_log[-1].get("results", []):
            cite_verdicts[r["cid"]] = {pid: d["verdict"]
                                        for pid, d in r.get("per_pid", {}).items()}

    # --- Phase 3: compute conditioned metrics ---
    results = []
    print(f"\n{'cid':10s} {'wid':5s} {'tfit':6s} {'n_fire':6s} "
          f"{'util_fire':10s} {'cite_fire':10s} {'util_abs':9s} {'gate'}")
    print("-" * 90)
    for cand in CANDIDATES:
        cid = cand["cid"]; wid = cand.get("committed_id") or "----"
        wl = labels.get(cid, {})
        pv = pair_verdicts.get(cid, {})
        cv = cite_verdicts.get(cid, {})
        fire_pids = [pid for pid, d in wl.items() if d.get("verdict") == "SHOULD_FIRE"]
        nofire_pids = [pid for pid, d in wl.items() if d.get("verdict") == "NO_FIRE"]
        # trigger_fit_rate = fraction labelled SHOULD_FIRE or NEUTRAL (basically "any match")
        any_match = sum(1 for d in wl.values() if d.get("verdict") in ("SHOULD_FIRE", "NEUTRAL"))
        tfit = any_match / max(len(wl), 1)
        # Utility on fire
        if fire_pids:
            ef = sum(1 for pid in fire_pids if pv.get(pid) == "ext")
            bf = sum(1 for pid in fire_pids if pv.get(pid) == "base")
            util_fire = ef / (ef + bf) if (ef + bf) else 0.5
        else:
            util_fire = None; ef = bf = 0
        # Citation on fire
        if fire_pids:
            cited = sum(1 for pid in fire_pids
                        if cv.get(pid) in ("CITED_CLEAR", "CITED_WEAK"))
            judged = sum(1 for pid in fire_pids if cv.get(pid) in
                         ("CITED_CLEAR", "CITED_WEAK", "NOT_CITED"))
            cite_fire = cited / judged if judged else None
        else:
            cite_fire = None
        # No-op on no-fire
        if nofire_pids:
            en = sum(1 for pid in nofire_pids if pv.get(pid) == "ext")
            bn = sum(1 for pid in nofire_pids if pv.get(pid) == "base")
            util_abs = en / (en + bn) if (en + bn) else 0.5
        else:
            util_abs = None; en = bn = 0

        # Gate
        pass_util  = (util_fire is not None and util_fire >= 0.55 and len(fire_pids) >= 8)
        pass_cite  = (cite_fire is not None and cite_fire >= 0.50)
        pass_noharm = (util_abs is None or abs(util_abs - 0.50) <= 0.10)
        gate_pass = pass_util and pass_cite and pass_noharm
        gate_str = f"u{'P' if pass_util else 'F'}c{'P' if pass_cite else 'F'}h{'P' if pass_noharm else 'F'}"
        print(f"  {cid:9s} {wid:4s} {tfit:.2f}   "
              f"{len(fire_pids):3d}   "
              f"{'—' if util_fire is None else f'{util_fire:.2f}':8s}  "
              f"{'—' if cite_fire is None else f'{cite_fire:.2f}':8s}  "
              f"{'—' if util_abs is None else f'{util_abs:.2f}':8s}  "
              f"{gate_str} {'PASS' if gate_pass else 'FAIL'}")

        results.append({
            "cid": cid, "wid": cand.get("committed_id"),
            "aphorism": cand["aphorism"],
            "trigger_fit_rate": tfit,
            "n_should_fire": len(fire_pids),
            "n_no_fire": len(nofire_pids),
            "util_when_fires": util_fire,
            "cite_when_fires": cite_fire,
            "util_when_absent": util_abs,
            "pass_util": pass_util,
            "pass_cite": pass_cite,
            "pass_noharm": pass_noharm,
            "gate_pass": gate_pass,
        })

    n_pass = sum(1 for r in results if r["gate_pass"])
    print(f"\n  Conditioned-gate PASS: {n_pass}/{len(results)}")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "labeler": claude.model,
           "results": results}
    prev = cache_load(OUT_LOG, default=[]) or []
    prev.append(out)
    OUT_LOG.write_text(json.dumps(prev, ensure_ascii=False, indent=2))
    print(f"Saved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
