"""Exp 82 v2 step C2: LLM-as-classifier feature detector.

The keyword-regex `feature` proposer produced detectors with 0-12% recall on
SHOULD_FIRE pids — keywords don't capture the abstract trigger condition.
This C2 path replaces the detector with an LLM call: given (wisdom, problem),
ask the LLM to classify fire/no-fire.

For each (wisdom, pid in all 50 holdout):
  - call cheap-tier LLM with a classifier prompt
  - parse {fired: 0/1, reason}
  - bucket by exp17 ground-truth label (SHOULD_FIRE / NO_FIRE / NEUTRAL)
  - compute selectivity (same metrics as feature_eval_v2)

Output: feature_classify_summary.json — per-wisdom selectivity vs label.
Also re-grades fairly across all 9 usable wisdoms.

Concurrency: low (2 workers) to share quota with concurrent regrade.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from model_router import cheap  # noqa: E402

EXP_DIR = Path(__file__).parent
GOLD = EXP_DIR / "gold_answers.json"
TRIGGER_LABELS = PROJECT / "phase four" / "autonomous" / "exp17_trigger_labels.json"
MATRIX = EXP_DIR / "verdict_matrix.json"
OUT_SUMMARY = EXP_DIR / "feature_classify_summary.json"
LOG = EXP_DIR / "feature_classify_log.jsonl"


CLASSIFIER_PROMPT = """You are deciding whether a Chinese problem-solving aphorism applies to a given problem.

WISDOM:
  Aphorism: {aphorism}
  Source:   {source}
  Trigger signal (when wisdom applies): {signal}
  Unpacked: {unpacked}

PROBLEM:
{problem}

Decide: does this wisdom genuinely fit this problem's situation?
- Output 1 (fired) if the problem matches the wisdom's trigger signal — the
  problem clearly exhibits the failure mode the wisdom warns about.
- Output 0 (not fired) if the wisdom is irrelevant, only superficially related,
  or the problem doesn't have the conditions the wisdom addresses.

Be strict: only output 1 if the wisdom would change the answer in a meaningful
way for this specific problem. Generic relevance is NOT enough.

Output ONLY this JSON: {{"fired": 0 or 1, "reason": "one sentence"}}
"""


_RE = re.compile(r"\{[^{}]*?\"fired\"[^{}]*?\}", re.DOTALL)


def classify(wisdom: dict, problem: str, judge_client, hid: str = "", pid: str = "") -> int:
    prompt = CLASSIFIER_PROMPT.format(
        aphorism=wisdom.get("aphorism", ""),
        source=wisdom.get("source", ""),
        signal=wisdom.get("signal", ""),
        unpacked=wisdom.get("unpacked", "")[:1500],
        problem=problem[:2500],
    )
    last_err = None
    for attempt in range(3):
        try:
            r = judge_client.generate(prompt, max_tokens=200, temperature=0.0)
            text = (r.get("text") or "").strip()
            text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
            m = _RE.search(text)
            if not m:
                raise ValueError(f"no JSON in: {text[:200]}")
            obj = json.loads(m.group(0))
            fired = int(obj.get("fired", 0))
            if fired not in (0, 1):
                raise ValueError(f"fired={fired!r}")
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"pid": pid, "cid": hid, "fired": fired,
                                      "reason": obj.get("reason", "")[:200],
                                      "model": r.get("model", "")}, ensure_ascii=False) + "\n")
            return fired
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(1 + attempt)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"pid": pid, "cid": hid, "fired": 0,
                              "error": str(last_err)[:200]}, ensure_ascii=False) + "\n")
    return 0


def main():
    print("Loading wisdoms, gold, trigger labels...", flush=True)
    M = json.loads(MATRIX.read_text(encoding="utf-8"))
    cands = {c["cid"]: c for c in M["candidates"]}
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    labels = json.loads(TRIGGER_LABELS.read_text(encoding="utf-8"))

    # Use the same 9 usable wisdoms as full ablation (skip those with <5 SHOULD_FIRE)
    usable = []
    for cid, lab in labels.items():
        sf = sum(1 for v in lab.values() if v.get("verdict") == "SHOULD_FIRE")
        if sf >= 5:
            usable.append(cid)
    print(f"  usable wisdoms: {len(usable)} ({usable})", flush=True)

    # Use 3 cheap-tier models as classifier panel (so we can also see cross-classifier agreement)
    classifiers = {
        "gpt_mini": cheap("gpt_mini"),
        "gemini": cheap("gemini"),
        "claude_haiku": cheap("claude_haiku"),
    }
    print(f"  classifiers: {list(classifiers.keys())}", flush=True)

    LOG.write_text("")

    # build work items
    work = []
    for cid in usable:
        wisdom = cands[cid]
        for pid, rec in labels[cid].items():
            if pid not in gold:
                continue
            verdict = rec.get("verdict", "NA")
            if verdict in ("SHOULD_FIRE", "NO_FIRE", "NEUTRAL"):
                problem = gold[pid]["description"]
                for clf_name in classifiers:
                    work.append((clf_name, cid, pid, verdict, wisdom, problem))
    print(f"  total classify calls: {len(work)}", flush=True)

    # results[(clf, cid, pid)] = fired (0/1)
    results = {}
    print(f"\nRunning classify calls (4 workers)...", flush=True)
    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        fut_map = {}
        for clf_name, cid, pid, verdict, wisdom, problem in work:
            f = ex.submit(classify, wisdom, problem, classifiers[clf_name], cid, pid)
            fut_map[f] = (clf_name, cid, pid, verdict)
        for f in as_completed(fut_map):
            clf, cid, pid, verdict = fut_map[f]
            results[(clf, cid, pid)] = (f.result(), verdict)
            n_done += 1
            if n_done % 100 == 0:
                print(f"  [{n_done}/{len(work)}] {time.time()-t0:.0f}s", flush=True)

    # Aggregate per (classifier, cid)
    print(f"\n=== LLM-classifier feature selectivity per (classifier × wisdom) ===\n", flush=True)
    print(f"  {'wisdom':12s} | {'clf':14s} | {'fire SHOULD':>12s} | {'fire NO':>9s} | {'fire NEUT':>10s} | {'lift':>6s} | {'P':>4s} | {'R':>4s} | {'F1':>4s}", flush=True)
    print(f"  {'-'*12}-+-{'-'*14}-+-{'-'*12}-+-{'-'*9}-+-{'-'*10}-+-{'-'*6}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}", flush=True)
    summary: dict = {"by_classifier_wisdom": {}, "consensus": {}}
    for cid in usable:
        for clf in classifiers:
            bucket = defaultdict(lambda: [0, 0])
            for pid, rec in labels[cid].items():
                if pid not in gold:
                    continue
                v = rec.get("verdict")
                if v not in ("SHOULD_FIRE", "NO_FIRE", "NEUTRAL"):
                    continue
                fired, _ = results.get((clf, cid, pid), (0, v))
                if fired:
                    bucket[v][0] += 1
                bucket[v][1] += 1
            sf = bucket["SHOULD_FIRE"]
            nf = bucket["NO_FIRE"]
            nu = bucket["NEUTRAL"]
            fr_t = sf[0]/sf[1] if sf[1] else 0
            fr_o = nf[0]/nf[1] if nf[1] else 0
            fr_n = nu[0]/nu[1] if nu[1] else 0
            tp = sf[0]; fp = nf[0] + nu[0]; fn = sf[1] - sf[0]
            P = tp/(tp+fp) if (tp+fp) else 0
            R = tp/(tp+fn) if (tp+fn) else 0
            F1 = 2*P*R/(P+R) if (P+R) else 0
            summary["by_classifier_wisdom"][f"{clf}/{cid}"] = {
                "fire_rate_should_fire": fr_t,
                "fire_rate_no_fire": fr_o,
                "fire_rate_neutral": fr_n,
                "selectivity_lift": fr_t - fr_o,
                "precision": P, "recall": R, "f1": F1,
                "n_should_fire": sf[1], "n_no_fire": nf[1], "n_neutral": nu[1],
                "tp": tp, "fp": fp, "fn": fn,
            }
            print(f"  {cid:12s} | {clf:14s} | {sf[0]:2d}/{sf[1]:2d} ({fr_t:>4.0%}) | {nf[0]:2d}/{nf[1]:2d} ({fr_o:>3.0%}) | {nu[0]:2d}/{nu[1]:2d} ({fr_n:>3.0%}) | {fr_t-fr_o:+5.2f} | {P:.2f} | {R:.2f} | {F1:.2f}", flush=True)
        # 3-classifier consensus per wisdom
        consensus_fired = {}
        for pid, rec in labels[cid].items():
            if pid not in gold:
                continue
            v = rec.get("verdict")
            if v not in ("SHOULD_FIRE", "NO_FIRE", "NEUTRAL"):
                continue
            votes = sum(results.get((clf, cid, pid), (0, v))[0] for clf in classifiers)
            consensus_fired[pid] = (1 if votes >= 2 else 0, v)
        bucket = defaultdict(lambda: [0, 0])
        for pid, (fired, v) in consensus_fired.items():
            if fired:
                bucket[v][0] += 1
            bucket[v][1] += 1
        sf = bucket["SHOULD_FIRE"]
        nf = bucket["NO_FIRE"]
        fr_t = sf[0]/sf[1] if sf[1] else 0
        fr_o = nf[0]/nf[1] if nf[1] else 0
        tp = sf[0]; fp = nf[0] + bucket["NEUTRAL"][0]; fn = sf[1] - sf[0]
        P = tp/(tp+fp) if (tp+fp) else 0
        R = tp/(tp+fn) if (tp+fn) else 0
        F1 = 2*P*R/(P+R) if (P+R) else 0
        summary["consensus"][cid] = {
            "fire_rate_should_fire": fr_t, "fire_rate_no_fire": fr_o,
            "selectivity_lift": fr_t - fr_o, "precision": P, "recall": R, "f1": F1,
        }

    # Print consensus row
    print(f"\n=== 3-classifier consensus (≥2 of 3 agree) ===\n", flush=True)
    print(f"  {'wisdom':12s} | {'fire SHOULD':>12s} | {'fire NO':>9s} | {'lift':>6s} | {'F1':>4s}", flush=True)
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*9}-+-{'-'*6}-+-{'-'*4}", flush=True)
    for cid, m in summary["consensus"].items():
        print(f"  {cid:12s} | {m['fire_rate_should_fire']:>11.0%}  |  {m['fire_rate_no_fire']:>6.0%}  | {m['selectivity_lift']:+5.2f} | {m['f1']:.2f}", flush=True)

    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_SUMMARY.relative_to(PROJECT)}", flush=True)


if __name__ == "__main__":
    main()
