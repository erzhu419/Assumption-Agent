"""Exp 82 v2 step A.2: STRICT cross-judge regrade.

The first cross-judge (regrade.py) showed gemini/haiku at 99%+ BASE pass
rate — they were grading too leniently to distinguish EXT from BASE. This
strict version uses a more demanding rubric: extract 3-5 key checkpoints
from the gold answer FIRST, then grade the candidate on coverage.

Output: cross_judge_strict_summary.json
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
FORENSIC = EXP_DIR / "forensic.jsonl"
HYPOS = EXP_DIR / "hypotheses.jsonl"
GOLD = EXP_DIR / "gold_answers.json"
SUMMARY = EXP_DIR / "cross_judge_strict_summary.json"
REGRADE_LOG = EXP_DIR / "regrade_strict_forensic.jsonl"

ROLE_TO_KIND_EXT = {
    "feature": None,
    "constraint": "constraint",
    "decomposition": "ext_decomposition",
    "verification": "verify_pass2",
    "hp_change": "ext_hp_change",
}


GRADE_PROMPT_STRICT = """You are a STRICT grader. You will compare a candidate answer to a reference (gold) answer.

PROBLEM:
{problem}

GOLD ANSWER:
{gold}

CANDIDATE ANSWER:
{candidate}

INSTRUCTIONS:
1. From the GOLD answer, identify 3-5 KEY CHECKPOINTS — specific facts, numerical results, named methods, decision criteria, or required reasoning steps that the gold answer makes.
2. For each checkpoint, decide whether the CANDIDATE addresses it correctly (matching the gold's stance — not just mentioning the topic).
3. Output the count of checkpoints addressed.

A candidate is CORRECT (correct=1) only if ALL or all-but-one of the checkpoints are addressed correctly.
A candidate is INCORRECT (correct=0) if it misses 2+ checkpoints, contradicts the gold, or is non-responsive.

Be strict: a candidate that paraphrases the problem, restates without solving, makes generic statements, or lacks specific reasoning is INCORRECT even if it sounds plausible.

Output ONLY this JSON object:
{{"checkpoints": ["checkpoint 1", "checkpoint 2", ...], "candidate_addresses": [0/1, 0/1, ...], "correct": 0 or 1, "reason": "one sentence"}}

No markdown, no commentary outside the JSON.
"""

_GRADE_RE = re.compile(r"\{[\s\S]*?\"correct\"[\s\S]*?\}", re.DOTALL)


def grade_call(judge_client, problem: str, gold: str, candidate: str,
                judge_name: str, hid: str, pid: str, condition: str) -> int:
    if not (candidate or "").strip():
        with open(REGRADE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"judge": judge_name, "hid": hid, "pid": pid,
                                  "condition": condition, "result": 0,
                                  "reason": "empty"}, ensure_ascii=False) + "\n")
        return 0
    prompt = GRADE_PROMPT_STRICT.format(problem=problem[:3000], gold=gold[:3000], candidate=candidate[:3000])
    last_err = None
    for attempt in range(3):
        try:
            r = judge_client.generate(prompt, max_tokens=500, temperature=0.0)
            text = (r.get("text") or "").strip()
            text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
            # Use a more permissive bracket match
            depth = 0
            start = -1
            end = -1
            for i, ch in enumerate(text):
                if ch == "{":
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if start < 0 or end < 0:
                raise ValueError(f"no JSON: {text[:200]}")
            obj = json.loads(text[start:end])
            correct = int(obj.get("correct", 0))
            if correct not in (0, 1):
                raise ValueError(f"correct={correct!r}")
            checkpoints = obj.get("checkpoints", [])
            addresses = obj.get("candidate_addresses", [])
            with open(REGRADE_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"judge": judge_name, "hid": hid, "pid": pid,
                                      "condition": condition, "result": correct,
                                      "n_checkpoints": len(checkpoints),
                                      "n_addressed": sum(1 for a in addresses if a),
                                      "reason": obj.get("reason", "")[:200],
                                      "judge_model": r.get("model", ""),
                                      "attempt": attempt}, ensure_ascii=False) + "\n")
            return correct
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(1 + attempt)
    with open(REGRADE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"judge": judge_name, "hid": hid, "pid": pid,
                              "condition": condition, "result": 0,
                              "error": str(last_err)[:200]}, ensure_ascii=False) + "\n")
    return 0


# ── reuse find_* helpers from regrade.py ─────────────────────────────
def find_answer_for(records, hid, pid, role_target):
    if role_target is None:
        return ""
    if role_target == "constraint":
        candidates = [r for r in records
                      if r.get("hid") == hid and r.get("pid") == pid
                      and r.get("role") in ("constraint_pass1", "constraint_retry1", "constraint_retry2")
                      and r.get("answer_chars")]
    else:
        candidates = [r for r in records
                      if r.get("hid") == hid and r.get("pid") == pid
                      and r.get("role") == role_target
                      and r.get("answer_chars")]
    if not candidates:
        return ""
    candidates.sort(key=lambda r: r.get("ts", ""))
    return candidates[-1]["answer_chars"]


def find_base(records, hid, pid):
    rs = [r for r in records
          if r.get("hid") == hid and r.get("pid") == pid
          and r.get("role") in ("base_cached", "base_fresh")
          and r.get("answer_chars")]
    if not rs:
        return ""
    rs.sort(key=lambda r: r.get("ts", ""))
    return rs[-1]["answer_chars"]


def find_generic(records, hid, pid):
    rs = [r for r in records
          if r.get("hid") == hid and r.get("pid") == pid
          and r.get("role") == "generic"
          and r.get("answer_chars")]
    if not rs:
        return ""
    rs.sort(key=lambda r: r.get("ts", ""))
    return rs[-1]["answer_chars"]


def main():
    print("Loading...", flush=True)
    records = [json.loads(l) for l in open(FORENSIC) if l.strip()]
    hypos = [json.loads(l) for l in open(HYPOS) if l.strip()]
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    print(f"  forensic: {len(records)}, hypotheses: {len(hypos)}, gold: {len(gold)}", flush=True)

    # Use 3 judges this time for STRICT cross-judge: gpt_mini + gemini + claude_haiku
    # All using the strict rubric — fairest comparison.
    judges = {
        "gpt_mini_strict": cheap("gpt_mini"),
        "gemini_strict": cheap("gemini"),
        "claude_haiku_strict": cheap("claude_haiku"),
    }
    print(f"  judges: {list(judges.keys())}", flush=True)

    # Build work items
    work_items = []
    per_hid_pids = {}
    for h in hypos:
        hid = h["hid"]
        kind = h["kind"]
        ext_role = ROLE_TO_KIND_EXT[kind]
        per_hid_pids[hid] = {"trigger": list(h["trigger_subset"]),
                              "outside": list(h["outside_subset"])}
        for pid in h["trigger_subset"] + h["outside_subset"]:
            if pid not in gold:
                continue
            problem = gold[pid]["description"]
            g = gold[pid]["gold"]
            base = find_base(records, hid, pid)
            ext = base if kind == "feature" else find_answer_for(records, hid, pid, ext_role)
            generic = find_generic(records, hid, pid)
            for jn in judges:
                if base:
                    work_items.append((jn, hid, pid, "base", problem, g, base))
                if ext:
                    work_items.append((jn, hid, pid, "ext", problem, g, ext))
                if generic:
                    work_items.append((jn, hid, pid, "generic", problem, g, generic))
    print(f"  total grade calls: {len(work_items)}", flush=True)

    REGRADE_LOG.write_text("")

    print(f"\nRunning grade calls (8 workers)...", flush=True)
    results = defaultdict(int)
    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut_map = {}
        for it in work_items:
            jn, hid, pid, cond, problem, g, candidate = it
            f = ex.submit(grade_call, judges[jn], problem, g, candidate,
                            jn, hid, pid, cond)
            fut_map[f] = (jn, hid, pid, cond)
        for f in as_completed(fut_map):
            jn, hid, pid, cond = fut_map[f]
            results[(jn, hid, pid, cond)] = f.result()
            n_done += 1
            if n_done % 200 == 0:
                print(f"  [{n_done}/{len(work_items)}] {time.time()-t0:.0f}s", flush=True)

    # Aggregate
    print(f"\nAggregating per hypothesis × judge...", flush=True)
    by_hyp = {}
    for h in hypos:
        hid = h["hid"]
        kind = h["kind"]
        seed = h["seed_cid"]
        trig = per_hid_pids[hid]["trigger"]
        out = per_hid_pids[hid]["outside"]
        per_hyp = {"hid": hid, "seed_cid": seed, "kind": kind, "claim": h["claim"][:120]}
        for jn in judges:
            n_t = sum(1 for pid in trig if pid in gold)
            n_o = sum(1 for pid in out if pid in gold)
            base_t = sum(results.get((jn, hid, pid, "base"), 0) for pid in trig if pid in gold)
            ext_t = sum(results.get((jn, hid, pid, "ext"), 0) for pid in trig if pid in gold)
            base_o = sum(results.get((jn, hid, pid, "base"), 0) for pid in out if pid in gold)
            ext_o = sum(results.get((jn, hid, pid, "ext"), 0) for pid in out if pid in gold)
            gen_t = sum(results.get((jn, hid, pid, "generic"), 0) for pid in trig if pid in gold)
            per_hyp[jn] = {
                "n_trigger": n_t, "n_outside": n_o,
                "base_t": base_t, "ext_t": ext_t, "gen_t": gen_t,
                "base_o": base_o, "ext_o": ext_o,
                "delta_ext_base": (ext_t - base_t) / n_t if n_t else 0,
                "outside_delta_ext_base": (ext_o - base_o) / n_o if n_o else 0,
                "delta_ext_generic": (ext_t - gen_t) / n_t if n_t else 0,
            }
        d_eb = [per_hyp[jn]["delta_ext_base"] for jn in judges]
        d_ob = [per_hyp[jn]["outside_delta_ext_base"] for jn in judges]
        d_eg = [per_hyp[jn]["delta_ext_generic"] for jn in judges]
        per_hyp["consensus"] = {
            "min_delta_ext_base": min(d_eb),
            "mean_delta_ext_base": sum(d_eb) / len(d_eb),
            "max_outside_delta_ext_base": max(d_ob),
            "min_delta_ext_generic": min(d_eg),
            "mean_delta_ext_generic": sum(d_eg) / len(d_eg),
            "all_judges_accept": all(d >= 0.05 for d in d_eb) and all(d >= -0.05 for d in d_ob),
            "specific_in_all_judges": all(d_eb[i] > d_ob[i] for i in range(len(d_eb))),
            "beats_generic_in_all_judges": all(d >= 0 for d in d_eg),
        }
        by_hyp[hid] = per_hyp

    # Print
    print(f"\n=== STRICT Δ(EXT-BASE) on trigger ===\n", flush=True)
    print(f"  {'wisdom/kind':28s} | {'gpt_mini':>9s} | {'gemini':>9s} | {'haiku':>9s} | {'min':>7s} | {'all3':>5s} | {'spec3':>5s} | {'>GEN3':>5s}", flush=True)
    print(f"  {'-'*28}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*7}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}", flush=True)
    for hid, ph in by_hyp.items():
        label = f"{ph['seed_cid']}/{ph['kind']}"
        c = ph["consensus"]
        ok = "✓" if c["all_judges_accept"] else " "
        sp = "✓" if c["specific_in_all_judges"] else " "
        bg = "✓" if c["beats_generic_in_all_judges"] else " "
        print(f"  {label:28s} | {ph['gpt_mini_strict']['delta_ext_base']:+8.2%} | {ph['gemini_strict']['delta_ext_base']:+8.2%} | {ph['claude_haiku_strict']['delta_ext_base']:+8.2%} | {c['min_delta_ext_base']:+6.2%} |   {ok}   |   {sp}   |   {bg}", flush=True)

    # Aggregate BASE/EXT/GEN pass rate per judge
    print(f"\n=== Pass rates under STRICT rubric ===\n", flush=True)
    for jn in judges:
        b = e = g = nt = 0
        for ph in by_hyp.values():
            d = ph[jn]
            b += d["base_t"]; e += d["ext_t"]; g += d["gen_t"]; nt += d["n_trigger"]
        print(f"  {jn:22s}: BASE {b/nt:5.1%} | EXT {e/nt:5.1%} | GEN {g/nt:5.1%} (n={nt})", flush=True)

    SUMMARY.write_text(json.dumps({"hypotheses": by_hyp,
                                     "n_grade_calls": len(work_items),
                                     "judges": list(judges)},
                                    ensure_ascii=False, indent=2))
    print(f"\nSaved → {SUMMARY.relative_to(PROJECT)}", flush=True)


if __name__ == "__main__":
    main()
