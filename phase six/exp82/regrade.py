"""Exp 82 v2 step A: cross-judge re-grade of all 15 ablation hypotheses.

Loads forensic.jsonl + hypotheses.jsonl. For each hypothesis, finds the
(BASE / EXT / GENERIC) candidate answer per pid, then re-grades each
candidate with TWO additional judge families:
  - gemini-3-flash       (Google)
  - claude-haiku-4-5     (Anthropic)
(The original judge was gpt-5.4-mini — OpenAI.)

Output: cross_judge_summary.json — per-hypothesis Δ under each judge,
plus a 3-judge consensus column.
"""
from __future__ import annotations

import json
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
SUMMARY = EXP_DIR / "cross_judge_summary.json"
REGRADE_LOG = EXP_DIR / "regrade_forensic.jsonl"

ROLE_TO_KIND_EXT = {
    "feature": None,                # feature has no ext call; ext_answer = base_answer
    "constraint": "constraint",     # take last constraint_pass*/retry* per pid
    "decomposition": "ext_decomposition",
    "verification": "verify_pass2",
    "hp_change": "ext_hp_change",
}


GRADE_PROMPT = """You are grading a candidate answer against a reference (gold) answer.

PROBLEM:
{problem}

GOLD ANSWER:
{gold}

CANDIDATE ANSWER:
{candidate}

Grade the candidate on a binary correctness scale:
  1 = correct (covers the key points of the gold answer; minor wording differences are fine)
  0 = incorrect (misses key points, contradicts the gold, or is non-responsive)

Output ONLY a JSON object: {{"correct": 0 or 1, "reason": "one sentence"}}
No markdown, no commentary outside the JSON.
"""

import re
_GRADE_RE = re.compile(r"\{[^{}]*?\"correct\"[^{}]*?\}", re.DOTALL)


def grade_call(judge_client, problem: str, gold: str, candidate: str,
                judge_name: str, hid: str, pid: str, condition: str) -> int:
    """One grading call with retries; returns 0 or 1."""
    if not (candidate or "").strip():
        rec = {"judge": judge_name, "hid": hid, "pid": pid, "condition": condition,
               "result": 0, "reason": "empty candidate"}
        with open(REGRADE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return 0
    prompt = GRADE_PROMPT.format(problem=problem[:3000], gold=gold[:3000], candidate=candidate[:3000])
    last_err = None
    for attempt in range(3):
        try:
            r = judge_client.generate(prompt, max_tokens=200, temperature=0.0)
            text = (r.get("text") or "").strip()
            text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
            m = _GRADE_RE.search(text)
            if not m:
                raise ValueError(f"no JSON in {text[:200]}")
            obj = json.loads(m.group(0))
            correct = int(obj.get("correct", 0))
            if correct not in (0, 1):
                raise ValueError(f"correct={correct!r}")
            rec = {"judge": judge_name, "hid": hid, "pid": pid, "condition": condition,
                   "result": correct, "reason": obj.get("reason", "")[:200],
                   "judge_model": r.get("model", ""), "attempt": attempt}
            with open(REGRADE_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return correct
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(1 + attempt)
    # all retries failed
    rec = {"judge": judge_name, "hid": hid, "pid": pid, "condition": condition,
           "result": 0, "error": str(last_err)[:200]}
    with open(REGRADE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 0


def find_answer_for(records: list, hid: str, pid: str, role_target) -> str:
    """Find the latest answer record for (hid, pid, role).
    For constraint kind, role_target='constraint' means take last of
    constraint_pass1 / constraint_retry1 / constraint_retry2 (whichever is latest).
    """
    if role_target is None:
        return ""
    if role_target == "constraint":
        candidates = [r for r in records
                      if r.get("hid") == hid and r.get("pid") == pid
                      and (r.get("role") in ("constraint_pass1", "constraint_retry1", "constraint_retry2"))
                      and r.get("answer_chars")]
    else:
        candidates = [r for r in records
                      if r.get("hid") == hid and r.get("pid") == pid
                      and r.get("role") == role_target
                      and r.get("answer_chars")]
    if not candidates:
        return ""
    # Take the latest by ts
    candidates.sort(key=lambda r: r.get("ts", ""))
    return candidates[-1]["answer_chars"]


def find_base(records: list, hid: str, pid: str) -> str:
    rs = [r for r in records
          if r.get("hid") == hid and r.get("pid") == pid
          and r.get("role") in ("base_cached", "base_fresh")
          and r.get("answer_chars")]
    if not rs:
        return ""
    rs.sort(key=lambda r: r.get("ts", ""))
    return rs[-1]["answer_chars"]


def find_generic(records: list, hid: str, pid: str) -> str:
    rs = [r for r in records
          if r.get("hid") == hid and r.get("pid") == pid
          and r.get("role") == "generic"
          and r.get("answer_chars")]
    if not rs:
        return ""
    rs.sort(key=lambda r: r.get("ts", ""))
    return rs[-1]["answer_chars"]


def main():
    print("Loading forensic + hypotheses + gold...", flush=True)
    records = [json.loads(l) for l in open(FORENSIC) if l.strip()]
    hypos = [json.loads(l) for l in open(HYPOS) if l.strip()]
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    print(f"  forensic: {len(records)} records, hypotheses: {len(hypos)}, gold: {len(gold)}", flush=True)

    judges = {
        "gemini": cheap("gemini"),
        "claude_haiku": cheap("claude_haiku"),
    }
    print(f"  judges: {list(judges.keys())}", flush=True)

    # Build the per-(hid, pid, condition) candidate answer pool
    print("\nBuilding (hid, pid, condition) → answer index...", flush=True)
    work_items = []  # list of (judge_name, hid, pid, condition, problem, gold_text, candidate)
    per_hid_pids: dict = {}
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
            for judge_name in judges:
                if base:
                    work_items.append((judge_name, hid, pid, "base", problem, g, base))
                if ext:
                    work_items.append((judge_name, hid, pid, "ext", problem, g, ext))
                if generic:
                    work_items.append((judge_name, hid, pid, "generic", problem, g, generic))
    print(f"  total grade calls: {len(work_items)}", flush=True)

    # Reset regrade log
    REGRADE_LOG.write_text("")

    # Run all grade calls in parallel
    print(f"\nRunning grade calls (8 workers)...", flush=True)
    results: dict = defaultdict(int)  # (judge, hid, pid, cond) → 0/1
    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut_map = {}
        for it in work_items:
            judge_name, hid, pid, cond, problem, g, candidate = it
            f = ex.submit(grade_call, judges[judge_name], problem, g, candidate,
                          judge_name, hid, pid, cond)
            fut_map[f] = (judge_name, hid, pid, cond)
        for f in as_completed(fut_map):
            jn, hid, pid, cond = fut_map[f]
            results[(jn, hid, pid, cond)] = f.result()
            n_done += 1
            if n_done % 100 == 0:
                print(f"  [{n_done}/{len(work_items)}] {time.time()-t0:.0f}s", flush=True)

    # Aggregate per hypothesis × judge
    print(f"\nAggregating per hypothesis × judge...", flush=True)
    by_hyp: dict = {}
    for h in hypos:
        hid = h["hid"]
        kind = h["kind"]
        seed = h["seed_cid"]
        trig = per_hid_pids[hid]["trigger"]
        out = per_hid_pids[hid]["outside"]
        per_hyp = {"hid": hid, "seed_cid": seed, "kind": kind, "claim": h["claim"][:120]}
        for jn in list(judges) + ["gpt_mini_orig"]:
            n_t = sum(1 for pid in trig if pid in gold)
            n_o = sum(1 for pid in out if pid in gold)
            if jn == "gpt_mini_orig":
                # Use original ablation evidence
                ev = h.get("evidence", {})
                base_t = ev.get("base_correct_trigger", 0)
                ext_t = ev.get("ext_correct_trigger", 0)
                base_o = ev.get("base_correct_outside", 0)
                ext_o = ev.get("ext_correct_outside", 0)
                gen_t = ev.get("generic_correct_trigger", 0)
            else:
                base_t = sum(results.get((jn, hid, pid, "base"), 0) for pid in trig if pid in gold)
                ext_t = sum(results.get((jn, hid, pid, "ext"), 0) for pid in trig if pid in gold)
                base_o = sum(results.get((jn, hid, pid, "base"), 0) for pid in out if pid in gold)
                ext_o = sum(results.get((jn, hid, pid, "ext"), 0) for pid in out if pid in gold)
                gen_t = sum(results.get((jn, hid, pid, "generic"), 0) for pid in trig if pid in gold)
            per_hyp[jn] = {
                "n_trigger": n_t,
                "n_outside": n_o,
                "base_t": base_t, "ext_t": ext_t, "gen_t": gen_t,
                "base_o": base_o, "ext_o": ext_o,
                "delta_ext_base": (ext_t - base_t) / n_t if n_t else 0,
                "outside_delta_ext_base": (ext_o - base_o) / n_o if n_o else 0,
                "delta_ext_generic": (ext_t - gen_t) / n_t if n_t else 0,
            }
        # consensus across 3 judges
        d_eb = [per_hyp[jn]["delta_ext_base"] for jn in ("gpt_mini_orig", "gemini", "claude_haiku")]
        d_ob = [per_hyp[jn]["outside_delta_ext_base"] for jn in ("gpt_mini_orig", "gemini", "claude_haiku")]
        d_eg = [per_hyp[jn]["delta_ext_generic"] for jn in ("gpt_mini_orig", "gemini", "claude_haiku")]
        per_hyp["consensus"] = {
            "min_delta_ext_base": min(d_eb),
            "mean_delta_ext_base": sum(d_eb) / 3,
            "max_outside_delta_ext_base": max(d_ob),
            "mean_outside_delta_ext_base": sum(d_ob) / 3,
            "min_delta_ext_generic": min(d_eg),
            "mean_delta_ext_generic": sum(d_eg) / 3,
            # 3-judge accept: every judge says trigger Δ ≥ 5pp AND outside Δ ≥ -5pp
            "all_judges_accept": all(d >= 0.05 for d in d_eb) and all(d >= -0.05 for d in d_ob),
            # specificity: trigger Δ > outside Δ in every judge
            "specific_in_all_judges": all(d_eb[i] > d_ob[i] for i in range(3)),
        }
        by_hyp[hid] = per_hyp

    # Print summary table
    print(f"\n=== Per-hypothesis × judge Δ(EXT-BASE) on trigger ===\n", flush=True)
    print(f"  {'wisdom/kind':28s} | {'gpt_mini':>9s} | {'gemini':>9s} | {'haiku':>9s} | {'min':>7s} | {'all3?':>6s} | {'spec3?':>7s}", flush=True)
    print(f"  {'-'*28}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}", flush=True)
    for hid, ph in by_hyp.items():
        label = f"{ph['seed_cid']}/{ph['kind']}"
        c = ph["consensus"]
        ok = "✓" if c["all_judges_accept"] else " "
        sp = "✓" if c["specific_in_all_judges"] else " "
        print(f"  {label:28s} | {ph['gpt_mini_orig']['delta_ext_base']:+8.2%} | {ph['gemini']['delta_ext_base']:+8.2%} | {ph['claude_haiku']['delta_ext_base']:+8.2%} | {c['min_delta_ext_base']:+6.2%} |   {ok}    |    {sp}", flush=True)

    SUMMARY.write_text(json.dumps({"hypotheses": by_hyp,
                                     "n_grade_calls": len(work_items),
                                     "judges": list(judges) + ["gpt_mini_orig"]},
                                    ensure_ascii=False, indent=2))
    print(f"\nSaved → {SUMMARY.relative_to(PROJECT)}", flush=True)
    print(f"        {REGRADE_LOG.relative_to(PROJECT)}", flush=True)


if __name__ == "__main__":
    main()
