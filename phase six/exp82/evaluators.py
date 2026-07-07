"""Exp 82 v2: 5 evaluators, one injection mechanism per `kind`.

Common pipeline for a Hypothesis h on its trigger_subset:

  for pid in h.trigger_subset:
      base_answer    = cached or fresh BASE solver call
      ext_answer     = kind-specific injection of h.expr → fresh solver call
      generic_answer = (optional) generic 'be careful' warning → fresh solver call
      grade each vs gold → 0/1 correctness
      log ALL prompts/responses/chars to forensic.jsonl

Then aggregate Δ(EXT-BASE), optionally Δ(EXT-GENERIC), and the same on
outside_subset for specificity.

Forensic logging mandatory — every solver call and every grading call goes
into forensic.jsonl (per-trial JSON record with full prompt + response +
chars + timestamps + retries). User directive: "漏一个标点找你"
"""
from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from model_router import cheap, expensive  # noqa: E402
from hypothesis import Hypothesis, accept_decide  # noqa: E402

EXP_DIR = Path(__file__).parent
GOLD_PATH = EXP_DIR / "gold_answers.json"
BASE_CACHE_PATH = PROJECT / "phase two" / "analysis" / "cache" / "answers" / "_valp_v20p1_base_answers.json"
FORENSIC_LOG = EXP_DIR / "forensic.jsonl"

GENERIC_WARNING = "请在回答前仔细思考，注意可能的边界情况和常见错误，确保答案完整且正确。\n\n"


# ───────────────────────────────────────────────────────────────────────
# Forensic logging
# ───────────────────────────────────────────────────────────────────────

def _log_forensic(record: dict, path: Path = FORENSIC_LOG) -> None:
    record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ───────────────────────────────────────────────────────────────────────
# Solver
# ───────────────────────────────────────────────────────────────────────

def call_solver(prompt: str, solver_client, hp: dict = None,
                solver_name: str = "", role: str = "solver",
                pid: str = "", hid: str = "", retries: int = 4) -> dict:
    """Single solver call with forensic logging.

    `hp` overrides max_tokens / temperature / top_p (passed via OpenAI API).
    """
    hp = hp or {}
    max_tokens = hp.get("max_tokens", 1500)
    temperature = hp.get("temperature", 0.3)
    last_err = None
    t0 = time.time()
    for attempt in range(retries):
        try:
            r = solver_client.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            text = (r.get("text") or "").strip()
            if not text:
                raise RuntimeError("empty response")
            rec = {"role": role, "pid": pid, "hid": hid, "solver": solver_name,
                   "prompt_len": len(prompt), "answer_len": len(text),
                   "answer_chars": text, "prompt_chars": prompt,
                   "max_tokens": max_tokens, "temperature": temperature,
                   "elapsed": time.time() - t0, "attempt": attempt,
                   "model": r.get("model", "")}
            _log_forensic(rec)
            return {"text": text, "model": r.get("model", ""), "elapsed": rec["elapsed"], "ok": True}
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    rec = {"role": role, "pid": pid, "hid": hid, "solver": solver_name,
           "prompt_len": len(prompt), "prompt_chars": prompt,
           "error": str(last_err)[:300], "attempts": retries,
           "elapsed": time.time() - t0}
    _log_forensic(rec)
    return {"text": "", "model": "", "elapsed": time.time() - t0, "ok": False, "error": str(last_err)[:300]}


# ───────────────────────────────────────────────────────────────────────
# Grading vs gold
# ───────────────────────────────────────────────────────────────────────

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

_GRADE_RE = re.compile(r"\{[^{}]*?\"correct\"[^{}]*?\}", re.DOTALL)


def grade(problem: str, gold: str, candidate: str, judge_client,
          pid: str = "", hid: str = "", role: str = "grade") -> dict:
    """Binary grading — 0 or 1. Forensic-logged."""
    if not candidate.strip():
        rec = {"role": role, "pid": pid, "hid": hid, "graded_correct": 0, "reason": "empty candidate"}
        _log_forensic(rec)
        return {"correct": 0, "reason": "empty candidate", "ok": True}
    prompt = GRADE_PROMPT.format(problem=problem[:3000], gold=gold[:3000], candidate=candidate[:3000])
    last_err = None
    t0 = time.time()
    for attempt in range(3):
        try:
            r = judge_client.generate(prompt, max_tokens=200, temperature=0.0)
            text = (r.get("text") or "").strip()
            text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
            m = _GRADE_RE.search(text)
            if not m:
                raise ValueError(f"no grading JSON: {text[:200]}")
            obj = json.loads(m.group(0))
            correct = int(obj.get("correct", 0))
            if correct not in (0, 1):
                raise ValueError(f"invalid correct value {correct!r}")
            rec = {"role": role, "pid": pid, "hid": hid, "graded_correct": correct,
                   "reason": obj.get("reason", "")[:300], "judge_model": r.get("model", ""),
                   "candidate_len": len(candidate), "elapsed": time.time() - t0, "attempt": attempt}
            _log_forensic(rec)
            return {"correct": correct, "reason": obj.get("reason", ""), "ok": True}
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(1 + attempt)
    rec = {"role": role, "pid": pid, "hid": hid,
           "error": str(last_err)[:300], "elapsed": time.time() - t0}
    _log_forensic(rec)
    return {"correct": 0, "reason": "grade_error", "ok": False, "error": str(last_err)[:300]}


# ───────────────────────────────────────────────────────────────────────
# Per-kind injection — produce the EXT prompt / wrap solver call
# ───────────────────────────────────────────────────────────────────────

def _inject_decomposition(problem: str, expr: dict) -> str:
    steps = expr.get("steps", [])
    block = "\n".join(steps)
    return (
        f"Please solve the following problem. First, follow these {len(steps)} preparatory steps "
        f"in order, then write your final answer.\n\n{block}\n\n"
        f"PROBLEM:\n{problem}\n\nFINAL ANSWER:"
    )


def _inject_verification(problem: str, expr: dict, solver_client, hp: dict,
                          solver_name: str, pid: str, hid: str) -> str:
    """Two-pass: first BASE answer, then verify-per-instruction → revised answer."""
    instr = expr.get("instruction", "verify your answer carefully")
    # pass 1
    p1 = problem
    r1 = call_solver(p1, solver_client, hp=hp, solver_name=solver_name,
                      role="verify_pass1", pid=pid, hid=hid)
    if not r1["ok"]:
        return ""
    # pass 2 — feed back into solver with verify instruction
    p2 = (
        f"PROBLEM:\n{problem}\n\nYOUR INITIAL ANSWER:\n{r1['text']}\n\n"
        f"NOW VERIFY: {instr}\n"
        f"If you find a problem, write a corrected final answer below. "
        f"If the initial answer is correct, restate it.\n\nFINAL ANSWER:"
    )
    r2 = call_solver(p2, solver_client, hp=hp, solver_name=solver_name,
                      role="verify_pass2", pid=pid, hid=hid)
    return r2.get("text", "")


def _check_constraint(answer: str, expr: dict) -> tuple[bool, str]:
    """Return (passes, reason)."""
    req = expr.get("required_substrings", [])
    forb = expr.get("forbidden_substrings", [])
    has_any_req = any(s in answer for s in req) if req else True
    has_no_forb = not any(s in answer for s in forb) if forb else True
    if not has_any_req:
        return False, f"missing required: {req}"
    if not has_no_forb:
        return False, f"contains forbidden"
    return True, "ok"


def _inject_constraint_retry(problem: str, expr: dict, solver_client, hp: dict,
                              solver_name: str, pid: str, hid: str) -> str:
    """BASE call; if constraint fails, retry with explicit hint up to max_retries."""
    max_retries = expr.get("max_retries", 2)
    req = expr.get("required_substrings", [])
    hint = (
        f"\n\nIMPORTANT: Your answer must address the following — include at least one of "
        f"these aspects in your reasoning: {req}"
    )
    cur = call_solver(problem, solver_client, hp=hp, solver_name=solver_name,
                       role="constraint_pass1", pid=pid, hid=hid)
    if not cur["ok"]:
        return ""
    answer = cur["text"]
    for retry in range(max_retries):
        passes, _reason = _check_constraint(answer, expr)
        if passes:
            return answer
        # retry with hint
        p_retry = problem + hint + (
            f"\n\nYour previous attempt did not satisfy this. Try again with the required content."
        )
        r = call_solver(p_retry, solver_client, hp=hp, solver_name=solver_name,
                         role=f"constraint_retry{retry+1}", pid=pid, hid=hid)
        if not r["ok"]:
            break
        answer = r["text"]
    return answer


def _detect_feature(problem: str, expr: dict) -> bool:
    """Run feature detector on problem text — return True if any signal fires."""
    kws_zh = expr.get("keywords_zh", [])
    kws_en = expr.get("keywords_en", [])
    regs = expr.get("regex", [])
    for kw in kws_zh + kws_en:
        if kw and kw in problem:
            return True
    for r in regs:
        if r and re.search(r, problem):
            return True
    return False


def _hp_override(hp_default: dict, expr: dict) -> dict:
    """Merge default hp with hypothesis-prescribed override."""
    out = dict(hp_default)
    for k in ("temperature", "top_p", "max_tokens"):
        if k in expr:
            out[k] = expr[k]
    return out


# ───────────────────────────────────────────────────────────────────────
# Per-kind evaluator — the A/B test
# ───────────────────────────────────────────────────────────────────────

def _evaluate_one_kind(h: Hypothesis, problem: str, gold: str, base_cached: str,
                        solver_client, judge_client, solver_name: str,
                        pid: str, with_generic: bool) -> dict:
    """Run BASE / GENERIC / EXT for one (hypothesis, problem) cell."""
    hp = {"temperature": 0.3, "top_p": 0.95, "max_tokens": 1500}

    # BASE: prefer cached if we have it; otherwise fresh
    if base_cached:
        base_answer = base_cached
        _log_forensic({"role": "base_cached", "pid": pid, "hid": h.hid,
                        "answer_len": len(base_answer), "answer_chars": base_answer})
    else:
        r = call_solver(problem, solver_client, hp=hp, solver_name=solver_name,
                         role="base_fresh", pid=pid, hid=h.hid)
        base_answer = r.get("text", "")

    # GENERIC: only if requested
    generic_answer = ""
    if with_generic:
        gp = GENERIC_WARNING + problem
        rg = call_solver(gp, solver_client, hp=hp, solver_name=solver_name,
                          role="generic", pid=pid, hid=h.hid)
        generic_answer = rg.get("text", "")

    # EXT: dispatch by kind
    ext_answer = ""
    if h.kind == "feature":
        # feature-only — no solver call. We just record whether the detector
        # fires. Correctness Δ for feature is computed from the BASE/EXT pair
        # by COMPARING THE OTHER KINDS' triggering. For pure feature ablation
        # we use the BASE answer as the "EXT" too — feature kind tests
        # whether the detector itself usefully picks up the trigger subset.
        # (See aggregation: feature kind is judged on trigger-fit-rate, not
        # on solver Δ.)
        fired = _detect_feature(problem, h.expr)
        ext_answer = base_answer  # placeholder
        _log_forensic({"role": "feature_detect", "pid": pid, "hid": h.hid,
                        "fired": int(fired), "expr": h.expr})
        return {"base_answer": base_answer, "generic_answer": generic_answer,
                "ext_answer": ext_answer, "feature_fired": fired}
    elif h.kind == "decomposition":
        prompt = _inject_decomposition(problem, h.expr)
        r = call_solver(prompt, solver_client, hp=hp, solver_name=solver_name,
                         role="ext_decomposition", pid=pid, hid=h.hid)
        ext_answer = r.get("text", "")
    elif h.kind == "verification":
        ext_answer = _inject_verification(problem, h.expr, solver_client, hp,
                                            solver_name, pid, h.hid)
    elif h.kind == "constraint":
        ext_answer = _inject_constraint_retry(problem, h.expr, solver_client, hp,
                                                solver_name, pid, h.hid)
    elif h.kind == "hp_change":
        hp_eff = _hp_override(hp, h.expr)
        r = call_solver(problem, solver_client, hp=hp_eff, solver_name=solver_name,
                         role="ext_hp_change", pid=pid, hid=h.hid)
        ext_answer = r.get("text", "")

    return {"base_answer": base_answer, "generic_answer": generic_answer,
            "ext_answer": ext_answer}


def evaluate(h: Hypothesis, gold_answers: dict, base_cache: dict,
             solver_client, judge_client, solver_name: str = "",
             with_generic: bool = False, n_trials: int = 1,
             max_workers: int = 4) -> dict:
    """Evaluate a Hypothesis on its trigger_subset (and outside_subset for specificity).

    Returns evidence dict ready to feed into Hypothesis.record_outcome.
    """
    trigger = [pid for pid in h.trigger_subset if pid in gold_answers]
    outside = [pid for pid in h.outside_subset if pid in gold_answers]

    def _do_one(pid: str, _trial: int = 0):
        rec = gold_answers[pid]
        problem = rec["description"]
        gold = rec["gold"]
        base_cached = base_cache.get(pid, "")
        run = _evaluate_one_kind(h, problem, gold, base_cached, solver_client,
                                   judge_client, solver_name, pid, with_generic)
        out = {"pid": pid, "trial": _trial, "feature_fired": run.get("feature_fired", None)}
        out["base_correct"] = grade(problem, gold, run["base_answer"], judge_client,
                                      pid=pid, hid=h.hid, role="grade_base")["correct"]
        if with_generic:
            out["generic_correct"] = grade(problem, gold, run["generic_answer"], judge_client,
                                              pid=pid, hid=h.hid, role="grade_generic")["correct"]
        if h.kind == "feature":
            out["ext_correct"] = out["base_correct"]
        else:
            out["ext_correct"] = grade(problem, gold, run["ext_answer"], judge_client,
                                          pid=pid, hid=h.hid, role="grade_ext")["correct"]
        return out

    cells = [(pid, t) for pid in trigger for t in range(n_trials)]
    out_cells = [(pid, t) for pid in outside for t in range(n_trials)]

    trigger_results = []
    outside_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fut in as_completed([ex.submit(_do_one, pid, t) for pid, t in cells]):
            trigger_results.append(fut.result())
        for fut in as_completed([ex.submit(_do_one, pid, t) for pid, t in out_cells]):
            outside_results.append(fut.result())

    n_t = len(trigger_results)
    n_o = len(outside_results)
    base_t = sum(r["base_correct"] for r in trigger_results)
    ext_t = sum(r["ext_correct"] for r in trigger_results)
    base_o = sum(r["base_correct"] for r in outside_results)
    ext_o = sum(r["ext_correct"] for r in outside_results)

    evidence = {
        "n_trigger": n_t,
        "n_outside": n_o,
        "base_correct_trigger": base_t,
        "ext_correct_trigger": ext_t,
        "delta_ext_base": (ext_t / n_t - base_t / n_t) if n_t else 0.0,
        "base_correct_outside": base_o,
        "ext_correct_outside": ext_o,
        "outside_delta_ext_base": (ext_o / n_o - base_o / n_o) if n_o else 0.0,
        "trigger_results": trigger_results,
        "outside_results": outside_results,
    }
    if with_generic:
        gen_t = sum(r.get("generic_correct", 0) for r in trigger_results)
        evidence["generic_correct_trigger"] = gen_t
        evidence["delta_ext_generic"] = (ext_t / n_t - gen_t / n_t) if n_t else 0.0
    if h.kind == "feature":
        n_fire_t = sum(int(bool(r.get("feature_fired"))) for r in trigger_results)
        n_fire_o = sum(int(bool(r.get("feature_fired"))) for r in outside_results)
        evidence["feature_fire_rate_trigger"] = (n_fire_t / n_t) if n_t else 0.0
        evidence["feature_fire_rate_outside"] = (n_fire_o / n_o) if n_o else 0.0

    return evidence


def load_gold() -> dict:
    if not GOLD_PATH.exists():
        raise RuntimeError(f"gold not found at {GOLD_PATH}; run generate_gold.py first")
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def load_base_cache() -> dict:
    """Load v1 cached BASE answers (pid → answer_text)."""
    if not BASE_CACHE_PATH.exists():
        return {}
    return json.loads(BASE_CACHE_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    # smoke test: evaluate one hypothesis on a few cells
    from proposers import propose
    M = json.loads((EXP_DIR / "verdict_matrix.json").read_text(encoding="utf-8"))
    cands = {c["cid"]: c for c in M["candidates"]}
    wisdom = cands["WCAND10"]

    # use just 3 trigger pids and 1 outside pid
    import json as _json
    labels = _json.loads((PROJECT / "phase four" / "autonomous" / "exp17_trigger_labels.json").read_text(encoding="utf-8"))
    fire_pids = [pid for pid, e in labels["WCAND10"].items() if e.get("verdict") == "SHOULD_FIRE"][:3]
    nofire_pids = [pid for pid, e in labels["WCAND10"].items() if e.get("verdict") == "NO_FIRE"][:1]
    print(f"trigger sample: {fire_pids}")
    print(f"outside sample: {nofire_pids}")

    h = propose(wisdom, "decomposition",
                trigger_subset=fire_pids, outside_subset=nofire_pids)
    print(f"\n=== smoke evaluating: {h.claim[:100]} ===")
    gold = load_gold()
    base_cache = load_base_cache()
    print(f"  gold count: {len(gold)}, base cache size: {len(base_cache)}")
    print(f"  pids with gold: trigger={[p for p in fire_pids if p in gold]}, outside={[p for p in nofire_pids if p in gold]}")

    solver = cheap("gpt_mini")
    judge = cheap("gpt_mini")
    ev = evaluate(h, gold, base_cache, solver, judge, solver_name="gpt_mini",
                  with_generic=True, n_trials=1, max_workers=2)
    decision, reason = accept_decide(ev)
    h.record_outcome(ev, decision, reason)
    print(f"\n  evidence: {_json.dumps({k:v for k,v in ev.items() if not k.endswith('_results')}, indent=2)}")
    print(f"  decision: {decision} (reason: {reason})")
