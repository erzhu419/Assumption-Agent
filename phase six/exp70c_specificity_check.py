"""Exp 70c — specificity check on the ABL_TIGHT form.

Exp 70b found that the tightened-ablation form (trigger + failure
label only, no procedure, no verification) produced HUGE accuracy
gains on hard slices: constraint 6%->60%, counterfactual 34%->80%.

Two competing explanations for that effect:
  (A) The trigger + failure-label is content-specifically helpful —
      naming the cognitive error redirects the solver's thinking.
  (B) Any extra prefix that says 'be careful' would do the same job;
      the effect is just 'slow down' / 'verify' generalization.

To distinguish, we add a GENERIC_WARNING condition: same prefix length
and structural shape as ABL_TIGHT, but with NO problem-class trigger
and NO specific failure name. If ABL_TIGHT clearly beats GENERIC_WARNING
on the same problems with the same judges, the specificity matters.
If they tie, ABL_TIGHT was just a slow-down effect.

Cost: 5 slices x 50 problems = 250 generic-warning solves +
5 x 1 pair x 3 judges x 50 = 750 judgments. ~$3, ~10 min.
"""
import json, os, random, re, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase one" / "scripts" / "validation"))
sys.path.insert(0, str(PROJECT / "phase six" / "cards"))
sys.path.insert(0, str(PROJECT / "phase six" / "slices"))

def _load_api_keys():
    if os.environ.get("RUOLI_GPT_KEY") and os.environ.get("RUOLI_BASE_URL"): return
    keyfile = Path.home() / ".api_keys"
    if not keyfile.exists(): return
    pat = re.compile(r'^\s*export\s+(\w+)=("([^"]*)"|\'([^\']*)\'|(\S+))')
    for line in keyfile.read_text().splitlines():
        m = pat.match(line)
        if not m: continue
        name = m.group(1)
        val = m.group(3) if m.group(3) is not None else (m.group(4) if m.group(4) is not None else m.group(5))
        os.environ.setdefault(name, val)
        if name == "RUOLI_BASE_URL":
            base = val + "/v1" if not val.endswith("/v1") else val
            os.environ.setdefault("CLAUDE_PROXY_BASE_URL", base)
            os.environ.setdefault("GPT5_BASE_URL", base)
            os.environ.setdefault("GEMINI_PROXY_BASE_URL", base)
        if name == "RUOLI_GEMINI_KEY": os.environ.setdefault("GEMINI_PROXY_API_KEY", val)
        if name == "RUOLI_GPT_KEY": os.environ.setdefault("GPT5_API_KEY", val)
        if name == "RUOLI_CLAUDE_KEY": os.environ.setdefault("CLAUDE_PROXY_API_KEY", val)
_load_api_keys()

from model_router import cheap
from cached_framework import judge_pair, _save_content_cache
from slices import ALL_SLICES

PARALLEL = 8
AUTO = PROJECT / "phase six" / "autonomous"
OUT_LOG = AUTO / "exp70c_specificity_check_log.json"
PRIOR70 = AUTO / "exp70_sub_mvp_log.json"
PRIOR70B = AUTO / "exp70b_tightened_ablation_log.json"

# Generic-warning prefix, same shape as ABL_TIGHT but content-free.
GENERIC_WARNING = (
    "## METHODOLOGICAL HINT: careful_reasoning\n"
    "Trigger: This problem may be tricky.\n"
    "Failure to avoid: Hasty conclusions; missing important details."
)

SOLVE_PROMPT = """{card_section}## Problem
{problem}

## Output
Reason step by step concisely (3-8 sentences), then on the LAST LINE write exactly:
ANSWER: <answer>
"""


def solve(client, problem_text, card_render):
    card_section = (card_render + "\n\n") if card_render else ""
    try:
        r = client.generate(
            SOLVE_PROMPT.format(card_section=card_section, problem=problem_text),
            max_tokens=900, temperature=0.0,
        )
        return r["text"].strip()
    except Exception as e:
        return f"[err: {e}]"


def extract_answer(text):
    m = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    return (m.group(1) if m else text[-200:]).strip()


def score(answer_text, gold, tol):
    raw = extract_answer(answer_text)
    if isinstance(gold, str):
        return 1 if gold.lower().strip() in raw.lower() else 0
    s = raw.replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")
    nm = re.search(r'-?\d+(?:\.\d+)?(?:[eE]-?\d+)?', s)
    if not nm: return 0
    try: val = float(nm.group())
    except Exception: return 0
    if not isinstance(tol, (int, float)) or tol >= 1.001:
        if gold == 0: return 1 if abs(val) < 0.001 else 0
        rel = abs(val - gold) / abs(gold)
        return 1 if rel <= ((tol or 1.05) - 1.0) else 0
    return 1 if abs(val - gold) <= tol else 0


def judge_with_side_randomize(client, problem, ans_a, ans_b, label_a, label_b, seed):
    rng = random.Random(seed)
    if rng.random() < 0.5:
        left, right, a_was = ans_a, ans_b, "A"
    else:
        left, right, a_was = ans_b, ans_a, "B"
    try:
        v = judge_pair(client, problem, left, right)
        w = v.get("winner", "tie")
        if w == "tie": return "tie"
        return label_a if w == a_was else label_b
    except Exception:
        return "tie"


def run_pairwise_panel(judges, ans_dict_a, ans_dict_b, label_a, label_b, problems):
    results = {}
    for jname, jclient in judges:
        wins_a = wins_b = ties = 0
        per_pid = {}
        def one(p):
            pid = p["pid"]
            a = ans_dict_a.get(pid, ""); b = ans_dict_b.get(pid, "")
            if not a or not b: return pid, "tie"
            seed = hash(pid + label_a + label_b + jname + "spec") % (2**32)
            return pid, judge_with_side_randomize(jclient, p["prompt"], a, b, label_a, label_b, seed)
        with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
            futs = [ex.submit(one, p) for p in problems]
            for f in as_completed(futs):
                pid, w = f.result()
                per_pid[pid] = w
                if w == label_a: wins_a += 1
                elif w == label_b: wins_b += 1
                else: ties += 1
        n_eff = wins_a + wins_b
        wr = wins_a / n_eff if n_eff else 0.5
        results[jname] = {"wins_a": wins_a, "wins_b": wins_b, "ties": ties,
                              "n_eff": n_eff, "wr_a": wr, "per_pid": per_pid}
    _save_content_cache()
    wrs = [results[j]["wr_a"] for j in [j for j, _ in judges]]
    results["mean_wr"] = sum(wrs) / len(wrs)
    results["min_wr"] = min(wrs)
    results["max_wr"] = max(wrs)
    return results


def main():
    print(f"=== Exp 70c: specificity check on ABL_TIGHT form ===", flush=True)

    e70 = json.loads(PRIOR70.read_text(encoding="utf-8"))
    e70b = json.loads(PRIOR70B.read_text(encoding="utf-8"))
    abl_tight_answers = e70b["answers_ablated_tight"]
    base_answers = {s: e70["answers"][s]["BASE"] for s in abl_tight_answers}
    print(f"  Loaded BASE and ABL_TIGHT cached answers from prior experiments", flush=True)

    solver = cheap("gemini")
    judges = [
        ("gemini", cheap("gemini")),
        ("claude_haiku", cheap("claude_haiku")),
        ("gpt_mini", cheap("gpt_mini")),
    ]

    # ---- Stage 1: solve GENERIC_WARNING condition --------------------
    print(f"\n[1/3] Solving GENERIC_WARNING condition (5 x 50 = 250 solves)...", flush=True)
    t0 = time.time()
    generic_answers = {}
    for slice_name in abl_tight_answers:
        problems = ALL_SLICES[slice_name]
        generic_answers[slice_name] = {}
        def solve_task(p):
            return p["pid"], solve(solver, p["prompt"], GENERIC_WARNING)
        done = 0
        with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
            futs = [ex.submit(solve_task, p) for p in problems]
            for f in as_completed(futs):
                pid, ans = f.result()
                generic_answers[slice_name][pid] = ans
                done += 1
                if done % 25 == 0:
                    print(f"  [{slice_name}] GENERIC {done}/{len(problems)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  All GENERIC solves done in {time.time()-t0:.0f}s", flush=True)

    # ---- Stage 2: objective accuracy ---------------------------------
    print(f"\n[2/3] Objective grading vs gold...", flush=True)
    obj = {}
    for slice_name in abl_tight_answers:
        problems = ALL_SLICES[slice_name]
        b = sum(score(base_answers[slice_name].get(p["pid"],""),
                       p["gold"], p.get("tol", 1.05)) for p in problems)
        t = sum(score(abl_tight_answers[slice_name].get(p["pid"],""),
                       p["gold"], p.get("tol", 1.05)) for p in problems)
        g = sum(score(generic_answers[slice_name].get(p["pid"],""),
                       p["gold"], p.get("tol", 1.05)) for p in problems)
        obj[slice_name] = {"BASE_acc": b/len(problems),
                              "ABL_TIGHT_acc": t/len(problems),
                              "GENERIC_acc": g/len(problems)}
        print(f"  {slice_name:14s}: BASE={obj[slice_name]['BASE_acc']:.1%} "
              f"ABL_TIGHT={obj[slice_name]['ABL_TIGHT_acc']:.1%} "
              f"GENERIC={obj[slice_name]['GENERIC_acc']:.1%}", flush=True)

    # ---- Stage 3: ABL_TIGHT vs GENERIC ------------------------------
    print(f"\n[3/3] Pairwise ABL_TIGHT vs GENERIC (3 judges x 5 slices x 50 = 750 judgments)...", flush=True)
    t1 = time.time()
    pairwise = {}
    for slice_name in abl_tight_answers:
        problems = ALL_SLICES[slice_name]
        r = run_pairwise_panel(judges,
                                abl_tight_answers[slice_name],
                                generic_answers[slice_name],
                                "ABL_TIGHT", "GENERIC", problems)
        pairwise[slice_name] = r
        print(f"  [{slice_name:14s}] ABL_TIGHT_vs_GENERIC: mean wr={r['mean_wr']:.3f} "
              f"(min={r['min_wr']:.3f}) per-judge: gem={r['gemini']['wr_a']:.3f} "
              f"hai={r['claude_haiku']['wr_a']:.3f} gpt={r['gpt_mini']['wr_a']:.3f}", flush=True)
    print(f"  Judging done in {time.time()-t1:.0f}s", flush=True)

    # ---- Verdict per slice ------------------------------------------
    print(f"\n=== SPECIFICITY VERDICT PER SLICE ===", flush=True)
    verdicts = {}
    for slice_name in abl_tight_answers:
        delta_obj = obj[slice_name]["ABL_TIGHT_acc"] - obj[slice_name]["GENERIC_acc"]
        wr = pairwise[slice_name]["mean_wr"]
        if delta_obj >= 0.10 and wr >= 0.60:
            v = "SPECIFIC: trigger+failure-label is content-causal"
        elif delta_obj >= 0.05 and wr >= 0.55:
            v = "TENTATIVE SPECIFIC: signal exists but not large"
        elif abs(delta_obj) < 0.05 and 0.45 <= wr <= 0.55:
            v = "SLOW-DOWN ONLY: no specificity beyond generic warning"
        else:
            v = f"MIXED (Δacc={delta_obj:+.1%}, wr={wr:.3f})"
        verdicts[slice_name] = {
            "verdict": v,
            "delta_acc_tight_minus_generic": delta_obj,
            "wr_tight_vs_generic": wr,
        }
        print(f"  {slice_name:14s}: {v}", flush=True)
        print(f"    Δacc(TIGHT - GENERIC) = {delta_obj:+.1%}, "
              f"wr(TIGHT vs GENERIC) = {wr:.3f}", flush=True)

    n_specific = sum(1 for v in verdicts.values() if "SPECIFIC" in v["verdict"] and "TENTATIVE" not in v["verdict"])
    n_tent = sum(1 for v in verdicts.values() if "TENTATIVE" in v["verdict"])
    n_slow = sum(1 for v in verdicts.values() if "SLOW-DOWN" in v["verdict"])
    print(f"\n=== RESULT: {n_specific}/5 SPECIFIC, {n_tent}/5 TENTATIVE, {n_slow}/5 SLOW-DOWN-ONLY ===", flush=True)

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "objective_accuracy": obj,
        "pairwise_tight_vs_generic": pairwise,
        "verdicts": verdicts,
        "answers_generic": generic_answers,
    }
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}", flush=True)


if __name__ == "__main__":
    main()
