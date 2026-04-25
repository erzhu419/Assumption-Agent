"""Exp 61 — Cross-language validation of Stage 1 scheduler.

Closes the standing concern that the agent stack has only been
validated in English. This experiment translates Exp 56's 60
problems into Spanish, French, German, and Japanese, then runs
the same protocol in each language.

If the LLM scheduler beats fixed priors and beats baseline in EACH
of the 5 languages (English baseline + 4 new), Stage 1's value is
not English-specific. The scheduler prompt itself is in English
(reasoning about which abstract methodological strategy applies),
but the solver and the problem text are in the target language.

Conditions per language (60 problems each):
  baseline + 4 fixed priors + scheduler-picked + oracle = 7 conds
  × 4 new languages = 1680 solver calls
  + 60 problems × 4 langs translation = 240 translation calls
  + 60 problems × 4 langs scheduler picks = 240 scheduler calls
  Total ~2160 calls. ~$15-25 cheap-tier.

Expected runtime: ~60-90 min wall.
"""
import json, os, random, re, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
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
            os.environ.setdefault("CLAUDE_PROXY_BASE_URL", base); os.environ.setdefault("GPT5_BASE_URL", base); os.environ.setdefault("GEMINI_PROXY_BASE_URL", base)
        if name == "RUOLI_GEMINI_KEY": os.environ.setdefault("GEMINI_PROXY_API_KEY", val)
        if name == "RUOLI_GPT_KEY": os.environ.setdefault("GPT5_API_KEY", val)
        if name == "RUOLI_CLAUDE_KEY": os.environ.setdefault("CLAUDE_PROXY_API_KEY", val)
_load_api_keys()
from model_router import cheap

PARALLEL = 6
AUTO = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO / "exp61_crosslanguage_log.json"

LANGUAGES = {
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
}

PRIORS = {
    "decompose": "Before answering, decompose into atomic substeps.",
    "restate":   "Before answering, RE-READ the question carefully; check if the obvious interpretation is a trap.",
    "estimate":  "Before answering, give an order-of-magnitude estimate; sanity-check final answer.",
    "constraints": "Before answering, enumerate explicit constraints; satisfy all simultaneously.",
    "none":      "",
}

TRANSLATE_PROMPT = """Translate the following problem from English into {target_language}.
Preserve all numbers, names, units, and the structure of the question
exactly. Translate only the natural-language content.

## English
{problem}

## {target_language}
"""

# Solver prompt template — the "ANSWER:" tag stays in English so we can
# extract numerically. The reasoning block is in the target language.
SOLVE_PROMPT = """## Problem (in {language})
{problem}

## Approach hint (in English)
{approach}

## Output
Reason step by step in {language} (1-3 sentences), then on the LAST LINE
write exactly (use English keyword "ANSWER:"):
ANSWER: <your final answer (numbers/text only)>
"""

# Scheduler prompt is in English; problem text inside is target-language.
SCHED_PROMPT = """You are a strategy selector. The problem below may be in
a non-English language. Choose ONE prior from:
{{decompose, restate, estimate, constraints, none}}.

- decompose: best for multi-step arithmetic problems.
- restate: best for trick questions where the obvious interpretation is a trap.
- estimate: best for Fermi/order-of-magnitude problems.
- constraints: best for logic puzzles with explicit constraints.
- none: only if no prior is clearly more applicable.

## Problem
{problem}

## Output (JSON only)
{{"choice": "decompose"|"restate"|"estimate"|"constraints"|"none",
  "reason": "1 short sentence in English"}}
"""

def translate(client, problem, target_lang_name):
    try:
        r = client.generate(
            TRANSLATE_PROMPT.format(target_language=target_lang_name, problem=problem),
            max_tokens=600, temperature=0.0)
        # Strip any meta-text; take just the result
        text = r["text"].strip()
        # Sometimes the model echoes labels; strip lines starting with "##"
        lines = [l for l in text.split('\n') if not l.strip().startswith("##")]
        return "\n".join(lines).strip()
    except Exception as e:
        return f"[err: {e}]"

def solve(client, problem, prior_name, language_name):
    p = PRIORS[prior_name]
    approach = f"Use this strategy: {p}" if p else "Use any approach you think appropriate."
    try:
        r = client.generate(
            SOLVE_PROMPT.format(problem=problem, approach=approach, language=language_name),
            max_tokens=600, temperature=0.0)
        return r["text"].strip()
    except Exception as e:
        return f"[err: {e}]"

def schedule(client, problem):
    try:
        r = client.generate(SCHED_PROMPT.format(problem=problem),
                             max_tokens=200, temperature=0.0)
        m = re.search(r'"choice"\s*:\s*"(decompose|restate|estimate|constraints|none)"', r["text"])
        if m: return m.group(1)
    except Exception: pass
    return "none"

def extract_answer(text):
    m = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    return (m.group(1) if m else text[-200:]).strip()

def score_numeric(ext, gold, tol=None):
    s = ext.replace(",", "").replace("$", "").replace("%", "").replace("¥", "").strip().rstrip(".")
    m = re.search(r'-?\d+(?:\.\d+)?', s)
    if not m: return 0
    try: ev = float(m.group())
    except: return 0
    try: gv = float(gold)
    except: return 0
    if tol is not None:
        if gv == 0: return 1 if ev == 0 else 0
        ratio = abs(ev / gv) if gv != 0 else float('inf')
        return 1 if (1.0 / tol) <= ratio <= tol else 0
    return 1 if abs(ev - gv) < 0.02 else 0

def score_text_loose(ext, gold):
    """For cross-language, accept English gold matched in either English or
    target-language answers. Substring match only."""
    e = ext.lower().strip().rstrip(".")
    g = str(gold).lower().strip()
    if g in e: return 1
    g_words = set(re.findall(r'\w+', g))
    meaningful = {w for w in g_words if len(w) >= 3}
    if not meaningful: return 0
    e_words = set(re.findall(r'\w+', e))
    return 1 if len(meaningful & e_words) / len(meaningful) >= 0.5 else 0

def score(ext, gold, tol=None):
    if isinstance(gold, (int, float)): return score_numeric(ext, gold, tol)
    return score_text_loose(ext, gold)


def run_language(lang_code, lang_name, all_problems_en, translate_client, solver_client, sched_client):
    """Translate, schedule, solve all 7 conditions in one language."""
    print(f"\n=== Language: {lang_name} ({lang_code}) ===")

    # 1) Translate all problems
    print(f"[1/4] Translating {len(all_problems_en)} problems to {lang_name}...")
    translated_problems = []
    t0 = time.time()
    def trans(p):
        return p, translate(translate_client, p["prompt"], lang_name)
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(trans, p) for p in all_problems_en]
        for i, f in enumerate(as_completed(futs), 1):
            p, t = f.result()
            translated_problems.append({**p, "prompt_en": p["prompt"], "prompt": t})
            if i % 20 == 0: print(f"  translate {i}/60 ({time.time()-t0:.0f}s)")

    # Sort to preserve order
    translated_problems.sort(key=lambda x: x["pid"])

    # 2) Scheduler picks
    print(f"\n[2/4] Scheduler picks ({lang_name} problems)...")
    sched_picks = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(schedule, sched_client, p["prompt"]): p for p in translated_problems}
        for i, f in enumerate(as_completed(futs), 1):
            p = futs[f]; sched_picks[p["pid"]] = f.result()
    print(f"  schedule done ({time.time()-t0:.0f}s)")

    # 3) Fixed conditions + scheduler + oracle = 7 × 60 = 420
    conditions = ["baseline", "decompose", "restate", "estimate", "constraints"]
    answers = {c: {} for c in conditions}; answers["scheduler"] = {}; answers["oracle"] = {}

    print(f"\n[3/4] Running fixed conditions (5 x 60 = 300 calls)...")
    tasks = [(c, p) for c in conditions for p in translated_problems]
    def run_fixed(c, p):
        prior = "none" if c == "baseline" else c
        return c, p["pid"], solve(solver_client, p["prompt"], prior, lang_name)
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(run_fixed, c, p) for c, p in tasks]
        for f in as_completed(futs):
            c, pid, ans = f.result(); answers[c][pid] = ans; done += 1
            if done % 60 == 0: print(f"  fixed {done}/300 ({time.time()-t0:.0f}s)")

    print(f"\n[4/4] Scheduler-solve + Oracle (60 + 60)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(solve, solver_client, p["prompt"],
                            sched_picks[p["pid"]], lang_name): p for p in translated_problems}
        for f in as_completed(futs):
            p = futs[f]; answers["scheduler"][p["pid"]] = f.result()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(solve, solver_client, p["prompt"],
                            p["optimal_prior"], lang_name): p for p in translated_problems}
        for f in as_completed(futs):
            p = futs[f]; answers["oracle"][p["pid"]] = f.result()
    print(f"  done ({time.time()-t0:.0f}s)")

    # Score
    fams = ["A", "B", "C", "D"]
    cond_scores = {c: {f: [] for f in fams} for c in conditions + ["scheduler", "oracle"]}
    for p in translated_problems:
        f = p["family"][0]
        for c in conditions + ["scheduler", "oracle"]:
            ext = extract_answer(answers[c].get(p["pid"], ""))
            sc = score(ext, p["gold"], p.get("tol"))
            cond_scores[c][f].append(sc)

    # Scheduler pick correctness
    correct_picks = sum(1 for p in translated_problems if sched_picks[p["pid"]] == p["optimal_prior"])

    print(f"\n--- {lang_name} accuracy ---")
    print(f"{'Cond':14s} {'A':>6s} {'B':>6s} {'C':>6s} {'D':>6s} {'overall':>8s}")
    summary = {}
    for c in conditions + ["scheduler", "oracle"]:
        accs = [sum(cond_scores[c][f]) / len(cond_scores[c][f]) for f in fams]
        n_correct = sum(sum(cond_scores[c][f]) for f in fams)
        overall = n_correct / 60
        print(f"{c:14s} {accs[0]:>6.3f} {accs[1]:>6.3f} {accs[2]:>6.3f} {accs[3]:>6.3f} {overall:>8.3f}")
        summary[c] = {"A_acc": accs[0], "B_acc": accs[1], "C_acc": accs[2],
                       "D_acc": accs[3], "overall": overall, "n_correct": n_correct}
    print(f"  scheduler correct picks: {correct_picks}/60 = {correct_picks/60:.1%}")
    return summary, sched_picks, correct_picks, translated_problems


def main():
    print(f"=== Exp 61: cross-language validation ===")
    e56 = json.loads((AUTO / "exp56_stage1_broader_log.json").read_text())
    pp = e56["per_problem"]

    # Reconstruct problems (English)
    all_problems_en = []
    for pid, d in pp.items():
        all_problems_en.append({"pid": pid, "prompt": d["prompt"], "gold": d["gold"],
                                  "family": d["family"],
                                  "optimal_prior": {"A_decompose": "decompose",
                                                      "B_restate": "restate",
                                                      "C_estimate": "estimate",
                                                      "D_constraints": "constraints"}[d["family"]],
                                  "tol": 10 if d["family"] == "C_estimate" else None})

    translate_client = cheap("gemini")
    solver_client = cheap("gemini")
    sched_client = cheap("gemini")

    all_results = {}
    all_translations = {}
    all_picks = {}

    # English baseline (from Exp 56)
    en_summary = {c: e56["scores"][c] for c in
                    ["baseline", "decompose", "restate", "estimate", "constraints",
                     "scheduler", "oracle"]}
    for c in en_summary:
        en_summary[c]["overall"] = e56["scores_overall"][c]
    all_results["en"] = en_summary
    en_correct = e56["scheduler_correct_picks"]
    all_picks["en"] = e56["scheduler_picks"]

    # Run each non-English language
    for lang_code, lang_name in LANGUAGES.items():
        try:
            summary, picks, correct, translations = run_language(
                lang_code, lang_name, all_problems_en,
                translate_client, solver_client, sched_client)
            all_results[lang_code] = summary
            all_picks[lang_code] = picks
            all_translations[lang_code] = [{"pid": p["pid"], "prompt_en": p["prompt_en"],
                                              "prompt_translated": p["prompt"]}
                                             for p in translations]
        except Exception as e:
            print(f"  Failed on {lang_name}: {e}")
            continue

    # Cross-language summary
    print(f"\n\n========================================================")
    print(f"=== Cross-language summary (overall accuracy on n=60) ===")
    print(f"========================================================")
    conds = ["baseline", "decompose", "restate", "estimate", "constraints",
             "scheduler", "oracle"]
    print(f"{'Lang':6s} " + " ".join(f"{c:>10s}" for c in conds) + "  schedacc")
    print("-" * 100)
    for lc in ["en", "es", "fr", "de", "ja"]:
        if lc not in all_results: continue
        row = [all_results[lc][c]["overall"] for c in conds]
        sched_correct = (e56["scheduler_correct_picks"] if lc == "en"
                          else sum(1 for p in all_problems_en
                                    if all_picks[lc].get(p["pid"]) == p["optimal_prior"]))
        print(f"{lc:6s} " + " ".join(f"{v:>10.3f}" for v in row) +
              f"  {sched_correct/60:.1%}")

    print(f"\n=== Scheduler-vs-best-fixed delta per language ===")
    for lc in all_results:
        sched = all_results[lc]["scheduler"]["overall"]
        fixed = max(all_results[lc][c]["overall"] for c in
                      ["decompose", "restate", "estimate", "constraints"])
        print(f"  {lc}: scheduler={sched:.3f}  best_fixed={fixed:.3f}  "
              f"delta={sched - fixed:+.3f}")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "n_problems": 60,
           "languages": list(all_results.keys()),
           "results_per_language": all_results,
           "scheduler_picks_per_language": all_picks,
           "translations": all_translations}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
