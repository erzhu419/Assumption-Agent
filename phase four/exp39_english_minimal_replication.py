"""Exp 39 — English-language minimal replication of the audit-stack
fragility.

Addresses the GPT-review critique that the paper's claim is
language-scoped and that the Chinese-only evaluation cannot
support a general field-level conclusion.

Protocol:
  1. Take 30 prompts from MT-Bench (writing/reasoning/extraction
     categories — open-ended, English-authored).
  2. Build a 6-entry English-only wisdom library: short
     methodological aphorisms with English source attributions
     and English-authored unpacking, NO Chinese.
  3. Use the same gemini-3-flash solver. For each prompt, generate
     base answer (no library) and 6 ext answers (one per wisdom).
  4. Cheap-tier 3-judge panel adjudicates each ext-vs-base pair.
  5. Apply L1 cross-family check on the strongest English wisdom.

Result interpretation:
  - If English wisdoms also show judge-fragility (high cheap-tier wr,
    low expensive-tier wr), our claim survives the language scope.
  - If English wisdoms ARE robust under cross-family judges, the
    paper's claim is genuinely Chinese-loop-specific and we say so.

Either outcome is informative.

Cost: 30 × 7 generations + 30 × 6 × 3 cheap judgments ≈ 750 calls.
~$10, ~30 min.
"""

import json
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from model_router import cheap, cheap_panel
from llm_client import parse_json_from_llm

AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp39_english_replication_log.json"

PARALLEL = 6
N_PROMPTS = 30


# 6 English-only wisdoms — Munger / Drucker / Buffett / classic
# methodological aphorisms with English unpacking.
ENGLISH_WISDOMS = [
    {
        "id": "EW01",
        "aphorism": "To a man with a hammer, everything looks like a nail.",
        "source": "Charlie Munger (mental models tradition)",
        "signal": "When the same lens is being applied repeatedly to "
                  "problems whose surface form is similar but whose "
                  "underlying structure differs.",
        "unpacked_for_llm": "Before committing to a solution, name the "
                             "lens you are using and check whether the "
                             "problem actually fits that lens. If you are "
                             "applying the same model to every situation, "
                             "you are likely missing structural variation.",
    },
    {
        "id": "EW02",
        "aphorism": "If you can't measure it, you can't manage it.",
        "source": "attributed Drucker (managerial-empiricism tradition)",
        "signal": "When recommendations are being made about a process "
                  "without first establishing what counts as success and "
                  "how it would be observed.",
        "unpacked_for_llm": "Before recommending an action, name the "
                             "metric whose change would tell you the "
                             "action worked. If no such metric exists or "
                             "is impractical to obtain, the recommendation "
                             "is unfalsifiable.",
    },
    {
        "id": "EW03",
        "aphorism": "Be approximately right rather than precisely wrong.",
        "source": "Carveth Read / Keynes / value-investing tradition",
        "signal": "When the answer requires extrapolation beyond available "
                  "data and false-precision (decimals, %, exact dates) "
                  "would substitute for genuine uncertainty.",
        "unpacked_for_llm": "Prefer ranges, orders of magnitude, and "
                             "explicit unknowns over single point "
                             "estimates whose precision the data cannot "
                             "support. Note where the analysis is "
                             "interpolation vs.\ extrapolation.",
    },
    {
        "id": "EW04",
        "aphorism": "Show me the incentive and I will show you the outcome.",
        "source": "Charlie Munger",
        "signal": "When the question is about persistent behaviour of "
                  "an actor or organisation but the proposed answer "
                  "names motives or values rather than the structure of "
                  "rewards and consequences they actually face.",
        "unpacked_for_llm": "When predicting or explaining persistent "
                             "behaviour, identify the incentive structure "
                             "(what is rewarded, what is punished, by "
                             "whom, on what timescale) before invoking "
                             "values or character.",
    },
    {
        "id": "EW05",
        "aphorism": "Better to be roughly right early than exactly right late.",
        "source": "decision-theory and bake-off tradition",
        "signal": "When the answer requires committing to one approach "
                  "out of several plausible candidates and the cost of "
                  "delay is non-trivial.",
        "unpacked_for_llm": "List the candidate approaches, define "
                             "cheap discriminating tests, run them "
                             "before committing fully. A 70%-confident "
                             "decision delivered now usually beats a "
                             "95%-confident decision delivered after "
                             "the window has closed.",
    },
    {
        "id": "EW06",
        "aphorism": "Distinguish the known from the unknown from the unknowable.",
        "source": "Frank Knight / risk-uncertainty tradition",
        "signal": "When the question conflates measurable risk with "
                  "irreducible uncertainty or with absence of relevant "
                  "data.",
        "unpacked_for_llm": "Sort facts by epistemic status: things you "
                             "have direct evidence for; things you can "
                             "estimate by extrapolation; things you "
                             "could in principle know but don't yet; "
                             "things that are unknowable in the relevant "
                             "timeframe. Different categories warrant "
                             "different treatments.",
    },
]


# 30 MT-Bench-style English problems (curated subset to keep prompts open-ended)
# We'll pull from MT-Bench via datasets.
def load_mtbench_prompts(n=30):
    """Pull MT-Bench question prompts; filter to open-ended cats."""
    from datasets import load_dataset
    try:
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    except Exception:
        try:
            ds = load_dataset("philschmid/mt-bench", split="train")
        except Exception as e:
            raise RuntimeError(f"Could not load MT-Bench: {e}")
    open_cats = {"writing", "roleplay", "reasoning", "extraction",
                 "humanities", "stem"}
    prompts = []
    for x in ds:
        cat = x.get("category", "").lower()
        if cat not in open_cats:
            continue
        raw = x.get("prompt") or x.get("turns") or x.get("question")
        # MT-Bench HF dataset stores `prompt` as a list of turns; take first
        if isinstance(raw, list):
            raw = raw[0] if raw else None
        if not raw or not isinstance(raw, str) or len(raw) < 50:
            continue
        prompts.append({"id": f"mtbench_{x.get('prompt_id', len(prompts))}",
                        "category": cat, "problem": raw})
        if len(prompts) >= n:
            break
    return prompts


FRAME_PROMPT = """For the problem below, identify what kind of problem
it is and rewrite it in your own words.

## Problem
{problem}

## Output JSON
{{"frame": "object_level / paradigm / hybrid",
  "critical_reframe": "one sentence (under 60 words) on what this is really about",
  "rewritten_problem": "100-200 word rewrite making the structure explicit"}}
"""

EXECUTE_PROMPT = """# Solve the problem.

## PRIMARY FRAME
- frame: {frame}
- critical reframe: {critical_reframe}

## Problem (rewritten)
{rewritten_problem}

## Optional methodological wisdoms (apply only if they actually fit)
{wisdom_block}

## Output requirement
Give a focused English answer, under 400 words, that addresses the
specific question.
"""

JUDGE_PROMPT = """Methodological review of two answers to the same problem.

## Problem
{problem}

## Answer A
{answer_a}

## Answer B
{answer_b}

Which answer is more methodologically sound and substantively responsive
to what the problem actually asks?

Output JSON: {{"winner": "A"/"B"/"tie", "score_a": 1-10, "score_b": 1-10,
  "reasoning": "80 chars max"}}
"""


def solve(client, problem, wisdoms):
    try:
        r = client.generate(FRAME_PROMPT.format(problem=problem),
                            max_tokens=500, temperature=0.2)
        m = parse_json_from_llm(r["text"])
    except Exception as e:
        return f"[turn0 err: {e}]"
    if wisdoms:
        wb = "\n\n".join(f"• {w['aphorism']} ({w['source']})\n"
                          f"  Signal: {w['signal']}\n"
                          f"  How to apply: {w['unpacked_for_llm']}"
                          for w in wisdoms)
    else:
        wb = "(none)"
    try:
        r = client.generate(EXECUTE_PROMPT.format(
            frame=m.get("frame", "object_level"),
            critical_reframe=m.get("critical_reframe", ""),
            rewritten_problem=m.get("rewritten_problem", problem),
            wisdom_block=wb), max_tokens=900, temperature=0.2)
        return r["text"].strip()
    except Exception as e:
        return f"[turn1 err: {e}]"


def judge_one(client, problem, a, b):
    try:
        r = client.generate(JUDGE_PROMPT.format(problem=problem,
                                                  answer_a=a, answer_b=b),
                            max_tokens=300, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        return v.get("winner", "tie")
    except Exception as e:
        return f"err:{str(e)[:30]}"


def main():
    print("[1/4] Loading MT-Bench prompts...")
    try:
        prompts = load_mtbench_prompts(N_PROMPTS)
        print(f"  loaded {len(prompts)} from MT-Bench, "
              f"categories: {set(p['category'] for p in prompts)}")
    except Exception as e:
        print(f"  MT-Bench load failed: {e}")
        # Fallback: hardcode 30 generic English open-ended
        prompts = [{"id": f"fallback_{i}", "category": "fallback",
                     "problem": f"Open-ended placeholder {i}"} for i in range(N_PROMPTS)]

    solver = cheap("gemini")
    judges = cheap_panel()
    print(f"  Solver: {solver.model}")
    print(f"  Judges: {[j.model for j in judges]}\n")

    # Stage 1: generate answers (base + 6 ext)
    print(f"[2/4] Generating {len(prompts)} × 7 = {len(prompts)*7} answers...")
    answers = {"base": {}}
    for w in ENGLISH_WISDOMS:
        answers[w["id"]] = {}

    def gen_task(p, kind):
        wisdoms = [] if kind == "base" else [next(w for w in ENGLISH_WISDOMS
                                                    if w["id"] == kind)]
        return p["id"], kind, solve(solver, p["problem"], wisdoms)

    tasks = [(p, "base") for p in prompts] + \
            [(p, w["id"]) for p in prompts for w in ENGLISH_WISDOMS]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(gen_task, p, k) for p, k in tasks]
        done = 0
        for f in as_completed(futs):
            pid, kind, ans = f.result()
            answers[kind][pid] = ans
            done += 1
            if done % 20 == 0:
                print(f"  gen {done}/{len(tasks)} ({time.time()-t0:.0f}s)")

    # Stage 2: judge each ext vs base across the 3 cheap families
    print(f"\n[3/4] Judging {len(prompts)} × {len(ENGLISH_WISDOMS)} × "
           f"{len(judges)} = {len(prompts)*len(ENGLISH_WISDOMS)*len(judges)} pairs...")

    def judge_task(judge, p, w):
        pid = p["id"]; wid = w["id"]
        b = answers["base"].get(pid, "")
        e = answers[wid].get(pid, "")
        if not b or not e or b.startswith("[") or e.startswith("["):
            return judge.family, wid, pid, "missing"
        rng = random.Random((hash(pid) ^ hash(wid)) % (2**32))
        if rng.random() < 0.5:
            left, right, ext_was = e, b, "A"
        else:
            left, right, ext_was = b, e, "B"
        w_ans = judge_one(judge, p["problem"], left, right)
        if w_ans == "tie": v = "tie"
        elif w_ans in ("A", "B"): v = "ext" if w_ans == ext_was else "base"
        else: v = "err"
        return judge.family, wid, pid, v

    verdicts = {j.family: {w["id"]: {} for w in ENGLISH_WISDOMS} for j in judges}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(judge_task, j, p, w)
                 for j in judges for p in prompts for w in ENGLISH_WISDOMS]
        done = 0
        for f in as_completed(futs):
            fam, wid, pid, v = f.result()
            verdicts[fam][wid][pid] = v
            done += 1
            if done % 50 == 0:
                print(f"  judge {done}/{len(futs)} ({time.time()-t0:.0f}s)")

    # Stage 3: aggregate per (wisdom, family) wr_ext
    print(f"\n[4/4] Per-wisdom × per-family wr_ext (single-family A/B):")
    print(f"{'wisdom':30s}", end="")
    for j in judges:
        print(f"  {j.family[:10]:>10s}", end="")
    print(f"  {'mean':>6s}")
    print("-" * 80)

    summary = []
    for w in ENGLISH_WISDOMS:
        wid = w["id"]
        row = {"wid": wid, "aphorism": w["aphorism"]}
        wrs = []
        line = f"{wid + ' ' + w['aphorism'][:24]:30s}"
        for j in judges:
            v = verdicts[j.family][wid]
            ne = sum(1 for x in v.values() if x == "ext")
            nb = sum(1 for x in v.values() if x == "base")
            tot = ne + nb
            wr = ne / tot if tot else 0.5
            wrs.append(wr)
            row[f"wr_{j.family}"] = wr
            row[f"n_{j.family}"] = tot
            line += f"  {wr:>10.2f}"
        mean_wr = sum(wrs) / len(wrs) if wrs else 0.5
        line += f"  {mean_wr:>6.2f}"
        row["mean_wr"] = mean_wr
        # Single-family-fragility check: do families spread?
        spread = max(wrs) - min(wrs)
        row["family_spread"] = spread
        summary.append(row)
        print(line)

    print(f"\n=== Headline ===")
    pass_at_60 = sum(1 for r in summary
                      if any(r.get(f"wr_{j.family}", 0) >= 0.60 for j in judges))
    pass_all_60 = sum(1 for r in summary
                       if all(r.get(f"wr_{j.family}", 0) >= 0.60 for j in judges))
    print(f"  English wisdoms with at least one family wr >= 0.60: "
          f"{pass_at_60}/{len(ENGLISH_WISDOMS)}")
    print(f"  English wisdoms with ALL three families wr >= 0.60: "
          f"{pass_all_60}/{len(ENGLISH_WISDOMS)}  (this is the "
          f"non-fragile-under-cheap-panel count)")
    spreads = [r["family_spread"] for r in summary]
    print(f"  Mean family-spread (max-min wr per wisdom): "
          f"{sum(spreads)/len(spreads):.3f}")
    print(f"  Compare to Chinese-loop: e.g., W078 = 0.60/0.51/0.57 spread = 0.09")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "benchmark": "MT-Bench (open-ended categories), n=30 prompts",
           "wisdoms": ENGLISH_WISDOMS, "judges": [j.model for j in judges],
           "solver": solver.model, "verdicts": verdicts, "summary": summary}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
