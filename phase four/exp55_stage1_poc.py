"""Exp 55 — Stage 0.5 + Stage 1 minimal proof of concept.

Closes four standing reviewer concerns simultaneously:
  W1. ``Five-stage roadmap is speculative and not validated by experiments''
  W2. ``Architectural claims are broader than the evidence supports''
  W3. ``Empirical unit too narrow'' (one Chinese wisdom loop)
  W4. ``Positive controls failed'' (need a known-positive task)

Design: a 60-problem dataset spanning two task families where
DIFFERENT methodological priors are optimal:

  Family A (30 multi-step arithmetic word problems):
      OPTIMAL PRIOR = ``decompose'': break the problem into
      atomic arithmetic substeps before computing.
      Wrong prior (``restate'') wastes tokens on trivially clear
      questions.

  Family B (30 CRT-style trick questions, e.g. bat-and-ball):
      OPTIMAL PRIOR = ``restate'': re-read the question carefully
      and identify what is being asked before answering, because
      the obvious decompose-and-compute path is exactly the trap.
      Wrong prior (``decompose'') triggers the System-1 wrong
      answer.

Stages instantiated:
  STAGE 0   = library of 3 priors (decompose, restate, none)
  STAGE 0.5 = no separate world model in this minimal POC --- the
              solver itself is the predictor (we measure outcome
              directly rather than predict-then-execute)
  STAGE 1   = LLM scheduler that, given a problem, selects which
              prior to invoke

Conditions (each on all 60 problems):
  C1. baseline (no prior)
  C2. always-decompose
  C3. always-restate
  C4. LLM-scheduler (picks per-problem)
  C5. ORACLE (always picks the family-optimal prior; upper bound)

If C4 > max(C2, C3) and approaches C5, we have minimal Stage 1
validation: a learned scheduler that selects the right prior
beats any fixed-prior baseline.

Cost: 60 problems x 5 conditions = 300 solver calls + 60
scheduler-pick calls = 360 total. ~$3-5 cheap-tier.
Ground truth: GSM8K-style problems have numeric answers, which we
extract by regex; CRT trick questions have hand-coded answers.
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))


def _load_api_keys():
    if os.environ.get("RUOLI_GPT_KEY") and os.environ.get("RUOLI_BASE_URL"):
        return
    keyfile = Path.home() / ".api_keys"
    if not keyfile.exists():
        return
    pat = re.compile(r'^\s*export\s+(\w+)=("([^"]*)"|\'([^\']*)\'|(\S+))')
    for line in keyfile.read_text().splitlines():
        m = pat.match(line)
        if not m: continue
        name = m.group(1)
        val = m.group(3) if m.group(3) is not None else (
              m.group(4) if m.group(4) is not None else m.group(5))
        os.environ.setdefault(name, val)
        if name == "RUOLI_BASE_URL":
            base = val + "/v1" if not val.endswith("/v1") else val
            os.environ.setdefault("CLAUDE_PROXY_BASE_URL", base)
            os.environ.setdefault("GPT5_BASE_URL", base)
            os.environ.setdefault("GEMINI_PROXY_BASE_URL", base)
        if name == "RUOLI_GEMINI_KEY":
            os.environ.setdefault("GEMINI_PROXY_API_KEY", val)
        if name == "RUOLI_GPT_KEY":
            os.environ.setdefault("GPT5_API_KEY", val)
        if name == "RUOLI_CLAUDE_KEY":
            os.environ.setdefault("CLAUDE_PROXY_API_KEY", val)

_load_api_keys()
from model_router import cheap

PARALLEL = 6
AUTO = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO / "exp55_stage1_poc_log.json"


# ======================================================================
# Task set: 30 multi-step arithmetic (Family A) + 30 CRT trick (Family B)
# Each problem: {pid, family, prompt, gold, gold_explain}
# ======================================================================

FAMILY_A_DECOMPOSE = []  # multi-step arithmetic
FAMILY_B_RESTATE = []    # CRT trick

# ---------- Family A: multi-step arithmetic word problems ----------
# These benefit from DECOMPOSE: break into atomic ops before computing.
FAMILY_A_DECOMPOSE = [
    {"prompt": "A school has 24 classrooms. Each classroom has 35 students. If each student needs 3 notebooks per term and a notebook costs $2, what is the total cost of notebooks for all students for one term?", "gold": 5040},
    {"prompt": "A factory produces 480 widgets per day. 15% of widgets are defective and discarded. The remaining widgets are sold at $7 each. What is the daily revenue?", "gold": 2856},
    {"prompt": "Alice runs 8 km per day on weekdays and 12 km per day on weekends. How many km does she run in 4 weeks?", "gold": 256},
    {"prompt": "A box contains 6 red balls, 8 blue balls, and 10 green balls. If 25% of the balls are removed (rounded down), how many balls remain?", "gold": 18},
    {"prompt": "A car travels at 60 km/h for 2 hours, then at 90 km/h for 3 hours. What is its average speed in km/h over the entire 5-hour journey?", "gold": 78},
    {"prompt": "A baker uses 250g of flour for each cake. He has 8 kg of flour. After baking, 10% of cakes burn and are discarded. How many edible cakes does he produce?", "gold": 28},
    {"prompt": "A library has 1200 books. 35% are fiction, 25% are science, and the rest are history. If 40 fiction books are checked out, how many fiction books remain?", "gold": 380},
    {"prompt": "A cyclist covers 18 km in the first hour, then his speed decreases by 1 km/h every subsequent hour. What is the total distance after 4 hours?", "gold": 60},
    {"prompt": "A store sells T-shirts at $15 each. They offer a discount of 20% if you buy 5 or more. How much do you pay for 7 T-shirts?", "gold": 84},
    {"prompt": "A water tank holds 5000 liters. It loses 8 liters per hour due to evaporation and is refilled at 12 liters per hour from a pipe. Starting full, how full is it after 24 hours?", "gold": 5096},
    {"prompt": "A worker earns $18 per hour for the first 40 hours and $27 per hour for overtime. He worked 52 hours this week. What is his total pay?", "gold": 1044},
    {"prompt": "A garden has 3 rows of tomato plants and 4 rows of pepper plants. Each row contains 12 plants. Each tomato plant yields 8 tomatoes; each pepper plant yields 5 peppers. What is the total harvest count (tomatoes + peppers)?", "gold": 528},
    {"prompt": "A teacher distributes 360 candies among her students. She keeps 12 for herself, gives 8 candies to each student, and has 4 left over. How many students are there?", "gold": 43},
    {"prompt": "A pool measures 20 m by 10 m by 2 m deep. Water fills it at 200 liters per minute (1 cubic meter = 1000 liters). How many minutes to fill?", "gold": 2000},
    {"prompt": "A train leaves at 9:00 AM, travels 60 km in the first hour, 75 km in the second, and 90 km in the third. What is its average speed for the 3 hours?", "gold": 75},
    {"prompt": "A city has 8 parks. Each park has 25 benches. Each bench costs $120 to install. What is the total installation cost for all benches?", "gold": 24000},
    {"prompt": "Each box contains 24 cans. Each crate holds 6 boxes. A truck can carry 50 crates. How many cans does a fully loaded truck carry?", "gold": 7200},
    {"prompt": "A man's age is 4 times his son's age. The sum of their ages is 60. In how many years will the son be half the father's current age?", "gold": 18},
    {"prompt": "A printer prints 45 pages per minute. It takes a 5-minute break after every 30 minutes of printing. How many pages does it print in 2 hours of total elapsed time?", "gold": 4725},
    {"prompt": "A movie theater sells 200 adult tickets at $12 each and 150 child tickets at $7 each. Concessions sales total $850. What is total revenue?", "gold": 4300},
    {"prompt": "A square garden has side 15 meters. A circular fountain of radius 3 meters is at the center. How much grass area, in square meters, surrounds the fountain (use pi=3.14)? Round to integer.", "gold": 197},
    {"prompt": "A delivery van carries 80 packages. 25% are large (3 kg each) and the rest are small (1 kg each). What is the total weight of all packages in kg?", "gold": 120},
    {"prompt": "A class of 30 students has 18 girls. The average score of girls is 85 and of boys is 78. What is the class average?", "gold": 82},
    {"prompt": "A factory has 5 machines. Each runs 16 hours per day and produces 30 units per hour. 4% of units fail QA. How many good units per day?", "gold": 2304},
    {"prompt": "Anna saves $50 per week. After 8 weeks she spent half her savings on a gift. From the remaining, she spends $20 per week. After how many more weeks is her balance zero?", "gold": 10},
    {"prompt": "A company has 240 employees. 5/8 work in engineering and 1/4 work in sales. The rest work in HR. How many work in HR?", "gold": 30},
    {"prompt": "A 240-page book is being copied. The first 30 pages take 4 minutes each; remaining pages take 2 minutes each. Total time in minutes?", "gold": 540},
    {"prompt": "A plant grows 3 cm per week for the first 4 weeks and 2 cm per week thereafter. What is its height after 10 weeks if it starts at 5 cm?", "gold": 29},
    {"prompt": "A taxi charges $3 base + $1.50 per km. A trip of 12 km takes 18 minutes. The driver receives a 15% tip on the fare. What is the driver's total revenue from this trip?", "gold": 24.15},
    {"prompt": "A school orders 144 chairs in 12 boxes equally. They need 8 boxes for the auditorium and the rest for the cafeteria. The cafeteria already has 18 chairs. How many chairs does the cafeteria have now?", "gold": 66},
]
for i, p in enumerate(FAMILY_A_DECOMPOSE):
    p["pid"] = f"A_{i:02d}"
    p["family"] = "A_decompose"
    p["optimal_prior"] = "decompose"
assert len(FAMILY_A_DECOMPOSE) == 30

# ---------- Family B: CRT-style trick questions ----------
# These benefit from RESTATE: re-read the question, decompose triggers
# the System-1 wrong answer.
FAMILY_B_RESTATE = [
    {"prompt": "A bat and a ball together cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost in dollars?", "gold": 0.05},
    {"prompt": "If 5 machines take 5 minutes to make 5 widgets, how many minutes does it take 100 machines to make 100 widgets?", "gold": 5},
    {"prompt": "In a lake there is a patch of lily pads. Every day the patch doubles in size. If it takes 48 days to cover the entire lake, how many days does it take for the patch to cover half the lake?", "gold": 47},
    {"prompt": "Emily's father has three daughters. The first two are named April and May. What is the third daughter's name?", "gold": "Emily"},
    {"prompt": "A farmer has 17 sheep. All but 9 die. How many sheep are left?", "gold": 9},
    {"prompt": "If a plane crashes on the border of the US and Canada, where do you bury the survivors?", "gold": "you don't bury survivors"},
    {"prompt": "How many of each animal did Moses take on the ark?", "gold": 0},
    {"prompt": "If you overtake the second-place runner in a race, what position are you in now?", "gold": 2},
    {"prompt": "A doctor gives you 3 pills and tells you to take one every half hour. How many minutes pass before all pills are taken?", "gold": 60},
    {"prompt": "I have two coins totaling 30 cents. One is not a nickel. What are the two coins?", "gold": "a quarter and a nickel"},
    {"prompt": "A man is looking at a photograph and says 'Brothers and sisters I have none, but that man's father is my father's son.' Who is in the photograph?", "gold": "his son"},
    {"prompt": "If it takes 8 men 10 hours to build a wall, how long would it take 4 men to build the same wall? (Assume identical work conditions.) Answer in hours.", "gold": 20},
    {"prompt": "A bottle of wine costs $10. The wine costs $9 more than the bottle. How much does the bottle alone cost in dollars?", "gold": 0.50},
    {"prompt": "If you have 6 apples and you take away 4, how many do you have?", "gold": 4},
    {"prompt": "Some months have 31 days; others have 30. How many months have 28 days?", "gold": 12},
    {"prompt": "A rooster lays an egg on the apex of a slanted roof. Which side does the egg roll down?", "gold": "neither, roosters don't lay eggs"},
    {"prompt": "Mary's mother has four daughters: Penny, Nickel, Dime, and ___. What is the fourth daughter's name?", "gold": "Mary"},
    {"prompt": "If a red house is made of red bricks and a blue house is made of blue bricks, what is a greenhouse made of?", "gold": "glass"},
    {"prompt": "Two fathers and two sons go fishing. Each catches one fish. Total fish caught = 3. How is this possible?", "gold": "grandfather, father, son (3 people)"},
    {"prompt": "A clerk in a butcher shop is 5 ft 10 in tall and wears size 13 shoes. What does he weigh?", "gold": "meat"},
    {"prompt": "If the day before yesterday is two days after Monday, what day is today?", "gold": "Friday"},
    {"prompt": "There is one room with no doors and no windows. Inside there is a man hanging from the ceiling and a puddle on the floor. How did he die?", "gold": "stood on ice that melted"},
    {"prompt": "A woman shoots her husband, then holds him underwater for 5 minutes, then hangs him. Five minutes later they go out for dinner. How?", "gold": "she's a photographer"},
    {"prompt": "Three lawyers stand under one umbrella, but none of them get wet. How?", "gold": "it's not raining"},
    {"prompt": "A boy was at a carnival and went to a booth where a man said 'If I write your exact weight on this paper then you have to give me $50, but if I cannot, I will pay you $50.' The boy walks away $50 richer without weighing himself. How?", "gold": "the man wrote 'your exact weight'"},
    {"prompt": "Before Mt. Everest was discovered, what was the highest mountain in the world?", "gold": "Mt. Everest"},
    {"prompt": "How many seconds are in a year?", "gold": 12},
    {"prompt": "Susan has 3 brothers. Each brother has 2 sisters. How many sisters does Susan have?", "gold": 1},
    {"prompt": "If you're running a race and you pass the person in 4th place, what place are you in?", "gold": 4},
    {"prompt": "A man drives 1 mile south, 1 mile east, and 1 mile north and arrives back at his starting point. He sees a bear. What color is the bear?", "gold": "white"},
]
for i, p in enumerate(FAMILY_B_RESTATE):
    p["pid"] = f"B_{i:02d}"
    p["family"] = "B_restate"
    p["optimal_prior"] = "restate"
assert len(FAMILY_B_RESTATE) == 30

ALL_PROBLEMS = FAMILY_A_DECOMPOSE + FAMILY_B_RESTATE


# ======================================================================
# Priors (Stage 0)
# ======================================================================

PRIORS = {
    "none": "",
    "decompose":
        "Before answering, decompose the problem into atomic substeps. "
        "Identify the smallest computational unit, list each step, then "
        "combine.",
    "restate":
        "Before answering, RE-READ the question carefully and restate "
        "in your own words what is actually being asked. Do not jump "
        "to compute; first identify whether the obvious decomposition "
        "is the correct interpretation.",
}


SOLVE_PROMPT_TEMPLATE = """## Problem
{problem}

## Approach hint
{approach}

## Output format
Reason step by step in 1-3 sentences, then on the LAST LINE write
exactly:
ANSWER: <your final answer>
"""

SCHEDULER_PROMPT = """You are a strategy selector. Given a problem,
choose ONE prior from this set: {{decompose, restate, none}}.

- decompose: best for multi-step arithmetic / counting problems where
  the right answer requires explicitly walking through subcomputations.
- restate: best for questions where the obvious computational
  interpretation is a trap; re-reading the question carefully reveals
  the actual ask.
- none: choose this only if neither prior helps.

## Problem
{problem}

## Output (JSON)
{{"choice": "decompose"|"restate"|"none", "reason": "1 sentence"}}
"""


# ======================================================================
# Solver and scoring
# ======================================================================

def solve(client, problem, prior_name):
    prior_text = PRIORS[prior_name]
    if prior_text:
        approach = f"Use this strategy: {prior_text}"
    else:
        approach = "Use any approach you think appropriate."
    try:
        r = client.generate(
            SOLVE_PROMPT_TEMPLATE.format(problem=problem, approach=approach),
            max_tokens=600, temperature=0.0)
        return r["text"].strip()
    except Exception as e:
        return f"[err: {e}]"


def schedule(client, problem):
    """Stage 1 LLM scheduler picks a prior given the problem."""
    try:
        r = client.generate(SCHEDULER_PROMPT.format(problem=problem),
                             max_tokens=200, temperature=0.0)
        text = r["text"].strip()
        # Extract JSON
        m = re.search(r'\{[^}]*"choice"\s*:\s*"(decompose|restate|none)"',
                       text)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "none"


def extract_answer(text):
    """Extract the answer after 'ANSWER:'."""
    m = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if not m:
        return text[-200:].strip()  # fallback: last 200 chars
    return m.group(1).strip()


def score_numeric(extracted, gold):
    """Returns (score: 0/1, normalized_extracted, gold)."""
    s = extracted.replace(",", "").replace("$", "").replace("%", "").strip()
    s = s.rstrip(".").strip()
    # Try to parse first number
    m = re.search(r'-?\d+(?:\.\d+)?', s)
    if not m:
        return 0, extracted, gold
    try:
        ext_val = float(m.group())
    except ValueError:
        return 0, extracted, gold
    try:
        gold_val = float(gold)
    except (ValueError, TypeError):
        return 0, extracted, gold
    # tolerance for floats
    if abs(ext_val - gold_val) < 0.02:
        return 1, ext_val, gold_val
    return 0, ext_val, gold_val


def score_text(extracted, gold):
    """Substring-match scoring for trick questions with text answers."""
    e = extracted.lower().strip().rstrip(".")
    g = str(gold).lower().strip()
    # split gold into key terms
    if g in e:
        return 1, extracted, gold
    # tolerate slight phrasings: e.g. gold "stood on ice that melted",
    # accepted if 'ice' and ('melt' or 'melted') in extracted
    g_words = set(re.findall(r'\w+', g))
    e_words = set(re.findall(r'\w+', e))
    overlap = g_words & e_words
    # Require overlap of half of meaningful gold words
    meaningful_gold = {w for w in g_words if len(w) >= 3}
    overlap_meaningful = meaningful_gold & e_words
    if meaningful_gold and len(overlap_meaningful) / len(meaningful_gold) >= 0.5:
        return 1, extracted, gold
    return 0, extracted, gold


def score(extracted, gold):
    if isinstance(gold, (int, float)):
        return score_numeric(extracted, gold)
    return score_text(extracted, gold)


# ======================================================================
# Main experiment
# ======================================================================

def main():
    print(f"=== Exp 55: Stage 0.5 + Stage 1 minimal POC ===")
    print(f"  Problems: {len(ALL_PROBLEMS)} ({len(FAMILY_A_DECOMPOSE)} "
          f"family A decompose-optimal + {len(FAMILY_B_RESTATE)} "
          f"family B restate-optimal)")
    print(f"  Conditions: baseline / decompose / restate / "
          f"LLM-scheduler / oracle\n")

    solver = cheap("gemini")
    scheduler_client = cheap("gemini")

    # Run all 4 fixed conditions on all problems
    conditions = ["baseline", "decompose", "restate"]
    answers = {c: {} for c in conditions}
    answers["scheduler"] = {}
    answers["oracle"] = {}
    sched_picks = {}

    print(f"[1/3] Running fixed conditions ({len(conditions)} x "
          f"{len(ALL_PROBLEMS)} = {len(conditions)*len(ALL_PROBLEMS)} calls)...")
    tasks = [(c, p) for c in conditions for p in ALL_PROBLEMS]

    def run_fixed(c, p):
        prior = "none" if c == "baseline" else c
        return c, p["pid"], solve(solver, p["prompt"], prior)

    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(run_fixed, c, p) for c, p in tasks]
        for f in as_completed(futs):
            c, pid, ans = f.result()
            answers[c][pid] = ans
            done += 1
            if done % 30 == 0:
                print(f"  fixed {done}/{len(tasks)} ({time.time()-t0:.0f}s)")

    print(f"\n[2/3] Stage-1 LLM scheduler (60 picks + 60 solves)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(schedule, scheduler_client, p["prompt"]):
                  p for p in ALL_PROBLEMS}
        for i, f in enumerate(as_completed(futs), 1):
            p = futs[f]
            sched_picks[p["pid"]] = f.result()
            if i % 20 == 0:
                print(f"  schedule {i}/60 ({time.time()-t0:.0f}s)")

    # Now solve with the scheduler-picked prior
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(solve, solver, p["prompt"], sched_picks[p["pid"]]):
                  p for p in ALL_PROBLEMS}
        for i, f in enumerate(as_completed(futs), 1):
            p = futs[f]
            answers["scheduler"][p["pid"]] = f.result()
            if i % 20 == 0:
                print(f"  scheduler-solve {i}/60 ({time.time()-t0:.0f}s)")

    print(f"\n[3/3] Oracle (always-optimal-prior, 60 calls)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(solve, solver, p["prompt"], p["optimal_prior"]):
                  p for p in ALL_PROBLEMS}
        for i, f in enumerate(as_completed(futs), 1):
            p = futs[f]
            answers["oracle"][p["pid"]] = f.result()
            if i % 20 == 0:
                print(f"  oracle {i}/60 ({time.time()-t0:.0f}s)")

    # Score and aggregate
    print(f"\n=== Scoring ===")
    cond_scores = {c: {"A": [], "B": []} for c in
                    ["baseline", "decompose", "restate", "scheduler", "oracle"]}
    per_problem = {}
    for p in ALL_PROBLEMS:
        per_problem[p["pid"]] = {
            "family": p["family"], "gold": p["gold"], "prompt": p["prompt"]}
        fam = "A" if p["family"].startswith("A") else "B"
        for c in ["baseline", "decompose", "restate", "scheduler", "oracle"]:
            ans = answers[c].get(p["pid"], "")
            ext = extract_answer(ans)
            sc, _, _ = score(ext, p["gold"])
            cond_scores[c][fam].append(sc)
            per_problem[p["pid"]][f"{c}_answer"] = ans
            per_problem[p["pid"]][f"{c}_extracted"] = ext
            per_problem[p["pid"]][f"{c}_score"] = sc

    print(f"\n{'Condition':14s} {'Family A acc':>14s} {'Family B acc':>14s} "
          f"{'Overall':>10s}")
    print("-" * 60)
    for c in ["baseline", "decompose", "restate", "scheduler", "oracle"]:
        a_acc = sum(cond_scores[c]["A"]) / len(cond_scores[c]["A"])
        b_acc = sum(cond_scores[c]["B"]) / len(cond_scores[c]["B"])
        overall = (sum(cond_scores[c]["A"]) + sum(cond_scores[c]["B"])) / 60
        print(f"{c:14s} {a_acc:>14.3f} {b_acc:>14.3f} {overall:>10.3f}")

    # Scheduler quality: how often does it pick the optimal prior?
    print(f"\n=== Scheduler pick quality ===")
    correct_picks = 0
    a_picks = {"decompose": 0, "restate": 0, "none": 0}
    b_picks = {"decompose": 0, "restate": 0, "none": 0}
    for p in ALL_PROBLEMS:
        pick = sched_picks[p["pid"]]
        fam = "A" if p["family"].startswith("A") else "B"
        if fam == "A": a_picks[pick] = a_picks.get(pick, 0) + 1
        else: b_picks[pick] = b_picks.get(pick, 0) + 1
        if pick == p["optimal_prior"]:
            correct_picks += 1
    print(f"  Family A picks (optimal=decompose): {a_picks}")
    print(f"  Family B picks (optimal=restate):   {b_picks}")
    print(f"  Total correct picks: {correct_picks}/60 = "
          f"{correct_picks/60:.1%}")

    # Save
    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "n_problems": len(ALL_PROBLEMS),
           "n_family_A": len(FAMILY_A_DECOMPOSE),
           "n_family_B": len(FAMILY_B_RESTATE),
           "conditions": list(cond_scores.keys()),
           "scheduler_picks": sched_picks,
           "scheduler_correct_picks": correct_picks,
           "scores": {c: {"A_acc": sum(cond_scores[c]["A"]) / len(cond_scores[c]["A"]),
                           "B_acc": sum(cond_scores[c]["B"]) / len(cond_scores[c]["B"]),
                           "overall_acc": (sum(cond_scores[c]["A"]) + sum(cond_scores[c]["B"])) / 60,
                           "A_n_correct": sum(cond_scores[c]["A"]),
                           "B_n_correct": sum(cond_scores[c]["B"])}
                       for c in cond_scores},
           "per_problem": per_problem}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
