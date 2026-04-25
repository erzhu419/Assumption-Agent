"""Exp 56 — Stage 1 validation across 4 task families and 4 priors.

Closes the ``empirical unit too narrow'' concern by extending Exp 55
(2 priors × 2 families) to 4 priors × 4 task families. Each family
has a different methodologically optimal prior; the LLM scheduler
must learn to pick correctly across the broader space.

Priors (Stage 0):
  decompose:    break into atomic substeps (best for arithmetic)
  restate:      re-read the question (best for trick questions)
  estimate:     order-of-magnitude estimate first (best for Fermi)
  constraints:  list hidden constraints before solving (best for
                logic puzzles)

Task families (15 problems each, 60 total):
  A: multi-step arithmetic (decompose-optimal)
  B: CRT trick questions (restate-optimal)
  C: Fermi / estimation problems (estimate-optimal)
  D: logic puzzles with constraints (constraints-optimal)

Conditions:
  baseline + 4 fixed priors + LLM-scheduler + oracle = 7 conditions
  Total: 60 problems × 7 = 420 solver calls + 60 schedule = 480
  Cost: ~$5 cheap-tier.
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
        val = m.group(3) if m.group(3) is not None else (
              m.group(4) if m.group(4) is not None else m.group(5))
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

PARALLEL = 6
AUTO = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO / "exp56_stage1_broader_log.json"

# Reuse 15 from each family of Exp 55, plus add families C and D
FAMILY_A = [  # multi-step arithmetic, decompose-optimal
    {"prompt": "A school has 24 classrooms. Each classroom has 35 students. If each student needs 3 notebooks per term and a notebook costs $2, what is the total cost of notebooks?", "gold": 5040},
    {"prompt": "A factory produces 480 widgets per day. 15% are defective and discarded. The remaining are sold at $7 each. Daily revenue?", "gold": 2856},
    {"prompt": "Alice runs 8 km per day on weekdays and 12 km per day on weekends. How many km in 4 weeks?", "gold": 256},
    {"prompt": "A box has 6 red, 8 blue, 10 green balls. If 25% are removed (rounded down), how many balls remain?", "gold": 18},
    {"prompt": "A car: 60 km/h for 2h, 90 km/h for 3h. Average speed over 5h in km/h?", "gold": 78},
    {"prompt": "A baker uses 250g flour per cake. 8 kg flour. 10% of cakes burn. How many edible cakes?", "gold": 28},
    {"prompt": "1200 books: 35% fiction, 25% science, rest history. 40 fiction checked out. Fiction remaining?", "gold": 380},
    {"prompt": "A cyclist: 18 km in hour 1, speed -1 km/h each subsequent hour. Total distance after 4 hours?", "gold": 60},
    {"prompt": "T-shirts $15 each, 20% off if buying 5+. How much for 7 T-shirts?", "gold": 84},
    {"prompt": "Tank holds 5000 L. Loses 8 L/h evaporation, refilled at 12 L/h. Starting full, how full after 24h?", "gold": 5096},
    {"prompt": "Worker: $18/h first 40h, $27/h overtime. Worked 52h. Total pay?", "gold": 1044},
    {"prompt": "Garden: 3 rows tomatoes, 4 rows peppers, 12 plants/row. Tomatoes 8 each, peppers 5 each. Total harvest count?", "gold": 528},
    {"prompt": "Teacher distributes 360 candies. Keeps 12, gives 8 each, has 4 left. How many students?", "gold": 43},
    {"prompt": "Pool 20m × 10m × 2m. Fills at 200 L/min (1 m³ = 1000 L). Minutes to fill?", "gold": 2000},
    {"prompt": "A class of 30 has 18 girls. Girl avg 85, boy avg 78. Class avg?", "gold": 82},
]
FAMILY_B = [  # CRT trick questions, restate-optimal
    {"prompt": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much is the ball?", "gold": 0.05},
    {"prompt": "5 machines take 5 minutes to make 5 widgets. How long for 100 machines to make 100 widgets?", "gold": 5},
    {"prompt": "A lily pad patch doubles daily. It covers the lake in 48 days. When did it cover half?", "gold": 47},
    {"prompt": "Emily's father has three daughters. The first two are April and May. What is the third's name?", "gold": "Emily"},
    {"prompt": "A farmer has 17 sheep. All but 9 die. How many remain?", "gold": 9},
    {"prompt": "How many of each animal did Moses take on the ark?", "gold": 0},
    {"prompt": "If you overtake the second-place runner, what position are you in now?", "gold": 2},
    {"prompt": "Doctor gives you 3 pills, take one every half hour. How many minutes total?", "gold": 60},
    {"prompt": "Two coins total 30 cents. One is not a nickel. What are they?", "gold": "a quarter and a nickel"},
    {"prompt": "If 8 men take 10 hours to build a wall, how long for 4 men? Hours.", "gold": 20},
    {"prompt": "A bottle of wine costs $10. The wine costs $9 more than the bottle. The bottle alone in dollars?", "gold": 0.50},
    {"prompt": "How many months have 28 days?", "gold": 12},
    {"prompt": "Mary's mother has four daughters: Penny, Nickel, Dime, and ___. The fourth's name?", "gold": "Mary"},
    {"prompt": "If a red house is made of red bricks and a blue house of blue bricks, what is a greenhouse made of?", "gold": "glass"},
    {"prompt": "Susan has 3 brothers. Each brother has 2 sisters. How many sisters does Susan have?", "gold": 1},
]
FAMILY_C = [  # Fermi / estimation, estimate-first optimal
    # The "right" answer for these is an order-of-magnitude estimate;
    # we mark gold as a numeric value in the right ballpark and accept
    # within 1 order of magnitude (factor of 10).
    {"prompt": "Roughly how many piano tuners are there in Chicago? Give a single integer estimate.", "gold": 100, "tol": 10},
    {"prompt": "Estimate the number of golf balls that fit inside a typical school bus. Single integer.", "gold": 500000, "tol": 10},
    {"prompt": "Estimate how many trees there are on Earth (in billions). Single integer.", "gold": 3000, "tol": 10},
    {"prompt": "How many heartbeats does an average human have in a 75-year lifetime? Single integer.", "gold": 3000000000, "tol": 10},
    {"prompt": "Estimate the number of atoms in a single grain of sand. Order of magnitude (10^N), give N.", "gold": 19, "tol": 2},
    {"prompt": "How many words does an average adult speak in a day? Single integer.", "gold": 16000, "tol": 5},
    {"prompt": "Estimate the total mass of all humans alive on Earth, in kilograms. Single integer.", "gold": 400000000000, "tol": 10},
    {"prompt": "How many cells are in the average adult human body? Order of magnitude (10^N), give N.", "gold": 13, "tol": 2},
    {"prompt": "Estimate the number of pizzas eaten in the US per year, in millions. Single integer.", "gold": 3000, "tol": 5},
    {"prompt": "How many words are in the average novel? Single integer.", "gold": 80000, "tol": 3},
    {"prompt": "Estimate how many gallons of water a person uses per day in the US. Single integer.", "gold": 80, "tol": 3},
    {"prompt": "How many cars cross the Brooklyn Bridge per day? Single integer estimate.", "gold": 116000, "tol": 5},
    {"prompt": "Estimate the number of stars in the Milky Way, in billions. Single integer.", "gold": 200, "tol": 5},
    {"prompt": "How many breaths does an average person take per day? Single integer.", "gold": 22000, "tol": 3},
    {"prompt": "Estimate the number of grains of rice in a 1 kg bag. Single integer.", "gold": 50000, "tol": 3},
]
FAMILY_D = [  # logic puzzles with constraints, constraints-first optimal
    # Each has a unique answer derivable by enumerating constraints.
    {"prompt": "Three boxes: one has only apples, one only oranges, one mixed. All three are mislabeled. You may pick one fruit from one box without looking. Which box should you pick from to label all correctly?", "gold": "mixed"},
    {"prompt": "Five people in a row: Alice is left of Bob, Bob is left of Carol, Dave is to Alice's immediate right, Eve is at the rightmost end. Who is in the middle?", "gold": "Dave"},
    {"prompt": "A man wants to ferry a wolf, goat, and cabbage across a river. His boat holds him plus one item. The wolf eats the goat alone, the goat eats the cabbage alone. What does he take across first?", "gold": "goat"},
    {"prompt": "Three switches in one room control three bulbs in another room you cannot see into. You may go to the bulbs only ONCE. How can you determine which switch controls which bulb? Describe the method briefly.", "gold": "turn one on for a while, turn off; turn another on; check"},
    {"prompt": "A book is on top of a table. The table is on a chair. The chair is in the kitchen. Where is the book?", "gold": "kitchen"},
    {"prompt": "A and B are siblings. A is older. C is younger than A but older than B. Order them youngest to oldest.", "gold": "B C A"},
    {"prompt": "If Alice is in front of Bob and Bob is behind Carol, who is in front?", "gold": "Carol"},
    {"prompt": "There are 100 lockers numbered 1-100, all closed. 100 students take turns. Student n toggles every n-th locker. After all 100, which lockers are open? (Pattern, not list.)", "gold": "perfect squares"},
    {"prompt": "Two doors: one leads to safety, one to death. Two guards: one always lies, one always tells truth. You ask one question. What do you ask?", "gold": "what would the other guard say"},
    {"prompt": "A clock shows 3:15. What is the angle (in degrees) between the hour and minute hands?", "gold": 7.5},
    {"prompt": "I am thinking of a 2-digit number. The digits sum to 9. The number is divisible by 3 (always true if digits sum to 9). When you reverse the digits, the new number is 27 less than the original. What is the number?", "gold": 63},
    {"prompt": "5 friends sit in a row. Alice is at one end. Bob is not next to Alice. Carol is next to Bob. Dave is next to Alice. Eve is next to Carol. Where is Eve relative to Alice (1=adjacent to 5=opposite end)?", "gold": 5},
    {"prompt": "A fair coin is flipped 3 times. What is the probability of getting at least 2 heads? Give a fraction (e.g. 1/2) or decimal.", "gold": 0.5},
    {"prompt": "Train A leaves station X at 10:00 AM going east at 60 km/h. Train B leaves station Y (180 km east of X) at 10:30 AM going west at 90 km/h. At what time do they meet (HH:MM)?", "gold": "11:24"},
    {"prompt": "Three cards: Ace of Spades, King of Hearts, Queen of Clubs. They are shuffled face down. You pick two without looking. What is the probability both are red? (Hint: count colors.) Decimal.", "gold": 0.0},
]

for i, p in enumerate(FAMILY_A):
    p["pid"] = f"A_{i:02d}"; p["family"] = "A_decompose"; p["optimal_prior"] = "decompose"
for i, p in enumerate(FAMILY_B):
    p["pid"] = f"B_{i:02d}"; p["family"] = "B_restate"; p["optimal_prior"] = "restate"
for i, p in enumerate(FAMILY_C):
    p["pid"] = f"C_{i:02d}"; p["family"] = "C_estimate"; p["optimal_prior"] = "estimate"
for i, p in enumerate(FAMILY_D):
    p["pid"] = f"D_{i:02d}"; p["family"] = "D_constraints"; p["optimal_prior"] = "constraints"

ALL_PROBLEMS = FAMILY_A + FAMILY_B + FAMILY_C + FAMILY_D
assert len(ALL_PROBLEMS) == 60

PRIORS = {
    "none": "",
    "decompose":
        "Before answering, decompose the problem into atomic substeps. "
        "List each step explicitly, then combine.",
    "restate":
        "Before answering, RE-READ the question carefully and restate "
        "in your own words what is actually being asked. Do not rush "
        "to compute; the obvious computational interpretation may be a trap.",
    "estimate":
        "Before answering, estimate the order of magnitude of the "
        "answer using rough numbers and known anchors. Then sanity-check "
        "your final answer against the estimate.",
    "constraints":
        "Before answering, explicitly enumerate all the constraints "
        "given in the problem. Identify what must be true. Then derive "
        "the answer by satisfying all constraints simultaneously.",
}

SOLVE_PROMPT = """## Problem
{problem}

## Approach hint
{approach}

## Output
Reason step by step in 1-3 sentences, then on the LAST LINE write exactly:
ANSWER: <your final answer>
"""

SCHED_PROMPT = """You are a strategy selector. Given a problem, choose ONE
prior from this set: {{decompose, restate, estimate, constraints, none}}.

- decompose: best for multi-step arithmetic / counting where the right
  answer requires explicit subcomputations.
- restate: best for trick questions where the obvious computational
  interpretation is a trap.
- estimate: best for Fermi/estimation problems where the answer is an
  order-of-magnitude quantity with no exact data given.
- constraints: best for logic puzzles where the answer must satisfy a
  set of explicit constraints.
- none: only if no prior is clearly more applicable than baseline.

## Problem
{problem}

## Output (JSON only)
{{"choice": "decompose"|"restate"|"estimate"|"constraints"|"none",
  "reason": "1 short sentence"}}
"""

def solve(client, problem, prior_name):
    prior = PRIORS[prior_name]
    approach = f"Use this strategy: {prior}" if prior else "Use any approach you think appropriate."
    try:
        r = client.generate(SOLVE_PROMPT.format(problem=problem, approach=approach),
                             max_tokens=600, temperature=0.0)
        return r["text"].strip()
    except Exception as e:
        return f"[err: {e}]"

def schedule(client, problem):
    try:
        r = client.generate(SCHED_PROMPT.format(problem=problem),
                             max_tokens=200, temperature=0.0)
        m = re.search(r'"choice"\s*:\s*"(decompose|restate|estimate|constraints|none)"',
                       r["text"])
        if m: return m.group(1)
    except Exception: pass
    return "none"

def extract_answer(text):
    m = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    return (m.group(1) if m else text[-200:]).strip()

def score_numeric(ext, gold, tol=None):
    s = ext.replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")
    m = re.search(r'-?\d+(?:\.\d+)?', s)
    if not m: return 0
    try: ev = float(m.group())
    except: return 0
    try: gv = float(gold)
    except: return 0
    if tol is not None:
        # Order-of-magnitude tolerance: accept if within factor `tol`
        if gv == 0: return 1 if ev == 0 else 0
        ratio = abs(ev / gv) if gv != 0 else float('inf')
        return 1 if (1.0 / tol) <= ratio <= tol else 0
    return 1 if abs(ev - gv) < 0.02 else 0

def score_text(ext, gold):
    e = ext.lower().strip().rstrip(".")
    g = str(gold).lower().strip()
    if g in e: return 1
    g_words = set(re.findall(r'\w+', g))
    meaningful = {w for w in g_words if len(w) >= 3}
    if not meaningful: return 0
    e_words = set(re.findall(r'\w+', e))
    return 1 if len(meaningful & e_words) / len(meaningful) >= 0.5 else 0

def score(ext, gold, tol=None):
    if isinstance(gold, (int, float)):
        return score_numeric(ext, gold, tol)
    return score_text(ext, gold)


def main():
    print(f"=== Exp 56: Stage 1 broader validation (4 priors x 4 task families) ===")
    print(f"  Problems: {len(ALL_PROBLEMS)} (15 each across A/B/C/D)\n")
    solver = cheap("gemini")
    sched_client = cheap("gemini")
    conditions = ["baseline", "decompose", "restate", "estimate", "constraints"]
    answers = {c: {} for c in conditions}
    answers["scheduler"] = {}; answers["oracle"] = {}
    sched_picks = {}

    print(f"[1/3] Fixed conditions: {len(conditions)} x {len(ALL_PROBLEMS)} = "
          f"{len(conditions)*len(ALL_PROBLEMS)} calls...")
    tasks = [(c, p) for c in conditions for p in ALL_PROBLEMS]
    def run_fixed(c, p):
        prior = "none" if c == "baseline" else c
        return c, p["pid"], solve(solver, p["prompt"], prior)
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(run_fixed, c, p) for c, p in tasks]
        for f in as_completed(futs):
            c, pid, ans = f.result(); answers[c][pid] = ans; done += 1
            if done % 30 == 0: print(f"  fixed {done}/{len(tasks)} ({time.time()-t0:.0f}s)")

    print(f"\n[2/3] Stage-1 LLM scheduler: 60 picks + 60 solves...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(schedule, sched_client, p["prompt"]): p for p in ALL_PROBLEMS}
        for i, f in enumerate(as_completed(futs), 1):
            p = futs[f]; sched_picks[p["pid"]] = f.result()
            if i % 20 == 0: print(f"  schedule {i}/60 ({time.time()-t0:.0f}s)")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(solve, solver, p["prompt"], sched_picks[p["pid"]]): p
                  for p in ALL_PROBLEMS}
        for i, f in enumerate(as_completed(futs), 1):
            p = futs[f]; answers["scheduler"][p["pid"]] = f.result()
            if i % 20 == 0: print(f"  scheduler-solve {i}/60 ({time.time()-t0:.0f}s)")

    print(f"\n[3/3] Oracle (60 calls)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(solve, solver, p["prompt"], p["optimal_prior"]): p
                  for p in ALL_PROBLEMS}
        for i, f in enumerate(as_completed(futs), 1):
            p = futs[f]; answers["oracle"][p["pid"]] = f.result()
            if i % 20 == 0: print(f"  oracle {i}/60 ({time.time()-t0:.0f}s)")

    # Score
    print(f"\n=== Per-family accuracy ===")
    fams = ["A", "B", "C", "D"]
    cond_scores = {c: {f: [] for f in fams} for c in conditions + ["scheduler", "oracle"]}
    per_problem = {}
    for p in ALL_PROBLEMS:
        per_problem[p["pid"]] = {"family": p["family"], "gold": p["gold"], "prompt": p["prompt"]}
        f = p["family"][0]
        for c in conditions + ["scheduler", "oracle"]:
            ext = extract_answer(answers[c].get(p["pid"], ""))
            sc = score(ext, p["gold"], tol=p.get("tol"))
            cond_scores[c][f].append(sc)
            per_problem[p["pid"]][f"{c}_extracted"] = ext
            per_problem[p["pid"]][f"{c}_score"] = sc

    print(f"\n{'Condition':14s} {'A_dec':>7s} {'B_res':>7s} {'C_est':>7s} {'D_con':>7s} {'overall':>9s}")
    print("-" * 60)
    for c in conditions + ["scheduler", "oracle"]:
        accs = [sum(cond_scores[c][f]) / len(cond_scores[c][f]) for f in fams]
        n_correct = sum(sum(cond_scores[c][f]) for f in fams)
        overall = n_correct / 60
        print(f"{c:14s} {accs[0]:>7.3f} {accs[1]:>7.3f} {accs[2]:>7.3f} {accs[3]:>7.3f} {overall:>9.3f}")

    # Scheduler quality
    correct_picks = 0
    pick_dist = {f: {} for f in fams}
    for p in ALL_PROBLEMS:
        pick = sched_picks[p["pid"]]
        f = p["family"][0]
        pick_dist[f][pick] = pick_dist[f].get(pick, 0) + 1
        if pick == p["optimal_prior"]: correct_picks += 1

    print(f"\n=== Scheduler pick distribution ===")
    for f in fams:
        opt = {"A": "decompose", "B": "restate", "C": "estimate", "D": "constraints"}[f]
        print(f"  Family {f} (optimal={opt}): {pick_dist[f]}")
    print(f"  Total correct picks: {correct_picks}/60 = {correct_picks/60:.1%}")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "n_problems": 60, "n_per_family": 15,
           "scheduler_picks": sched_picks,
           "scheduler_correct_picks": correct_picks,
           "scores": {c: {f"{f}_acc": sum(cond_scores[c][f]) / len(cond_scores[c][f])
                           for f in fams}
                       for c in cond_scores},
           "scores_overall": {c: sum(sum(cond_scores[c][f]) for f in fams) / 60
                                for c in cond_scores},
           "pick_distribution": pick_dist,
           "per_problem": per_problem}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
