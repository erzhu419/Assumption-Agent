"""Exp 62 — Harder task families where prior choice swings accuracy.

The core empirical concern with Exp 56 is that gemini-3-flash already
gets 73-100% baseline accuracy on the 4 task families, so the prior
moves wr by only a few percentage points. To demonstrate that the
agent stack genuinely matters, we add 5 HARDER task families designed
so that (a) gemini is NOT at ceiling, and (b) the wrong prior actively
hurts.

Hard families (15 problems each, 75 total):
  E (Bayesian/probabilistic traps):  optimal = constraints
      Disease-test base-rate, Monty Hall variants, two-envelope, etc.
  F (multi-hop arithmetic, 6+ steps): optimal = decompose
      Compound interest with multiple rate changes, multi-stage
      production with rejection, etc.
  G (subtle word-trap problems):     optimal = restate
      Variants of CRT but novel surface forms not in pretraining.
  H (formal logic):                  optimal = constraints
      Knights and knaves, syllogism chains, set membership puzzles.
  I (Fermi with multi-factor):       optimal = estimate
      Hidden parameters need to be estimated as a chain.

Conditions: baseline + 4 priors + scheduler + oracle = 7 × 75 = 525
calls + 75 schedule = 600 calls. ~$5-10.
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
OUT_LOG = AUTO / "exp62_harder_tasks_log.json"

# Family E — Bayesian / probabilistic traps. Optimal = constraints
FAMILY_E = [
    {"prompt": "A disease affects 1% of the population. A test is 99% sensitive (correctly identifies sick) and 99% specific (correctly identifies healthy). Given a positive test, what is the probability the person is actually sick? Give a percentage.", "gold": 50, "tol": 1.2},
    {"prompt": "There are 3 doors. Behind one is a car, behind the other two are goats. You pick door 1. Monty (who knows) opens door 3 to reveal a goat. He offers you to switch to door 2. What is the probability of winning the car if you switch? Give a fraction or decimal.", "gold": 0.667, "tol": 1.05},
    {"prompt": "A box has 10 balls: 3 red, 7 blue. You draw 2 without replacement. What is the probability both are red? Decimal.", "gold": 0.0667, "tol": 1.1},
    {"prompt": "A coin is flipped 10 times and lands heads 8 times. Which is more likely on the 11th flip: heads or tails? Answer 'heads', 'tails', or 'equal'.", "gold": "equal"},
    {"prompt": "Two children. At least one is a boy. What is the probability both are boys? Fraction.", "gold": 0.333, "tol": 1.05},
    {"prompt": "Out of 1000 patients, 990 are healthy and 10 are sick. A test correctly identifies 90% of sick patients and incorrectly flags 5% of healthy ones. Of all positive tests, what fraction are actually sick? Decimal.", "gold": 0.154, "tol": 1.15},
    {"prompt": "A bag has 4 red, 6 blue marbles. You draw with replacement 3 times. P(all three red)? Decimal.", "gold": 0.064, "tol": 1.1},
    {"prompt": "A rare event has probability 1/1000 per day. Probability the event occurs at least once in a year (365 days)? Decimal.", "gold": 0.306, "tol": 1.1},
    {"prompt": "Birthday problem: how many people in a room before P(at least 2 share a birthday) > 0.5? Integer.", "gold": 23, "tol": 1.1},
    {"prompt": "A family has 4 children. P(at least 2 are girls) assuming equal probability? Decimal.", "gold": 0.6875, "tol": 1.05},
    {"prompt": "You roll 2 fair dice. Given the sum is 7, what is the probability one die shows 4? Fraction.", "gold": 0.333, "tol": 1.05},
    {"prompt": "A test has 90% sensitivity and 80% specificity for a condition with 5% prevalence. P(condition | positive test)? Decimal.", "gold": 0.191, "tol": 1.2},
    {"prompt": "Three cards: red/red, red/blue, blue/blue. Pick one card, see one side is red. What is the probability the other side is also red? Fraction.", "gold": 0.667, "tol": 1.05},
    {"prompt": "Two envelopes: one has $X, the other $2X. You pick one. Should you switch? Answer 'yes', 'no', or 'doesn't matter'.", "gold": "doesn't matter"},
    {"prompt": "Lottery has 1 in 1,000,000 odds. You buy 100 tickets. Probability of winning? Decimal.", "gold": 0.0001, "tol": 1.1},
]
for i, p in enumerate(FAMILY_E):
    p["pid"] = f"E_{i:02d}"; p["family"] = "E_probabilistic"; p["optimal_prior"] = "constraints"

# Family F — multi-hop arithmetic, 6+ steps. Optimal = decompose
FAMILY_F = [
    {"prompt": "$1000 invested at 5% compounded annually for 3 years, then 7% for 2 years. Final amount in dollars? Round to integer.", "gold": 1325},
    {"prompt": "A factory has 3 lines. Line A: 100 units/h, 5% defect. Line B: 80/h, 3% defect. Line C: 120/h, 7% defect. After 8 hours total good units?", "gold": 2360},
    {"prompt": "Train A leaves at 9am 60 km/h, train B leaves at 10am 90 km/h same direction. When does B catch A (HH:MM)?", "gold": "12:00"},
    {"prompt": "$5000 loan at 6% annual, paid back in 4 equal annual installments. Total interest paid (rounded to integer dollars)?", "gold": 644},
    {"prompt": "A school: 240 students. 60% boys. 40% of boys play football. 30% of girls play football. Total football players?", "gold": 86},
    {"prompt": "Cylinder: r=5, h=10. Sphere: r=3. Volume of cylinder minus volume of sphere? (use pi=3.14, round to integer)", "gold": 672},
    {"prompt": "30% off, then 20% off the discounted price. What is the total percentage discount from original?", "gold": 44},
    {"prompt": "A car uses 8 L/100km in city, 6 L/100km on highway. Trip: 50km city + 200km highway. Total fuel?", "gold": 16},
    {"prompt": "Tank A holds 100L, drains at 5 L/min. Tank B holds 50L, fills at 3 L/min. After how many minutes is total water in both tanks equal?", "gold": 6.25, "tol": 1.05},
    {"prompt": "Salary of $60,000. After-tax (25%): take-home. Then save 20% of take-home. Annual savings in dollars?", "gold": 9000},
    {"prompt": "5 workers do a job in 12 days. After 4 days, 2 more workers join. Total days from start?", "gold": 9.71, "tol": 1.1},
    {"prompt": "Investment grows 10%/year. Initial $5000. After 5 years, withdraw $2000. Continues at 10%/year for 3 more. Final amount, rounded.", "gold": 8051},
    {"prompt": "12 oz coffee: 80% water. Add 4 oz water. New % water?", "gold": 85},
    {"prompt": "Box: 12 ft × 8 ft × 6 ft. 1 small cube = 8 cu ft. How many small cubes fit? Integer.", "gold": 72},
    {"prompt": "Rectangle: perimeter 40, area 96. Length and width sum?", "gold": 20},
]
for i, p in enumerate(FAMILY_F):
    p["pid"] = f"F_{i:02d}"; p["family"] = "F_multistep"; p["optimal_prior"] = "decompose"

# Family G — subtle word-trap problems. Optimal = restate
FAMILY_G = [
    {"prompt": "I am the brother of the blind man's brother but the blind man is not my brother. How is this possible? One sentence.", "gold": "blind man is myself"},
    {"prompt": "A man builds a house with all four walls facing south. He sees a bear walk by. What color is the bear?", "gold": "white"},
    {"prompt": "There are 30 cows in a field. 28 chickens. How many didn't?", "gold": 10},
    {"prompt": "If you have only one match in a freezing cabin with a candle, an oil lamp, and a wood stove, which do you light first?", "gold": "match"},
    {"prompt": "A doctor and a boy are fishing. The boy is the doctor's son but the doctor is not his father. How is this possible? One word.", "gold": "mother"},
    {"prompt": "A truck driver is going down a one-way street the wrong way and passes 10 police cars without being stopped. Why?", "gold": "walking"},
    {"prompt": "What is so fragile that just saying its name will break it?", "gold": "silence"},
    {"prompt": "A rope ladder hangs from a ship. Each rung is 30 cm apart. There are 10 rungs and the lowest is just above sea level at low tide. Tide rises 1.5 m. How many rungs are now under water?", "gold": 0},
    {"prompt": "I went to the store to buy 6 apples and 4 bananas. On the way back I dropped some. Now I have 4 apples and 3 bananas. How many fruit did I drop?", "gold": 3},
    {"prompt": "How can you drop a raw egg onto a concrete floor without cracking it?", "gold": "concrete is hard to crack"},
    {"prompt": "A man pushes his car to a hotel and tells the owner he is bankrupt. Why?", "gold": "Monopoly"},
    {"prompt": "Forward I am heavy, backward I am not. What am I?", "gold": "ton"},
    {"prompt": "A woman in a wheelchair asked her husband to bring her some books from the upper shelf. He couldn't reach. What did he do?", "gold": "ask her, she's in a wheelchair so probably not the issue"},
    {"prompt": "Two boxers in a championship fight. Round 5, neither has thrown a punch. How can this be?", "gold": "they are women"},
    {"prompt": "A man went outside in the rain with no umbrella and no hat, but not a hair on his head got wet. How?", "gold": "bald"},
]
for i, p in enumerate(FAMILY_G):
    p["pid"] = f"G_{i:02d}"; p["family"] = "G_wordtrap"; p["optimal_prior"] = "restate"

# Family H — formal logic. Optimal = constraints
FAMILY_H = [
    {"prompt": "Knights always tell truth, knaves always lie. A says 'B is a knight'. B says 'A is a knave'. What is each?", "gold": "A knave, B knave"},
    {"prompt": "All cats are mammals. Some mammals are pets. Therefore all cats are pets. Valid or invalid?", "gold": "invalid"},
    {"prompt": "Three friends - Alice (A), Bob (B), Carol (C) - sit in a row. A is not next to B. C is to the right of A. Position from left?", "gold": "A C B"},
    {"prompt": "A says 'B and C lie'. B says 'A lies'. C says 'A tells truth'. Who tells truth (assuming exactly one tells truth)?", "gold": "B"},
    {"prompt": "If P then Q. Q is true. Can we conclude P? Yes/No.", "gold": "no"},
    {"prompt": "Box A says: 'B contains gold'. Box B says: 'Both boxes have gold'. Exactly one box has gold and exactly one box's claim is true. Which has gold?", "gold": "A"},
    {"prompt": "Five people: 3 always lie, 2 always tell truth. A says 'B lies'. B says 'C tells truth'. C says 'D lies'. D says 'E tells truth'. E says 'A tells truth'. How many liars among A,B?", "gold": 1},
    {"prompt": "If it rains, the ground is wet. The ground is dry. What can we conclude about rain?", "gold": "it did not rain"},
    {"prompt": "Some birds fly. Penguins are birds. Therefore penguins fly. Valid or invalid?", "gold": "invalid"},
    {"prompt": "All A are B. No B are C. Therefore no A are C. Valid or invalid?", "gold": "valid"},
    {"prompt": "Three sisters: youngest is the best painter. Middle is best singer. Oldest is best dancer. Beth is older than the painter. Carla is younger than the dancer. Beth is not the singer. What is Beth best at?", "gold": "dancer"},
    {"prompt": "A says: 'I am a knave or B is a knight'. What can A be?", "gold": "knight"},
    {"prompt": "Every student passed at least one test. No student passed all tests. There are 4 tests. Min number of students?", "gold": 1},
    {"prompt": "If A → B, B → C, ¬C. Conclude:", "gold": "not A"},
    {"prompt": "5 hats: 3 red, 2 blue. 3 prisoners in a line. Prisoner 1 (back) sees 2 ahead. P2 sees P3. P3 sees nothing. P1 doesn't know. P2 doesn't know. What color is P3 wearing?", "gold": "red"},
]
for i, p in enumerate(FAMILY_H):
    p["pid"] = f"H_{i:02d}"; p["family"] = "H_logic"; p["optimal_prior"] = "constraints"

# Family I — Fermi with multi-factor estimation
FAMILY_I = [
    {"prompt": "Estimate total annual energy in MWh consumed by all data centers globally. Single integer.", "gold": 200000000, "tol": 10},
    {"prompt": "Estimate total daily volume in liters of all soft drinks consumed worldwide. Single integer.", "gold": 1000000000, "tol": 10},
    {"prompt": "How many words does ChatGPT generate per day worldwide? Single integer.", "gold": 100000000000, "tol": 10},
    {"prompt": "Estimate lifetime kWh consumption of one US household over 50 years. Single integer.", "gold": 500000, "tol": 5},
    {"prompt": "Total number of phone calls made in the US per year. Single integer.", "gold": 200000000000, "tol": 10},
    {"prompt": "Number of unique books currently in print worldwide. Single integer.", "gold": 130000000, "tol": 5},
    {"prompt": "Estimate daily steps of all people in NYC combined. Single integer.", "gold": 70000000000, "tol": 10},
    {"prompt": "Total weight of all the world's ants in kg. Single integer.", "gold": 12000000000, "tol": 10},
    {"prompt": "Total cars manufactured globally per year. Single integer.", "gold": 90000000, "tol": 5},
    {"prompt": "Number of bytes in all the world's photos. Single integer.", "gold": 10000000000000000000, "tol": 100},
    {"prompt": "Total volume of all the world's oceans in cubic kilometers. Single integer.", "gold": 1300000000, "tol": 5},
    {"prompt": "Total number of trees cut down annually globally. Single integer.", "gold": 15000000000, "tol": 10},
    {"prompt": "Total annual CO2 emissions in metric tons globally. Single integer.", "gold": 36000000000, "tol": 5},
    {"prompt": "Number of tweets per second worldwide. Single integer.", "gold": 6000, "tol": 5},
    {"prompt": "Total mass of all stars in the Milky Way in solar masses. Single integer.", "gold": 1500000000000, "tol": 5},
]
for i, p in enumerate(FAMILY_I):
    p["pid"] = f"I_{i:02d}"; p["family"] = "I_fermi"; p["optimal_prior"] = "estimate"

ALL_PROBLEMS = FAMILY_E + FAMILY_F + FAMILY_G + FAMILY_H + FAMILY_I
assert len(ALL_PROBLEMS) == 75

PRIORS = {
    "decompose": "Before answering, decompose into atomic substeps.",
    "restate": "Before answering, RE-READ the question carefully; check if the obvious interpretation is a trap.",
    "estimate": "Before answering, give an order-of-magnitude estimate; sanity-check final answer.",
    "constraints": "Before answering, enumerate explicit constraints; satisfy all simultaneously.",
    "none": "",
}

SOLVE = """## Problem
{problem}

## Approach hint
{approach}

## Output
Reason step by step in 1-3 sentences, then on the LAST LINE write exactly:
ANSWER: <your final answer>
"""

SCHED = """You are a strategy selector. Given a problem, choose ONE prior from:
{{decompose, restate, estimate, constraints, none}}.

- decompose: best for multi-step arithmetic / counting where the right answer requires explicit subcomputations.
- restate: best for trick questions where the obvious computational interpretation is a trap.
- estimate: best for Fermi/order-of-magnitude problems with no exact data.
- constraints: best for logic puzzles or probabilistic traps where the answer must satisfy explicit constraints simultaneously.
- none: only if no prior is clearly more applicable than baseline.

## Problem
{problem}

## Output (JSON only)
{{"choice": "decompose"|"restate"|"estimate"|"constraints"|"none", "reason": "1 sentence"}}
"""

def solve(client, problem, prior):
    p = PRIORS[prior]
    appr = f"Use this strategy: {p}" if p else "Use any approach you think appropriate."
    try:
        r = client.generate(SOLVE.format(problem=problem, approach=appr), max_tokens=600, temperature=0.0)
        return r["text"].strip()
    except Exception as e: return f"[err: {e}]"

def schedule(client, problem):
    try:
        r = client.generate(SCHED.format(problem=problem), max_tokens=200, temperature=0.0)
        m = re.search(r'"choice"\s*:\s*"(decompose|restate|estimate|constraints|none)"', r["text"])
        if m: return m.group(1)
    except Exception: pass
    return "none"

def extract(text):
    m = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    return (m.group(1) if m else text[-200:]).strip()

def score_n(ext, gold, tol=None):
    s = ext.replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")
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
    return 1 if abs(ev - gv) < 0.05 else 0

def score_t(ext, gold):
    e = ext.lower().strip().rstrip(".")
    g = str(gold).lower().strip()
    if g in e: return 1
    g_w = set(re.findall(r'\w+', g))
    mean = {w for w in g_w if len(w) >= 3}
    if not mean: return 0
    e_w = set(re.findall(r'\w+', e))
    return 1 if len(mean & e_w) / len(mean) >= 0.5 else 0

def sc(ext, gold, tol=None):
    if isinstance(gold, (int, float)): return score_n(ext, gold, tol)
    return score_t(ext, gold)


def main():
    print(f"=== Exp 62: harder task families (5 families × 15 = 75 problems) ===")
    solver = cheap("gemini")
    sched_c = cheap("gemini")
    conds = ["baseline", "decompose", "restate", "estimate", "constraints"]
    answers = {c: {} for c in conds}; answers["scheduler"] = {}; answers["oracle"] = {}
    sched_picks = {}

    print(f"\n[1/3] Fixed conds: 5 × 75 = 375 calls...")
    tasks = [(c, p) for c in conds for p in ALL_PROBLEMS]
    def run_fix(c, p):
        prior = "none" if c == "baseline" else c
        return c, p["pid"], solve(solver, p["prompt"], prior)
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(run_fix, c, p) for c, p in tasks]
        for f in as_completed(futs):
            c, pid, ans = f.result(); answers[c][pid] = ans; done += 1
            if done % 75 == 0: print(f"  fixed {done}/375 ({time.time()-t0:.0f}s)")

    print(f"\n[2/3] Schedule + scheduler-solve (75 + 75 calls)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(schedule, sched_c, p["prompt"]): p for p in ALL_PROBLEMS}
        for f in as_completed(futs):
            p = futs[f]; sched_picks[p["pid"]] = f.result()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(solve, solver, p["prompt"], sched_picks[p["pid"]]): p for p in ALL_PROBLEMS}
        for f in as_completed(futs):
            p = futs[f]; answers["scheduler"][p["pid"]] = f.result()

    print(f"\n[3/3] Oracle (75 calls)...")
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(solve, solver, p["prompt"], p["optimal_prior"]): p for p in ALL_PROBLEMS}
        for f in as_completed(futs):
            p = futs[f]; answers["oracle"][p["pid"]] = f.result()

    # Score
    fams = ["E", "F", "G", "H", "I"]
    cs = {c: {f: [] for f in fams} for c in conds + ["scheduler", "oracle"]}
    per_problem = {}
    for p in ALL_PROBLEMS:
        per_problem[p["pid"]] = {"family": p["family"], "gold": p["gold"]}
        f = p["family"][0]
        for c in conds + ["scheduler", "oracle"]:
            ext = extract(answers[c].get(p["pid"], ""))
            s = sc(ext, p["gold"], p.get("tol"))
            cs[c][f].append(s)
            per_problem[p["pid"]][f"{c}_score"] = s

    print(f"\n=== Per-family accuracy on harder tasks ===")
    print(f"{'Cond':14s} {'E_prob':>7s} {'F_mult':>7s} {'G_word':>7s} {'H_log':>7s} {'I_fer':>7s} {'overall':>9s}")
    print("-" * 70)
    summary = {}
    for c in conds + ["scheduler", "oracle"]:
        accs = [sum(cs[c][f]) / len(cs[c][f]) for f in fams]
        n_correct = sum(sum(cs[c][f]) for f in fams)
        overall = n_correct / 75
        print(f"{c:14s} " + " ".join(f"{a:>7.3f}" for a in accs) + f" {overall:>9.3f}")
        summary[c] = {"per_family": {f: accs[i] for i, f in enumerate(fams)},
                       "overall": overall, "n_correct": n_correct}

    correct_picks = sum(1 for p in ALL_PROBLEMS if sched_picks[p["pid"]] == p["optimal_prior"])
    print(f"\nScheduler correct picks: {correct_picks}/75 = {correct_picks/75:.1%}")
    print(f"\n=== Scheduler vs best fixed: {summary['scheduler']['overall']:.3f} vs "
          f"{max(summary[c]['overall'] for c in conds[1:]):.3f}")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "n_problems": 75,
           "scheduler_picks": sched_picks,
           "scheduler_correct_picks": correct_picks,
           "summary": summary,
           "per_problem": per_problem}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
