"""Exp 71 — OOD-procedural benchmark: does specific wisdom help when the
procedure is GENUINELY novel to the model?

This is the experiment the user's meta-observation pointed to. Exp 70a-d
showed that on textbook problems (in-distribution for gemini-3-flash),
GENERIC 'be careful' prefixes match or beat slice-specific wisdom. The
parsimonious reading was 'specificity doesn't help' — but this is a
category error: we tested wisdom-injection on problems the model already
knew, where elicitation is sufficient. The right test is on problems
where the model GENUINELY lacks a needed procedure.

We use two procedural-bottleneck domains that have specific algorithms
the model is unlikely to apply correctly without explicit instruction:

  HAMMING(7,4) SYNDROME DECODING
    A 7-bit codeword may have one bit flipped. Compute parity-check
    syndrome via XORs over specific position subsets, identify error,
    flip, extract data bits at positions 3,5,6,7. Without explicit
    algorithm, the model gets bit-position conventions wrong.

  SPRAGUE-GRUNDY SUBTRACTION GAME {1,3,4}
    Three piles; each turn remove 1, 3, or 4 from one pile; loser is
    the one who can't move. Compute Grundy values via mex recurrence,
    XOR-compose across piles. Determine FIRST/SECOND winner. Without
    the recurrence, model treats it like Nim (wrong) or simulates
    incorrectly.

Four conditions (per problem):
  BASE          : no card.
  GENERIC       : 'This problem may be tricky.' (matched in shape to
                   the cards but no class-specificity, no procedure).
  SPECIFIC_LITE : trigger + failure-label only (no procedure, no example).
  SPECIFIC_FULL : trigger + failure-label + step-by-step procedure +
                   worked example.

If wisdom-content effects exist when the procedure is genuinely needed:
  BASE & GENERIC & SPECIFIC_LITE all stay low (<30%)
  SPECIFIC_FULL jumps up (>60%)
  Δacc(SPECIFIC_FULL - GENERIC) >> 0
  Pairwise wr(SPECIFIC_FULL vs GENERIC) >> 0.6 from cross-family judges.

If wisdom-content STILL doesn't show up here, we have to admit
prompt-injection wisdom is fundamentally insufficient — the wisdom
must be transmitted at a different substrate (fine-tune, retrieval
policy, search control).

Cost: 4 conditions x 24 problems = 96 solves + judging
~3 pair types x 3 judges x 24 = 216 judgments. ~$2, ~10 min.
"""
import json, os, random, re, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase one" / "scripts" / "validation"))


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


PARALLEL = 8
AUTO = PROJECT / "phase six" / "autonomous"
OUT_LOG = AUTO / "exp71_ood_benchmark_log.json"


# ============================================================
# Problem set: HAMMING(7,4) decoding
# ============================================================
# Convention: positions 1..7 left-to-right; parity bits at 1,2,4;
# even parity; data bits at positions 3,5,6,7. At most one bit flipped.
# Gold = the corrected 4-bit data string.
HAMMING_PROBLEMS = [
    # Each: (received_codeword, correct_data_bits)
    ("0110111", "1011"),   # gpt-5.5 sample 1
    ("0000101", "0101"),   # gpt-5.5 sample 2
    ("0010111", "1110"),   # gpt-5.5 sample 3
    ("0001000", "0000"),   # gpt-5.5 sample 4
    ("0010101", "1101"),   # gpt-5.5 sample 5
    ("0001101", "0111"),   # gpt-5.5 sample 6
    ("1110000", "1000"),   # gpt-5.5 sample 7
    ("1101010", "0010"),   # gpt-5.5 sample 8
    # 4 more, programmatically encoded + verified:
    ("0000000", "0000"),   # all zero, no error
    ("1111111", "1111"),   # all one, no error
    ("1010101", "1101"),   # data=1101, no error (encoded directly)
    ("1001011", "0011"),   # data=0011, error at pos 4
]
HAMMING_TASK_DESCRIPTION = (
    "Hamming(7,4) decoding. Convention: 7 positions numbered 1-7 left-to-right; "
    "parity bits at positions 1, 2, 4; data bits at positions 3, 5, 6, 7; "
    "even parity. The received codeword may have one bit flipped. Decode "
    "to recover the 4 original data bits."
)

# ============================================================
# Problem set: SPRAGUE-GRUNDY {1,3,4} subtraction game
# ============================================================
# 3 independent piles; on a turn, remove exactly 1, 3, or 4 from one pile.
# Loser cannot move (normal play). Output: FIRST or SECOND.
# Grundy values g(n) for subtraction set {1,3,4} are periodic mod 7:
#   n mod 7: 0 1 2 3 4 5 6
#   g(n):    0 1 0 1 2 3 2
# A position is losing (SECOND wins) iff XOR of g(pile_i) = 0.
SG_PROBLEMS = [
    # gpt-5.5 samples (verified):
    ((17, 24, 35), "SECOND"),  # 17%7=3 g=1; 24%7=3 g=1; 35%7=0 g=0; XOR=0
    ((19, 26, 28), "SECOND"),  # g=3,2,0; XOR=1??? let me recompute
    ((20, 25, 31), "FIRST"),   # 20%7=6 g=2; 25%7=4 g=2; 31%7=3 g=1; XOR=1
    ((18, 23, 30), "FIRST"),   # g=2,1,2; XOR=1
    ((15, 16, 21), "FIRST"),   # g=1,0,0; XOR=1
    ((27, 34, 41), "FIRST"),   # g=3,2,1; XOR=0??? recompute
    ((10, 24, 28), "SECOND"),  # g=1,1,0; XOR=0
    ((12, 33, 47), "FIRST"),   # g=3,3,1; XOR=1
    # New ones verified:
    ((7, 14, 21), "SECOND"),   # all g=0; XOR=0
    ((8, 9, 16), "FIRST"),     # 8%7=1 g=1; 9%7=2 g=0; 16%7=2 g=0; XOR=1
    ((11, 18, 25), "FIRST"),   # g=2,2,2; XOR=2
    ((13, 20, 27), "FIRST"),   # g=2(13%7=6), 2, 3(27%7=6); XOR = 2^2^2 = 2
]
SG_TASK_DESCRIPTION = (
    "Sprague-Grundy / impartial-game theory. Three independent piles; a "
    "turn consists of removing exactly 1, 3, or 4 counters from one pile. "
    "Normal play: a player who cannot move loses. Determine whether the "
    "FIRST or SECOND player has a winning strategy from the given position."
)

# Verify SG gold answers programmatically before launching
def _verify_sg_gold():
    g_table = {0: 0, 1: 1, 2: 0, 3: 1, 4: 2, 5: 3, 6: 2}
    bad = []
    for piles, gold in SG_PROBLEMS:
        gs = [g_table[n % 7] for n in piles]
        xor = 0
        for v in gs: xor ^= v
        expected = "SECOND" if xor == 0 else "FIRST"
        if expected != gold:
            bad.append((piles, gold, expected, gs, xor))
    return bad

# Verify Hamming gold answers
def _verify_hamming_gold():
    bad = []
    for received, gold in HAMMING_PROBLEMS:
        bits = [int(c) for c in received]
        # 1-indexed
        s1 = bits[0] ^ bits[2] ^ bits[4] ^ bits[6]   # positions 1,3,5,7
        s2 = bits[1] ^ bits[2] ^ bits[5] ^ bits[6]   # positions 2,3,6,7
        s4 = bits[3] ^ bits[4] ^ bits[5] ^ bits[6]   # positions 4,5,6,7
        synd = s1 + 2*s2 + 4*s4
        if synd != 0 and 1 <= synd <= 7:
            bits[synd-1] ^= 1
        data = "".join(str(bits[i-1]) for i in (3, 5, 6, 7))
        if data != gold:
            bad.append((received, gold, data, (s1, s2, s4), synd))
    return bad


# ============================================================
# Conditions
# ============================================================
GENERIC = (
    "## METHODOLOGICAL HINT: careful_reasoning\n"
    "Trigger: This problem may be tricky.\n"
    "Failure to avoid: Hasty conclusions; missing important details."
)

# Hamming-specific
HAMMING_LITE = (
    "## METHODOLOGICAL HINT: hamming_decode\n"
    "Trigger: A 7-bit string is a (potentially corrupted) Hamming(7,4) codeword. "
    "You must decode the 4 data bits.\n"
    "Failure to avoid: Confusing parity-bit positions (1,2,4) with data-bit "
    "positions (3,5,6,7); computing parity over wrong subsets."
)
HAMMING_FULL = HAMMING_LITE + """

Procedure:
1. Number positions 1 through 7 left-to-right.
2. Compute three parity-check syndrome bits using even parity (XOR):
   s1 = XOR over positions {1, 3, 5, 7}
   s2 = XOR over positions {2, 3, 6, 7}
   s4 = XOR over positions {4, 5, 6, 7}
3. Compute syndrome value s = s1 + 2*s2 + 4*s4 (in {0,...,7}).
4. If s = 0: codeword is uncorrupted. Otherwise: flip the bit at position s.
5. Read the data bits from positions 3, 5, 6, 7 (in that order).
6. Output as a 4-bit string.

Worked example: Received '0110111'.
  Positions: 1 2 3 4 5 6 7
  Bits:      0 1 1 0 1 1 1
  s1 = 0 XOR 1 XOR 1 XOR 1 = 1
  s2 = 1 XOR 1 XOR 1 XOR 1 = 0
  s4 = 0 XOR 1 XOR 1 XOR 1 = 1
  Syndrome = 1 + 0 + 4 = 5, so flip position 5.
  Corrected:  0 1 1 0 0 1 1
  Data at positions 3,5,6,7: 1, 0, 1, 1.
  Answer: 1011"""

# Sprague-Grundy specific
SG_LITE = (
    "## METHODOLOGICAL HINT: sprague_grundy_subtraction\n"
    "Trigger: An impartial combinatorial game with multiple independent piles "
    "where a move removes exactly 1, 3, or 4 counters from a single pile, "
    "and the player who cannot move loses (normal play).\n"
    "Failure to avoid: Treating this like Nim (where moves can take any "
    "number); intuitive guessing without computing Grundy values."
)
SG_FULL = SG_LITE + """

Procedure:
1. The game value of each pile is its Grundy value (nimber).
2. For subtraction set {1,3,4}, Grundy values are periodic with period 7:
     n mod 7:  0  1  2  3  4  5  6
     g(n):     0  1  0  1  2  3  2
3. For each pile of size n, look up g(n) using n mod 7.
4. Compute the XOR of all Grundy values across the piles.
5. If the XOR equals 0, the position is losing for the player to move:
   answer SECOND. Otherwise the position is winning for the player to
   move: answer FIRST.

Worked example: Piles 10, 24, 28.
  10 mod 7 = 3 -> g = 1
  24 mod 7 = 3 -> g = 1
  28 mod 7 = 0 -> g = 0
  XOR = 1 XOR 1 XOR 0 = 0.
  Answer: SECOND"""


SOLVE_PROMPT = """{card_section}## Task description
{task_description}

## Problem
{problem}

## Output
Reason step by step concisely (3-8 sentences), then on the LAST LINE write exactly:
ANSWER: <answer>
"""


def make_problem_text(domain, payload):
    if domain == "hamming":
        received = payload
        return f"Received codeword: '{received}'. Decode the 4 data bits and output them as a 4-character binary string."
    else:  # sg
        piles = payload
        return f"Piles: {piles[0]}, {piles[1]}, {piles[2]}. Output 'FIRST' or 'SECOND'."


def solve(client, problem_text, task_description, card_render):
    card_section = (card_render + "\n\n") if card_render else ""
    try:
        r = client.generate(
            SOLVE_PROMPT.format(card_section=card_section,
                                  task_description=task_description,
                                  problem=problem_text),
            max_tokens=1100, temperature=0.0,
        )
        return r["text"].strip()
    except Exception as e:
        return f"[err: {e}]"


def extract_answer(text):
    m = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    return (m.group(1) if m else text[-200:]).strip()


def score_hamming(answer_text, gold):
    raw = extract_answer(answer_text).strip().strip("'\"`")
    # extract first 4-bit substring
    m = re.search(r'[01]{4,}', raw)
    if m:
        candidate = m.group()[:4]
        return 1 if candidate == gold else 0
    return 0


def score_sg(answer_text, gold):
    raw = extract_answer(answer_text).upper()
    if "FIRST" in raw and "SECOND" not in raw: return 1 if gold == "FIRST" else 0
    if "SECOND" in raw and "FIRST" not in raw: return 1 if gold == "SECOND" else 0
    return 0


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


def run_pairwise_panel(judges, ans_dict_a, ans_dict_b, label_a, label_b, problems_with_text):
    """problems_with_text: list of (pid, problem_text)."""
    results = {}
    for jname, jclient in judges:
        wins_a = wins_b = ties = 0
        per_pid = {}
        def one(pid_text):
            pid, ptext = pid_text
            a = ans_dict_a.get(pid, ""); b = ans_dict_b.get(pid, "")
            if not a or not b: return pid, "tie"
            seed = hash(pid + label_a + label_b + jname + "ood") % (2**32)
            return pid, judge_with_side_randomize(jclient, ptext, a, b, label_a, label_b, seed)
        with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
            futs = [ex.submit(one, pt) for pt in problems_with_text]
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
    print(f"=== Exp 71: OOD-procedural benchmark ===", flush=True)

    bad_h = _verify_hamming_gold()
    bad_sg = _verify_sg_gold()
    if bad_h:
        print(f"  ! Hamming gold check FAILED for {len(bad_h)} problems:", flush=True)
        for b in bad_h: print(f"      {b}", flush=True)
    else:
        print(f"  ✓ Hamming gold answers verified ({len(HAMMING_PROBLEMS)} problems)", flush=True)
    if bad_sg:
        print(f"  ! Sprague-Grundy gold check FAILED for {len(bad_sg)} problems:", flush=True)
        for b in bad_sg: print(f"      {b}", flush=True)
    else:
        print(f"  ✓ Sprague-Grundy gold answers verified ({len(SG_PROBLEMS)} problems)", flush=True)
    if bad_h or bad_sg:
        print(f"  Aborting due to gold mismatches. Fix and retry.", flush=True)
        return

    # Construct unified problem list
    all_problems = []
    for i, (received, gold) in enumerate(HAMMING_PROBLEMS):
        all_problems.append({
            "pid": f"H_{i:02d}", "domain": "hamming",
            "payload": received, "gold": gold,
            "prompt": make_problem_text("hamming", received),
            "task_description": HAMMING_TASK_DESCRIPTION,
        })
    for i, (piles, gold) in enumerate(SG_PROBLEMS):
        all_problems.append({
            "pid": f"S_{i:02d}", "domain": "sg",
            "payload": piles, "gold": gold,
            "prompt": make_problem_text("sg", piles),
            "task_description": SG_TASK_DESCRIPTION,
        })
    n_total = len(all_problems)
    print(f"  Total problems: {n_total} ({len(HAMMING_PROBLEMS)} Hamming + {len(SG_PROBLEMS)} SG)", flush=True)

    solver = cheap("gemini")
    judges = [
        ("gemini", cheap("gemini")),
        ("claude_haiku", cheap("claude_haiku")),
        ("gpt_mini", cheap("gpt_mini")),
    ]

    # ---- Stage 1: solve all 4 conditions x 24 problems = 96 ---------
    print(f"\n[1/3] Solving 4 conditions x {n_total} = {4*n_total} solves...", flush=True)
    t0 = time.time()
    answers = {c: {} for c in ["BASE", "GENERIC", "SPECIFIC_LITE", "SPECIFIC_FULL"]}

    def get_card(cond, domain):
        if cond == "BASE": return ""
        if cond == "GENERIC": return GENERIC
        if cond == "SPECIFIC_LITE":
            return HAMMING_LITE if domain == "hamming" else SG_LITE
        if cond == "SPECIFIC_FULL":
            return HAMMING_FULL if domain == "hamming" else SG_FULL

    def solve_task(cond, p):
        return cond, p["pid"], solve(solver, p["prompt"], p["task_description"],
                                          get_card(cond, p["domain"]))

    tasks = [(c, p) for c in answers for p in all_problems]
    done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(solve_task, c, p) for c, p in tasks]
        for f in as_completed(futs):
            c, pid, ans = f.result()
            answers[c][pid] = ans
            done += 1
            if done % 24 == 0:
                print(f"  {done}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  All solves done in {time.time()-t0:.0f}s", flush=True)

    # ---- Stage 2: objective grading ---------------------------------
    print(f"\n[2/3] Objective grading vs gold...", flush=True)
    obj = {}
    per_pid_correct = {c: {} for c in answers}
    for cond in answers:
        h_correct = h_total = s_correct = s_total = 0
        for p in all_problems:
            ans = answers[cond].get(p["pid"], "")
            if p["domain"] == "hamming":
                sc = score_hamming(ans, p["gold"])
                h_correct += sc; h_total += 1
            else:
                sc = score_sg(ans, p["gold"])
                s_correct += sc; s_total += 1
            per_pid_correct[cond][p["pid"]] = sc
        obj[cond] = {
            "hamming_acc": h_correct/h_total,
            "sg_acc": s_correct/s_total,
            "overall_acc": (h_correct+s_correct)/(h_total+s_total),
        }
        print(f"  {cond:14s}: Hamming={obj[cond]['hamming_acc']:.1%} "
              f"SG={obj[cond]['sg_acc']:.1%} | "
              f"Overall={obj[cond]['overall_acc']:.1%}", flush=True)

    delta_full_minus_generic = obj["SPECIFIC_FULL"]["overall_acc"] - obj["GENERIC"]["overall_acc"]
    delta_lite_minus_generic = obj["SPECIFIC_LITE"]["overall_acc"] - obj["GENERIC"]["overall_acc"]
    print(f"\n  Δacc(SPECIFIC_FULL - GENERIC) = {delta_full_minus_generic:+.1%}", flush=True)
    print(f"  Δacc(SPECIFIC_LITE - GENERIC) = {delta_lite_minus_generic:+.1%}", flush=True)

    # ---- Stage 3: pairwise (3 pair types x 3 judges) -----------------
    print(f"\n[3/3] Pairwise judging (3 pair types x 3 judges x {n_total} = "
          f"{3*3*n_total} judgments)...", flush=True)
    t1 = time.time()
    pairwise = {}
    pid_text = [(p["pid"], p["prompt"]) for p in all_problems]
    pair_specs = [
        ("SPECIFIC_FULL", "GENERIC"),
        ("SPECIFIC_FULL", "SPECIFIC_LITE"),
        ("SPECIFIC_LITE", "GENERIC"),
    ]
    for la, lb in pair_specs:
        pname = f"{la}_vs_{lb}"
        r = run_pairwise_panel(judges, answers[la], answers[lb], la, lb, pid_text)
        pairwise[pname] = r
        print(f"  {pname:32s}: mean wr={r['mean_wr']:.3f} "
              f"(min={r['min_wr']:.3f}) gem={r['gemini']['wr_a']:.3f} "
              f"hai={r['claude_haiku']['wr_a']:.3f} gpt={r['gpt_mini']['wr_a']:.3f}", flush=True)
    print(f"  Judging done in {time.time()-t1:.0f}s", flush=True)

    # ---- Verdict -----------------------------------------------------
    wr_full_vs_gen = pairwise["SPECIFIC_FULL_vs_GENERIC"]["mean_wr"]
    wr_lite_vs_gen = pairwise["SPECIFIC_LITE_vs_GENERIC"]["mean_wr"]
    wr_full_vs_lite = pairwise["SPECIFIC_FULL_vs_SPECIFIC_LITE"]["mean_wr"]

    print(f"\n=== VERDICT ===", flush=True)
    if delta_full_minus_generic >= 0.30 and wr_full_vs_gen >= 0.60:
        v = "WISDOM-CONTENT VALIDATED: specific procedure with worked example produces large objective gain over generic warning"
    elif delta_full_minus_generic >= 0.15 and wr_full_vs_gen >= 0.55:
        v = "WISDOM-CONTENT TENTATIVELY VALIDATED"
    elif delta_full_minus_generic < 0.05 and 0.45 <= wr_full_vs_gen <= 0.55:
        v = "NO WISDOM-CONTENT EFFECT EVEN ON OOD-PROCEDURAL TASKS — prompt-injection wisdom paradigm broadly invalidated"
    else:
        v = f"MIXED (Δacc={delta_full_minus_generic:+.1%}, wr_full_vs_gen={wr_full_vs_gen:.3f})"

    print(f"  Δacc(FULL - GENERIC) = {delta_full_minus_generic:+.1%}", flush=True)
    print(f"  Δacc(LITE - GENERIC) = {delta_lite_minus_generic:+.1%}", flush=True)
    print(f"  wr(FULL vs GENERIC)  = {wr_full_vs_gen:.3f}", flush=True)
    print(f"  wr(LITE vs GENERIC)  = {wr_lite_vs_gen:.3f}", flush=True)
    print(f"  wr(FULL vs LITE)     = {wr_full_vs_lite:.3f}", flush=True)
    print(f"  Verdict: {v}", flush=True)

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "domains": ["hamming(7,4)", "sprague-grundy {1,3,4}"],
        "n_problems": n_total,
        "objective_accuracy": obj,
        "delta_acc_full_minus_generic": delta_full_minus_generic,
        "delta_acc_lite_minus_generic": delta_lite_minus_generic,
        "pairwise": pairwise,
        "verdict": v,
        "answers": answers,
        "per_pid_correct": per_pid_correct,
    }
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}", flush=True)


if __name__ == "__main__":
    main()
