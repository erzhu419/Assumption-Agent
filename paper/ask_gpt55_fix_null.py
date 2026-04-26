"""Ask gpt-5.5 how to fix the 0/12 result on the wisdom-library loop."""
import os, re, sys, time
from pathlib import Path

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
_load_api_keys()
from openai import OpenAI

KEY = os.environ.get("RUOLI_GPT_KEY") or os.environ.get("GPT5_API_KEY", "")
BASE_RAW = os.environ.get("RUOLI_BASE_URL") or os.environ.get("GPT5_BASE_URL", "https://ruoli.dev")
BASE = BASE_RAW.rstrip("/")
if not BASE.endswith("/v1"): BASE += "/v1"
client = OpenAI(base_url=BASE, api_key=KEY)

PROMPT = """You are a senior ML researcher. A colleague has built and audited
a closed-loop "wisdom library" agent. The data is below. Question at the end.

LOOP DESIGN:
- Solver: gemini-3-flash with a multi-stage Chinese reasoning scaffold (v20)
- Library: 75 starter wisdoms (aphorisms from Confucius, Drucker, Munger, etc.,
  plus modern ones), each with {aphorism, source, signal, unpacked_for_llm}.
  Examples: "兼听则明，偏信则暗" (listen to all sides for clarity, not partial
  belief for darkness); "解题三步：列前提、定指标、再推论" (solve in 3 steps:
  list premises, fix metric, deduce).
- Orchestrator: 4 modules generate 21 candidate new wisdoms across the cycle;
  upstream filtering keeps 12 reaching the gate.
- Gate: same-family +10pp pairwise A/B test on n=50 held-out Chinese
  open-ended problems, judge=gemini-3-flash. KEEP if wr_ext >= 0.60.

ORIGINAL RESULT: gate accepts 3/12 candidates (W076, W077, W078).

FULL AUDIT FINDS:
1. Cross-family rejudgment (Claude/GPT/Opus): all 3 KEEPs drop to wr 0.40-0.55,
   none clear 0.60.
2. Cached drops are statistically consistent with top-3-of-12 winner's curse on
   noisy n=50 measurements (empirical Bayes: drops at 48-72nd percentile of
   pure regression-to-mean).
3. PREREGISTERED fresh-data replication at n=100, no exemplar mechanism: 0/12.
4. PREREGISTERED fresh-data replication at n=100, WITH original v20 in-pool
   exemplar mechanism: 0/12. Exemplar adds 0.06-0.18 win-rate uniformly.
5. PREREGISTERED fresh-data replication at n=200, WITH exemplar, fresh seed:
   0/12 again. Top three subthreshold: W078 inner=0.590, wcand03=0.580.
6. Hierarchical Bayesian model at n=30 gave W078 posterior 0.763 of passing
   joint rule; FALSIFIED at n>=100.

NOW THE INTERESTING PART — POSITIVE CONTROL (Exp 67/68/69):
- Exp 67: 16 problems gemini gets WRONG baseline. Generic algorithmic prior
  ("list constraints, apply procedure, verify"): Δacc=0 (floor effect). But
  pairwise: NEGATIVE-vs-BASE wr=0.286 (gate rejects reasoning-suppression),
  POS-vs-NEG wr=0.733. Gemini and Claude-haiku judges κ=1.0.
- Exp 68: 15 hand-curated Bayesian counter-intuitive problems. Strong
  task-specific Bayesian template + 2 worked examples (two-children, base-rate
  disease). Δacc=+6.7% (rescued 1 problem; gemini baseline was 14/15 ceiling).
  POS-vs-BASE wr=0.846 (11-2-2). NEG-vs-BASE wr=0.133. POS-vs-NEG wr=1.000.
  Both judges identical splits, κ=1.0.
- Exp 69: same as Exp 68 but solver=gpt-5.4-mini. Baseline 15/15 (ceiling).
  POS-vs-BASE wr=0.933. NEG-vs-BASE wr=0.000.

So:
- Original wisdoms (aphorisms): 0/12 survives audit
- Exp 68/69 strong algorithmic Bayesian template: gate accepts at wr=0.846/0.933,
  cross-family agrees, NOT structurally anti-additive

QUESTION:
Given this exact pattern of evidence, how do we redesign the loop so it
actually produces wisdoms that survive the audit?

Specifically:
(1) Which of these is the bottleneck: the wisdom FORM (aphorism vs algorithmic
    template), the SOLVER (already too strong), the TASK distribution
    (Chinese open-ended is too noisy for n=50), the GATE (same-family is too
    stylistic), or the candidate GENERATOR (orchestrator distills paraphrases
    of existing library)?
(2) Concretely: what is the minimal redesign that would plausibly produce
    wisdom_count > 0 survivors at n=200 cross-family preregistered?
(3) Is there a DIFFERENT kind of "wisdom" — not the aphorism shape, not the
    full algorithmic template — that could plausibly work in this loop?

Be concrete and willing to be wrong. ~600 words max."""


def main():
    t0 = time.time()
    r = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=3000,
        temperature=0.5,
    )
    out = r.choices[0].message.content
    elapsed = time.time() - t0
    print(f"[{elapsed:.0f}s, {r.usage.total_tokens} tokens]\n")
    print(out)
    Path(__file__).parent.joinpath("GPT55_FIX_NULL.md").write_text(out)
    print("\n\nSaved → paper/GPT55_FIX_NULL.md")


if __name__ == "__main__":
    main()
