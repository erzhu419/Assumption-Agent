"""Consult gpt-5.5 on Exp 71 benchmark design."""
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
BASE = (os.environ.get("RUOLI_BASE_URL") or "https://ruoli.dev").rstrip("/")
if not BASE.endswith("/v1"): BASE += "/v1"
client = OpenAI(base_url=BASE, api_key=KEY)

PROMPT = """You are a senior ML researcher. A colleague has run a sequence of
experiments on LLM-as-judge self-improvement loops and arrived at a deep
methodological problem. Help them design the next experiment.

CONTEXT — what we already know
==============================
v1 paper: a Chinese 'wisdom library' agent uses an LLM-judge (gemini-3-flash)
to gate proposed methodological priors at +10pp pairwise win rate, n=50.
Original gate accepts 3/12 candidates; 3 preregistered fresh-data
replications (n=100, n=100, n=200) all yield 0/12 surviving candidates.

Loop v2 redesign attempted: replace aphorisms with 'triggered cognitive
patches' (trigger + failure-mode label + procedural steps + worked example).

Loop v2 sub-MVP results (just completed):
- 5 problem slices, 50 problems each, hand-curated, all in-distribution
  for gemini-3-flash (textbook Bayesian, classic logic puzzles, multi-step
  arithmetic, constraint puzzles, lateral-thinking word problems).
- 3 conditions: BASE / WITH-CARD / ABLATED-TIGHT (trigger+label only).
- Plus a GENERIC condition: 'This problem may be tricky. Watch for hasty
  conclusions; missing important details.' — same shape as ABLATED-TIGHT
  but with NO problem-class trigger, NO specific failure name.

Headline result on objective accuracy (% correct against gold):
  bayesian: BASE=98%, WITH=98%, ABL_TIGHT=98%, GENERIC=96% (ceiling)
  quantifier: 92% / 92% / 92% / 94% (ceiling)
  multistep: 88% / 88% / 88% / 90% (ceiling)
  constraint: 6% / 12% / 60% / 70%   <-- generic warning rescues from 6 to 70
  counterfactual: 34% / 36% / 80% / 84%  <-- generic warning rescues from 34 to 84

Pairwise judge verdict (3-judge cross-family panel: gemini-3-flash,
claude-haiku-4.5, gpt-5.4-mini):
  ABL_TIGHT vs GENERIC, mean wr (across slices): 0.220, 0.477, 0.414, 0.197, 0.574
  i.e. generic warning is matched or PREFERRED over slice-specific
  trigger+failure-label cards on 4/5 slices.

THE COLLEAGUE'S META-OBSERVATION
================================
The conclusion 'generic warning matches specific wisdom' implicitly says
'all relevant wisdom is already in the model; prompt only elicits.' But
this contradicts everyday research practice — humans face problems where
they need NEW methodology, and that new methodology IS information not
previously available to them. The contradiction's resolution: our
benchmark falls within model competence. We are testing wisdom-injection
on problems the model can ALREADY do (or ALREADY know how to be careful
about) — so 'be careful' prefix elicitation is sufficient and
specific-content contributions are physically undetectable.

THE NEXT EXPERIMENT (Exp 71)
============================
We need to construct a benchmark where:
  C1. Model baseline accuracy (with no wisdom prompt) is LOW (say, <30%)
  C2. Even maximum 'be careful / think harder' generic prompting can't
      rescue accuracy meaningfully (delta < 10pp)
  C3. A specific wisdom intervention WITH an appropriate worked example
      CAN rescue accuracy (delta > 30pp)
  C4. Problems must be objectively scorable (numeric or short-string gold)
  C5. The 'wisdom' must be encodable in <300 words including a worked
      example — this is the realistic bandwidth of a methodological prior

C1+C2 means: problems must be GENUINELY out of distribution in a way
that elicitation alone can't fix. C3 means: there must be a learnable
PROCEDURE that, if conveyed, fixes them. C4+C5 keep evaluation tractable.

Suggest concrete problem domains and example problems that might satisfy
C1-C3. Be specific:
- What is the problem class?
- Why does gemini-3-flash baseline likely fail? (training distribution
  argument)
- Why does generic 'be careful' likely not help? (the missing piece is
  procedural, not attention-allocation)
- What is the wisdom that fixes it? Sketch it.
- What's the worked example?

Constraints:
- Avoid relying on post-cutoff knowledge — that's factual recall, not
  methodology
- Avoid constructed-from-scratch fairy-tale problems — those don't
  generalize
- Lean toward problems that humans need explicit METHODOLOGY for (where
  natural intuition fails specifically), since that's the wisdom-paper's
  spirit

Give 3-5 concrete candidate problem domains, with reasoning. Then pick
the strongest 1-2 and propose 5-10 sample problems each, with gold
answers. ~700 words max.
"""


def main():
    t0 = time.time()
    r = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=4000,
        temperature=0.4,
    )
    out = r.choices[0].message.content
    elapsed = time.time() - t0
    print(f"[{elapsed:.0f}s, {r.usage.total_tokens} tokens]\n")
    print(out)
    Path(__file__).parent.joinpath("GPT55_EXP71_DESIGN.md").write_text(out)
    print("\n\nSaved → paper/GPT55_EXP71_DESIGN.md")


if __name__ == "__main__":
    main()
