"""Consult gpt-5.5: can dialogue-Claude's self-hypothesize-and-validate
pattern be replicated for OTHER problems?"""
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

PROMPT = """You are a senior ML researcher. We have a deep methodological
question that emerged from an empirical research project. Help us think
through it carefully.

CONTEXT — confirmed empirical event
=====================================
On 2026-04-24, in a Claude dialogue session, Claude (opus 4.6) was
working on the v1 'wisdom-library self-improvement loop' paper. After
16 prior audit experiments all returned null (0/12 KEEP candidates
survived), Claude in dialogue:

1. Independently proposed a redesigned gate ('trigger-conditioned gate'):
   decompose pair-wr into util_fire (utility when wisdom triggers),
   cite_fire (citation rate when triggered), and util_abs.
2. Implemented it (Exp 17), running on the same 12 candidates.
3. Got 4/12 PASS — the first positive signal in 17 experiments.
4. Wrote up the result in the paper.

The user only 'approved' Claude's proposals; the user was clear they
did not contribute the design idea. So: Claude in dialogue performed
SOMETHING resembling autonomous self-hypothesizing.

CONTEXT — important caveat
==========================
Exp 33, a separate prospective rerun on 9 fresh candidates with the
same gate, returned 0/9 PASS. Combined: 4/21 ≈ 19%, Wilson 95% CI
[0.08, 0.38], which fails to clear ~30% chance baseline (one-sided
binomial p=0.14). The 4/12 was a tuning-set over-fit. Claude did NOT
catch this in dialogue; only the separate prospective experiment did.

So strictly: dialogue-Claude self-PROPOSED a new method, partially
self-VALIDATED it (the 4/12 looked positive in tuning), but failed to
self-CATCH its own over-fit. The prospective falsification came from
the experimental protocol, not from Claude's reflective process.

QUESTION
========
The user's research question: 'If dialogue-Claude can do this once
(propose + partially validate a methodology under in-loop human
approval), can the SAME pattern be replicated for OTHER stuck
problems? What's the structural recipe?'

Specifically:
1. Is the 'partial self-hypothesizing capability' that we observed
   real, or is it artifact of cherry-picking? (Across the v1 paper +
   Exp 70 series + Exp 71 = ~30 distinct experimental designs, how
   often does Claude in dialogue produce a genuinely novel
   contribution vs just executing user instructions?)
2. The over-fit failure: dialogue-Claude proposed a methodology that
   over-fit to the tuning data and could not see this. Is that a
   universal failure mode, or specific to wisdom-loop debugging? In
   other research domains, do we expect dialogue-Claude to over-fit
   in the same way?
3. The replication recipe: if we wanted to replicate the
   'dialogue-Claude proposes a real new method' phenomenon for OTHER
   stuck problems, what minimal structural ingredients do we need?
   Is it just (LLM in dialogue + human approval + execution loop), or
   are there subtler requirements?
4. Most importantly for paper v2: does the existence of even ONE
   successful instance of dialogue-Claude self-hypothesizing change
   the paper's central thesis? The thesis was 'current LLM agents
   cannot self-hypothesize'. If dialogue-Claude can, even partially
   and even with over-fit, then the thesis is wrong as stated. The
   correct thesis might be: 'current AUTONOMOUS LLM agents cannot;
   dialogue-mediated LLM CAN, with two failure modes (over-fit and
   non-encodable structure).'

Be honest. If the empirical instance is too thin to support
generalization, say so. If you think the over-fit is universal, say
so. If you think the replication recipe requires something we
haven't identified, propose what.

~700 words max.
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
    Path(__file__).parent.joinpath("GPT55_REPLICATION.md").write_text(out)
    print("\n\nSaved → paper/GPT55_REPLICATION.md")


if __name__ == "__main__":
    main()
