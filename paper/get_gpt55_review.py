"""Ask gpt-5.5 for a detailed top-venue reviewer-style critique
of the current paper draft.

Sends main.tex (main body up to \appendix; appendices are summarised
in the prompt) and asks for the review structured as NeurIPS/ICLR
review format: summary, strengths, weaknesses, questions, rating.
"""

import os
import re
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))


def _load_api_keys():
    """Source ~/.api_keys (shell-export format) into os.environ if RUOLI_* are missing."""
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

_load_api_keys()
from openai import OpenAI

KEY = os.environ.get("RUOLI_GPT_KEY") or os.environ.get("GPT5_API_KEY", "")
BASE_RAW = os.environ.get("RUOLI_BASE_URL") or os.environ.get("GPT5_BASE_URL", "https://ruoli.dev")
# Ensure /v1 suffix for OpenAI-compat path
BASE = BASE_RAW.rstrip("/")
if not BASE.endswith("/v1"):
    BASE = BASE + "/v1"
if not KEY:
    raise SystemExit("Missing RUOLI_GPT_KEY env var (and no GPT5_API_KEY fallback).")
client = OpenAI(base_url=BASE, api_key=KEY)
print(f"using base={BASE}, key=...{KEY[-8:]}")

MAIN_TEX = Path(__file__).parent / "main.tex"
OUT = Path(__file__).parent / "GPT55_REVIEW.md"


def main_body(text):
    """Return everything up to and excluding \\appendix."""
    idx = text.find("\\appendix")
    return text[:idx] if idx > 0 else text


REVIEW_PROMPT = """You are a senior reviewer for a top ML venue
(NeurIPS / ICML / ICLR main track). Read the LaTeX submission below
carefully.  It claims to be a methodology paper / case study about
auditing a self-improving LLM loop.

Produce a thorough review in the following structure (be specific,
cite line numbers where useful, do not pad):

# Summary
2--4 sentences in your own words on what the paper claims and how
it argues for that claim.

# Strengths
A bulleted list (5--8 items). Each item names a specific positive
aspect, where in the paper it appears, and why it strengthens the
submission.

# Weaknesses
A bulleted list (8--15 items), ordered roughly from most to least
severe. Each item:
- Names the specific issue (with line / section reference if possible)
- Explains why it is a weakness for THIS class of paper
- Suggests what would fix it

# Questions to the authors
A list of 4--8 specific questions that, if answered, would change
your rating. These should be questions whose answers are not
already in the submission.

# Rating
Pick exactly one of:  Strong Accept / Accept / Weak Accept /
Borderline / Weak Reject / Reject / Strong Reject

Then a one-paragraph justification (3--5 sentences) that names
which weaknesses drove the rating most.

# Confidence
1 (low) -- 5 (high), with one sentence explaining your domain
familiarity with self-improving LLM loops, LLM-as-judge audit
methods, and reproducibility / preregistration discussions.

---

Important review guidance:
- Be candid. If the work is interesting but not yet a top-venue
  contribution, say so without softening.
- Look hard for: claim-vs-evidence mismatches, internal
  inconsistencies, statistical overstatement, post-hoc analysis,
  selection effects, unsupported generalisations.
- Penalise self-praise and assertion-without-evidence; reward
  honest negative results, scoped claims, and reproducible
  artefacts.
- Note any statistical or theoretical errors directly. Do not
  hedge.
- Do not invent issues. Every weakness must point at a specific
  passage in the submission.

The paper has appendices (not pasted below) covering Exp 18-19
(agent designs and writes a replacement gate), a multi-step
research-task pilot, full prompt templates, a compute-and-cost
breakdown table, a wisdom-record schema example, and a
reproducibility appendix specifying model identities, temperatures,
seeds, and proxy details. You may treat their existence as evidence
but do not request the appendices unless something specific is
unclear in the main body.

=== BEGIN SUBMISSION (main body, LaTeX source) ===

{paper_text}

=== END SUBMISSION ===

Now write the review.
"""


def main():
    raw = MAIN_TEX.read_text(encoding="utf-8")
    body = main_body(raw)
    print(f"Sending {len(body)} chars (main body, ~{len(body)//4} tokens) "
          f"to gpt-5.5...")

    prompt = REVIEW_PROMPT.format(paper_text=body)

    t0 = time.time()
    r = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=6000,
        temperature=0.3,
    )
    review = r.choices[0].message.content
    elapsed = time.time() - t0
    usage = r.usage.model_dump() if r.usage else {}
    print(f"  done in {elapsed:.0f}s, usage: {usage}")

    OUT.write_text(review)
    print(f"\nReview saved → {OUT}")
    print(f"\n{'=' * 70}\nReview length: {len(review)} chars\n{'=' * 70}\n")
    print(review)


if __name__ == "__main__":
    main()
