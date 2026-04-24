"""Regenerate the 3 old intuition figures via gpt-5.5 (prompt design)
+ gpt-image-2 (render).

Old versions were generated with gpt-mini prompts and feel flat. The
point of each:
  1) audit_sieve         — the headline visual: 6 independent layers
                            filter a self-improvement loop's 3 KEEPs
                            to 0 survivors. Must feel like a sieve /
                            funnel that is SUPPOSED to catch things.
  2) gray_zone_fragility — measurement fragility at n=50 in the
                            [0.50, 0.65] band. Should FEEL fragile:
                            same content, different verdict.
  3) wisdom_prosthesis   — the scaffold setup. Cognitive prosthesis:
                            LLM alone 74% vs. LLM + wisdom library
                            86% on the same substrate.

Overwrites existing files in paper/figs/.
"""

import base64
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "phase zero" / "scripts"))
from dotenv import load_dotenv
root = Path(__file__).parent.parent
for p in (root / ".env", root / "phase zero" / ".env"):
    if p.exists():
        load_dotenv(p)

from openai import OpenAI

BASE = os.environ["GPT5_BASE_URL"]
KEY = os.environ["GPT5_API_KEY"]
client = OpenAI(base_url=BASE, api_key=KEY)

FIGS_DIR = Path(__file__).parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)


# ---- design briefs ----
BRIEFS = {
    "audit_sieve": {
        "filename": "intuition_audit_sieve.png",
        "size": "1024x1536",
        "brief": """Design a scientific intuition figure titled
'The six-layer audit stack as a sieve.'

CONCEPT:
A self-improving LLM loop produces 12 candidate wisdoms. A held-out
+10pp single-family A/B gate commits 3 of them as KEEPs
(W076, W077, W078). These 3 KEEPs then face a six-layer independent
audit stack, each layer attacking a DIFFERENT non-specific source of
lift. ZERO candidates exit the bottom of the stack. Final validated
library delta: +0.

DESIRED VISUAL:
A vertical pipeline or stacked-funnel motif. Top: 12 small grey
candidate tokens being poured into an initial gate that lets 3 green
tokens through (labeled W076, W077, W078). The 3 tokens then descend
through 6 differently-shaped filters, each clearly labelled:
  L1 cross-family judge
  L2 side-randomization reseed
  L3 sample extension (n=50 to n=100)
  L4 cross-solver replication
  L5 fresh domain / GSM8K port
  L6 faithfulness (non-pair-wr)
At each filter, at least one green token is visibly rejected (shown
exiting sideways with a small red X). By the last filter, all 3 are
rejected. At the very bottom: an empty output bin with the label
'final library delta: +0'.

STYLE:
Clean scientific diagram, flat vector aesthetic (think Distill.pub or
a high-end textbook). Pastel but distinct colors for the 6 filters
(different shape for each: mesh, funnel, strainer, grate, membrane,
needle-eye). Soft shadows. Sans-serif labels, no stray text. White
background. Vertical portrait composition. Do not include any numbers
other than the layer labels L1-L6 and 'final library delta: +0'.

Output only the final text-to-image prompt, under 320 words.""",
    },

    "gray_zone_fragility": {
        "filename": "intuition_gray_zone_fragility.png",
        "size": "1024x1024",
        "brief": """Design a scientific intuition figure titled
'Measurement fragility at n=50 in the 0.50-0.65 band.'

CONCEPT:
Pairwise win-rate at n=50 is measurement-fragile in the band
0.50-0.65 because a ONE-PAIR flip moves wr by 0.02. The default
self-improvement KEEP threshold (+10pp, so wr ≥ 0.60) sits INSIDE
this fragile band. Single-family A/B measurements in this band can
cross the decision boundary without any change in the underlying
content — just a different judge family, a different A/B side seed,
or a different sample extension.

DESIRED VISUAL:
A horizontal wr-axis from 0.40 to 0.75, with a shaded 'gray zone'
rectangle spanning 0.50 to 0.65. A dashed vertical line at 0.60
labeled 'KEEP threshold (+10pp)'. Above the axis, three clusters of
small dots, each cluster a different KEEP candidate (W076, W077,
W078). Within each cluster, draw multiple dots at slightly different
positions all within the gray zone, with small labeled arrows
between them saying 'same content, different verdict' — one dot per
arrow representing an audit layer (L1 Opus, L1 Haiku, L2 reseed, L3
n=100). The arrows cross back and forth over the dashed 0.60
threshold. Annotation at top-right: 'One-pair flip = +0.02.' Bottom
caption: 'Single-family A/B at n=50 is measurement-fragile in this
band.'

STYLE:
Clean statistical-figure aesthetic (think Tufte or the ggplot
scientific look). Use a single muted palette: grey axis, light-rose
gray-zone fill, three distinct candidate colors (steel blue, warm
orange, deep purple) for the dot clusters. Sans-serif labels.
Square canvas. No decorative elements; every mark carries
information. No numbers other than axis ticks, the 0.60 threshold
label, and the 0.02 one-pair-flip annotation.

Output only the final text-to-image prompt, under 280 words.""",
    },

    "wisdom_prosthesis": {
        "filename": "intuition_wisdom_prosthesis.png",
        "size": "1024x1024",
        "brief": """Design a scientific intuition figure titled
'Cognitive prosthesis: scaffolded solver with a wisdom library.'

CONCEPT:
A raw large language model given 'just solve it' underperforms the
same model when it is given retrieval-augmented methodological
scaffolding: problem reframes, anti-patterns to avoid, evaluation
criteria, and exemplars of parallel reasoning. Each retrieved
'wisdom' is a short aphorism (from Yi Jing, Munger, Drucker, Mao,
folk sayings) unpacked into LLM-legible form. The paper's v20
scaffold uses top-K=2 retrieval over a 75-entry wisdom library.
Empirically: 74% win rate alone, 86% with the library.

DESIRED VISUAL:
LEFT HALF: A stark lonely LLM — a simple labeled cube with a small
speech bubble saying 'just solve it', no context, no tools. Beneath
it a small bar showing 74%.

CENTER (arrow): a single thick arrow labeled '+ retrieve top-K=2
from wisdom library'.

RIGHT HALF: The same LLM cube, now surrounded by 6-8 small floating
wisdom cards arranged in a loose orbit. Each card has a short
aphorism ('list candidates; run tests; then decide', 'investigate
before speaking', 'sharpen the tool first', 'treat the root cause,
not the symptom', 'look at incentives to understand behavior',
'urgent: symptoms; stable: roots'). Two of the cards have a subtle
glow (they are the K=2 retrieved wisdoms for this problem). Beneath,
a bar showing 86%.

BOTTOM: A tiny caption 'v20 scaffold: frame + rewrite + execute +
audit, with top-K wisdoms injected at frame time.'

STYLE:
Clean pedagogical diagram, flat vector. Soft pastel palette. Each
wisdom card in a different soft color. The two glowing retrieved
cards use a subtle warm highlight. Sans-serif labels. Wide landscape
OR square composition. Think 3Blue1Brown or Distill.pub. No stray
text. No numbers beyond 74% and 86% and 'top-K=2'.

Output only the final text-to-image prompt, under 300 words.""",
    },
}


def run_one(slug, spec):
    print(f"\n=== {slug} ===")
    print("[1/2] gpt-5.5 designing prompt...")
    r = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": spec["brief"]}],
        max_tokens=900, temperature=0.3,
    )
    prompt = r.choices[0].message.content.strip()
    print(f"   prompt ({len(prompt)} chars, preview): {prompt[:180]}...")

    print("[2/2] gpt-image-2 rendering...")
    t0 = time.time()
    resp = client.images.generate(
        model="gpt-image-2", prompt=prompt, size=spec["size"], n=1,
    )
    d = resp.data[0]
    out = FIGS_DIR / spec["filename"]
    if d.b64_json:
        out.write_bytes(base64.b64decode(d.b64_json))
    elif d.url:
        import urllib.request
        urllib.request.urlretrieve(d.url, out)
    else:
        print(f"   ERROR: no b64 or url in response: {vars(d)}")
        return False
    print(f"   saved → {spec['filename']} ({out.stat().st_size} bytes, {time.time()-t0:.0f}s)")
    return True


if __name__ == "__main__":
    slugs = sys.argv[1:] if len(sys.argv) > 1 else list(BRIEFS)
    for s in slugs:
        if s not in BRIEFS:
            print(f"unknown slug: {s}")
            continue
        run_one(s, BRIEFS[s])
