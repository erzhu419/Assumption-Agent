"""Claude-designed image prompts for the 4 intuition figures.

Different conceptual metaphors from the gpt-5.5 versions — not cosmetic
variants but genuinely different framings:

  audit_sieve      → ladder of invariances (each layer shows WHAT it
                      holds constant, not just what it rejects)
  gray_zone        → Galton-board / quincunx (measurement physics:
                      same ball, chaotic landing in [0.50, 0.65])
  wisdom_prosthesis → v20 pipeline architecture (frame → rewrite →
                      execute → audit, with top-K retrieval at frame)
  decomposition    → onion-shell peeling (observed lift as nested
                      rings, each audit layer peels one ring,
                      innermost core Z_specific is tiny)

Output filenames end in _claude to avoid overwriting gpt-5.5 versions.
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


# ---- prompts written directly by Claude ----

PROMPTS = {
    "audit_sieve_claude": {
        "size": "1024x1536",
        "prompt": """Vertical scientific figure, clean flat-vector aesthetic,
white background, pastel palette, sans-serif typography, the look of a
modern statistics textbook or a Distill.pub article.

Concept: a LADDER OF INVARIANCES. Six horizontal platforms stacked
vertically, each platform a different "invariance test" for three
identical green tokens (labelled W076, W077, W078) falling downward.

At the top: a small source-hopper labelled "inner-loop gate: +10pp at
n=50, gemini-judged" dropping the three green tokens.

Platform 1 ("L1 cross-family judge"): Show TWO stylized judge icons
(gavel silhouettes) side by side, connected by a dashed equals-sign.
The three tokens rest briefly; one token gets a red cross and falls
off the side.

Platform 2 ("L2 side-reseed"): Two position cards A|B and B|A with a
double-arrow between them. A token falls off sideways.

Platform 3 ("L3 n=50 -> n=100"): A small histogram doubling its bin
count, with a CI bar shrinking. A token falls off.

Platform 4 ("L4 cross-solver"): Three small cube icons side-by-side
(three different solver families). A token falls off.

Platform 5 ("L5 fresh domain + GSM8K"): Two domain-labelled trays
next to each other. A token falls off.

Platform 6 ("L6 faithfulness, non-pair-wr"): A small sentence icon
and an embedding dotted scatter, with an arrow between them.
The last token falls off.

At the very bottom: an empty square tray labelled "final library
delta: +0". No tokens in it.

Small sub-caption on the side of each platform: "what this layer
holds constant." Each platform in a slightly different soft pastel
(lavender, mint, pale-blue, peach, cream, rose). Each token is a
small green disc with the wisdom ID in white.

No numbers other than the L1-L6 labels and "+0" in the bottom tray.
Portrait aspect. Clean, pedagogical, no decorative clutter.""",
    },

    "gray_zone_fragility_claude": {
        "size": "1024x1024",
        "prompt": """Square scientific figure, clean flat-vector aesthetic,
white background, modern-textbook style (Tufte / Edward-Tufte-meets-
Distill.pub). Sans-serif typography.

Concept: a GALTON BOARD / QUINCUNX specialised for pair-wise win
rate. Demonstrates measurement fragility in the [0.50, 0.65] band.

Top of canvas: a small hopper labelled "same answer pairs (identical
content)". From the hopper, a single ball drops downward into a
lattice of small pegs arranged in a triangular Galton pattern. The
pegs are not labelled individually; they collectively represent
"nuisance factors" (judge family, side seed, sample subset).

Bottom half: a horizontal wr axis from 0.40 to 0.75, with 0.50 and
0.60 clearly marked. A light-rose shaded band highlights
[0.50, 0.65] labelled "gray zone at n=50". A dashed vertical line at
0.60 labelled "KEEP threshold (+10pp)".

At the bottom where the balls land: stacked histogram bins showing
where the identical-content ball landed under different nuisance
realisations. The distribution is wide, centred near 0.56, with
noticeable mass on both sides of the 0.60 threshold — some balls
land in "KEEP" territory, others in "REVERT".

Two small annotations:
  - "one-pair flip = ±0.02" (pointing at the ball lattice)
  - "same content, different verdict" (pointing at the bimodal
    straddle of the 0.60 line)

Palette: muted greys for the lattice, a soft teal for the balls, and
a warm rose for the gray zone band. One warm-orange dashed line for
the KEEP threshold.

No numbers except the axis ticks (0.40, 0.50, 0.60, 0.70, 0.75), the
"±0.02" annotation, and the axis label "pair win rate".

Clean, pedagogical, unified colour palette, no 3D or skeuomorphic
elements.""",
    },

    "wisdom_prosthesis_claude": {
        "size": "1536x1024",
        "prompt": """Wide landscape scientific figure, clean flat-vector
aesthetic, white background, soft pastel palette, sans-serif typography.
Aesthetic: 3Blue1Brown-meets-system-architecture-diagram.

Concept: the v20 scaffolded-solver PIPELINE with wisdom retrieval
called out as an architectural step, not as an orbital decoration.

Layout: a horizontal left-to-right pipeline with four boxes in sequence,
each labelled:

  1. "FRAME" (light lavender box)
  2. "REWRITE" (light mint)
  3. "EXECUTE" (light peach)
  4. "AUDIT" (light rose)

Above box 1 (FRAME): a small stack of "wisdom cards" labelled "75-entry
library". An arrow points down from the stack into box 1, labelled
"retrieve top-K=2". Two cards are highlighted (glowing soft yellow),
showing they are the retrieved pair for the current problem. The two
highlighted cards show sample aphorisms:
  - "investigate before speaking (Mao)"
  - "sharpen the tool first (Confucius)"

Between boxes 1-4: small arrows showing dataflow.

Below the pipeline: a single thin horizontal bar divided into two
regions. Left region (lighter shade) labelled "baseline solver, no
library: 74%". Right region (darker shade) labelled "v20 + library:
86%". A small "+12pp" annotation between them.

Top-left corner: a raw LLM cube icon with a thought bubble "just
solve it" — small, as a reference to the baseline.

Top of canvas (header): "v20 scaffold: top-K=2 wisdom retrieval at
frame time."

Palette: pastel lavender, mint, peach, rose for the four pipeline
boxes; warm-yellow glow for retrieved cards; soft neutral grey for
other cards; muted blue for the baseline bar, deeper teal for the
scaffolded bar.

No stray text. No numbers except "74%", "86%", "+12pp", "top-K=2",
"75".""",
    },

    "decomposition_claude": {
        "size": "1536x1024",
        "prompt": """Wide landscape scientific figure, clean flat-vector
aesthetic, white background, sans-serif typography. Style: Distill.pub
equation-visualisation.

Concept: ONION-SHELL PEELING of the pair-wr lift. The observed
inner-loop wr lift above 0.50 is shown as a set of concentric rings
around a tiny centre. Each audit layer peels off one outer ring,
revealing the core.

Layout (from left to right, like a time-lapse of peeling):

STAGE 0 (far left): A large bullseye-like disc. Outermost ring (red,
thick): "Z_style — judge preference lift". Middle ring (yellow,
medium): "Z_generic — any-extra-context lift". Inner small core
(blue, tiny): "Z_specific — wisdom-content lift".
Above: "observed wr_ext - 0.50 (inner-loop, n=50)".

STAGE 1 (next): Same disc with red ring crossed out. Arrow coming in
labelled "L1: cross-family judge swap". Label below: "Z_style cancels".

STAGE 2: Disc with yellow ring crossed out too. Arrow labelled "L4:
cross-solver + L5: fresh domain". Label: "Z_generic cancels".

STAGE 3 (far right): Just the tiny blue core, shown alone. Arrow
labelled "L6: faithfulness probe, directly estimates Z_specific".
Below it: "Z_specific approx 0 for W076, W077, W078".

Between stages: small horizontal arrows showing the peeling
progression.

Bottom-centre caption: "The null verdict is not 'no lift observed'.
It is 'no Z_specific lift remains after nuisance cancellation.'"

Palette: deep-red for Z_style, warm-yellow for Z_generic, steel-blue
for Z_specific. Light-grey for crossed-out rings. White background.

Small ticks on each disc edge show it is a "ring" not a solid pie
slice. The centre blue core in STAGE 3 should be visibly tiny — the
visual punchline.

No stray text. No numbers except "0.50" in the baseline reference.""",
    },
}


def run(slug, spec):
    print(f"\n=== {slug} ===")
    print(f"[prompt, {len(spec['prompt'])} chars]  {spec['prompt'][:160]}...")
    t0 = time.time()
    resp = client.images.generate(
        model="gpt-image-2", prompt=spec["prompt"], size=spec["size"], n=1,
    )
    d = resp.data[0]
    out = FIGS_DIR / f"intuition_{slug}.png"
    if d.b64_json:
        out.write_bytes(base64.b64decode(d.b64_json))
    elif d.url:
        import urllib.request
        urllib.request.urlretrieve(d.url, out)
    else:
        print(f"   ERROR: no b64/url: {vars(d)}")
        return
    print(f"   saved → {out.name} ({out.stat().st_size} bytes, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    slugs = sys.argv[1:] if len(sys.argv) > 1 else list(PROMPTS)
    for s in slugs:
        if s not in PROMPTS:
            print(f"unknown: {s}")
            continue
        run(s, PROMPTS[s])
