"""Claude-designed paper-overview figure, companion to the gpt-5.5
version. Different visual storytelling: the 3 KEEPs as persistent
tokens across all phases, the audit stack as a ladder echoing
fig:audit-sieve, and final +0 as a physical empty bucket not just
text.
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


PROMPT = """Wide landscape scientific-paper visual abstract, 1536x1024,
flat-vector aesthetic on a white background. Style: a high-end
scientific textbook or the Distill.pub article layout, clean
sans-serif typography, unified pastel palette (teal, sand, cream,
lavender) with ONE highlight color (warm coral-red) reserved strictly
for "rejection" or "failure" marks. No decorative clutter; every
visual element earns its place.

Concept: the paper's full arc in ONE picture, told as a horizontal
left-to-right narrative. Three small green wisdom tokens labelled
W076, W077, W078 are the recurring characters; they appear in
Phase 2, survive the inner-loop gate, then fail every audit layer,
then fail again prospectively.

LAYOUT: four unequal-width phases separated by subtle vertical
guide-lines. Left to right:

PHASE 1 (narrow, 15% width) — "BUILD".
A stylised solver cube labelled "v20 scaffold" with a thin arrow from
a small stack labelled "75-entry wisdom library" into the cube. Below,
a tiny horizontal bar with two segments reading "74% -> 86% win rate
against baseline". One sub-label: "top-K=2 retrieval".

PHASE 2 (medium, 25% width) — "EVOLVE".
Top: four small module icons arranged 2x2, labelled
"failure-gen / success-distill / cross-LLM / pruner". Arrows converge
downward to a row of 12 small grey candidate tokens. Below: a single
horizontal gate bar labelled "+10pp held-out A/B (gemini-judged,
n=50)". Below the gate: three GREEN tokens labelled W076, W077, W078
emerge. 9 grey tokens are shown OUTSIDE, marked with a small coral X
(refused). Sub-label: "3/12 commit, library 75 -> 77".

PHASE 3 (widest, 40% width) — "AUDIT".
This is the visual climax. The 3 green W076/W077/W078 tokens from
Phase 2 enter from the left onto a vertical 6-rung LADDER labelled
"6-layer audit stack". Rungs, top to bottom, each labelled:
L1 cross-family judge (gavels icon),
L2 side-reseed (A|B swap icon),
L3 n=50 -> 100 (histogram-widening icon),
L4 cross-solver (3-cube icon),
L5 fresh domain + GSM8K (two-tray icon),
L6 faithfulness, non-pair-wr (sentence+scatter icon).
At EACH rung: a small coral X appears beside one of the three tokens
and it falls off the side of the ladder. By the bottom rung, all
three tokens have fallen. Below the ladder: an empty tray labelled
"0/3 KEEPs survive". The three coral Xs on the ladder should be the
visual climax — unmissable.

PHASE 4 (medium, 20% width) — "RECOVER?".
Top: a smaller second gate labelled "trigger-conditioned gate
(redesigned, Exp 15)". Below it, on the LEFT side of this phase, a
small green group of 4 tokens labelled "4/12 local (tuning set)".
A dashed-line arrow points from this group to the RIGHT side,
labelled "apply prospectively (Exp 33, 9 new candidates)", where an
empty tray sits with a coral "0/9" annotation. Below both: a single
summary bar labelled "combined 3/22 (below 30% random baseline)"
with a coral dashed reference line at 30%.

FAR-RIGHT CODA (outside the 4 phases): a large, visually striking
EMPTY BUCKET or EMPTY HANDS icon in muted grey. Above it, in large
bold type: "final library delta = +0". Below it, a short handwritten-
looking sub-caption: "the audit stack is the contribution."

Top of the whole canvas (title band): "Scaffold, Evolve, Then Audit:
6 layers of independent re-judgment reject every committed KEEP."

Typography: large phase headers (BUILD / EVOLVE / AUDIT / RECOVER?)
in a distinct band above each column, mid-size labels inside, tiny
sub-captions below. Numbers integrated into visuals (inside tokens,
inside tray, inside bars) not as separate text blobs.

No stray text. No decorative faces or characters. No 3D. No
skeuomorphic paper textures. Use the highlight coral ONLY for Xs and
the 30% reference line.
"""


def main():
    print("=== Claude-designed prompt ===")
    print(f"  prompt length: {len(PROMPT)} chars\n")
    print("=== gpt-image-2 renders ===")
    t0 = time.time()
    resp = client.images.generate(
        model="gpt-image-2", prompt=PROMPT, size="1536x1024", n=1,
    )
    d = resp.data[0]
    out = FIGS_DIR / "intuition_paper_overview_claude.png"
    if d.b64_json:
        out.write_bytes(base64.b64decode(d.b64_json))
    elif d.url:
        import urllib.request
        urllib.request.urlretrieve(d.url, out)
    print(f"  saved → {out.name} ({out.stat().st_size} bytes, "
          f"{time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
