"""Paper overview figure — 'paper in one picture'.

Two passes: first gpt-5.5 designs a prompt from a brief (saved as
intuition_paper_overview_gpt55.png), then this script will be called
again for the Claude-designed variant. Filenames are disambiguated.
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


DESIGN_BRIEF_FOR_GPT55 = """You design a paper-overview intuition
figure. The paper's one-sentence claim: "A self-improving LLM loop
reports 3 new committed wisdoms under its own gate, but a six-layer
independent audit stack rejects all three, and a post-hoc
architectural recovery attempt fails to generalize; the audit stack
itself is the methodological contribution."

The figure must visualise the paper's full arc as a left-to-right
narrative in ONE landscape image. Treat it as the paper's visual
abstract — the first image a reader sees after the title.

The narrative has FOUR phases, each with concrete numbers:

Phase 1 — BUILD. A scaffolded solver (v20, Chinese open-ended reasoning)
retrieves top-K=2 wisdoms from a 75-entry library. Baseline solver
win-rate 74%, with library 86%.

Phase 2 — EVOLVE. A 4-module orchestrator (failure-generator,
success-distiller, cross-model-distiller, Darwinian-pruner) proposes
12 candidates; a +10pp held-out A/B gate commits 3 KEEPs
(W076, W077, W078) and refuses the other 9. Library grows 75 -> 77.

Phase 3 — AUDIT. The three committed KEEPs then face a 6-layer audit
stack (cross-family judge, side-randomization, sample-extension,
cross-solver, fresh-domain-and-GSM8K, faithfulness). Every layer is
independent of the others. Zero of three KEEPs survive any layer.

Phase 4 — RECOVER? A trigger-conditioned gate is redesigned on the
12 candidates and locally rescues 4 of them. But when the same gate
is applied prospectively to 9 new candidates (Exp 33), 0 of them pass,
combining to 3/22 below the 30% random-inclusion baseline. The
architectural redesign does not generalize.

FINAL VERDICT: library delta on external data = +0. The audit stack
is the contribution.

Design requirements:
- Landscape layout, 1536x1024
- Four visually distinct phase columns or bands
- Within each phase, show the key numbers (75, 12, 3/12, 0/3, 4/12,
  3/22, +0) as integrated visual elements (bars, tokens, funnel
  outputs), NOT as bullet lists
- Arrows between phases signal the passage of verdict
- A prominent concluding "+0" element at the far right, emphasised
- Do not use small dense text; image models garble it
- Use a unified pastel palette with ONE highlight color for the
  audit-stack rejection
- Scientific textbook aesthetic (Distill.pub / 3Blue1Brown), flat
  vector, sans-serif typography, white background
- No decorative clutter; every mark earns its place
- The visual climax is the transition Phase 3 -> Phase 4 (audit kills
  the 3 KEEPs, architectural recovery fails prospectively)

Output ONLY the final text-to-image prompt for the image model. Be
concrete about layout, visual elements, and typography placement.
Keep under 400 words. Do not include preamble, explanation, or
multiple drafts.
"""


def gpt55_designed():
    print("=== gpt-5.5 designs prompt ===")
    r = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": DESIGN_BRIEF_FOR_GPT55}],
        max_tokens=1200, temperature=0.3,
    )
    prompt = r.choices[0].message.content.strip()
    print(f"  prompt ({len(prompt)} chars)")
    print(f"  preview: {prompt[:300]}...\n")

    print("=== gpt-image-2 renders ===")
    t0 = time.time()
    resp = client.images.generate(
        model="gpt-image-2", prompt=prompt, size="1536x1024", n=1,
    )
    d = resp.data[0]
    out = FIGS_DIR / "intuition_paper_overview_gpt55.png"
    if d.b64_json:
        out.write_bytes(base64.b64decode(d.b64_json))
    elif d.url:
        import urllib.request
        urllib.request.urlretrieve(d.url, out)
    print(f"  saved → {out.name} ({out.stat().st_size} bytes, "
          f"{time.time()-t0:.0f}s)")
    return prompt


if __name__ == "__main__":
    gpt55_designed()
