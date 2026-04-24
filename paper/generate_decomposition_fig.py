"""Generate intuition_decomposition.png via gpt-5.5 (prompt design) + gpt-image-2.

The figure visualises the theoretical claim in §3.5:
  wr_ext = 0.5 + E[Z_specific] + E[Z_generic] + E[Z_style]
and how each audit layer (L1-L6) cancels one non-specific term,
leaving only Z_specific visible. For our 3 KEEPs, Z_specific ≈ 0,
which is why all three fail every layer.
"""

import base64
import os
import sys
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

OUT = Path(__file__).parent / "figs" / "intuition_decomposition.png"
OUT.parent.mkdir(exist_ok=True)

# ---- Step 1: ask gpt-5.5 to write a detailed image prompt ----
DESIGN_BRIEF = """You are designing a scientific-paper intuition figure for the
decomposition of pair-wise win rate (wr_ext) into three additive components,
and how a six-layer audit stack cancels the non-specific terms.

The concept (from the paper's §3.5):

  observed_wr_ext - 0.50 = Z_specific + Z_generic + Z_style

where:
  - Z_specific = lift from the wisdom's actual content
  - Z_generic  = lift from ANY extra context of matching length
  - Z_style    = lift from judge-preferred surface features

Audit layers cancel non-specific terms:
  - L1 (cross-family judge)   cancels Z_style
  - L4 (cross-solver)         cancels interaction effects
  - L5 (fresh domain/GSM8K)   cancels distribution-specific styling
  - L6 (faithfulness)         directly estimates Z_specific

The empirical finding: after all layers, Z_specific ≈ 0 for our 3 KEEPs.

Design the image as:

LEFT half: A tall stacked bar chart showing "Observed wr_ext" as three
colored slices (blue=Z_specific very thin, yellow=Z_generic medium,
pink=Z_style large), with a horizontal dashed line at 0.50 as baseline.
Above the bar: label "Inner-loop measurement at n=50".

RIGHT half: Six small icons/arrows labelled L1-L6, each pointing to
and "erasing" or "crossing out" the yellow and pink slices. A final
small bar on the far right shows just the tiny blue Z_specific slice
remaining — approximately zero.

Bottom caption area: "After audit: only Z_specific remains. For W076,
W077, W078: Z_specific approx 0."

Style: clean scientific diagram, isometric or flat, Matplotlib-like
aesthetics, no text artefacts, use soft pastel colors (#4A90E2 blue,
#F5A623 yellow, #D0021B red), white background, readable sans-serif
labels. Think of Distill.pub or 3Blue1Brown visual style — clear
and pedagogical.

Write the image prompt for a text-to-image model. Output only the
prompt, no preamble or explanation. Keep under 350 words.
"""

print("[1/2] Asking gpt-5.5 to design image prompt...")
r = client.chat.completions.create(
    model="gpt-5.5",
    messages=[{"role": "user", "content": DESIGN_BRIEF}],
    max_tokens=800, temperature=0.3,
)
img_prompt = r.choices[0].message.content.strip()
print(f"   Prompt ({len(img_prompt)} chars): {img_prompt[:200]}...")

# ---- Step 2: call gpt-image-2 ----
print("\n[2/2] Calling gpt-image-2...")
resp = client.images.generate(
    model="gpt-image-2",
    prompt=img_prompt,
    size="1536x1024",  # wide for left+right layout
    n=1,
)

d = resp.data[0]
if d.b64_json:
    OUT.write_bytes(base64.b64decode(d.b64_json))
    print(f"   Saved (b64) → {OUT}")
elif d.url:
    import urllib.request
    urllib.request.urlretrieve(d.url, OUT)
    print(f"   Saved (url) → {OUT}")
else:
    print(f"   ERROR: unknown response type {vars(d)}")
