"""Generate paper intuition figures via gpt-image-2 on ruoli.dev.

Each figure illustrates a CONCEPT from the paper (not a data plot).
Data plots are in figs/fig1..fig4.png; these land as figs/intuition_*.png.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
import urllib.request
import urllib.error


try:
    from dotenv import load_dotenv
    project_root = Path(__file__).resolve().parent.parent
    for candidate in (project_root / ".env", project_root / "phase zero" / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
    load_dotenv(override=False)
except ImportError:
    pass

API_KEY = os.environ.get("GPT5_API_KEY") or os.environ.get("RUOLI_API_KEY")
if not API_KEY:
    raise SystemExit("Missing GPT5_API_KEY / RUOLI_API_KEY env var. "
                     "Do NOT hardcode API keys in this file.")
API_URL = os.environ.get("GPT5_BASE_URL", "https://ruoli.dev/v1").rstrip("/v1") \
            + "/v1/images/generations"
OUT_DIR = Path(__file__).parent / "figs"
MODEL = "gpt-image-2"


FIGURES = [
    {
        "name": "intuition_wisdom_prosthesis",
        "size": "1024x1024",
        "prompt": (
            "Clean academic paper figure, minimal vector-graphics style, "
            "white background, black and dark-blue ink. Left side: a single "
            "large cube labeled 'LLM' alone with a small speech bubble "
            "reading 'just solve it'. Right side: the same cube labeled "
            "'LLM' surrounded by six small rectangular cards arranged in "
            "an arc, each card showing a short aphorism symbol (a scroll "
            "icon, an ideogram, a proverb mark). A horizontal arrow "
            "between left and right reads '+ wisdom library'. Below, a "
            "bar showing 74% filling up to 86%. No photorealistic detail, "
            "no human figures, pure technical diagram. Academic paper "
            "figure style like Distill.pub or NeurIPS."
        ),
    },
    {
        "name": "intuition_loop_vs_audit_disagreement",
        "size": "1024x1024",
        "prompt": (
            "Clean academic paper figure, minimal vector-graphics style, "
            "white background. Center: a single rectangular card labeled "
            "'candidate wisdom KEEP'. Two arrows fork from it. Upper arrow "
            "goes to a green check mark labeled 'inner-loop gate: +10pp "
            "n=50 single family'. Lower arrow goes to a stack of six "
            "small red X marks labeled L1 through L6, representing 'six-"
            "layer audit: cross-family, reseed, extend, cross-solver, "
            "fresh-domain, faithfulness'. Below the audit stack, three "
            "numbers in red: '-20pp, -13pp, -9pp'. The intuition: same "
            "content, same candidate, opposite verdict. No people, pure "
            "technical diagram, NeurIPS figure style."
        ),
    },
    {
        "name": "intuition_audit_sieve",
        "size": "1024x1536",
        "prompt": (
            "Clean academic paper figure, vertical layout, minimal "
            "vector-graphics style, white background. A wide funnel at "
            "top labeled '12 candidates'. Below it a narrower sieve "
            "labeled 'inner-loop gate' with '3 KEEPs' passing through. "
            "Below that, six horizontal filter bars stacked vertically, "
            "each labeled (L1: cross-family, L2: reseed, L3: n=50 to n=100, "
            "L4: cross-solver, L5: fresh domain, L6: faithfulness). Three "
            "candidates enter the top of the filter stack; at each level "
            "they are deflected to the side in red. Zero candidates exit "
            "the bottom, labeled 'final library delta: +0'. Every "
            "filter level in a different shade. No people, pure diagram, "
            "academic publication quality."
        ),
    },
    {
        "name": "intuition_gray_zone_fragility",
        "size": "1024x1024",
        "prompt": (
            "Clean academic paper figure, horizontal layout, minimal "
            "vector-graphics style, white background. A horizontal axis "
            "labeled 'pair win rate' running from 0.40 to 0.75. A shaded "
            "gray band spans the region 0.54 to 0.62, labeled "
            "'measurement gray zone'. Inside the band, three dot clusters "
            "each containing 5 dots with small jitter (same candidate, "
            "different reseed). Two of the clusters cross the 0.60 KEEP "
            "threshold under some seeds and fall under it in others, "
            "shown with arrow pairs indicating 'same content, different "
            "verdict'. Below: caption-style text 'single-family A/B at "
            "n=50 is measurement-fragile in this band'. No people, pure "
            "technical figure, NeurIPS style."
        ),
    },
    {
        "name": "intuition_trigger_gate_decomposition",
        "size": "1024x1024",
        "prompt": (
            "Clean academic paper figure, minimal vector-graphics style, "
            "white background. A single input labeled 'pair-wr gate' on "
            "the left splits by arrow into three orthogonal columns on "
            "the right. Column 1 labeled 'trigger fit: does context match "
            "the wisdom's pattern P?'. Column 2 labeled 'conditional "
            "utility: conditional on trigger firing, does the answer "
            "improve?'. Column 3 labeled 'citability: is the wisdom "
            "explicitly invoked?'. Each column shows a small check/cross "
            "symbol at the bottom (pass/fail). Three small check marks "
            "at the very bottom join with AND to give a single PASS. "
            "Caption: 'trigger-conditioned gate decomposition (L1/L2/L3)'. "
            "Pure diagram, no people, NeurIPS publication quality."
        ),
    },
]


def generate_one(fig: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{fig['name']}.png"
    payload = {
        "model": MODEL,
        "prompt": fig["prompt"],
        "size": fig.get("size", "1024x1024"),
        "n": 1,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} on {fig['name']}:\n{err_body}") from e
    data = json.loads(body)
    if "data" not in data or not data["data"]:
        raise RuntimeError(f"unexpected response for {fig['name']}:\n{body[:500]}")
    item = data["data"][0]
    if item.get("b64_json"):
        out_path.write_bytes(base64.b64decode(item["b64_json"]))
    elif item.get("url"):
        with urllib.request.urlopen(item["url"], timeout=120) as img_resp:
            out_path.write_bytes(img_resp.read())
    else:
        raise RuntimeError(f"no b64_json or url in response: {item!r}")
    elapsed = time.time() - t0
    print(f"  [{elapsed:5.1f}s] {out_path.name}  "
          f"({out_path.stat().st_size/1024:.0f} KB)")
    return out_path


def main():
    which = sys.argv[1:] if len(sys.argv) > 1 else None
    for fig in FIGURES:
        if which and fig["name"] not in which:
            continue
        print(f"Generating {fig['name']} ({fig.get('size', '1024x1024')})...")
        try:
            generate_one(fig, OUT_DIR)
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
