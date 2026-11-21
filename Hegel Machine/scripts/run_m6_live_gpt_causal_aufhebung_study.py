#!/usr/bin/env python3
"""Run Live Official ChatGPT-Subscription Benchmark `Hegel-M6-Live-Causal-24` (`288` episodes) on isolated branch `exp/m6-causal-wormhole-sandbox`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition.m6_live_gpt_causal_aufhebung_benchmark import (  # noqa: E402
    run_m6_live_gpt_causal_aufhebung_study,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run M6 vs M5 Live GPT-6 Causal Aufhebung Benchmark.")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs" / "evidence" / "m6_live_gpt_causal_aufhebung_288ep_20261121",
    )
    parser.add_argument("--models", nargs="+", default=["gpt-6-luna", "gpt-6.1-terra", "gpt-6.2-sol"])
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--skip-live-gpt", action="store_true")
    args = parser.parse_args()

    manifest = run_m6_live_gpt_causal_aufhebung_study(
        args.out.resolve(),
        run_live_gpt=not args.skip_live_gpt,
        models=tuple(args.models),
        max_workers=args.max_workers,
        max_turns=args.max_turns,
    )
    print(
        json.dumps(
            {
                "version": manifest["version"],
                "completed_episode_count": manifest.get("completed_episode_count"),
                "quota_paused_models": manifest.get("quota_paused_models"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
