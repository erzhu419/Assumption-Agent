#!/usr/bin/env python3
"""Run V18.4 `Hegel-Discovery-240` Publication-Scale Continual Scientific Discovery Benchmark (`240` tasks × `3` models × `3` arms = `2,160` episodes)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition.m5_v18_4_hegel_discovery_240 import (  # noqa: E402
    run_v18_4_hegel_discovery_240_study,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run V18.4 Hegel-Discovery-240 Publication-Scale Continual Benchmark."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs" / "evidence" / "v18_4_hegel_discovery_240_20261121",
        help="Target evidence directory (supports checkpoint-resume automatically).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-6-luna", "gpt-6.1-terra", "gpt-6.2-sol"],
        help="Subscription models via official ChatGPT OAuth Codex CLI.",
    )
    parser.add_argument(
        "--waves",
        nargs="+",
        default=["wave_1", "wave_2", "wave_3", "wave_4", "wave_5", "wave_6"],
        help="Ordered continual waves to execute/replay (default: wave_1 .. wave_6).",
    )
    parser.add_argument("--max-workers", type=int, default=8, help="Concurrent episode workers.")
    parser.add_argument("--max-turns", type=int, default=6, help="Max multi-turn budget per episode.")
    parser.add_argument("--reasoning-effort", type=str, default="medium", help="Codex reasoning effort.")
    parser.add_argument("--skip-live-gpt", action="store_true", help="Dry structure check without live subscription calls.")
    parser.add_argument("--clean-restart", action="store_true", help="Delete existing --out directory before starting.")
    args = parser.parse_args()

    manifest = run_v18_4_hegel_discovery_240_study(
        args.out.resolve(),
        run_live_gpt=not args.skip_live_gpt,
        models=tuple(args.models),
        wave_names=tuple(args.waves),
        max_workers=args.max_workers,
        max_turns=args.max_turns,
        reasoning_effort=args.reasoning_effort,
        clean_restart=args.clean_restart,
    )
    print(
        json.dumps(
            {
                "version": manifest["version"],
                "completed_episode_count": manifest.get("completed_episode_count"),
                "waves_completed": manifest.get("waves_completed"),
                "quota_paused_models": manifest.get("quota_paused_models"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
