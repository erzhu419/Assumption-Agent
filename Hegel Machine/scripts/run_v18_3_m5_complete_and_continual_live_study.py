"""CLI Runner for Hegel V18.3: Complete 3-Model Post-Audit Matrix + 3-Wave Complex Continual Loop + Amortization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hegel_machine.cognition.m5_v18_3_continual_and_complex_live import (
    run_v18_3_complete_and_continual_live_study,
)


def main() -> None:
    repo_hegel_root = Path(__file__).resolve().parents[1]
    default_out = (
        repo_hegel_root
        / "docs"
        / "evidence"
        / "v18_3_m5_complete_3model_continual_complex_and_amortization_20261121"
    )

    parser = argparse.ArgumentParser(description="Run Hegel V18.3 Complete 3-Model & Complex Continual Live Study")
    parser.add_argument("--out", type=Path, default=default_out, help="Output evidence directory")
    parser.add_argument("--skip-live-gpt", action="store_true", help="Dry check without live Codex calls")
    parser.add_argument("--max-workers", type=int, default=5, help="Concurrent Codex CLI workers per wave")
    parser.add_argument("--reasoning-effort", default="medium", help="Codex reasoning effort")
    args = parser.parse_args()

    manifest = run_v18_3_complete_and_continual_live_study(
        args.out,
        repo_hegel_root=repo_hegel_root,
        run_live_gpt=not args.skip_live_gpt,
        max_workers=args.max_workers,
        reasoning_effort=args.reasoning_effort,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
