"""CLI Runner for Hegel V18.2: Sovereign Audit Closure + 10-Arm Benchmark + ChatGPT Subscription Live Challenge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hegel_machine.cognition.m5_v18_2_sovereign_closure import run_v18_2_m5_sovereign_closure_study


def main() -> None:
    repo_hegel_root = Path(__file__).resolve().parents[1]
    default_out = repo_hegel_root / "docs" / "evidence" / "v18_2_m5_sovereign_closure_and_live_subscription_20261121"

    parser = argparse.ArgumentParser(description="Run Hegel V18.2 Sovereign Audit Closure & Live Subscription Study")
    parser.add_argument("--out", type=Path, default=default_out, help="Output evidence directory")
    parser.add_argument("--skip-live-gpt", action="store_true", help="Skip live ChatGPT subscription execution")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-6-luna", "gpt-6.1-terra"],
        help="Official ChatGPT subscription models to evaluate on L07 & L08",
    )
    parser.add_argument("--reasoning-effort", default="medium", help="Codex reasoning effort")
    parser.add_argument("--max-workers", type=int, default=3, help="Concurrent Codex CLI workers")
    args = parser.parse_args()

    manifest = run_v18_2_m5_sovereign_closure_study(
        args.out,
        repo_hegel_root=repo_hegel_root,
        run_live_gpt=not args.skip_live_gpt,
        gpt_models=tuple(args.models),
        reasoning_effort=args.reasoning_effort,
        max_workers=args.max_workers,
    )
    print(json.dumps({
        "version": manifest["version"],
        "all_audit_counterexamples_closed_v18_0": manifest["all_audit_counterexamples_closed_v18_0"],
        "all_audit_counterexamples_closed_fresh": manifest["all_audit_counterexamples_closed_fresh"],
        "core_12_controls_passed": manifest["core_12_controls_passed"],
        "twohop_and_narrow_stress_summaries": manifest["twohop_and_narrow_stress_summaries"],
        "live_chatgpt_subscription_L07_L08_challenge": manifest["live_chatgpt_subscription_L07_L08_challenge"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
