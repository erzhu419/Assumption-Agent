"""Runner for Hegel V18.1: Official OpenAI ChatGPT Subscription Multi-Model M5 Benchmark & Forensic Decision Provenance.

Usage (Official ChatGPT Subscription across `gpt-6-luna`, `gpt-6.1-terra`, `gpt-6.2-sol`):
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_v18_1_m5_live_gpt_and_provenance_study.py \
    --models gpt-6-luna gpt-6.1-terra gpt-6.2-sol

Usage (Offline Deterministic Provenance Only):
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_v18_1_m5_live_gpt_and_provenance_study.py --no-live-gpt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition.m5_live_gpt_and_provenance import run_v18_1_m5_live_gpt_and_provenance_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hegel V18.1 Official ChatGPT Subscription Multi-Model M5 Study")
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "evidence" / "v18_1_m5_live_gpt_and_decision_provenance_20261120",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-6-luna", "gpt-6.1-terra", "gpt-6.2-sol"],
        help="Official ChatGPT subscription model slugs (via Codex CLI `auth_mode: chatgpt`)",
    )
    parser.add_argument("--reasoning-effort", type=str, default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--max-workers", type=int, default=3, help="Parallel episode workers across subscription models")
    parser.add_argument("--no-live-gpt", action="store_true", help="Skip live subscription calls (run deterministic provenance only)")
    args = parser.parse_args()

    manifest = run_v18_1_m5_live_gpt_and_provenance_study(
        args.evidence_dir.resolve(),
        repo_root=REPO_ROOT,
        run_live_gpt=not args.no_live_gpt,
        gpt_models=args.models,
        reasoning_effort=args.reasoning_effort,
        max_workers=args.max_workers,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
