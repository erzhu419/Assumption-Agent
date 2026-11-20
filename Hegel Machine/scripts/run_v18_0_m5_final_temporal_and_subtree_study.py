"""Runner script for Hegel V18.0 (`M5` Final Culmination): Temporal Folding & Shared Base-Subtree Priors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition.m5_temporal_multiscale_and_base_subtree import run_full_v18_0_m5_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hegel V18.0 M5 Final Culmination Study")
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "evidence" / "v18_0_m5_final_temporal_multiscale_and_base_subtree_20261120",
    )
    args = parser.parse_args()
    manifest = run_full_v18_0_m5_study(args.evidence_dir.resolve())
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
