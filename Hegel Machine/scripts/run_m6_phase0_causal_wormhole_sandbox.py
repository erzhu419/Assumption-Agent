#!/usr/bin/env python3
"""Run M6 Phase-0 Standalone Causal-Wormhole Propagation & Recursive Sub-Assumption Sandbox on isolated branch `exp/m6-causal-wormhole-sandbox`."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition.m6_causal_wormhole_sandbox import (  # noqa: E402
    run_m6_phase0_complete_study,
)


def main() -> int:
    out_dir = REPO_ROOT / "docs" / "evidence" / "m6_phase0_causal_wormhole_sandbox_20261121"
    manifest = run_m6_phase0_complete_study(out_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
