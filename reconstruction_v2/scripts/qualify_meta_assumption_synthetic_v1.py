#!/usr/bin/env python3
"""Source-free numeric-mechanism qualification for meta-assumption compilation.

The worker accepts no arguments and has no dataset, network, model, evaluator,
or output-file channel.  Its only successful output is one canonical semantic
receipt on stdout.  The receipt includes recomputable probe-statistic
commitments, trust-anchored evidence bundles, two-stage minimum commitment,
and real ``PolicyRuntime`` differential/no-op execution evidence.  This is
development qualification, not reality-source efficacy evidence.
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_PACKAGE_ROOT))

from assumption_agent.benchmarks import (  # noqa: E402
    meta_assumption_synthetic_worlds_v1 as qualification,
)


def main() -> int:
    if len(sys.argv) != 1:
        return 2
    sys.stdout.buffer.write(
        qualification.canonical_bytes(qualification.qualify())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
