#!/usr/bin/env python3
"""Use the tracked runner for this preserved study protocol."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from run_risk_experiment_study import main

if __name__ == "__main__":
    main()
