"""Run the Phase 2 knowledge integrator over pending candidates."""

import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

import _config as cfg
from knowledge_integrator.integrator import integrate_candidate, route_candidate_file


def main():
    cand_files = sorted(cfg.PENDING_REVIEW.glob("upd_*.json"))
    print(f"Found {len(cand_files)} pending candidates")

    decisions = []
    for cp in cand_files:
        cand = json.loads(cp.read_text(encoding="utf-8"))
        dec = integrate_candidate(
            cand, cfg.KB_DIR, cfg.CHANGE_HISTORY_DIR, cfg.STABILITY_TIERS
        )
        route_candidate_file(
            cp, dec, cfg.APPLIED, cfg.REJECTED, cfg.PENDING_HUMAN
        )
        decisions.append(dec)

    by_action = Counter(d.action for d in decisions)
    print(f"\n  Decisions: {dict(by_action)}")

    # Reason breakdown
    reasons = Counter(d.reason.split(" (")[0] for d in decisions if d.action == "reject")
    if reasons:
        print(f"  Reject reasons (by category):")
        for r, n in reasons.most_common():
            print(f"    {n:>4}  {r}")

    out = cfg.PROJECT_ROOT / "analysis" / "integrator_decisions.json"
    out.write_text(json.dumps([asdict(d) for d in decisions], indent=2, ensure_ascii=False))
    print(f"\nSaved decisions to {out}")


if __name__ == "__main__":
    main()
