"""Run the Phase 2 distiller: produce UpdateCandidates from kept records."""

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

import _config as cfg
from experience_distiller.distiller import distill, save_candidates


def main():
    eval_path = cfg.PROJECT_ROOT / "analysis" / "evaluator_results.json"
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    keep_ids = {r["execution_id"] for r in data["results"] if r["keep"]}
    print(f"{len(keep_ids)} records flagged keep by evaluator")

    all_by_id = {}
    kept = []
    for f in sorted(cfg.EXECUTIONS_DIR.glob("exec_*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        all_by_id[rec["execution_id"]] = rec
        if rec["execution_id"] in keep_ids:
            kept.append(rec)

    print(f"Loaded {len(all_by_id)} total records, distilling {len(kept)} kept...")
    candidates = distill(kept, all_by_id)
    print(f"Produced {len(candidates)} update candidates")

    # Breakdown
    by_placement = Counter(c.placement for c in candidates)
    by_strength = Counter(c.evidence_strength for c in candidates)
    top_strategies = Counter(c.target_strategy for c in candidates).most_common(10)
    print(f"\n  by placement: {dict(by_placement)}")
    print(f"  by strength : {dict(by_strength)}")
    print(f"  top 10 strategies by candidate count: {top_strategies}")

    n = save_candidates(candidates, cfg.PENDING_REVIEW)
    print(f"\nSaved {n} candidates to {cfg.PENDING_REVIEW}")


if __name__ == "__main__":
    main()
