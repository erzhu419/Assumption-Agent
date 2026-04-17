"""
Phase 2 v2 pipeline: failure-weighted evaluator + LLM-rewritten conditions.

Steps:
  1. Evaluator (v2 weights) over all executions
  2. Distiller produces candidates
  3. LLM condition writer overrides template text; rejects redundant candidates
  4. Integrator applies surviving candidates to KB
  5. Diff report

Assumes:
  - KB already rolled back to pre-snapshot
  - Candidate/applied/rejected dirs have been cleared
"""

import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

import _config as cfg
from experience_evaluator.evaluator import evaluate_all, summarize
from experience_distiller.distiller import distill, save_candidates
from experience_distiller.condition_writer import ConditionWriter
from knowledge_integrator.integrator import integrate_candidate, route_candidate_file


def main():
    # -- Step 1: evaluator --
    print("[1/5] Evaluator (failure-weighted)...")
    results = evaluate_all(cfg.EXECUTIONS_DIR, info_threshold=cfg.INFO_SCORE_THRESHOLD)
    summary = summarize(results)
    print(f"       kept={summary['kept']}/{summary['total']} (rate={summary['keep_rate']:.1%})")
    print(f"       tag_counts={summary['tag_counts']}")

    kept_ids = {r.execution_id for r in results if r.keep}
    all_by_id = {}
    kept_records = []
    for f in sorted(cfg.EXECUTIONS_DIR.glob("exec_*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        all_by_id[rec["execution_id"]] = rec
        if rec["execution_id"] in kept_ids:
            kept_records.append(rec)

    # -- Step 2: distiller --
    print("[2/5] Distiller...")
    candidates = distill(kept_records, all_by_id)
    by_placement = Counter(c.placement for c in candidates)
    print(f"       {len(candidates)} candidates (by placement: {dict(by_placement)})")

    # -- Step 3: LLM condition writer --
    print("[3/5] LLM condition writer...")
    writer = ConditionWriter(kb_dir=cfg.KB_DIR, executions_dir=cfg.EXECUTIONS_DIR)
    approved: list = []
    rejected_by_llm: list = []
    for i, cand in enumerate(candidates):
        out = writer.rewrite(cand.to_dict())
        if out.get("rejected") or not out.get("condition"):
            rejected_by_llm.append((cand.candidate_id, out))
            continue
        cand.condition_text = out["condition"]
        approved.append(cand)
        if (i + 1) % 20 == 0:
            print(f"       processed {i+1}/{len(candidates)} (approved={len(approved)})")
    print(f"       approved={len(approved)}, rejected_by_llm={len(rejected_by_llm)}")

    # Clear pending dir and write approved candidates
    for f in cfg.PENDING_REVIEW.glob("*.json"):
        f.unlink()
    save_candidates(approved, cfg.PENDING_REVIEW)

    # -- Step 4: integrator --
    print("[4/5] Integrator...")
    decisions = []
    for cp in sorted(cfg.PENDING_REVIEW.glob("upd_*.json")):
        cand = json.loads(cp.read_text(encoding="utf-8"))
        dec = integrate_candidate(cand, cfg.KB_DIR, cfg.CHANGE_HISTORY_DIR, cfg.STABILITY_TIERS)
        route_candidate_file(cp, dec, cfg.APPLIED, cfg.REJECTED, cfg.PENDING_HUMAN)
        decisions.append(dec)
    by_action = Counter(d.action for d in decisions)
    print(f"       {dict(by_action)}")

    out = cfg.PROJECT_ROOT / "analysis" / "pipeline_v2_summary.json"
    out.write_text(json.dumps({
        "evaluator": summary,
        "distiller_count": len(candidates),
        "llm_approved": len(approved),
        "llm_rejected": len(rejected_by_llm),
        "integrator_decisions": {k: v for k, v in by_action.items()},
    }, indent=2, ensure_ascii=False))
    print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
