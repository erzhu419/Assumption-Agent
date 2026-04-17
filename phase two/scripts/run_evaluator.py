"""Run the Phase 2 experience evaluator over all logged executions."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

import _config as cfg
from experience_evaluator.evaluator import evaluate_all, summarize


def main():
    print(f"Scanning {cfg.EXECUTIONS_DIR}")
    results = evaluate_all(cfg.EXECUTIONS_DIR, info_threshold=cfg.INFO_SCORE_THRESHOLD)
    summary = summarize(results)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out = cfg.PROJECT_ROOT / "analysis" / "evaluator_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "summary": summary,
        "results": [asdict(r) for r in results],
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
