"""Delete only the ordinary arm's unconditional two extra queries, one fixed seed.

All candidate code, ordering, fits, independent confirmation and final evaluation
remain unmodified. This is a diagnostic of the claimed measurement-cost necessity,
not a replacement preregistered experiment. No API calls or repo writes.
"""
from __future__ import annotations

import json
from pathlib import Path

import hegel_machine.cognition.open_python_revision as revision

ROOT = Path(__file__).resolve().parent
REPO = Path(revision.__file__).resolve().parents[3]
SEED = 5026101

original_init = revision.ScriptedOpenGrammarCodeProvider.__init__
changes = []


def patched_init(self, *, query_batches, candidate_specs, diagnostic_script=None, tolerance=.035):
    if candidate_specs and all(name.startswith("session_prior_") for name, code in candidate_specs):
        assert len(query_batches) == 2
        assert len(query_batches[1]) == 2
        changes.append({"candidate_names": [name for name, _ in candidate_specs],
                        "removed_query_points": query_batches[1],
                        "retained_query_count": len(query_batches[0])})
        query_batches = [query_batches[0]]
    original_init(self, query_batches=query_batches, candidate_specs=candidate_specs,
                  diagnostic_script=diagnostic_script, tolerance=tolerance)


def main():
    published = json.loads((REPO / "docs/evidence/v16_9_m2_audited_open_workspace_20261119/open_author_revision_results.json").read_text())
    published = next(r for r in published if r["seed"] == SEED)
    revision.ScriptedOpenGrammarCodeProvider.__init__ = patched_init
    try:
        controlled = revision.run_open_revision_chain_experiment(SEED, ROOT / "open_forced_queries_workspace")
    finally:
        revision.ScriptedOpenGrammarCodeProvider.__init__ = original_init
    comparison = []
    for target, arms in controlled["downstream_targets"].items():
        for arm, row in arms.items():
            baseline = published["downstream_targets"][target][arm]
            comparison.append({
                "target": target, "arm": arm,
                "published_pass": baseline["strict_pass"], "controlled_pass": row["strict_pass"],
                "published_rmse": baseline["eval_rmse"], "controlled_rmse": row["eval_rmse"],
                "published_measurement_cost": baseline["costs"]["measurement_cost"],
                "controlled_measurement_cost": row["costs"]["measurement_cost"],
                "published_compute_cost": baseline["costs"]["compute_cost"],
                "controlled_compute_cost": row["costs"]["compute_cost"],
                "published_composite_cost": baseline["costs"]["composite_cost"],
                "controlled_composite_cost": row["costs"]["composite_cost"],
            })
    result = {"seed": SEED, "commit": "612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8",
              "controlled_changes": changes, "comparison": comparison, "controlled_full": controlled}
    (ROOT / "open_forced_queries_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
