#!/usr/bin/env python3
"""Run positive, reusable and broken finite structures with one shared library."""

import argparse
import json
from pathlib import Path

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.mechanism_discovery import run_mechanism_research
from hegel_machine.cognition.mechanism_lab import MechanismLab
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    library = KnowledgeLibrary()
    results = []
    for phase, variant in (("source", "regular"), ("transfer", "regular"), ("change", "broken")):
        if phase == "transfer":
            library = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
        for family in ("low_rank", "modules", "dynamics", "topology", "order"):
            lab = MechanismLab(family, variant)
            report = run_mechanism_research(lab, library=library,
                config=ResearchConfig(seed=89 if phase == "transfer" else 7))
            evaluator = MechanismLab(family, variant)
            rows = evaluator.evaluation_rows(seed=1701, length=32)
            program = Program.from_dict(report["episode"]["program"])
            loss = sum((program.predict(row) - row.target)**2 for row in rows) / len(rows)
            found = report["discovery"]
            results.append({"phase": phase, "family": family, "variant": variant,
                "status": report["status"], "fresh_loss": loss, "costs": report["costs"],
                "evaluation_costs": evaluator.costs,
                "hypotheses": [{"kind": item["kind"], "status": item["status"]} for item in found.get("hypotheses", [])],
                "rank": found.get("rank"), "slow_dimension": found.get("slow_dimension"),
                "coarse_loss": found.get("coarse_validation_loss"), "regimes": found.get("regimes"),
                "boundary": found.get("change_boundary"),
                "retained_components": found.get("retained_components", []),
                "applied_concept_ids": found.get("applied_concept_ids", [])})
    summary = {"scope": "Five finite controllable apparatuses; source, restored-library transfer, structural changes",
               "library_restored": True, "cases": results,
               "all_completed": all(row["status"] == "completed" for row in results),
               "maximum_fresh_loss": max(row["fresh_loss"] for row in results),
               "research_costs": {name: sum(row["costs"][name] for row in results)
                                  for name in ("observation_cost", "measurement_cost", "compute_cost")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
