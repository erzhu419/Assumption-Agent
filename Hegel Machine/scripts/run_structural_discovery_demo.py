"""Run object, graph, event, scale, and structural-prior revision examples."""

import argparse
import json
from pathlib import Path

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import EvaluationContext, Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.structural_discovery import run_structural_research
from hegel_machine.cognition.structure_lab import StructureLab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional path for the compact JSON report")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    library, reports = KnowledgeLibrary(), []
    config = ResearchConfig(observation_budget=192, compute_budget=160000, max_rounds=1, seed=args.seed)
    for readout, variant in (("total", "exchange"), ("focus", "exchange"), ("window", "exchange"),
                             ("event", "exchange"), ("total", "sink"), ("focus", "anchored")):
        lab = StructureLab(readout=readout, variant=variant)
        result = run_structural_research(lab, config=config, library=library)
        before = dict(lab.costs)
        fresh = lab.evaluation_rows(objects=result["selected_objects"], scale=result["selected_scale"],
                                    seed=args.seed + 900, length=48)
        program = Program.from_dict(json.loads(json.dumps(result["episode"]["program"])))
        context = EvaluationContext(fresh[-1])
        program.predict(fresh[-1], context=context)
        reports.append({"readout": readout, "apparatus_variant": variant,
            "fresh_mse": loss(program, fresh, result["episode"]["fallback"], .5),
            "expression": program.describe(), "selected_objects": result["selected_objects"],
            "selected_scale": result["selected_scale"], "selection": result["selection"],
            "measurement_reconfiguration": result["measurement_reconfiguration"],
            "hypotheses": [{key: item[key] for key in ("kind", "status", "counterexamples")}
                           for item in result["discovery"]["hypotheses"]],
            "prior_assessments": [{"record_id": item["record_id"], "status": item["status"], "loss": item["loss"]}
                                  for item in result["prior_assessments"]],
            "boundary_refinements": result["boundary_refinements"],
            "research_costs": result["costs"],
            "evaluation_costs": {name: lab.costs[name] - before[name] for name in before},
            "prediction_execution": {"primitive_steps": context.primitive_steps,
                "call_evaluations": context.call_evaluations, "cache_hits": context.cache_hits},
            "core_observations": result["episode"]["observation_cost"],
            "wall_time_seconds": result["wall_time_seconds"]})
        # The next apparatus receives the same ordinary persisted knowledge.
        library = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    payload = {"seed": args.seed, "results": reports, "structural_priors": len(library.structural_priors),
               "supported_family": "scalar object exchange, shared finite-radius graph, window and event readouts",
               "remaining_uncertainty": "Interaction boundaries are finite intervals; untested laws remain open."}
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(json.dumps({"output": str(args.output.resolve()),
                          "results": [{"readout": row["readout"], "variant": row["apparatus_variant"],
                                       "fresh_mse": row["fresh_mse"], "scale": row["selected_scale"]}
                                      for row in reports]}, indent=2))
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
