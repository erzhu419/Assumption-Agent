#!/usr/bin/env python3
"""Measure autonomous world-claim discovery, transfer, refutation and reduction."""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from random import Random
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import EvaluationContext, Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.meta_prior_lab import ScalarApparatus, make_meta_prior_sequence as sequence
from hegel_machine.cognition.models import Probe, ResearchConfig


def fresh_rows(environment, seed, count=256):
    """The scorer queries a separate apparatus after all learning has ended."""
    rng, rows = Random(seed), []
    while len(rows) < count:
        length = min(environment.spec.max_probe_length, count-len(rows))
        probe = Probe(f"evaluation_{len(rows)}", tuple(
            {name: rng.uniform(*environment.spec.feature_bounds[name])
             for name in environment.spec.feature_names} for _ in range(length)))
        rows.extend(environment.observe(probe))
    return rows


def study(seeds=(31001, 51001, 71001), profiles=("quadratic", "nested_readout")):
    from hegel_machine.cognition.meta_prior_research import MetaPriorResearch

    config = ResearchConfig(observation_budget=96, compute_budget=60000,
                            minimum_improvement=1e-7, max_rounds=5)
    reports, final_libraries = [], []
    for profile in profiles:
        for seed in seeds:
            for arm in ("shared", "cold"):
                library = KnowledgeLibrary()
                for position, environment in enumerate(sequence(seed, profile)):
                    if arm == "cold":
                        library = KnowledgeLibrary()
                    result = MetaPriorResearch(replace(config, seed=seed+1009*position), library).run(environment)
                    reports.append({"profile": profile, "seed": seed, "arm": arm, "position": position,
                        "actual_observations": environment.observation_count,
                        "knowledge_size_before_next_task": len(library.concepts),
                        "research": deepcopy(result)})
                    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
                final_libraries.append({"profile": profile, "seed": seed, "arm": arm, "library": library.to_dict()})

    # No frozen program receives any evaluation data or subsequent revisions.
    for report in reports:
        environment = sequence(report["seed"], report["profile"])[report["position"]]
        rows = fresh_rows(environment, report["seed"]+1000003+report["position"]*137)
        result = report["research"]
        program = Program.from_dict(result["program"])
        call_evaluations, primitive_steps, source_ids = 0, 0, set()
        for row in rows:
            context = EvaluationContext(row)
            program.predict(row, context=context)
            call_evaluations += context.call_evaluations
            primitive_steps += context.primitive_steps
            source_ids.update(definition.source_concept_id for definition, _ in context._calls
                              if definition.source_concept_id)
        nodes = result["research_state"].get("graph", {}).get("nodes", {})
        report["evaluation"] = {"mse": loss(program, rows, result["fallback"], .5),
            "observations": environment.observation_count, "call_evaluations": call_evaluations,
            "primitive_steps": primitive_steps, "called_source_ids": sorted(source_ids)}
        report["recursive_questions"] = {"nodes": len(nodes),
            "dependent_nodes": sum(bool(node.get("depends_on")) for node in nodes.values()),
            "residual_nodes": sum(bool(node.get("residual_base")) for node in nodes.values()),
            "selected_dependent_answer": any(node.get("depends_on") and node.get("answer")
                and Program.from_dict(node["answer"]).expanded().simplified() == program.expanded().simplified()
                for node in nodes.values())}

    def summarize(selected):
        return {"tasks": len(selected), "completed": sum(report["research"]["status"] == "completed" for report in selected),
            "mean_mse": mean(report["evaluation"]["mse"] for report in selected),
            "max_mse": max(report["evaluation"]["mse"] for report in selected),
            "costs": {name: sum(report["research"]["costs"][name] for report in selected)
                      for name in ("observation_cost", "compute_cost")},
            "tasks_executing_learned_calls": sum(bool(report["evaluation"]["call_evaluations"]) for report in selected),
            "tasks_with_recursive_residuals": sum(bool(report["recursive_questions"]["residual_nodes"]) for report in selected),
            "tasks_selecting_dependent_answers": sum(report["recursive_questions"]["selected_dependent_answer"] for report in selected)}

    aggregate = {arm: summarize([report for report in reports if report["arm"] == arm]) for arm in ("shared", "cold")}
    by_profile = {profile: {arm: summarize([report for report in reports if report["arm"] == arm
                                          and report["profile"] == profile]) for arm in ("shared", "cold")}
                  for profile in profiles}
    return {"protocol": {"seeds": list(seeds), "profiles": list(profiles), "config": asdict(config),
        "tasks_per_sequence": 4, "arms": ["shared", "cold"], "evaluation_rows": 256,
        "learning_completed_before_evaluation": True,
        "public_interface": "TaskSpec and observe; no mechanism family or target coefficients",
        "evaluation_stage_order": ["low_rank", "changed_readout", "orthogonal_break", "simple_response"]},
        "aggregate": aggregate, "by_profile": by_profile, "tasks": reports, "final_libraries": final_libraries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[31001, 51001, 71001])
    parser.add_argument("--profiles", nargs="+", choices=("quadratic", "nested_readout"),
                        default=["quadratic", "nested_readout"])
    args = parser.parse_args()
    report = study(args.seeds, args.profiles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"output": str(args.output.resolve()), "aggregate": report["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
