#!/usr/bin/env python3
"""Learn, call, compose and restore executable concepts on three small tasks.

Run directly with --output artifacts/callable_concept_demo.json. The comparison
keeps the candidate-row budget equal and measures expression execution work
separately. No model service or external dataset is needed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import EvaluationContext, Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.sparse_search import sparse_candidates


@dataclass
class Fitted:
    candidate: Candidate
    training_loss: float
    budget: int
    search_cost: int
    selection_cost: int
    candidates: int


def measured_rows(seed, count, level, names=("x", "z")):
    rng, rows = Random(seed), []
    for _ in range(count):
        x, z = rng.uniform(-1., 1.), rng.uniform(-1., 1.)
        target = x * x + x * z
        for _ in range(level - 1):
            target = target * target + target
        rows.append(Observation(dict(zip(names, (x, z))), target))
    return rows


def task_spec(identifier, names=("x", "z")):
    return TaskSpec(identifier, "Infer the measured numerical response.", names, (), (),
                    feature_bounds={name: (-1., 1.) for name in names})


def fit_rows(rows, *, atoms=(), budget=3600):
    # Both arms reserve the same part of the budget for selecting a candidate
    # on their shared acquired observations. Final evaluation is separate.
    search = SearchBudget(3 * budget // 4)
    selection = SearchBudget(budget - search.limit)
    candidates = sparse_candidates(rows, budget=search, max_history=0, additional_terms=atoms)
    measured = []
    for candidate in candidates:
        if not selection.spend(len(rows)):
            break
        measured.append((loss(candidate.program, rows, 0., .5), candidate.program.size, candidate))
    if not measured:
        raise ValueError("the demonstration budget cannot generate and compare a candidate")
    error, _, chosen = min(measured, key=lambda item: (item[0], item[1]))
    return Fitted(chosen, error, budget, search.spent, selection.spent, len(measured))


def call_structure(program):
    definitions, sources = set(), set()

    def visit(node, depth=0):
        maximum = depth
        if node.op == "call":
            definitions.add(node.definition.definition_id)
            if node.definition.source_concept_id:
                sources.add(node.definition.source_concept_id)
            maximum = max(maximum, visit(node.definition.body, depth + 1))
        return max([maximum, *(visit(child, depth) for child in node.children)])

    depth = visit(program)
    return {"definition_ids": sorted(definitions), "source_concept_ids": sorted(sources),
            "maximum_call_depth": depth, "contains_call": bool(definitions)}


def execution_comparison(program, rows):
    expanded = program.expanded()
    totals = {name: {"primitive_steps": 0, "call_evaluations": 0, "cache_hits": 0}
              for name in ("callable", "expanded")}
    largest_difference = 0.
    for row in rows:
        contexts = {name: EvaluationContext(row) for name in totals}
        direct = contexts["callable"].evaluate(program)
        reference = contexts["expanded"].evaluate(expanded)
        largest_difference = max(largest_difference, abs(direct - reference))
        for name, context in contexts.items():
            for key in totals[name]:
                totals[name][key] += getattr(context, key)
    return {"rows": len(rows), "max_absolute_prediction_difference": largest_difference, **totals}


def learn_three_tasks(seed=113, budget=3600):
    library = KnowledgeLibrary()
    a_rows = measured_rows(seed, 40, 1)
    a = fit_rows(a_rows, budget=budget)
    a_id, _ = library.learn(a.candidate, task_spec("A"), observations=a_rows)

    b_rows = measured_rows(seed + 101, 40, 2)
    b_spec = task_spec("B")
    b_atoms = library.callable_atoms(b_spec, limit=8)
    b = fit_rows(b_rows, atoms=b_atoms, budget=budget)
    raw = fit_rows(b_rows, budget=budget)
    b_id, _ = library.learn(b.candidate, b_spec, observations=b_rows)
    checkpoint = json.loads(json.dumps(library.to_dict()))
    restored = KnowledgeLibrary.from_dict(checkpoint)

    c_spec = task_spec("C", ("u", "v"))
    c_rows = measured_rows(seed + 211, 40, 3, c_spec.feature_names)
    c_atoms = restored.callable_atoms(c_spec, limit=8)
    c = fit_rows(c_rows, atoms=c_atoms, budget=budget)
    c_id, _ = restored.learn(c.candidate, c_spec, observations=c_rows)
    return {"fits": {"A": a, "B_callable": b, "B_raw": raw, "C": c},
            "fresh": {"A": measured_rows(seed + 307, 64, 1),
                      "B_callable": measured_rows(seed + 401, 64, 2),
                      "B_raw": measured_rows(seed + 401, 64, 2),
                      "C": measured_rows(seed + 503, 64, 3, c_spec.feature_names)},
            "source_ids": {"A": a_id, "B": b_id, "C": c_id},
            "checkpoint": checkpoint, "library": restored,
            "atoms": {"B": b_atoms, "C": c_atoms}}


def run_demo(seed=113, budget=3600):
    run = learn_three_tasks(seed, budget)
    results = {}
    for name, fitted in run["fits"].items():
        program, fresh = fitted.candidate.program, run["fresh"][name]
        results[name] = {"training_rows": 40, "evaluation_rows": len(fresh),
                         "training_mse": fitted.training_loss,
                         "evaluation_mse": loss(program, fresh, 0., .5),
                         "candidate_row_budget": fitted.budget,
                         "search_candidate_rows": fitted.search_cost,
                         "selection_candidate_rows": fitted.selection_cost,
                         "total_candidate_rows": fitted.search_cost + fitted.selection_cost,
                         "candidates_compared": fitted.candidates,
                         "expression": program.describe(), **call_structure(program)}
    saved_b = next(concept for concept in KnowledgeLibrary.from_dict(run["checkpoint"]).concepts
                   if concept.concept_id == run["source_ids"]["B"])
    original_b = run["fits"]["B_callable"].candidate.program
    round_trip_difference = max(abs(original_b.predict(row) - saved_b.program.predict(row))
                                for row in run["fresh"]["B_callable"])
    return {"experiment": "learned_callable_concept_composition", "seed": seed,
            "tasks": {"A": "g(x,z) = x*x + x*z", "B": "g*g + g",
                      "C": "h*h + h with h learned on B and controls renamed to u,v"},
            "source_concept_ids": run["source_ids"], "results": results,
            "checkpoint_max_prediction_difference": round_trip_difference,
            "execution": {name: execution_comparison(run["fits"][name].candidate.program, run["fresh"][name])
                          for name in ("B_callable", "C")},
            "interpretation": {
                "comparison": "B arms share observations, candidate-row cap and final evaluation; A acquisition is reported separately.",
                "execution_counts": "Primitive steps and call cache hits measure actual expression execution; they do not replace candidate-row accounting.",
                "scope": "A local composition example: the raw sparse dictionary is quadratic, while the learned coordinate composes a fourth-degree law. This does not establish superiority over unrestricted program search or code agents.",
            }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--budget", type=int, default=3600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_demo(args.seed, args.budget)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
