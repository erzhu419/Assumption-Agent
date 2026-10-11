"""Executable dependent revision plans and bounded recursive investigation."""

from __future__ import annotations

from dataclasses import dataclass, field

from .evaluation import initial_prediction, loss
from .models import Candidate, Observation, ResearchQuestion, SearchBudget
from .synthesis import synthesize


@dataclass
class RevisionPlan:
    plan_id: str
    question: str
    operations: tuple[str, ...]
    steps: list[dict] = field(default_factory=list)

    def to_dict(self):
        return {"plan_id": self.plan_id, "question": self.question,
                "operations": list(self.operations), "steps": self.steps}


def revision_plans(question: ResearchQuestion, *, joint: bool = True,
                   depth: int = 2) -> list[RevisionPlan]:
    operations = list(dict.fromkeys(question.operations))
    plans = []
    if not joint:
        return [RevisionPlan(f"single_{op}", question.question, (op,)) for op in operations]
    if "refit" in operations:
        plans.append(RevisionPlan("refit", "Test parameter explanations", ("refit",)))
    if "representation" in operations or "composition" in operations:
        sequence = ("representation", "composition") if depth > 1 else ("representation",)
        plans.append(RevisionPlan("representation_then_relation",
                                  "Construct a representation, then solve the relation in that representation", sequence))
    if "relation" in operations:
        sequence = ("relation", "composition") if depth > 1 else ("relation",)
        plans.append(RevisionPlan("relation_then_composition",
                                  "Retain useful intermediate predicates for a composed explanation", sequence))
    if "robustification" in operations:
        sequence = ("refit", "robustification") if depth > 1 else ("robustification",)
        plans.append(RevisionPlan("model_then_uncertainty",
                                  "Separate systematic model error from residual uncertainty", sequence))
    if "idealization" in operations:
        plans.append(RevisionPlan("simplification", "Reduce repeated computation even when prediction fits", ("idealization",)))
    return plans


def execute_revision_plan(plan: RevisionPlan, observations: list[Observation], *, incumbent,
                          budget: SearchBudget, risk_quantile: float, max_history: int = 4,
                          beam_width: int = 3) -> list[Candidate]:
    """A child returns executable candidates, including temporarily worse ones.

    Intermediate branches are evaluated for exploration; only the controller's
    final comparison decides whether they replace the working hypothesis.
    """
    beam = [incumbent]
    all_candidates = []
    fallback = initial_prediction(observations, risk_quantile)
    for index, operation in enumerate(plan.operations):
        start = budget.spent
        remaining_steps = len(plan.operations) - index
        quota = max(0, budget.remaining // remaining_steps)
        generated = []
        branches = beam[:beam_width] if index else beam[:1]
        for parent in branches:
            rows = observations
            if operation in {"representation", "composition"}:
                # Different history lengths are real alternative observation
                # representations. Each is fitted only where the history exists.
                depths = [depth for depth in range(1, max_history + 1)
                          if sum(len(row.history) >= depth for row in observations) >= 4]
                desired = max(depths, default=0)
                if desired:
                    rows = [row for row in observations if len(row.history) >= desired]
            local = SearchBudget(min(budget.remaining, quota // (2 * max(1, len(branches)))))
            batch = synthesize(rows, operations=(operation,), budget=local, incumbent=parent,
                               risk_quantile=risk_quantile, max_history=max_history,
                               seed_programs=tuple(beam) if index else ())
            budget.spend(local.spent)
            generated.extend(Candidate(f"{plan.plan_id}:{index}:{c.candidate_id}", c.program, operation,
                                       c.rationale, (parent.describe(),)) for c in batch[:12])
        ranked = []
        for candidate in generated:
            if not budget.spend(len(observations)):
                break
            measured = loss(candidate.program, observations, fallback, risk_quantile)
            ranked.append((measured, candidate.program.size, candidate))
        ranked.sort(key=lambda item: (item[0], item[1]))
        # At least one intermediate from each expression size survives; a short
        # tied atom should not erase every useful composed representation.
        survivors = []
        seen_sizes = set()
        for measured, size, candidate in ranked:
            if not survivors or size not in seen_sizes:
                survivors.append(candidate)
                seen_sizes.add(size)
            if len(survivors) >= beam_width:
                break
        if survivors:
            beam = [candidate.program for candidate in survivors]
        all_candidates.extend(generated)
        plan.steps.append({"step": index, "operation": operation,
                           "depends_on": [] if index == 0 else [index - 1],
                           "status": "evaluated" if ranked else "budget_exhausted",
                           "compute_cost": budget.spent - start,
                           "generated_candidates": len(generated),
                           "best_training_loss": ranked[0][0] if ranked else None,
                           "intermediate_programs": [p.describe() for p in beam]})
        if not ranked:
            break
    return all_candidates
