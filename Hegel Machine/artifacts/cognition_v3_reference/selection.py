"""Representative experiments and prospective comparison of executable models."""

from dataclasses import dataclass
from random import Random
from statistics import mean

from .evaluation import loss
from .language import Program
from .models import Candidate, Observation, Probe, SearchBudget
from .synthesis import synthesize


def representative_probe(spec, *, seed, index, remaining, used=()):
    """A stratified, independently permuted control sequence, with no labels."""
    if not spec.feature_bounds:
        used_ids = {probe.probe_id for probe in used}
        available = [probe for probe in spec.available_probes
                     if probe.probe_id not in used_ids and probe.cost <= remaining]
        return available[index % len(available)] if available else None
    length = min(8, spec.max_probe_length, remaining)
    if length < 3:
        return None
    rng = Random(seed + 700_001 + index * 1777)
    columns = {}
    for name in spec.feature_names:
        low, high = spec.feature_bounds[name]
        values = [low + (high - low) * (i + rng.random()) / length for i in range(length)]
        rng.shuffle(values)
        columns[name] = values
    return Probe(f"representative_{index}", tuple(
        {name: columns[name][i] for name in spec.feature_names} for i in range(length)))


@dataclass
class AssessedCandidate:
    candidate: Candidate
    score: float
    reference_loss: float
    evidence_loss: float
    fresh_loss: float | None = None
    origins: tuple[Candidate, ...] = ()


def rank_candidates(candidates, observations, reference, *, fallback, q, budget,
                    complexity_weight, fresh=(), fresh_weight=0.5):
    """Screen on evidence, then compare on representative and new observations.

    Reference queries cover the public control domain. Targeted queries remain
    useful for fitting and falsification without defining the task distribution.
    """
    ranked = []
    programs = {}
    for candidate in candidates:
        origins = programs.setdefault(candidate.program, [])
        if all(origin.candidate_id != candidate.candidate_id for origin in origins):
            origins.append(candidate)
    for origins in programs.values():
        candidate = origins[0]
        units = len(observations) + len(reference) + len(fresh)
        if not budget.spend(units):
            break
        empirical = loss(candidate.program, observations, fallback, q)
        representative = loss(candidate.program, reference, fallback, q) if reference else empirical
        prospective = loss(candidate.program, fresh, fallback, q) if fresh else None
        score = 0.8 * representative + 0.2 * empirical
        if prospective is not None:
            score = (1.0 - fresh_weight) * score + fresh_weight * prospective
        score += complexity_weight * candidate.program.size
        ranked.append(AssessedCandidate(candidate, score, representative, empirical, prospective, tuple(origins)))
    return sorted(ranked, key=lambda row: (row.score, row.candidate.program.size))


def warm_start_candidates(candidates, observations, *, budget, q, complexity_weight):
    """Learn startup predictions rather than discarding all current signals."""
    startup_models = {}
    results = []
    for candidate in candidates:
        depth = candidate.program.minimum_history_depth
        if depth == 0:
            continue
        if depth not in startup_models:
            rows = [row for row in observations if len(row.history) < depth]
            if len(rows) < 3:
                startup_models[depth] = []
                continue
            current_rows = [Observation(row.features, row.target) for row in rows]
            local = SearchBudget(min(budget.remaining, 360))
            generated = synthesize(current_rows, operations=("refit", "composition", "relation"),
                                   risk_quantile=q, budget=local)
            budget.spend(local.spent)
            # These are fitting scores; the controller still measures the full
            # guarded program on an experiment not used to build it.
            scored = rank_candidates(generated[:12], current_rows, [], fallback=mean(row.target for row in rows),
                                     q=q, budget=budget, complexity_weight=complexity_weight)
            startup_models[depth] = [row.candidate.program for row in scored[:2]]
        for startup in startup_models[depth]:
            program = Program("if", children=(Program("available", lag=depth), candidate.program, startup))
            results.append(Candidate(f"startup:{candidate.candidate_id}", program,
                                     "abstract_reuse" if candidate.operation == "reuse" else candidate.operation,
                                     "Learn the initial regime and retain the mature historical representation",
                                     candidate.parents, candidate.memory_id))
    return results


class PlanAllocation:
    """Revise search allocations from measured gain per unit of work."""
    def __init__(self):
        self.experience = {}

    def allocate(self, plans, allowance):
        weights = []
        for plan in plans:
            previous = self.experience.get(plan.plan_id)
            weight = 1.0 if previous is None else 0.5 + min(3.0, max(0.0, previous) * allowance)
            weights.append(weight)
        total = sum(weights) or 1
        return [int(allowance * weight / total) for weight in weights]

    def update(self, plans, assessed, baseline_loss):
        for plan in plans:
            outcomes = [row.fresh_loss for row in assessed
                        if any(origin.candidate_id.removeprefix("startup:").startswith(plan.plan_id + ":")
                               for origin in (row.origins or (row.candidate,)))
                        and row.fresh_loss is not None]
            cost = sum(step["compute_cost"] for step in plan.steps)
            if outcomes and cost:
                rate = (baseline_loss - min(outcomes)) / cost
                self.experience[plan.plan_id] = 0.5 * self.experience.get(plan.plan_id, rate) + 0.5 * rate
