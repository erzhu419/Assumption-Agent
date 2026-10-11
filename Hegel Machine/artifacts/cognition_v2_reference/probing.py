"""Choose experiments by disagreement over complete predicted trajectories."""

from __future__ import annotations

from statistics import pvariance
from random import Random

from .evaluation import predict
from .models import Candidate, Observation, Probe, SearchBudget, TaskSpec


def construct_probes(spec: TaskSpec, candidates: list[Candidate], *, round_number: int,
                     seed: int = 7, max_history: int = 4) -> list[Probe]:
    """Build interventions from program boundaries and alternate histories.

    This uses only the public controllable ranges and candidate programs. It
    does not consult a task generator or identify the hidden mechanism.
    """
    if not spec.feature_bounds:
        return []
    rng = Random(seed + round_number * 113)
    middle = {name: (low + high) / 2 for name, (low, high) in spec.feature_bounds.items()}
    length = min(spec.max_probe_length, max(4, max_history + 2))
    results = []

    def add(sequence):
        results.append(Probe(f"constructed_{round_number}_{len(results)}", tuple(sequence)))

    boundaries = {name: [] for name in spec.feature_bounds}

    def visit(program):
        if program.op in {"ge", "lt"}:
            left, right = program.children
            if left.op == "feature" and right.op == "const" and left.name in boundaries:
                boundaries[left.name].append(right.value)
        for child in program.children:
            visit(child)

    for candidate in candidates:
        visit(candidate.program)
    for name, (low, high) in spec.feature_bounds.items():
        ascending = [low + (high - low) * i / max(1, length - 1) for i in range(length)]
        add([{**middle, name: value} for value in ascending])
        add([{**middle, name: value} for value in reversed(ascending)])
        # Equal final input with different histories can split explanations
        # that agree on all currently observed memoryless points.
        add([{**middle, name: low if i < length - 1 else middle[name]} for i in range(length)])
        add([{**middle, name: high if i < length - 1 else middle[name]} for i in range(length)])
        for boundary in sorted(set(boundaries[name]))[:2]:
            epsilon = (high - low) / 25
            left, right = max(low, boundary - epsilon), min(high, boundary + epsilon)
            if low <= left < right <= high:
                add([{**middle, name: left if i % 2 == 0 else right} for i in range(length)])
    for _ in range(2):
        add([{name: rng.uniform(*spec.feature_bounds[name]) for name in spec.feature_names}
             for _ in range(length)])
    return results


def observed_classes(candidates: list[Candidate], observations: list[Observation],
                     fallback: float, budget: SearchBudget) -> list[list[str]]:
    """Report empirical tolerance groups without deleting their programs."""
    groups = {}
    for candidate in candidates:
        if not budget.spend(len(observations)):
            break
        vector = tuple(round(predict(candidate.program, row, fallback), 6) for row in observations)
        groups.setdefault(vector, []).append(candidate.candidate_id)
    return list(groups.values())


def choose_probe(probes: list[Probe], candidates: list[Candidate], fallback: float,
                 budget: SearchBudget) -> tuple[Probe | None, float]:
    best_probe, best_value = None, -1.0
    for probe in probes:
        cost = probe.cost * len(candidates)
        if not budget.spend(cost):
            break
        predictions = []
        for candidate in candidates:
            history, outcomes = [], []
            for features in probe.sequence:
                value = predict(candidate.program, Observation(features, 0.0, tuple(history)), fallback)
                outcomes.append(value)
                history.append({**features, "__target__": value})
            predictions.append(outcomes)
        # Average disagreement can reward a useful split even when some pair
        # of Boolean candidates agrees. Maximin pair distance cannot do this.
        value = sum(pvariance(column) for column in zip(*predictions)) / probe.cost
        if value > best_value:
            best_probe, best_value = probe, value
    return best_probe, max(0.0, best_value)
