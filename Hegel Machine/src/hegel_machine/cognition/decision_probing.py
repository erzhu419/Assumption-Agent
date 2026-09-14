"""Paid history interventions for two different executable decision explanations."""

from itertools import combinations
import json
from statistics import mean

from .branch_support import history_branch_support
from .decision_frontier import _alternative_condition
from .evaluation import predict
from .models import Observation, Probe
from .probing import construct_probes
from .risk_search import trajectory_blocks
from .selection import representative_probe
from .uncertainty import action_and_interval


def _key(probe):
    return tuple(tuple(sorted(point.items())) for point in probe.sequence)


def decision_probe_condition(spec, policies):
    """The executable question's public structure, before paid design work."""
    alternatives = list({json.dumps(_alternative_condition(spec, candidate, model), sort_keys=True):
                         _alternative_condition(spec, candidate, model)
                         for candidate, model in policies}.values())
    alternatives.sort(key=lambda item: json.dumps(item, sort_keys=True))
    depth = max((history_branch_support(candidate.program, ())["required_depth"]
                 for candidate, _ in policies), default=0)
    return {"alternatives": alternatives, "history_depth": depth,
            "preparation": "different_control_histories_same_current_control"}


def _empirical_paths(observations, length):
    blocks = trajectory_blocks(observations)
    indices = sorted({0, len(blocks) // 2, len(blocks) - 1})
    return [blocks[index][:length] for index in indices]


def _terminal_actions(probe, generator, residuals, path, policies, fallback, costs):
    """Policies see the same possible observations, not their own past actions.

    A candidate supplies one possible response mechanism. Its errors on an
    acquired trajectory supply an empirical innovation path; uncontrolled
    measurements travel with that path. These are design scenarios, not draws
    from a known outcome distribution and not observations of the new probe.
    """
    history = []
    for index, point in enumerate(probe.sequence):
        features = {**path[index % len(path)].features, **point}
        row = Observation(features, 0., tuple(history))
        if index == probe.cost - 1:
            return [action_and_interval(candidate.program, row, fallback, model, costs)[0]
                    for candidate, model in policies]
        response = predict(generator.program, row, fallback) + residuals[index % len(residuals)]
        history.append({**features, "__target__": response})


def plan_decision_probes(spec, policies, observations, fallback, budget, *, remaining,
                         round_number, seed=7, used=(), recalled_probes=()):
    """Return a payable history contrast and two untouched representative trials.

    ``policies`` contains already fitted ``(Candidate, uncertainty_model)``
    pairs, with the deployed policy first. Recalled experiments may contribute
    their actual control preparations. Only public controls, these programs,
    and already acquired response/measurement paths enter design selection.

    Every generator prediction and complete policy execution is charged to
    ``budget``. The returned observation quote reserves both representative
    trials before the targeted pair. Targeted outcomes can falsify or fit an
    explanation; only the subsequent representative trials can establish
    deployment gain. No simulated risk reduction is reported as a reward.
    """
    if len(policies) < 2 or not observations or not spec.feature_bounds or spec.decision_costs is None:
        return {}
    distinct = []
    for candidate, model in policies:
        if not any(candidate.program == old.program and model == old_model
                   for old, old_model in distinct):
            distinct.append((candidate, model))
    policies = distinct
    if len(policies) < 2:
        return {}
    candidates = [candidate for candidate, _ in policies]
    depth = max(history_branch_support(candidate.program, observations)["required_depth"]
                for candidate in candidates)
    if depth == 0 or depth >= spec.max_probe_length:
        return {}
    length = min(spec.max_probe_length, max(4, depth + 2))
    reference_length = min(spec.max_probe_length, max(8, 4 * depth + 4))
    if reference_length <= depth or remaining < 2 * (length + reference_length):
        return {}
    references = []
    for index in range(2):
        probe = representative_probe(spec, seed=seed, index=round_number * 2 + 6000 + index,
            remaining=reference_length, used=(*used, *references), candidates=candidates)
        if probe is None:
            return {}
        references.append(probe)

    middle = {name: (low + high) / 2 for name, (low, high) in spec.feature_bounds.items()}
    sources = [*recalled_probes, *construct_probes(spec, candidates,
        round_number=round_number, seed=seed, max_history=depth)]
    seen = {_key(probe) for probe in used}
    preparations = []
    for source in sources:
        prefix = source.sequence[:length - 1]
        if len(prefix) < depth:
            continue
        # The current control is held fixed. Only previous controllable inputs
        # differ, including when a useful preparation was recalled from memory.
        probe = Probe(f"decision_history_{round_number}_{len(preparations)}",
                      tuple({name: point[name] for name in middle} for point in prefix) + (middle,))
        key = _key(probe)
        if key not in seen:
            seen.add(key)
            preparations.append((probe, source.probe_id))
    if len(preparations) < 2:
        return {}

    started = budget.spent
    generators = list(dict.fromkeys(candidate.program for candidate in candidates))
    generators = [next(candidate for candidate in candidates if candidate.program == program)
                  for program in generators]
    paths = _empirical_paths(observations, length)
    # Prefer complete, shared scenarios over partially pricing one history.
    # A small allowance still compares both preparations under each possible
    # response mechanism; additional empirical paths are paid when affordable.
    minimum_pair_cost = 2 * len(generators) * (length - 1 + len(policies))
    chosen_paths = []
    for path in paths:
        residual_cost = len(generators) * len(path)
        if budget.remaining < residual_cost + minimum_pair_cost * (len(chosen_paths) + 1):
            break
        budget.spend(residual_cost)
        errors = [[row.target - predict(generator.program, row, fallback) for row in path]
                  for generator in generators]
        chosen_paths.append((path, errors))
    if not chosen_paths:
        return {}

    scored = []
    for probe, source_id in preparations:
        price = len(generators) * len(chosen_paths) * (probe.cost - 1 + len(policies))
        if not budget.spend(price):
            break
        outcomes = []
        for path, errors in chosen_paths:
            for generator, residuals in zip(generators, errors):
                outcomes.append(_terminal_actions(probe, generator, residuals, path,
                    policies, fallback, spec.decision_costs))
        scored.append((probe, source_id, outcomes))

    best = None
    for left, right in combinations(scored, 2):
        differences = [abs((after[index] - before[index]) - (after[0] - before[0]))
            for before, after in zip(left[2], right[2]) for index in range(1, len(policies))]
        disagreement = mean(differences)
        if disagreement <= 1e-12:
            continue
        price = left[0].cost + right[0].cost
        value = disagreement / price
        if best is None or value > best[0]:
            best = value, disagreement, left, right
    if best is None:
        return {}
    _, disagreement, left, right = best
    targets = [left[0], right[0]]
    condition = decision_probe_condition(spec, policies)
    return {"targeted_probes": targets, "reference_probes": references,
        "observation_cost": sum(probe.cost for probe in (*targets, *references)),
        "simulation_compute_cost": budget.spent - started,
        "comparison_compute_cost": 2 * sum(probe.cost for probe in references) * (len(policies) - 1),
        "design": {"policy": "decision_history_contrast", "condition": condition,
            "candidate_disagreement": disagreement, "target_control": middle,
            "preparations": [{"probe_id": probe.probe_id, "source_probe_id": source,
                              "controls": list(probe.sequence[:-1])}
                             for probe, source, _ in (left, right)],
            "empirical_paths": len(chosen_paths), "possible_mechanisms": len(generators),
            "evaluated_preparations": len(scored),
            "comparison_scope": "simulated_terminal_action_history_contrast",
            "gain_comparison_scope": "subsequent_paid_representative_trajectories"}}
