"""Save a short comparison of already fitted, still plausible decision rules."""

from copy import deepcopy
import json
from math import sqrt
from statistics import mean, stdev

from .risk_search import trajectory_blocks
from .method_conditions import method_profile
from .uncertainty import action_and_interval, decision_loss
from .decision_evidence import decision_evidence_quote


def _policy(candidate, model):
    return {"candidate": {"candidate_id": candidate.candidate_id,
                "program": candidate.program.to_dict(), "operation": candidate.operation,
                "rationale": candidate.rationale, "parents": list(candidate.parents),
                "memory_id": candidate.memory_id}, "model": deepcopy(model)}


def _alternative_condition(spec, candidate, model):
    representation = method_profile(spec, [], candidate)["representation"]
    feature = model.get("feature")
    return {"representation": {key: representation[key] for key in (
                "kind", "families", "history_depth", "structure")},
            "decision_rule": {"kind": model.get("kind", "nominal"),
                "feature_role": spec.feature_names.index(feature) if feature in spec.feature_names else None,
                "transform": model.get("transform"), "action_rule": model.get("action_rule", "prediction"),
                "calibration_method": model.get("calibration_method")}}


def decision_evidence_frontier(spec, workspace, config, previous, candidate, evidence):
    """Retain the exact unresolved pair and its paid comparison for later work."""
    depth = max(previous.program.history_depth, candidate.program.history_depth)
    length = min(max(8, 4 * depth + 4), spec.max_probe_length)
    quote = decision_evidence_quote(evidence, trajectory_length=length,
        observation_remaining=config.observation_budget, compute_remaining=config.compute_budget)
    if not quote['needs_evidence']:
        return {}
    return {'policies': [_policy(previous, evidence['previous_model']),
                         _policy(candidate, evidence['candidate_model'])],
        'evidence_states': [deepcopy(evidence)],
        'design': {'policy': 'unresolved_decision_comparison',
            'alternatives': [_alternative_condition(spec, candidate, evidence['candidate_model'])],
            'unresolved_comparison': True, 'comparison_scope': 'frozen_complete_decision_followup',
            'comparisons': [{'policy_index': 1, **deepcopy(evidence['comparison'])}]},
        'evidence_rows': len(workspace.observations), 'probe_lengths': quote['probe_lengths'],
        'observation_cost': quote['observation_cost'], 'comparison_compute_cost': quote['compute_cost'],
        'compute_cost': quote['compute_cost'],
        'question': 'Do additional representative trajectories resolve the conflicting complete decisions?'}


def plan_decision_frontier(spec, workspace, config, budget, *, proposed_model=None):
    """Price new paired trajectories for a concrete unresolved decision choice.

    Old acquired trajectories screen already fitted programs and calibration
    rules. Their possible gain nominates a question, not an adoption or an
    expected payoff. The caller must compare these frozen policies on new
    complete trajectories before changing the deployed decision.
    """
    current = workspace.incumbent
    if spec.decision_costs is None or current is None:
        return {}
    rows = workspace.reference or workspace.observations
    if not rows:
        return {}
    deployed = workspace.uncertainty_model or {"kind": "nominal"}
    proposals = []
    if proposed_model is not None and proposed_model != deployed:
        proposals.append((current, proposed_model))
    programs = {current.program}
    for candidate in [*workspace.bank, workspace.evidence_champion, workspace.parent]:
        if candidate is not None and candidate.program not in programs:
            programs.add(candidate.program)
            proposals.append((candidate, {"kind": "nominal"}))
    if not proposals or not budget.spend(len(rows)):
        return {}

    costs = spec.decision_costs
    before_actions = [action_and_interval(current.program, row, workspace.fallback,
                                         deployed, costs)[0] for row in rows]
    before_losses = [decision_loss(action, row.target, costs)
                     for action, row in zip(before_actions, rows)]
    sizes = [len(block) for block in trajectory_blocks(rows)]

    def block_means(values):
        result, start = [], 0
        for length in sizes:
            result.append(mean(values[start:start + length]))
            start += length
        return result

    before_trials = block_means(before_losses)
    plausible = []
    for candidate, model in proposals:
        if not budget.spend(len(rows)):
            break
        actions = [action_and_interval(candidate.program, row, workspace.fallback,
                                       model, costs)[0] for row in rows]
        disagreement = mean(abs(before - after) for before, after in zip(before_actions, actions))
        if disagreement <= 1e-12:
            continue
        after_trials = block_means([decision_loss(action, row.target, costs)
                                    for action, row in zip(actions, rows)])
        gains = [before - after for before, after in zip(before_trials, after_trials)]
        gain = mean(gains)
        margin = 2 * stdev(gains) / sqrt(len(gains)) if len(gains) > 1 else 0.
        # A large residual or a different action alone does not establish a
        # question worth buying. At least one acquired trajectory must favor
        # the alternative, and the paired variation must leave a useful gain
        # plausible. Consistently inferior alternatives are already settled.
        if max(gains) <= config.minimum_improvement or gain + margin <= config.minimum_improvement:
            continue
        plausible.append((candidate, model, {"before_loss": mean(before_trials),
            "after_loss": mean(after_trials), "gain": gain, "support_margin": margin,
            "trajectory_gains": gains, "candidate_disagreement": disagreement}))
    plausible.sort(key=lambda row: (-row[2]["gain"], row[0].program.size))
    selected = plausible[:2]
    if not selected:
        return {}
    policies = [_policy(current, deployed), *[_policy(candidate, model) for candidate, model, _ in selected]]
    depth = max([current.program.history_depth, *(candidate.program.history_depth for candidate, _, _ in selected)])
    length = min(max(8, 4 * depth + 4), spec.max_probe_length)
    if length <= depth or 2 * length > config.observation_budget:
        return {}
    observation_cost = 2 * length
    comparison_cost = 2 * observation_cost * len(selected)
    alternatives = []
    for candidate, model, _ in selected:
        condition = _alternative_condition(spec, candidate, model)
        if condition not in alternatives:
            alternatives.append(condition)
    alternatives.sort(key=lambda item: json.dumps(item, sort_keys=True))
    return {"policies": policies,
            "design": {"policy": "unresolved_decision_comparison",
                "alternatives": alternatives,
                "candidate_disagreement": max(item[2]["candidate_disagreement"] for item in selected),
                "comparisons": [{"policy_index": index, **details}
                                for index, (_, _, details) in enumerate(selected, 1)],
                "comparison_scope": "already_acquired_trajectory_screening"},
            "evidence_rows": len(workspace.observations),
            "probe_lengths": [length, length], "observation_cost": observation_cost,
            "comparison_compute_cost": comparison_cost, "compute_cost": comparison_cost,
            "question": "Which fitted prediction and calibration rule makes better decisions on two new trajectories?"}
