"""Executable distinguishing questions that survive a bounded research visit."""

from dataclasses import asdict

from .models import Probe
from .probing import transition_probe
from .selection import representative_probe


def plan_experiment_frontier(spec, candidates, observations, queried, config, budget, *, turn):
    """Retain a concrete payable experiment, including its subsequent comparison.

    A zero fitting error does not resolve an interval containing several laws.
    This plan is rebuilt from the final paid evidence, then consumed on resume.
    """
    if not candidates or not config.active_probes or not config.transition_probes:
        return {}
    comparison = representative_probe(spec, seed=config.seed, index=turn + 3000,
        remaining=config.observation_budget, used=queried, candidates=candidates,
        adaptive_horizon=config.adaptive_horizon)
    if comparison is None:
        return {}
    probe, details = transition_probe(spec, candidates, observations, budget=budget,
        remaining=config.observation_budget - comparison.cost, round_number=turn + 3000,
        used=queried, joint_controls=config.joint_state_probes,
        followup_observation_cost=comparison.cost, remaining_rounds=1)
    uncertainty = max(details.get("unresolved_boundary_width", 0.),
                      details.get("candidate_disagreement", 0.))
    if probe is None or uncertainty <= 2 * config.minimum_improvement:
        return {}
    return {"probe": asdict(probe), "design": details, "uncertainty": uncertainty,
            "evidence_rows": len(observations), "observation_cost": probe.cost,
            "comparison_observation_cost": comparison.cost,
            "question": "Which remaining startup or mature transition law matches a new controlled trajectory?"}


def frontier_probe(frontier):
    payload = frontier["probe"]
    return Probe(payload["probe_id"], tuple(dict(row) for row in payload["sequence"]))


def frontier_is_payable(frontier, observation_budget):
    return bool(frontier and frontier["observation_cost"]
                + frontier["comparison_observation_cost"] <= observation_budget)
