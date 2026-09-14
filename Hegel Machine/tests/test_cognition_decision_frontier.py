"""Already fitted decision choices receive a small, measured next experiment."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.decision_frontier import plan_decision_frontier
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_conditions import method_profile
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.research_opportunities import ResearchOpportunities, opportunity_condition
from hegel_machine.cognition.uncertainty import select_decision_plan
from hegel_machine.cognition.workspace import ResearchWorkspace, candidate_from_dict


def _risk_fixture(name, seed):
    data = json.loads((Path(__file__).parent / "fixtures" /
        f"cognition_v020_risk_stop_{name}_seed{seed}.json").read_text())
    raw = data["public_spec"]
    spec = TaskSpec(raw["task_id"], raw["description"], tuple(raw["feature_names"]), (), (),
        risk_quantile=raw["risk_quantile"], feature_bounds=raw["feature_bounds"],
        max_probe_length=raw["max_probe_length"], decision_costs=tuple(raw["decision_costs"]))
    return spec, ResearchWorkspace.from_dict(data["workspace"]), ResearchConfig(**data["config"])


def _trajectory(target, *, length=4):
    result, history = [], []
    for index in range(length):
        features = {"x": index / length}
        result.append(Observation(features, target, tuple(history)))
        history.append({**features, "__target__": target})
    return result


def _question():
    spec = TaskSpec("measured_response", "Choose the action from public responses", ("x",), (), (),
        feature_bounds={"x": (-1., 1.)}, max_probe_length=32, risk_quantile=.9, decision_costs=(.9, .1))
    initial = _trajectory(1.) + _trajectory(1.)
    current = Candidate("deployed", Program("const", value=0.), "refit")
    workspace = ResearchWorkspace(spec.task_id, observations=initial, reference=initial,
        incumbent=current, bank=[current], fallback=0., initial_loss=.9,
        uncertainty_model={"kind": "nominal"})
    return spec, workspace, ResearchConfig(observation_budget=64, compute_budget=2000)


def _calibration(offset=.8):
    return {"kind": "global_interval", "residual_interval": [-1., 1.],
            "action_rule": "empirical_cost_quantile", "action_offset": offset}


@pytest.mark.parametrize("name,seed,planning_cost", [("n11", 2027, 396), ("s05", 3181, 352)])
def test_actual_stopped_risk_workspaces_offer_only_fitted_complete_policies(name, seed, planning_cost):
    spec, workspace, config = _risk_fixture(name, seed)
    before = workspace.to_dict()
    budget = SearchBudget(1200)
    frontier = plan_decision_frontier(spec, workspace, config, budget)
    assert workspace.to_dict() == before
    assert budget.spent == planning_cost
    assert len(frontier["policies"]) == 3
    assert frontier["policies"][0]["candidate"] == before["incumbent"]
    assert frontier["policies"][0]["model"] == before["uncertainty_model"]
    assert before["uncertainty_model"]["calibration_offset"] != 0.
    original = {candidate.candidate_id for candidate in workspace.bank}
    for policy in frontier["policies"][1:]:
        assert policy["candidate"]["candidate_id"] in original
        assert policy["candidate"]["program"] != before["incumbent"]["program"]
        assert policy["model"] == {"kind": "nominal"}
    assert frontier["probe_lengths"] == [8, 8]
    assert frontier["observation_cost"] == 16
    assert frontier["comparison_compute_cost"] == 64
    assert frontier["evidence_rows"] == len(workspace.observations)
    assert frontier["design"]["candidate_disagreement"] > 0.
    for comparison in frontier["design"]["comparisons"]:
        assert max(comparison["trajectory_gains"]) > config.minimum_improvement
        assert comparison["gain"] + comparison["support_margin"] > config.minimum_improvement
    assert name not in json.dumps(frontier["design"]["alternatives"])
    assert "w0" not in json.dumps(frontier["design"]["alternatives"])


def test_fitted_calibration_remains_attached_to_its_predictor_and_wins_only_on_new_trials():
    spec, workspace, config = _question()
    proposed = _calibration()
    budget = SearchBudget(2000)
    frontier = plan_decision_frontier(spec, workspace, config, budget, proposed_model=proposed)
    assert len(frontier["policies"]) == 2 and budget.spent == 2 * len(workspace.reference)
    assert workspace.uncertainty_model == {"kind": "nominal"}
    candidate = candidate_from_dict(frontier["policies"][1]["candidate"])
    assert candidate.program == workspace.incumbent.program
    assert frontier["policies"][1]["model"] == proposed
    fresh = _trajectory(1., length=8) + _trajectory(1., length=8)
    work = SearchBudget(frontier["comparison_compute_cost"])
    program, model, result = select_decision_plan(workspace.incumbent.program, workspace.uncertainty_model,
        candidate.program, proposed, fresh, fallback=workspace.fallback, costs=spec.decision_costs, budget=work)
    assert result["adopted"] and result["gain"] > .7
    assert program == candidate.program and model == proposed
    assert work.spent == frontier["comparison_compute_cost"] == 32


@pytest.mark.parametrize("case", ["only_residual", "same_action", "consistently_inferior"])
def test_residual_or_model_metadata_alone_cannot_create_a_decision_question(case):
    spec, workspace, config = _question()
    proposed = None
    if case == "same_action":
        proposed = _calibration(0.)
    elif case == "consistently_inferior":
        proposed = _calibration(-.5)
    frontier = plan_decision_frontier(spec, workspace, config, SearchBudget(2000), proposed_model=proposed)
    assert frontier == {}


def test_planning_cannot_turn_an_unpaid_alternative_into_a_question():
    spec, workspace, config = _question()
    budget = SearchBudget(2 * len(workspace.reference) - 1)
    assert plan_decision_frontier(spec, workspace, config, budget, proposed_model=_calibration()) == {}
    assert budget.spent == len(workspace.reference)


def test_historical_alternative_quotes_two_trajectories_that_reach_its_mature_state():
    spec, workspace, config = _question()
    workspace.observations = workspace.reference = _trajectory(1., length=6) + _trajectory(1., length=6)
    program = Program("if", children=(Program("available", lag=4),
        Program("mul", children=(Program("const", value=.8), Program("lag", name="__target__", lag=4))),
        Program("const", value=0.)))
    workspace.bank.append(Candidate("mature", program, "state"))
    frontier = plan_decision_frontier(spec, workspace, config, SearchBudget(2000))
    assert frontier["probe_lengths"] == [20, 20]
    assert frontier["observation_cost"] == 40 < config.observation_budget
    assert plan_decision_frontier(replace(spec, max_probe_length=4), workspace, config, SearchBudget(2000)) == {}


def test_actual_zero_return_blocks_parameter_drift_but_a_new_decision_structure_can_compete():
    spec, workspace, config = _question()
    first = plan_decision_frontier(spec, workspace, config, SearchBudget(2000), proposed_model=_calibration())
    profile = method_profile(spec, workspace.observations, workspace.incumbent,
        residual_loss=.9, baseline_loss=.9)

    def offer(frontier):
        return {"kind": "decision", "action": "compare_decisions", "profile": profile,
                "design": frontier["design"], "observation_cost": frontier["observation_cost"],
                "compute_cost": frontier["comparison_compute_cost"]}

    fresh = _trajectory(0., length=8) + _trajectory(0., length=8)
    selected, model, outcome = select_decision_plan(workspace.incumbent.program, workspace.uncertainty_model,
        workspace.incumbent.program, _calibration(), fresh, fallback=0., costs=spec.decision_costs,
        budget=SearchBudget(32))
    assert not outcome["adopted"] and outcome["gain"] < 0.
    assert selected == workspace.incumbent.program and model == workspace.uncertainty_model
    book = ResearchOpportunities()
    # The proposal deteriorated, so the deployed continuation retained exactly
    # its previous decisions and earned zero, not the proposed negative gain.
    book.observe(offer(first), {"execution_complete": True, "reference_rows": len(fresh),
        "gain": 0., "normalized_gain": 0.}, observation_cost=len(fresh), compute_cost=32)
    drifted = plan_decision_frontier(spec, workspace, config, SearchBudget(2000), proposed_model=_calibration(.81))
    assert opportunity_condition(offer(drifted)) == opportunity_condition(offer(first))
    restored = ResearchOpportunities(json.loads(json.dumps(book.to_dict())))
    assert not restored.rank([offer(drifted)], observation_remaining=100, compute_remaining=10000)
    conditional = {**_calibration(), "kind": "conditional_interval", "feature": "x",
        "transform": "identity", "threshold": .3, "intervals": [[-1., 1.], [-1., 1.]],
        "action_offsets": [.5, .8]}
    changed = plan_decision_frontier(spec, workspace, config, SearchBudget(2000), proposed_model=conditional)
    assert opportunity_condition(offer(changed)) != opportunity_condition(offer(first))
    assert restored.rank([offer(changed)], observation_remaining=100, compute_remaining=10000)[0]["exploration"]
