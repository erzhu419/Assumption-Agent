"""Each research direction must improve on the explanation already available."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.structure_lab import StructureLab
from hegel_machine.cognition.unified_research import _fit, _terms
from test_cognition_unified_action_policy import CONFIG, ObservedResearch, learned_directions


def feedback(result):
    return {event["research_action"]: event for event in result["actions"]
            if event["action"] == "research_action_feedback"}


def direction_group(result, direction):
    return next(group for group in result["agent"].assessments if any(
        claim.get("research_action_origin", {}).get("action") == direction for claim in group[0]))


def test_worse_callable_does_not_earn_the_improvement_already_found_by_structure(learned_directions):
    result, saved = learned_directions
    events = feedback(result)
    program, structure, callable_ = (events[name] for name in
        ("program_revision", "structural_research", "callable_transfer"))
    # These are actual fits of the same measured focus response. The late
    # callable beats the original affine guess, but loses to acquired structure.
    assert structure["after_loss"] < program["after_loss"] < callable_["after_loss"] < program["before_loss"]
    assert structure["before_loss"] == pytest.approx(program["after_loss"])
    assert callable_["before_loss"] == pytest.approx(structure["after_loss"])
    assert callable_["feedback"]["last_gain"] < 0
    record = saved["action_experience"][callable_["context"]]["callable_transfer"]
    assert record["last_gain"] < 0 and record["expected_gain"] < 0
    assert result["best"]["validation_loss"] <= structure["after_loss"]


def test_direction_origins_freeze_the_best_preceding_explanation(learned_directions):
    result, _ = learned_directions
    events = feedback(result)
    affine = result["agent"]._best(result["agent"].assessments[0][0])
    # The new mechanism direction also freezes the actual alternative even
    # when this source has no decomposable branch and produces no candidate.
    mechanism = events["mechanism_revision"]
    assert mechanism["baseline_program"] == affine["candidate"].program.to_dict()
    assert mechanism["before_loss"] == pytest.approx(loss(affine["candidate"].program,
        result["validation"], result["agent"].assessments[0][3], .5))
    assert mechanism["after_loss"] == mechanism["before_loss"]
    program = result["agent"]._best(direction_group(result, "program_revision")[0])
    structure = result["agent"]._best(direction_group(result, "structural_research")[0])
    predecessors = {"program_revision": affine, "structural_research": program,
                    "callable_transfer": structure}
    for direction, predecessor in predecessors.items():
        candidates, rows, _, _, _ = direction_group(result, direction)
        event = events[direction]
        for candidate in candidates:
            origin = candidate["research_action_origin"]
            assert origin["baseline_program"] == predecessor["candidate"].program.to_dict()
            assert origin["baseline_program"] == event["baseline_program"]
            assert origin["trial"] == event["feedback"]["trials"]
            measured = loss(Program.from_dict(origin["baseline_program"]), rows,
                origin["fallback"], origin["risk_quantile"])
            assert measured == pytest.approx(event["before_loss"])
    assert events["callable_transfer"]["baseline_program"] != events["program_revision"]["baseline_program"]


def test_parent_fitted_on_comparison_rows_is_remeasured_without_independent_return():
    lab = StructureLab(readout="focus", seed=91)
    training = lab.acquire(lab.random_probe(11, length=24))
    validation = lab.acquire(lab.random_probe(41, length=24))
    parent = _fit(_terms(lab.spec.feature_names), [*training.observations, *validation.observations],
        SearchBudget(100000), "pooled_parent", q=.5)
    parent["selection_basis"] = "all_acquired_evidence"
    # A parent's cached score can describe an earlier comparison. The new
    # view must measure its executable program on the current paid responses.
    parent["validation_loss"] = 0.
    original = deepcopy(parent)
    library = KnowledgeLibrary()
    agent, budget, actions = ObservedResearch(CONFIG, library=library), SearchBudget(CONFIG.compute_budget), []
    claims = agent._research_view(lab.spec, training.observations, validation.observations,
        budget, actions, [], ResearchGraph(), parent=parent, transitions=training.transitions)
    retained_group = next(group for group in agent.assessments
                          if any(claim["kind"] == "pooled_parent" for claim in group[0]))
    retained, = retained_group[0]
    assert retained["validation_loss"] == pytest.approx(loss(parent["candidate"].program,
        validation.observations, 0., .5))
    assert retained["validation_loss"] > CONFIG.minimum_improvement
    assert retained_group[2] == len(validation.observations)
    assert retained_group[4] - agent.assessments[0][4] == len(training.observations) + len(validation.observations)
    pending = [event for event in actions if event["action"] == "research_action_pending"]
    assert any(event["research_action"] == "structural_research"
               and event["evidence_basis"] == "reference_fitted_comparison" for event in pending)
    structural = [claim for claim in claims if claim.get("research_action_origin", {}).get("action")
                  == "structural_research"]
    assert structural
    assert all(claim["research_action_origin"]["trial"] is None
               and claim["research_action_origin"]["baseline_program"] == parent["candidate"].program.to_dict()
               for claim in structural)
    assert not [event for event in actions if event["action"] == "research_action_feedback"]
    assert not any(library.action_experience.values())
    assert parent == original and budget.spent <= budget.limit
