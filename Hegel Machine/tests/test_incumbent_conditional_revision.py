"""Retained conditions are hypotheses to revise using current paid evidence."""

from random import Random

import pytest

from hegel_machine.cognition.conditional_coordinates import (
    _masked, propose_conditional_coordinates, propose_experimental_conditional_coordinates,
)
from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.research_programs import replay_research_program


U = Program("feature", name="u")
ONE = Program("const", value=1.)
LAG = Program("if", children=(Program("available", lag=1),
    Program("lag", name="u", lag=1), Program("const", value=0.)))
SPEC = TaskSpec("condition_revision", "Predict purchased trajectories", ("u",), (), ())


def predictor(*, width=3, threshold=.15, placement="outside_window", history=False, gate=U):
    condition = Program("ge", children=(gate, Program("const", value=threshold)))
    basis = [U, ONE, LAG] if history else [U, ONE]
    terms = [_masked(term, condition, side, width, placement)
             for term in basis for side in (True, False)]
    coefficients = [2.3, -1.4, .2, -.1, .5, .7][:len(terms)]
    return _sum_terms(terms, coefficients, .17)


def observations(program, seed=71):
    rng = Random(seed)
    rows = []
    for _ in range(2):
        history = []
        for _ in range(48):
            features = {"u": rng.choice((-.9, -.6, -.3, 0., .3, .6, .9))}
            row = Observation(features, 0., tuple(history))
            rows.append(Observation(features, program.predict(row), tuple(history)))
            history.append(features)
    return rows


def step(claim):
    return claim["research_program"]["selection_steps"][0]


@pytest.mark.parametrize("changed", [dict(), {"threshold": -.15}, {"width": 4},
                                     {"placement": "inside_window"}])
def test_retained_joint_response_is_completed_before_local_parameter_revisions(changed):
    original = predictor()
    frozen = original.to_dict()
    truth = predictor(history=True, **changed)
    rows = observations(truth)
    work, receipt = SearchBudget(120000), {}
    claims = propose_conditional_coordinates(SPEC, rows, work, basis=[U, LAG],
        retained_programs=[original], gate_coordinates=[U, LAG], receipt=receipt)
    best = min(claims, key=lambda item: item["training_loss"])
    assert best["training_loss"] < 1e-10
    assert loss(best["candidate"].program, observations(truth, 1201), 0., .5) < 1e-10
    assert original.to_dict() == frozen
    assert receipt["seed_count"] == 1
    assert receipt["completed_baseline_fits"] == 1
    assert receipt["completed_joint_fits"] == 1
    assert [trial["stage"] for trial in receipt["parameter_trials"][:2]] == [
        "retained_baseline", "add_branch_coordinate"]
    assert receipt["parameter_trials"][0]["width"] == 3
    assert receipt["parameter_trials"][0]["threshold"] == .15
    assert any(step(claim)["revision_stage"] == "retained_baseline" for claim in claims)
    assert LAG.to_dict() in step(best)["branch_basis"]
    assert 0 < work.spent <= work.limit
    if changed:
        assert step(best)["revision_stage"] == "revise_condition"
        for key, value in changed.items():
            assert step(best)[key] == value


def test_new_history_coordinate_can_replace_the_old_current_gate():
    truth = predictor(history=True, gate=LAG)
    work, receipt = SearchBudget(120000), {}
    claims = propose_conditional_coordinates(SPEC, observations(truth), work,
        basis=[U, LAG], retained_programs=[predictor()], gate_coordinates=[U, LAG],
        receipt=receipt)
    best = min(claims, key=lambda item: item["training_loss"])
    assert step(best)["gate_coordinate"] == LAG.to_dict()
    assert step(best)["revision_stage"] == "revise_condition"
    assert loss(best["candidate"].program, observations(truth, 517), 0., .5) < 1e-10
    assert any(trial["gate_coordinate"] == U.to_dict()
               for trial in receipt["parameter_trials"])


def test_retained_history_gate_preserves_the_per_sample_prefix():
    original = predictor(placement="inside_window", gate=LAG)
    truth = predictor(history=True, placement="inside_window", gate=LAG, width=4)
    claims = propose_conditional_coordinates(SPEC, observations(truth), SearchBudget(120000),
        basis=[U, LAG], retained_programs=[original], gate_coordinates=[LAG])
    best = min(claims, key=lambda item: item["training_loss"])
    assert step(best)["gate_coordinate"] == LAG.to_dict()
    assert step(best)["width"] == 4
    assert loss(best["candidate"].program, observations(truth, 813), 0., .5) < 1e-10


def test_complete_baseline_and_first_joint_fit_are_reserved_before_column_work():
    work, receipt = SearchBudget(100), {}
    claims = propose_conditional_coordinates(SPEC, observations(predictor(history=True)), work,
        basis=[LAG], retained_programs=[predictor()], receipt=receipt)
    assert claims == [] and work.spent == 0
    assert receipt["completed_baseline_fits"] == receipt["completed_joint_fits"] == 0
    assert receipt["minimum_complete_fit_compute"] > work.limit
    assert work.last_denied_request["required_compute"] == receipt["minimum_complete_fit_compute"]


def test_saved_actual_answer_can_seed_the_next_conditional_revision():
    source = propose_conditional_coordinates(SPEC, observations(predictor(history=True)),
        SearchBudget(120000), basis=[U, LAG], retained_programs=[predictor()])
    saved = min(source, key=lambda item: item["training_loss"])["research_program"]
    truth = predictor(history=True, width=4)
    receipt = {}
    revised = propose_conditional_coordinates(SPEC, observations(truth), SearchBudget(120000),
        basis=[U, LAG], retained_research_programs=[saved], receipt=receipt)
    best = min(revised, key=lambda item: item["training_loss"])
    assert receipt["seed_count"] > 0
    assert step(best)["width"] == 4
    assert loss(best["candidate"].program, observations(truth, 2019), 0., .5) < 1e-10


def test_experimental_entry_keeps_a_measured_history_response_coordinate():
    truth = predictor(history=True)
    claims = propose_experimental_conditional_coordinates(SPEC, observations(truth),
        SearchBudget(120000), basis=[U, LAG], evidence=[{"control": "u", "gain": 1.,
            "width": 3, "placement": "outside_window", "threshold_interval": [0., .3],
            "condition_identified": True, "conditional_loss": 0.}])
    best = min(claims, key=lambda item: item["training_loss"])
    assert best["training_loss"] < 1e-10
    assert loss(best["candidate"].program, observations(truth, 173), 0., .5) < 1e-10


def test_saved_seeded_method_revises_boundary_without_dropping_background():
    v = Program("feature", name="v")
    background = Program("mul", children=(v, v))
    def with_background(program):
        return _sum_terms([program, background], [1., .81])
    def measured(program, seed):
        rng, result = Random(seed), []
        for row in observations(predictor(), seed):
            features = {**row.features, "v": rng.uniform(-1., 1.)}
            history = tuple({**point, "v": rng.uniform(-1., 1.)} for point in row.history)
            record = Observation(features, 0., history)
            result.append(Observation(features, program.predict(record), history))
        return result
    spec = TaskSpec("background", "Reuse the complete response", ("u", "v"), (), ())
    old = with_background(predictor())
    source = with_background(predictor(history=True))
    claims = propose_conditional_coordinates(spec, measured(source, 71), SearchBudget(120000),
        basis=[U, LAG], retained_programs=[old])
    saved = min(claims, key=lambda item: item["training_loss"])["research_program"]
    changed = with_background(predictor(history=True, threshold=-.15))
    replay = replay_research_program(saved, measured(changed, 1101), SearchBudget(120000))
    assert replay is not None and replay["research_program"]["reused_procedure"]
    assert step(replay)["threshold"] == -.15
    assert background.to_dict() in step(replay)["parameter_terms"]
    assert loss(replay["candidate"].program, measured(changed, 1407), 0., .5) < 1e-10
