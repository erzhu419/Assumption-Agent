"""A method's produced knowledge changes later research choices and investments."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.action_policy import ActionOption
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.method_conditions import method_profile
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from test_cognition_method_returns import method_evidence, measured_step


def producer():
    task, observations, profile, context, condition = method_evidence()
    library = KnowledgeLibrary()
    step = measured_step("representation", .4, context, profile, compute_cost=100, observation_cost=0)
    method_id = library.methods.observe(context, [step], condition=condition, source_task="source")
    alternative = library.methods.observe(context, [measured_step("refit", .6, context, profile,
        compute_cost=100, observation_cost=0)], condition=condition, source_task="another_source")
    program = Program("abs", children=(Program("feature", name="x"),))
    concept_id, _ = library.learn(Candidate("absolute", program, "representation"), task.spec, observations)
    step.update(produced_method_id=method_id, after_program=program.to_dict())
    links = library.record_knowledge_production(concept_id, task.spec, [step])
    assert links and links[0]["method_id"] == method_id
    return library, task, observations, profile, context, condition, concept_id, method_id, alternative, step


def consume(library, *, nonlinear=True, retained=True):
    spec = TaskSpec("consumer", "predict acquired output", ("u",), (), ())
    rows = [Observation({"u": value}, 4. * abs(value) + 1.) for value in (-2., -1., 0., 1., 2.)]
    library.begin_transfer_episode(spec, rows)
    budget = SearchBudget(100)
    candidates = library.propose(spec, rows, budget, preserve_parameters=False)
    fresh = [Observation({"u": value}, 4. * (abs(value) if nonlinear else value * value) + 1.)
             for value in (-2.5, -1.5, 1.5, 2.5)]
    best = min(candidates, key=lambda candidate: loss(candidate.program, fresh, 0., .5))
    for candidate in candidates:
        library.record_transfer_result(candidate, spec, accepted=retained and candidate == best,
            loss=loss(candidate.program, fresh, 0., .5), baseline_loss=1.,
            evidence_rows=len(fresh), comparison_compute=len(fresh), comparison_id=1)
    final = best if retained else Candidate("independent", Program("const", value=0.), "refit")
    result = library.finish_transfer_episode(spec, final_candidate=final, final_loss=0., baseline_loss=100.,
        compute_cost=9999, observation_cost=91, validation_rows=len(fresh))
    return result, budget


def select(library, profile, context, condition):
    options = [ActionOption("refit", 0, 100), ActionOption("representation", 0, 100)]
    return library.methods.choose(context, condition, options, [], task_id="new_producer", profile=profile,
        observation_remaining=100, compute_remaining=10000, ordinary=options[0])


def test_real_fitted_output_return_changes_method_choice_and_survives_json():
    library, _, _, profile, context, condition, concept, method_id, alternative, _ = producer()
    assert select(library, profile, context, condition).name == "refit"
    result, budget = consume(library)
    assert len(result["method_returns"]) == 1
    outcome = result["method_returns"][0]
    assert outcome["source_concept_id"] == concept and outcome["method_id"] == method_id
    assert outcome["retained_gain"] == pytest.approx(1.)
    assert outcome["exclusive_compute"] == budget.spent - 5 + 4
    assert outcome["shared_compute"] == 0
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    assert select(restored, profile, context, condition).name == "representation"
    estimate = restored.methods.last_choice["return_estimate"]
    assert estimate["gain"] == .4
    assert estimate["knowledge_return"]["gain"] == pytest.approx(1.)
    assert estimate["investment_gain"] == pytest.approx(1.4)
    assert estimate["investment_compute_cost"] == 100 + outcome["exclusive_compute"]


def test_future_output_value_is_not_capped_as_if_it_were_current_prediction_gain():
    library, _, _, profile, context, condition, _, identifier, _, _ = producer()
    consume(library)
    offer = library.methods.opportunity(context, condition, task_id="new_producer", profile=profile,
        observation_remaining=100, compute_remaining=10000, maximum_gain=.01,
        visit_observation_budget=1, visit_compute_budget=100)
    assert offer["method_id"] == identifier
    assert offer["expected_gain"] == .01
    assert offer["expected_knowledge_gain"] == pytest.approx(1.)
    assert offer["investment_gain"] == pytest.approx(1.01)
    assert offer["compute_cost"] == 100 < offer["investment_compute_cost"]
    assert offer["value"] == pytest.approx(1.01 / (offer["investment_compute_cost"] / 10000))


def test_negative_output_return_restricts_matching_production_condition_without_global_ban():
    library, task, observations, profile, context, condition, _, identifier, alternative, _ = producer()
    result, _ = consume(library, nonlinear=False, retained=False)
    assert result["method_returns"][0]["retained_gain"] < 0
    estimate = library.methods._return(library.methods.methods[0], [], profile)
    assert estimate["knowledge_return"]["gain"] < 0
    assert estimate["investment_gain"] < estimate["gain"]
    assert select(library, profile, context, condition).name == "refit"
    # A newly acquired control region changes the production problem. The old
    # consumer failure does not claim this new producer will output bad laws.
    changed_spec = replace(task.spec, feature_bounds={"x": (-10., 10.)})
    changed = method_profile(changed_spec, observations, Program("const", value=0.), residual_loss=1., baseline_loss=2.)
    reopened = library.methods._return(library.methods.methods[0], [], changed)
    assert reopened["knowledge_return"]["trials"] == 0
    assert reopened["investment_gain"] == reopened["gain"] == .4
    assert library.methods.methods[0]["status"] == "proposed"


def test_a_useful_but_discarded_candidate_does_not_earn_the_method_retained_gain():
    library, _, _, _, _, _, _, _, _, _ = producer()
    result, _ = consume(library, retained=False)
    assert result["method_returns"][0]["retained_gain"] == 0.
    assert result["method_returns"][0]["exclusive_compute"] > 0
    assert all(not row["retained"] for row in result["method_returns"][0]["comparisons"])


def test_only_the_actual_complete_producer_is_linked_once_and_calibration_is_not_a_program_producer():
    library, task, _, _, _, _, identifier, _, _, step = producer()
    assert library.record_knowledge_production(identifier, task.spec, [step]) == []
    original = deepcopy(library.concepts[0].production_methods)
    for updated in ({"execution_complete": False}, {"gain": 0.}, {"action": "calibration"},
                    {"after_program": Program("const", value=0.).to_dict()}):
        altered = {**step, **updated}
        assert library.record_knowledge_production(identifier, task.spec, [altered]) == []
    assert library.concepts[0].production_methods == original


def test_new_links_added_during_the_consumer_visit_do_not_receive_its_earlier_transfer_return():
    library, task, observations, _, _, _, identifier, _, _, step = producer()
    library.concepts[0].production_methods.clear()
    library.begin_transfer_episode(task.spec, observations)
    candidate = library.retrieve(task.spec)[0]
    library.record_transfer_result(candidate, task.spec, accepted=True, loss=0., baseline_loss=1.,
        evidence_rows=4, comparison_compute=4, comparison_id=1)
    other_producer = replace(task.spec, task_id="nominal_other_producer")
    assert library.record_knowledge_production(identifier, other_producer, [step])
    result = library.finish_transfer_episode(task.spec, final_candidate=candidate,
        final_loss=0., baseline_loss=1., compute_cost=100, observation_cost=4, validation_rows=4)
    assert result["returns"] and result["method_returns"] == []


def test_shared_shape_does_not_duplicate_gain_or_the_shared_fit_bill_into_one_producer():
    library, _, _, _, _, _, _, identifier, _, _ = producer()
    x = Program("feature", name="x")
    another = Program("mul", children=(Program("const", value=3.), Program("abs", children=(x,))))
    other_spec = TaskSpec("alternative_source", "predict", ("x",), (), ())
    # Keep the same affine-in-absolute parameterized shape on both sources.
    library.concepts[0].program = Program("mul", children=(Program("const", value=2.), Program("abs", children=(x,))))
    from hegel_machine.cognition.abstractions import extract_templates
    library.abstracts = extract_templates(library.concepts[0].program, library.concepts[0].concept_id)
    library.learn(Candidate("other", another, "representation"), other_spec)
    result, _ = consume(library)
    outcome = result["method_returns"][0]
    assert outcome["retained_gain"] is None
    assert outcome["shared_gain"] == pytest.approx(1.)
    assert outcome["exclusive_compute"] == 0 and outcome["shared_compute"] > 0
    assert all(not row["exclusive"] for row in outcome["comparisons"])
