"""Purchased relation procedures remain useful after names and ranges change."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.competing_object_experiments import (
    assess_competing_object_probe, plan_competing_object_probe,
)
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from hegel_machine.cognition.unified_structure import propose_structures
from test_cognition_adaptive_structure_boundary import (
    SETTINGS, _calibrate, _purchase, measured_joint_response,
)
from test_cognition_object_interventions import discover


def _problem(entry, paid):
    research = RelationIntervalResearch(entry["candidate"].program, [], SearchBudget(100000),
                                        structure=entry["structure"])
    alternatives = research.alternatives(paid, SearchBudget(100000))
    assert len(alternatives) >= 2
    return alternatives, research.questions(alternatives)


def _plan(lab, alternatives, questions, binding, methods=None, *, budget=40000, task="source"):
    return plan_competing_object_probe(lab.spec, alternatives, SearchBudget(budget),
        binding=binding, questions=questions, methods=methods, task_id=task, length=5)


@pytest.fixture(scope="module")
def learned_relation_method(measured_joint_response):
    lab, entry, paid, binding = measured_joint_response
    alternatives, questions = _problem(entry, paid)
    plan = _plan(lab, alternatives, questions, binding)
    evidence = assess_competing_object_probe(plan, _purchase(lab, plan["probe"]), SearchBudget(2000))
    assert evidence["status"] == "distinguished"
    assert evidence["calibration_status"] == "supported_on_probe"
    library = KnowledgeLibrary()
    learned = library.program_experiment_methods.observe(plan, evidence,
        task_id="source", command_spec=lab.spec, binding=binding)
    assert learned["action"] == "learn_program_experiment_method"
    # Select the question that produced the paid success, not a different pair
    # whose complete response may legitimately be inactive in this background.
    question = next(item for item in questions if item["question_id"] == plan["question"]["question_id"])
    return lab, alternatives, [question], binding, json.loads(json.dumps(library.to_dict()))


@pytest.fixture(scope="module")
def renamed_relation_problem():
    with AnonymousObjectLab(**{**SETTINGS, "name_seed": 97032077, "relation_radius": .81}) as lab:
        objects, _, _ = discover(lab)
        assert objects["complete"]
        schema = objects["schema"]
        spec = replace(lab.spec_for(("fine",)), context={"object_schema": json.dumps(schema)})
        paid = [row for seed in (89, 90, 91)
                for _, row in _purchase(lab, lab.random_probe(seed, length=24))]
        entry = propose_structures(spec, paid, SearchBudget(200000))[0]
        assert entry["kind"] == "structural:joint_window_event" and entry["training_loss"] < 1e-7
        alternatives, questions = _problem(entry, paid)
        binding = _calibrate(lab, spec, schema)
        yield lab, alternatives, questions, binding


def test_actual_relation_success_can_be_recalled_without_claiming_new_support(learned_relation_method):
    lab, alternatives, questions, binding, payload = learned_relation_method
    methods = KnowledgeLibrary.from_dict(payload).program_experiment_methods
    original = deepcopy(methods.records)
    replay = _plan(lab, alternatives, questions, binding, methods)
    assert replay["proposal_source"] == "learned_method" and replay["fresh_synthesis_skipped"]
    assert methods.records == original and len(original[0]["support"]) == 1
    evidence = assess_competing_object_probe(replay, _purchase(lab, replay["probe"]), SearchBudget(2000))
    assert evidence["status"] == "distinguished"
    action = methods.observe(replay, evidence, task_id="source", command_spec=lab.spec, binding=binding)
    assert action["action"] == "use_program_experiment_method" and not action["cross_task"]
    assert len(methods.records) == 1 and len(methods.records[0]["support"]) == 2


def test_serialized_relation_method_rebinds_new_names_and_range_from_current_evidence(
        learned_relation_method, renamed_relation_problem):
    source_lab, _, _, _, payload = learned_relation_method
    lab, alternatives, questions, binding = renamed_relation_problem
    assert not set(source_lab.spec.feature_names).intersection(lab.spec.feature_names)
    methods = KnowledgeLibrary.from_dict(payload).program_experiment_methods
    before = deepcopy(methods.records)
    replay = _plan(lab, alternatives, questions, binding, methods, task="renamed_successor")
    assert replay["proposal_source"] == "learned_method"
    assert methods.records == before
    boundary = replay["proposal"]["boundary"]
    assert boundary["thresholds"] != before[0]["source_boundary"]["thresholds"]
    assert all(set(point) == set(lab.spec.feature_bounds) for point in replay["probe"].sequence)
    evidence = assess_competing_object_probe(replay, _purchase(lab, replay["probe"]), SearchBudget(2000))
    assert evidence["status"] == "distinguished" and evidence["calibration_status"] == "supported_on_probe"
    action = methods.observe(replay, evidence, task_id="renamed_successor",
                             command_spec=lab.spec, binding=binding)
    if replay["proposal"].get("parent_method_id"):
        assert action["action"] == "learn_program_experiment_method"
        assert methods.records[0] == before[0]
        assert methods.records[-1]["parent_method_id"] == before[0]["method_id"]
        assert len(methods.records[-1]["support"]) == 1
    else:
        assert action["action"] == "use_program_experiment_method" and action["cross_task"]
        assert len(methods.records) == 1 and len(methods.records[0]["support"]) == 2


def test_current_relation_pair_rewrites_inactive_background_before_purchasing(
        learned_relation_method, renamed_relation_problem):
    _, _, _, _, payload = learned_relation_method
    lab, alternatives, questions, binding = renamed_relation_problem
    methods = KnowledgeLibrary.from_dict(payload).program_experiment_methods
    original = deepcopy(methods.records)
    # This pair was inactive in the source method's complete response setup.
    # Its condition must be coupled to the current pair's quantity/selector
    # roles, rather than paying for the source setup with different positions.
    replay = _plan(lab, alternatives, questions[:1], binding, methods, task="changed_pair")
    assert replay["forecast_status"] == "predicted_distinction"
    assert replay["proposal_source"] == "learned_method"
    assert replay["proposal"]["parent_method_id"] == original[0]["method_id"]
    assert methods.records == original
    evidence = assess_competing_object_probe(replay, _purchase(lab, replay["probe"]), SearchBudget(2000))
    assert evidence["status"] == "distinguished"
    learned = methods.observe(replay, evidence, task_id="changed_pair",
        command_spec=lab.spec, binding=binding)
    assert learned["action"] == "learn_program_experiment_method"
    assert methods.records[0] == original[0]
    assert len(methods.records) == 2 and len(methods.records[-1]["support"]) == 1


def test_actual_calibration_failure_of_rewritten_method_is_remembered_without_erasing_parent(
        learned_relation_method, renamed_relation_problem):
    _, _, _, _, payload = learned_relation_method
    lab, alternatives, questions, binding = renamed_relation_problem
    methods = KnowledgeLibrary.from_dict(payload).program_experiment_methods
    original = deepcopy(methods.records)
    plan = _plan(lab, alternatives, questions[:1], binding, methods, task="drifted_successor")
    assert plan["proposal"]["parent_method_id"] == original[0]["method_id"]
    # The actual actuator changes after calibration. Measurements come from
    # that changed apparatus; neither support nor field errors are fabricated.
    with AnonymousObjectLab(**{**SETTINGS, "name_seed": 97032077, "relation_radius": .81,
                               "actuator_curvature": .9}) as changed:
        evidence = assess_competing_object_probe(plan, _purchase(changed, plan["probe"]),
                                                 SearchBudget(2000))
    assert evidence["calibration_status"] == "counterexample"
    action = methods.observe(plan, evidence, task_id="drifted_successor",
                             command_spec=lab.spec, binding=binding)
    assert action["action"] == "revise_program_experiment_method"
    assert action["reason"] == "calibration_counterexample"
    assert methods.records[0] == original[0] and len(methods.records) == 2
    assert methods.records[-1]["parent_method_id"] == original[0]["method_id"]
    assert methods.records[-1]["status"] == "counterexample"
    assert not methods.records[-1]["support"] and len(methods.records[-1]["counterexamples"]) == 1
    remembered = deepcopy(methods.records)
    replay = _plan(lab, alternatives, questions[:1], binding, methods, task="drifted_successor")
    assert replay["forecast_status"] == "requires_paired_measurements"
    assert replay["proposal_source"] == "calibration" and replay["predicted_separation"] is None
    assert methods.records == remembered


def test_measured_relation_method_leaves_room_when_fresh_synthesis_exhausts_same_budget(
        learned_relation_method):
    lab, alternatives, questions, binding, payload = learned_relation_method
    methods = KnowledgeLibrary.from_dict(payload).program_experiment_methods
    replay = _plan(lab, alternatives, questions, binding, methods, budget=4000)
    erased = _plan(lab, alternatives, questions, binding, budget=4000)
    assert replay["forecast_status"] == "predicted_distinction"
    assert replay["proposal_source"] == "learned_method" and replay["compute_cost"] < 4000
    assert erased["probe"] is None and erased["forecast_status"] == "compute_exhausted"
    evidence = assess_competing_object_probe(replay, _purchase(lab, replay["probe"]), SearchBudget(2000))
    assert evidence["status"] == "distinguished"
    fresh = _plan(lab, alternatives, questions, binding)
    assert fresh["forecast_status"] == "predicted_distinction"
    assert replay["generation_cost"] < fresh["generation_cost"]
