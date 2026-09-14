"""Transferred laws earn future work through actual paired research outcomes."""

import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec


def task(name, feature="x", bounds=None):
    return TaskSpec(name, "predict the acquired response", (feature,), (), (),
                    feature_bounds={feature: bounds} if bounds else {})


def shelf():
    library = KnowledgeLibrary()
    x = Program("feature", name="x")
    absolute, _ = library.learn(Candidate("abs", Program("abs", children=(x,)), "representation"), task("source_abs"))
    square, _ = library.learn(Candidate("square", Program("mul", children=(x, x)), "representation"), task("source_square"))
    return library, absolute, square


def rows(feature="x"):
    return [Observation({feature: value}, value * value) for value in (-1., 0., 1.)]


def finish(library, spec, winner):
    return library.finish_transfer_episode(spec, final_candidate=winner, final_loss=0., baseline_loss=9.,
        compute_cost=9999, observation_cost=91, validation_rows=7)


def measured_transfer(library, spec, observations):
    library.begin_transfer_episode(spec, observations)
    budget = SearchBudget(100)
    candidates = library.propose(spec, observations, budget, preserve_parameters=False)
    fresh = [Observation({spec.feature_names[0]: 2.}, 4.)]
    winner = min(candidates, key=lambda candidate: loss(candidate.program, fresh, 0., .5))
    for candidate in candidates:
        library.record_transfer_result(candidate, spec, accepted=candidate == winner,
            loss=loss(candidate.program, fresh, 0., .5), baseline_loss=1., evidence_rows=1,
            comparison_compute=7, comparison_id=1, observations=observations)
    return finish(library, spec, winner), budget, winner


def test_paired_returns_pay_shared_fits_and_change_renamed_task_screening():
    library, absolute, square = shelf()
    source = task("first_target", "u")
    result, budget, _ = measured_transfer(library, source, rows("u"))
    records = result["returns"]
    assert {item["source_id"] for item in records} == {absolute, square}
    assert result["unique_adaptation_compute"] == budget.spent - len(rows())
    assert all(item["adaptation_compute"] < item["visit_context"]["compute_cost"] for item in records)
    bad = next(item for item in records if item["source_id"] == absolute)
    good = next(item for item in records if item["source_id"] == square)
    assert bad["paired_gain"] < 0 and not bad["retained"]
    assert good["paired_gain"] == pytest.approx(1.) and good["retained"]
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    target = task("new_task_and_input_name", "v")
    proposals = restored.propose(target, rows("v"), SearchBudget(12), preserve_parameters=False)
    # Only one complete structural screen fits; the actual useful law wins.
    assert proposals and {candidate.memory_id for candidate in proposals} == {square}
    assert proposals[0].program.predict(Observation({"v": 2.}, 4.)) == pytest.approx(4.)


def test_uncovered_conditions_reopen_a_previous_failed_structure():
    library, absolute, square = shelf()
    measured_transfer(library, task("first", bounds=(-2., 2.)), rows())
    same = library.propose(task("matching", bounds=(-2., 2.)), rows(), SearchBudget(12), preserve_parameters=False)
    assert {candidate.memory_id for candidate in same} == {square}
    changed = library.propose(task("different_control_support", bounds=(-10., 10.)),
                               rows(), SearchBudget(12), preserve_parameters=False)
    assert {candidate.memory_id for candidate in changed} == {absolute}


def test_visit_success_does_not_credit_a_rejected_transfer_or_unmeasured_proposal():
    library, absolute, square = shelf()
    spec = task("target")
    library.begin_transfer_episode(spec, rows())
    candidates = library.propose(spec, rows(), SearchBudget(100), preserve_parameters=False)
    rejected = next(candidate for candidate in candidates if candidate.memory_id == absolute)
    library.record_transfer_result(rejected, spec, accepted=False, loss=4., baseline_loss=1.,
        comparison_compute=7, evidence_rows=1, comparison_id=1)
    winner = next(candidate for candidate in candidates if candidate.memory_id == square)
    records = finish(library, spec, winner)["returns"]
    bad = next(item for item in records if item["source_id"] == absolute)
    unmeasured = next(item for item in records if item["source_id"] == square)
    assert bad["retained_normalized_gain"] < 0
    assert bad["visit_context"]["baseline_loss"] == 9.
    assert unmeasured["adaptation_compute"] > 0
    assert unmeasured["paired_gain"] is None and not unmeasured["retained"]
    assert unmeasured["evidence_status"] == "proposal_not_compared"


def test_a_positive_comparison_that_does_not_survive_has_no_retained_gain():
    library, absolute, _ = shelf()
    spec = task("target")
    library.begin_transfer_episode(spec, rows())
    candidate = next(candidate for candidate in library.propose(spec, rows(), SearchBudget(100))
                     if candidate.memory_id == absolute)
    library.record_transfer_result(candidate, spec, accepted=True, loss=1., baseline_loss=4.,
        evidence_rows=4, comparison_compute=12, comparison_id=1)
    winner = Candidate("later_better", Program("const", value=5.), "refit")
    record = next(item for item in finish(library, spec, winner)["returns"] if item["source_id"] == absolute)
    assert record["paired_gain"] == 3.
    assert record["retained_normalized_gain"] == 0. and not record["retained"]


def test_equal_program_origins_keep_one_shared_fit_bill_and_comparison_per_origin():
    library = KnowledgeLibrary()
    x = Program("feature", name="x")
    for factor in (2., 3.):
        program = Program("add", children=(Program("mul", children=(Program("const", value=factor), x)),
                                           Program("const", value=1.)))
        library.learn(Candidate(str(factor), program, "refit"), task(str(factor)))
    spec = task("target")
    observations = [Observation({"x": value}, 7. * value + 2.) for value in (-1., 0., 1.)]
    library.begin_transfer_episode(spec, observations)
    budget = SearchBudget(100)
    candidates = library.propose(spec, observations, budget, preserve_parameters=False)
    assert len(candidates) == 2 and candidates[0].program == candidates[1].program
    for candidate in candidates:
        for _ in range(2):
            library.record_transfer_result(candidate, spec, accepted=True, loss=0., baseline_loss=1.,
                evidence_rows=3, comparison_compute=9, comparison_id=1)
    result = finish(library, spec, candidates[0])
    assert result["unique_adaptation_compute"] == 6
    for item in result["returns"]:
        assert item["adaptation_compute"] == 6 and item["comparison_compute"] == 9
        assert len(item["comparisons"]) == 1
        assert len(item["shared_adaptation_work"][0]["source_ids"]) == 2


def test_direct_retrieval_uses_measured_return_and_episode_bookkeeping_resets():
    library, absolute, square = shelf()
    spec = task("target")
    library.begin_transfer_episode(spec, rows())
    candidates = library.retrieve(spec, observations=rows())
    for candidate in candidates:
        library.record_transfer_result(candidate, spec, accepted=candidate.memory_id == square,
            loss=4. if candidate.memory_id == absolute else 0., baseline_loss=1.,
            evidence_rows=1, comparison_compute=4, comparison_id=1)
    winner = next(candidate for candidate in candidates if candidate.memory_id == square)
    result = finish(library, spec, winner)
    assert all(item["adaptation_compute"] == 0 for item in result["returns"])
    assert library.retrieve(task("later"), observations=rows())[0].memory_id == square
    library.begin_transfer_episode(task("empty"), rows())
    assert finish(library, task("empty"), winner)["returns"] == []


def test_equivalent_zero_decay_concept_reuses_identity_without_duplicate_definitions():
    library = KnowledgeLibrary()
    x = Program("feature", name="x")
    redundant = Program("accumulate", value=0., children=(x, Program("const", value=0.)))
    identifier, created = library.learn(Candidate("redundant", redundant, "continuous_state"), task("first"))
    assert created and library.concepts[0].program == x
    same, created = library.learn(Candidate("direct", x, "representation"), task("next"))
    assert same == identifier and not created and len(library.concepts) == 1
    assert library.concepts[0].successful_tasks == ["first", "next"]


def test_learning_a_restored_redundant_concept_repairs_its_stored_program_and_templates():
    library = KnowledgeLibrary()
    x = Program("feature", name="x")
    identifier, _ = library.learn(Candidate("source", x, "representation"), task("first"))
    payload = library.to_dict()
    redundant = Program("accumulate", value=0., children=(x, Program("const", value=0.)))
    payload["concepts"][0]["program"] = redundant.to_dict()
    payload["abstracts"][0]["pattern"] = redundant.remap_features({"x": "role_0"}).to_dict()
    payload["abstracts"][0]["parameter_paths"] = [[1]]
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(payload)))
    same, created = restored.learn(Candidate("direct", x, "refit"), task("later"))
    assert same == identifier and not created
    assert restored.concepts[0].program == x
    assert restored.abstracts[0].pattern == Program("feature", name="role_0")
    assert restored.abstracts[0].parameter_paths == ()
    assert KnowledgeLibrary.from_dict(json.loads(json.dumps(restored.to_dict()))).concepts[0].program == x


def test_uncompared_proposal_cost_is_not_a_failed_experiment():
    library, absolute, _ = shelf()
    spec = task("target")
    library.begin_transfer_episode(spec, rows())
    candidates = library.propose(spec, rows(), SearchBudget(12), preserve_parameters=False)
    assert {candidate.memory_id for candidate in candidates} == {absolute}
    finish(library, spec, Candidate("other", Program("const", value=0.), "refit"))
    proposals = library.propose(task("later"), rows(), SearchBudget(12), preserve_parameters=False)
    assert {candidate.memory_id for candidate in proposals} == {absolute}
