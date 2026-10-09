"""Transferred laws earn future work through actual paired research outcomes."""

import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import Concept, KnowledgeLibrary
from hegel_machine.cognition.knowledge_returns import producer_outcomes, return_priority, transfer_condition
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.unified_session import UnifiedResearchSession


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


def test_transfer_condition_distinguishes_a_measured_response_regime_change():
    spec = task("response_regime")
    base = [Observation({"x": value}, target)
            for value, target in ((-1., -1.2), (0., 0.), (1., 1.2))]
    shifted = [Observation({"x": value}, target)
               for value, target in ((-1., -1.5), (0., 0.), (1., 1.5))]

    assert transfer_condition(spec, base)["response_signature"] != transfer_condition(
        spec, shifted)["response_signature"]


def test_transfer_condition_marks_unexplained_response_variation():
    spec = task("response_noise")
    base = [Observation({"x": value}, target)
            for value, target in ((-1., -1.2), (0., 0.), (1., 1.2), (2., 2.4))]
    noisy = [Observation({"x": value}, target)
             for value, target in ((-1., -1.2), (0., .2), (1., 1.0), (2., 2.4))]

    left = transfer_condition(spec, base)["response_signature"]
    right = transfer_condition(spec, noisy)["response_signature"]
    assert left["mean_bin"] == right["mean_bin"]
    assert left["spread_bin"] == right["spread_bin"]
    assert left["residual_bin"] != right["residual_bin"]


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


def _finish_completed_source_visit(library, spec, source_id, observations, serial,
                                   *, measured_loss=.1, program=None):
    condition = transfer_condition(spec, observations)
    library.begin_transfer_episode(spec, observations)
    candidate = Candidate(f"repeat_{serial}", program or library.concepts[0].program, "reuse",
                          memory_id=source_id)
    library.record_transfer_result(candidate, spec, accepted=True, loss=measured_loss,
        baseline_loss=1., evidence_rows=len(observations), comparison_compute=3,
        comparison_id=serial, observations=observations,
        origin={"template_id": None, "mapping": {}, "mapping_roles": {},
                "condition": condition, "adaptation_work": []})
    return library.finish_transfer_episode(spec, final_candidate=candidate, final_loss=measured_loss,
        baseline_loss=1., compute_cost=10, observation_cost=len(observations),
        validation_rows=len(observations), final_status="completed")


def test_noisy_source_return_needs_a_second_completed_confirmation():
    library, source_id, _ = shelf()
    spec = task("repeat_confirmation")
    first_observations = [Observation({"x": value}, target)
                          for value, target in ((-1., -1.2), (0., .2), (1., 1.0), (2., 2.4))]
    second_observations = [Observation({"x": value}, target)
                           for value, target in ((-1., -1.22), (0., .18), (1., .98), (2., 2.38))]

    first = _finish_completed_source_visit(library, spec, source_id, first_observations, 1)
    record = first["returns"][0]
    assert record["confirmation_status"] == "provisional"
    assert record["confirmation_replicates"] == 1

    second = _finish_completed_source_visit(library, spec, source_id, second_observations, 2)
    assert all(item["confirmation_status"] == "confirmed"
               for item in second["returns"] + [library.concepts[0].reuse_history[-2]])
    assert all(item["confirmation_replicates"] == 2
               for item in library.concepts[0].reuse_history[-2:])
    assert return_priority(library.concepts[0].reuse_history, record["condition"])[0] < 0.


@pytest.mark.parametrize("measured_loss", [1., 1. - 1e-13, 1. + 1e-13])
def test_near_zero_second_visit_does_not_confirm_a_provisional_source(measured_loss):
    library, source_id, _ = shelf()
    spec = task("near_zero_confirmation")
    observations = rows()
    first = _finish_completed_source_visit(library, spec, source_id, observations, 1)
    assert first["returns"][0]["confirmation_status"] == "provisional"

    second = _finish_completed_source_visit(library, spec, source_id, observations, 2,
                                            measured_loss=measured_loss)
    record, = second["returns"]
    assert record["value_status"] == record["confirmation_status"] == "neutral"
    assert record["confirmation_replicates"] == 0
    assert record["retained_normalized_gain"] == 0.
    assert library.concepts[0].reuse_history[-2]["confirmation_status"] == "provisional"
    assert return_priority(library.concepts[0].reuse_history, record["condition"]) == (0., 0.)


def test_distinct_singleton_programs_require_separate_repeat_confirmations():
    library, source_id, _ = shelf()
    spec = task("distinct_program_confirmation")
    observations = rows()
    first = _finish_completed_source_visit(library, spec, source_id, observations, 1)
    other_program = Program("const", value=2.)
    second = _finish_completed_source_visit(library, spec, source_id, observations, 2,
                                            program=other_program)
    assert second["returns"][0]["confirmation_status"] == "provisional"
    assert second["returns"][0]["confirmation_replicates"] == 1
    assert library.concepts[0].reuse_history[-2]["confirmation_status"] == "provisional"

    third = _finish_completed_source_visit(library, spec, source_id, observations, 3,
                                           program=other_program)
    assert third["returns"][0]["confirmation_status"] == "confirmed"
    assert third["returns"][0]["confirmation_replicates"] == 2
    assert library.concepts[0].reuse_history[0]["confirmation_status"] == "provisional"
    assert return_priority(library.concepts[0].reuse_history, first["returns"][0]["condition"],
                           program=library.concepts[0].program, source_ids=[source_id]) == (0., 0.)


def test_legacy_return_without_response_signature_is_not_applied_to_signed_condition():
    library, _, square = shelf()
    measured_transfer(library, task("legacy_source", "u"), rows("u"))
    source = next(concept for concept in library.concepts if concept.concept_id == square)
    record = next(item for item in source.reuse_history if item.get("kind") == "research_return")
    signed_condition = record["condition"]
    legacy_condition = json.loads(json.dumps(signed_condition))
    legacy_condition.pop("response_signature")
    payload = library.to_dict()
    restored_source = next(item for item in payload["concepts"] if item["concept_id"] == square)
    for item in restored_source["reuse_history"]:
        if item.get("kind") == "research_return":
            item["condition"] = legacy_condition
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(payload)))
    restored_source = next(concept for concept in restored.concepts if concept.concept_id == square)

    assert return_priority(restored_source.reuse_history, signed_condition) == (0.0, 0.0)
    assert return_priority(restored_source.reuse_history, legacy_condition)[0] < 0.0


def test_legacy_singleton_return_owner_is_not_reused_for_another_source():
    condition = {"objective": "predict", "data": {"boundary": "same"}}
    program = Program("feature", name="x")
    legacy = {"kind": "research_return", "source_id": "source_a",
              "condition": condition, "retained_normalized_gain": -.4,
              "adaptation_compute": 1, "comparison_compute": 1,
              "comparisons": [{"program": program.to_dict()}]}

    assert return_priority([legacy], condition, program=program,
                           source_ids=["source_a"])[0] > 0.
    assert return_priority([legacy], condition, program=program,
                           source_ids=["source_b"]) == (0., 0.)


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


def test_neutral_completed_fit_is_not_recorded_as_transfer_value():
    """A retained fit must improve before it can affect source reuse."""
    library, absolute, _ = shelf()
    spec = task("neutral_completed_fit")
    observations = rows()
    library.begin_transfer_episode(spec, observations)
    candidate = Candidate("neutral", library.concepts[0].program, "reuse",
                          parents=(absolute,), memory_id=absolute)
    library.record_transfer_result(candidate, spec, accepted=True, loss=1.,
        baseline_loss=1., evidence_rows=len(observations), comparison_compute=4,
        comparison_id="neutral")
    result = library.finish_transfer_episode(spec, final_candidate=candidate,
        final_loss=1., baseline_loss=1., compute_cost=8,
        observation_cost=len(observations), validation_rows=len(observations),
        final_status="completed")
    record, = result["returns"]
    assert record["retained"] and record["value_status"] == "neutral"
    assert record["confirmation_status"] == "neutral"
    assert record["retained_normalized_gain"] == 0.
    assert return_priority(library.concepts[0].reuse_history,
                           record["condition"]) == (0., 0.)


def test_numerical_residue_is_not_transfer_value():
    library, absolute, _ = shelf()
    spec = task("numerical_neutral_fit")
    observations = rows()
    library.begin_transfer_episode(spec, observations)
    candidate = Candidate("near_neutral", library.concepts[0].program, "reuse",
                          parents=(absolute,), memory_id=absolute)
    library.record_transfer_result(candidate, spec, accepted=True, loss=1. - 1e-13,
        baseline_loss=1., evidence_rows=len(observations), comparison_compute=4,
        comparison_id="near_neutral")
    result = library.finish_transfer_episode(spec, final_candidate=candidate,
        final_loss=1. - 1e-13, baseline_loss=1., compute_cost=8,
        observation_cost=len(observations), validation_rows=len(observations),
        final_status="completed")
    record, = result["returns"]
    assert record["value_status"] == "neutral"
    assert record["retained_normalized_gain"] == 0.
    assert return_priority(library.concepts[0].reuse_history,
                           record["condition"]) == (0., 0.)


@pytest.mark.parametrize("measured_loss", [1., 1. - 1e-13, 1. + 1e-13])
def test_unretained_near_zero_comparison_does_not_withdraw_an_offer(measured_loss):
    library, source_id, _ = shelf()
    spec = task("near_zero_withdrawal")
    observations = rows()
    library.begin_transfer_episode(spec, observations)
    candidate = Candidate("tested_source", library.concepts[0].program, "reuse",
                          memory_id=source_id)
    library.record_transfer_result(candidate, spec, accepted=False, loss=measured_loss,
        baseline_loss=1., evidence_rows=len(observations), comparison_compute=4,
        comparison_id="paid_comparison")
    winner = Candidate("retained_incumbent", Program("const", value=0.), "refit")
    result = library.finish_transfer_episode(spec, final_candidate=winner, final_loss=1.,
        baseline_loss=1., compute_cost=8, observation_cost=len(observations),
        validation_rows=len(observations), final_status="completed")
    record, = result["returns"]
    session = UnifiedResearchSession(ResearchConfig(),
        tasks=[{"task_index": 0, "seed": 1}], library=library, measurement_budget=8)
    session.state["pending_revisits"] = [{"task_position": 0,
        "source_task_position": 0, "candidate": {"program": candidate.program.to_dict(),
            "transfer_evidence": [{"source_id": source_id, "condition": record["condition"]}]}}]

    session._refresh_source_feedback()

    assert not record["retained"]
    assert record["value_status"] == record["confirmation_status"] == "neutral"
    assert record["retained_normalized_gain"] == 0.
    offer, = session.state["pending_revisits"]
    assert offer["source_return_status"] == ["neutral"]
    assert offer["knowledge_return_score"] == 0.
    assert not any(event["action"] == "withdraw_successor_after_negative_return"
                   for event in session.state["scheduler_events"])


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


def test_joint_lineage_reports_shared_comparison_once():
    library, absolute, square = shelf()
    spec = task("joint_lineage")
    observations = rows()
    library.begin_transfer_episode(spec, observations)
    program = Program("feature", name="x")
    winner = Candidate("joint", program, "reuse", parents=(absolute, square))
    condition = transfer_condition(spec, observations)
    shared_work = [{"work_id": "joint-fit", "compute_cost": 6,
                    "source_ids": [absolute, square], "template_ids": []}]
    for source in (absolute, square):
        candidate = Candidate(f"joint-{source}", program, "reuse", memory_id=source,
                              parents=(absolute, square))
        library.record_transfer_result(candidate, spec, accepted=True, loss=0.,
            baseline_loss=1., evidence_rows=3, comparison_compute=9,
            comparison_id=1, origin={"source_id": source, "template_id": None,
                "mapping": {}, "mapping_roles": {}, "condition": condition,
                "adaptation_work": shared_work, "lineage_id": "joint-lineage",
                "joint_source_ids": [absolute, square]})
    result = library.finish_transfer_episode(spec, final_candidate=winner,
        final_loss=0., baseline_loss=1., compute_cost=20, observation_cost=3,
        validation_rows=3)
    joint, = result["joint_returns"]
    assert joint["source_ids"] == sorted([absolute, square])
    assert joint["comparison_compute"] == 9
    assert joint["adaptation_compute"] == 6
    assert joint["shared_cost_accounted_once"]


def test_source_attribution_keeps_singleton_and_joint_trials_separate():
    """A shared probe id must not overwrite a source's singleton outcome."""
    program = Program("feature", name="x")

    def attempt(source_id, joint_source_ids, loss, lineage_id):
        row = {"program": program.to_dict(), "comparison_id": "same-probe",
               "baseline_loss": 1., "loss": loss, "evidence_rows": 1,
               "compute_cost": 1}
        return {"source_id": source_id, "condition": {},
                "joint_source_ids": joint_source_ids, "lineage_id": lineage_id,
                "comparisons": {("same-probe", program): row}}

    episode = {"task_id": "attribution_scope", "attempts": {
        ("a", "singleton"): attempt("a", ["a"], 0., "singleton"),
        ("a", "joint"): attempt("a", ["a", "b"], .5, "joint"),
        ("b", "joint"): attempt("b", ["a", "b"], .5, "joint"),
    }, "work": {}}
    outcomes = dict(producer_outcomes(episode, {program}))

    a_comparisons = outcomes["a"]["comparisons"]
    assert len(a_comparisons) == 2
    singleton = next(item for item in a_comparisons if item["exclusive"])
    shared = next(item for item in a_comparisons if not item["exclusive"])
    assert singleton["retained_gain"] == pytest.approx(1.)
    assert shared["retained_gain"] == pytest.approx(.5)
    assert outcomes["a"]["retained_gain"] == pytest.approx(1.)
    assert outcomes["a"]["shared_gain"] == pytest.approx(.5)
    assert outcomes["b"]["retained_gain"] is None
    assert outcomes["b"]["shared_gain"] == pytest.approx(.5)


def test_single_source_distinct_programs_keep_separate_return_attribution():
    library = KnowledgeLibrary(concepts=[Concept("source", Program("feature", name="x"), .5,
        "source", "refit")])
    spec = task("singleton_program_competition")
    observations = rows()
    condition = transfer_condition(spec, observations)
    library.begin_transfer_episode(spec, observations)
    first = Candidate("first", Program("feature", name="x"), "reuse", memory_id="source")
    second = Candidate("second", Program("const", value=1.), "reuse", memory_id="source")
    origin = {"source_id": "source", "template_id": "shared-template", "mapping": {},
              "mapping_roles": {}, "condition": condition, "adaptation_work": []}
    library.record_transfer_result(first, spec, accepted=True, loss=.1, baseline_loss=1.,
        evidence_rows=3, comparison_compute=2, comparison_id="first", origin=origin)
    library.record_transfer_result(second, spec, accepted=False, loss=.9, baseline_loss=1.,
        evidence_rows=3, comparison_compute=2, comparison_id="second", origin=origin)
    result = library.finish_transfer_episode(spec, final_candidate=first, final_loss=.1,
        baseline_loss=1., compute_cost=8, observation_cost=3, validation_rows=3,
        final_status="completed")
    assert len(result["returns"]) == 2
    assert {item["comparisons"][0]["comparison_id"] for item in result["returns"]} == {"first", "second"}
    assert all(len(item["comparisons"]) == 1 for item in result["returns"])


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
