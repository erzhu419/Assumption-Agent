import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec


def spec(task="device", names=("x", "z")):
    return TaskSpec(task, "measure output", names, (), (), context={"system_id": task})


def learn(rows, task=None):
    library = KnowledgeLibrary(action_experience={"numeric": {"repeat": {"gain": 0.2, "cost": 8}}})
    program = Program("feature", name="x")
    concept_id, _ = library.learn(Candidate("source", program, "refit"), task or spec(), rows)
    return library, concept_id


def test_joint_condition_preserves_old_law_inside_its_observed_box():
    rows = [Observation({"x": x, "z": z}, x) for x, z in [(-1, -1), (-1, 1), (1, -1), (0, 0)]]
    library, concept_id = learn(rows)
    failure = Observation({"x": 0.8, "z": 0.8}, -3)
    budget = SearchBudget(300)
    event = library.record_counterexample(concept_id, spec(), 4, 0, failure, budget=budget)
    assert event["inside_observed_support"]
    assert event["scope_change"] == "restricted_condition"
    assert event["condition_compute_cost"] == budget.spent > 0
    concept = library.concepts[0]
    condition = Program.from_dict(concept.scopes[0].predicate)
    assert condition.required_features() == {"x", "z"}
    guarded = library.retrieve(spec())[0].program
    assert all(guarded.predict(row) == row.target for row in rows)
    assert not condition.predict(failure)
    assert guarded.predict(failure) != -3  # The one-error repair is still a proposal.
    candidates = library.propose(spec(), [*rows, failure], SearchBudget(1800))
    assert any(all(abs(candidate.program.predict(row) - row.target) < 1e-5 for row in [*rows, failure])
               for candidate in candidates)
    assert all(candidate.program != concept.program for candidate in candidates)
    # A subsequent error in the fallback region preserves the valid old domain.
    further = library.record_counterexample(concept_id, spec(), 4, 0,
                                           Observation({"x": 0.9, "z": 0.9}, -3))
    assert further["scope_change"] == "outside_learned_condition"
    assert library.retrieve(spec())


def test_numeric_context_coordinate_is_learned_from_actual_agreement():
    rows = [Observation({"x": x, "z": z}, x) for x, z in [(-1, 0), (1, 0.1), (0, 0.2)]]
    library, concept_id = learn(rows)
    fresh = [Observation({"x": -0.5, "z": 0.15}, -0.5),
             Observation({"x": 0.3, "z": 0.8}, -3), Observation({"x": 0.6, "z": 0.9}, -3)]
    event = library.record_counterexample(concept_id, spec(), 4, 0, fresh[1],
                                         supporting_observations=fresh, budget=SearchBudget(300))
    assert event["scope_change"] == "restricted_condition"
    scope = library.concepts[0].scopes[0]
    condition = Program.from_dict(scope.predicate)
    assert condition.required_features() == {"z"}
    assert len(scope.support_examples) == 4
    assert len(scope.counter_examples) == 2
    assert condition.predict(Observation({"x": 0.9, "z": 0.1}, 0.9))
    assert not condition.predict(Observation({"x": -0.8, "z": 0.95}, -3))


def test_identical_control_and_history_conflict_still_revokes():
    rows = [Observation({"x": x, "z": z}, x) for x, z in [(-1, 0), (1, 0), (0, 1)]]
    library, concept_id = learn(rows)
    event = library.record_counterexample(concept_id, spec(), 4, 0, Observation({"x": 1, "z": 0}, -3),
                                         budget=SearchBudget(300))
    assert event["action"] == "revoked"
    assert library.retrieve(spec()) == []
    assert library.propose(spec(), rows, SearchBudget(300)) == []


def test_conditional_templates_transfer_roles_and_still_need_target_evidence():
    rows = [Observation({"x": x, "z": z}, x) for x, z in [(-1, -1), (-1, 1), (1, -1), (0, 0)]]
    library, concept_id = learn(rows)
    library.record_counterexample(concept_id, spec(), 4, 0, Observation({"x": 0.8, "z": 0.8}, -3))
    target = spec("other", ("u", "v"))
    target_rows = [Observation({"u": x, "v": z}, x) for x, z in [(-1, -1), (-1, 1), (1, -1), (0, 0)]]
    target_rows.append(Observation({"u": 0.8, "v": 0.8}, -3))
    proposed = library.propose(target, target_rows, SearchBudget(3000))
    matched = [candidate for candidate in proposed if ":condition:" in candidate.candidate_id]
    assert matched and any(all(abs(candidate.program.predict(row) - row.target) < 1e-5 for row in target_rows)
                           for candidate in matched)
    # Changing the environment's actual response invalidates a frozen transfer;
    # recording the measured rejection does not promote the source for this task.
    best = min(matched, key=lambda candidate: sum((candidate.program.predict(row) - row.target) ** 2 for row in target_rows))
    wrong = [Observation(row.features, row.target + 7) for row in target_rows]
    measured = sum((best.program.predict(row) - row.target) ** 2 for row in wrong) / len(wrong)
    assert measured > 40
    library.record_transfer_result(best, target, accepted=False, loss=measured, baseline_loss=1)
    assert target.task_id not in library.concepts[0].successful_tasks
    assert any(event["accepted"] is False for template in library.abstracts for event in template.transfer_history)


def test_condition_samples_and_action_experience_roundtrip_and_copy_independently():
    rows = [Observation({"x": x, "z": z}, x) for x, z in [(-1, -1), (-1, 1), (1, -1), (0, 0)]]
    library, concept_id = learn(rows)
    library.record_counterexample(concept_id, spec(), 4, 0, Observation({"x": 0.8, "z": 0.8}, -3))
    payload = json.loads(json.dumps(library.to_dict()))
    restored = KnowledgeLibrary.from_dict(payload)
    assert restored.to_dict() == payload
    assert restored.retrieve(spec())[0].program == library.retrieve(spec())[0].program
    independent = restored.without(set())
    independent.action_experience["numeric"]["repeat"]["gain"] = 9
    assert restored.action_experience["numeric"]["repeat"]["gain"] == pytest.approx(0.2)


def test_condition_learning_retains_guarded_history_after_reset_and_uses_no_future_target():
    from hegel_machine.cognition.condition_learning import learn_condition

    support = [Observation({'u': 0., '__target__': 100.}, 7., ({'u': value},))
               for value in (-2., -1.)]
    counterexamples = [Observation({'u': 0., '__target__': -100.}, -7., ({'u': value},))
                       for value in (1., 2.)]
    counterexamples.append(Observation({'u': 0., '__target__': -100.}, -7.))
    learned = learn_condition(support, counterexamples, SearchBudget(100))
    assert learned['status'] == 'separated'
    assert learned['scope_purpose'] == 'application'
    assert learned['measured_coverage']['scope_purpose'] == 'measured_coverage'
    condition = Program.from_dict(learned['condition'])
    assert condition.minimum_history_depth == 0
    assert '__target__' not in condition.required_features()
    assert all(condition.predict(row) for row in support)
    assert not any(condition.predict(row) for row in counterexamples)
    assert condition.predict(Observation({'u': 0.}, -999., ({'u': -1.5},)))
    assert not condition.predict(Observation({'u': 0.}, 999., ({'u': 1.5},)))
    assert not condition.predict(Observation({'u': 0.}, 999.))


def test_inseparable_application_scope_stays_unknown_and_coverage_creates_no_predicate():
    from hegel_machine.cognition.condition_learning import learn_condition

    support = [Observation({'u': 0.}, 0.), Observation({'u': 1.}, 1.)]
    counterexamples = [Observation({'u': 0.}, 7.)]
    unresolved = learn_condition(support, counterexamples, SearchBudget(100))
    assert unresolved['status'] == 'not_separable'
    assert unresolved['scope_status'] == 'unknown'
    coverage = learn_condition(support, counterexamples, SearchBudget(0),
                              scope_purpose='measured_coverage')
    assert coverage['status'] == 'measured_only' and coverage['condition'] is None
    assert coverage['scope_status'] == 'unknown'
    assert coverage['measured_coverage']['row_count'] == 3
    assert coverage['measured_coverage']['feature_ranges'] == {'u': [0., 1.]}


def test_relative_utility_mean_gain_differs_from_pointwise_without_world_support():
    from hegel_machine.cognition.condition_learning import learn_relative_utility_condition

    rows = [Observation({'u': 0.}, 0.) for _ in range(3)]
    baseline, proposed = [3., 3., 1.], [1., 1., 2.]
    pointwise = learn_relative_utility_condition(rows, baseline, proposed, SearchBudget(100))
    assert pointwise['status'] == 'not_separable' and pointwise['selector'] is None
    assert pointwise['strict_improvement_rows'] == 2
    mean = learn_relative_utility_condition(rows, baseline, proposed, SearchBudget(100),
                                            objective='mean_loss')
    assert mean['status'] == 'mean_improvement'
    assert Program.from_dict(mean['selector']).predict(Observation({'u': 0.}, 999.)) == 1.
    assert mean['nonimprovement_rows'] == 1 and mean['worst_loss_increase'] == 3.
    assert mean['scope_purpose'] == 'relative_utility'
    assert mean['application_scope_status'] == 'not_assessed'
    assert mean['world_mse_threshold'] == 1e-10
    assert mean['candidate_mse'] > mean['world_mse_threshold']
    assert mean['measured_coverage']['independent_confirmation'] is False
