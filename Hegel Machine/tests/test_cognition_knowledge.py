import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import EmpiricalScope, KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec


def spec(task, names=("x",), context=None):
    return TaskSpec(task, "predict measured output", names, (), (), context=context or {})


def const(value):
    return Program("const", value=value)


def feature(name):
    return Program("feature", name=name)


def add(left, right):
    return Program("add", children=(left, right))


def mul(left, right):
    return Program("mul", children=(left, right))


def error(program, rows):
    return sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows)


def test_parameter_template_transfers_roles_and_refits_from_few_observations():
    library = KnowledgeLibrary()
    program = add(add(mul(const(2), feature("x")), mul(const(-1), feature("z"))), const(0.5))
    source_rows = [Observation({"x": x, "z": z}, 2 * x - z + 0.5)
                   for x, z in ((0, 0), (1, 0), (0, 1), (2, -1))]
    source_id, _ = library.learn(Candidate("source", program, "refit"), spec("source", ("x", "z")), source_rows)
    target_rows = [Observation({"u": u, "v": v}, 3.5 * u + 0.7 * v - 2)
                   for u, v in ((0, 0), (1, 0), (0, 1), (-1, 2))]
    budget = SearchBudget(300)
    candidates = library.propose(spec("new", ("u", "v")), target_rows, budget)
    best = min(candidates, key=lambda candidate: error(candidate.program, target_rows))
    unseen = Observation({"u": 4.0, "v": -3.0}, 9.9)
    assert best.program.predict(unseen) == pytest.approx(unseen.target, abs=1e-6)
    assert best.memory_id == source_id and source_id in best.parents
    assert 0 < budget.spent <= budget.limit
    assert budget.spent % len(target_rows) == 0
    assert library.propose(spec("new", ("u", "v")), target_rows, SearchBudget(300), structural_transfer=False) == []


def test_extracted_nonlinear_subexpression_can_become_a_new_predictive_coordinate():
    library = KnowledgeLibrary()
    absolute = Program("abs", children=(feature("x"),))
    source = add(mul(const(2), absolute), const(1))
    source_id, _ = library.learn(Candidate("source", source, "representation"), spec("source"))
    assert any(template.kind == "subexpression" and template.pattern.op == "abs" for template in library.abstracts)
    rows = [Observation({"u": u, "v": v}, 4 * abs(u) + 1.5 * v - 0.2)
            for u in (-2, -1, 0, 1, 2) for v in (-1, 1)]
    candidates = library.propose(spec("target", ("u", "v")), rows, SearchBudget(2000))
    best = min(candidates, key=lambda candidate: error(candidate.program, rows))
    assert error(best.program, rows) < 1e-12
    assert any(parent.startswith(source_id + ":abstract:") for parent in best.parents)
    child_id, created = library.learn(best, spec("target", ("u", "v")), rows)
    assert created and child_id != source_id
    child = next(concept for concept in library.concepts if concept.concept_id == child_id)
    assert child.generation == 1
    assert library.concepts[0].reuse_history[-1]["kind"] == "refitted_abstraction"
    assert "target" not in library.concepts[0].successful_tasks


def test_threshold_parameters_are_searched_from_current_measurements():
    library = KnowledgeLibrary()
    program = Program("ge", children=(feature("x"), const(0.2)))
    library.learn(Candidate("source", program, "relation"), spec("source"))
    rows = [Observation({"u": x}, float(x >= 1.7)) for x in (0, 1, 2, 3)]
    candidates = library.propose(spec("target", ("u",)), rows, SearchBudget(500))
    best = min(candidates, key=lambda candidate: error(candidate.program, rows))
    assert error(best.program, rows) == 0
    assert best.program.predict(Observation({"u": 0.5}, 0)) == 0
    assert best.program.predict(Observation({"u": 2.5}, 1)) == 1


def test_scope_learns_an_executable_boundary_and_revokes_a_supported_counterexample():
    library = KnowledgeLibrary()
    original = feature("x")
    rows = [Observation({"x": x}, x) for x in (-1.0, 0.0, 1.0)]
    source_id, _ = library.learn(Candidate("source", original, "refit"), spec("same"), rows)
    event = library.record_counterexample(source_id, spec("same"), 9.0, 0.0, Observation({"x": 3.0}, 0.0))
    assert event["action"] == "restricted"
    assert event["scope_change"] == "restricted_interval"
    assert event["learned_boundary"]["upper"] == 2.0
    scoped = library.retrieve(spec("same"))[0].program
    assert scoped.predict(Observation({"x": 0.5}, 0.5)) == 0.5
    assert scoped.predict(Observation({"x": 3.0}, 0.0)) == 0.0
    assert library.concepts[0].excluded_tasks == []
    event = library.record_counterexample(source_id, spec("same"), 12.25, 0.0, Observation({"x": 0.5}, 4.0))
    assert event["action"] == "revoked"
    assert library.retrieve(spec("same")) == []
    assert all(template.status == "revoked" for template in library.abstracts)


def test_overlapping_features_in_another_system_do_not_revoke_original_knowledge():
    library = KnowledgeLibrary()
    common_interface = {"observation_unit": "dimensionless", "query_reset": "independent_sequence"}
    rows = [Observation({"x": x}, x) for x in (-1.0, 0.0, 1.0)]
    source_id, _ = library.learn(Candidate("source", feature("x"), "refit"), spec("first", context=common_interface), rows)
    event = library.record_counterexample(source_id, spec("other", context=common_interface), 9, 0, Observation({"x": 0.5}, 3.5))
    assert event["action"] == "restricted"
    assert event["scope_change"] == "restricted_context"
    assert library.concepts[0].status == "restricted"
    assert library.retrieve(spec("first", context=common_interface))
    assert library.retrieve(spec("other", context=common_interface)) == []


def test_explicit_system_identity_and_observed_interval_can_support_revocation():
    library = KnowledgeLibrary()
    context = {"system_id": "same-measured-device"}
    source_id, _ = library.learn(Candidate("source", feature("x"), "refit"), spec("one", context=context),
                                 [Observation({"x": x}, x) for x in (-1.0, 1.0)])
    event = library.record_counterexample(source_id, spec("two", context=context), 4, 0, Observation({"x": 0.0}, 2.0))
    assert event["action"] == "revoked"


def test_direct_reuse_measures_new_task_scope_before_classifying_counterexamples():
    library = KnowledgeLibrary()
    source_id, _ = library.learn(Candidate("source", feature("x"), "refit"), spec("source"),
                                 [Observation({"x": x}, x) for x in (-1.0, 1.0)])
    reused_spec = spec("reused")
    reused_rows = [Observation({"x": x}, x) for x in (4.0, 5.0)]
    library.record_use(source_id, reused_spec.task_id, spec=reused_spec, observations=reused_rows)
    reused_scope = next(scope for scope in library.concepts[0].scopes if scope.task_id == "reused")
    assert reused_scope.observed_rows == 2
    assert reused_scope.support == [{"name": "x", "lag": 0, "lower": 4.0, "upper": 5.0}]

    outside = library.record_counterexample(source_id, reused_spec, 49.0, 0.0,
                                            Observation({"x": 7.0}, 0.0))
    assert outside["same_supported_context"] and not outside["inside_observed_support"]
    assert outside["action"] == "restricted" and outside["scope_change"] == "restricted_interval"
    assert library.retrieve(spec("source"))
    inside = library.record_counterexample(source_id, reused_spec, 9.0, 0.0,
                                           Observation({"x": 4.5}, 7.5))
    assert inside["inside_observed_support"] and inside["action"] == "revoked"


def test_guard_fallback_observations_do_not_expand_original_program_support():
    library = KnowledgeLibrary()
    source_spec = spec("source", context={"system_id": "same-device"})
    source_id, _ = library.learn(Candidate("source", feature("x"), "refit"), source_spec,
                                 [Observation({"x": x}, x) for x in (-1.0, 0.0, 1.0)])
    library.record_counterexample(source_id, source_spec, 9.0, 0.0, Observation({"x": 3.0}, 0.0))
    new_spec = spec("reused", context={"system_id": "same-device"})
    guarded = library.retrieve(new_spec)[0].program
    actual_rows = [Observation({"x": 0.5}, 0.5), Observation({"x": 3.0}, 0.0)]
    assert all(guarded.predict(row) == row.target for row in actual_rows)
    library.record_use(source_id, new_spec.task_id, spec=new_spec, observations=actual_rows)
    scope = next(scope for scope in library.concepts[0].scopes if scope.task_id == "reused")
    assert scope.observed_rows == 1
    assert scope.support == [{"name": "x", "lag": 0, "lower": 0.5, "upper": 0.5}]
    assert scope.conditions == [{"name": "x", "lag": 0, "lower": None, "upper": 2.0}]
    assert scope.supports(actual_rows[0]) and not scope.supports(actual_rows[1])


def test_empirical_support_also_respects_the_executable_guard_conditions():
    scope = EmpiricalScope.measured(spec("measured"), feature("x"),
                                    [Observation({"x": x}, x) for x in (-1.0, 1.0)])
    scope.conditions = [{"name": "x", "lag": 0, "lower": 0.0, "upper": 0.5}]
    assert scope.supports(Observation({"x": 0.0}, 0.0))
    assert scope.supports(Observation({"x": 0.25}, 0.25))
    assert not scope.supports(Observation({"x": -0.1}, -0.1))
    assert not scope.supports(Observation({"x": 0.5}, 0.5))


def test_exact_deletion_removes_descendant_templates_and_preserves_independent_knowledge():
    library = KnowledgeLibrary()
    source_id, _ = library.learn(Candidate("source", feature("x"), "refit"), spec("source"))
    template_id = library.abstracts[0].template_id
    child_id, _ = library.learn(Candidate("child", mul(const(2), feature("x")), "abstract_reuse",
                                           parents=(source_id, template_id), memory_id=source_id), spec("child"))
    grandchild_id, _ = library.learn(Candidate("grandchild", mul(const(3), feature("x")), "abstract_reuse",
                                                parents=(child_id,), memory_id=child_id), spec("grandchild"))
    unrelated_id, _ = library.learn(Candidate("other", const(7), "refit"), spec("other"))
    removed = library.without({source_id})
    assert [concept.concept_id for concept in removed.concepts] == [unrelated_id]
    assert {source_id, child_id, grandchild_id, template_id}.issubset(removed.deleted_ids)
    assert not removed.abstracts
    assert len(library.concepts) == 4
    assert removed.next_concept_index == library.next_concept_index
    removed.record_use(unrelated_id, "intervention-only")
    assert "intervention-only" not in library.concepts[-1].successful_tasks


def test_old_json_library_and_new_scope_template_library_both_roundtrip():
    library = KnowledgeLibrary()
    source_id, _ = library.learn(Candidate("source", add(feature("x"), const(2)), "refit"), spec("source"),
                                 [Observation({"x": 0.0}, 2.0), Observation({"x": 1.0}, 3.0)])
    payload = json.loads(json.dumps(library.to_dict()))
    restored = KnowledgeLibrary.from_dict(payload)
    assert restored.to_dict() == payload
    old_row = {key: value for key, value in payload["concepts"][0].items()
               if key not in {"generation", "scopes", "reuse_history"}}
    old = KnowledgeLibrary.from_dict({"concepts": [old_row]})
    assert old.retrieve(spec("next"))[0].program.predict(Observation({"x": 1.0}, 0.0)) == 3.0
    assert old.abstracts[0].source_concept_id == source_id
    assert old.propose(spec("new", ("u",)), [Observation({"u": 0.0}, 4.0), Observation({"u": 1.0}, 6.0)], SearchBudget(200))
