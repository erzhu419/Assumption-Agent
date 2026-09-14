"""Mechanisms and empirical concept merges change on actual new evidence."""

import json

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.knowledge_evolution import KnowledgeEvolution
from hegel_machine.cognition.language import EvaluationContext, Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec


def feature(name):
    return Program("feature", name=name)


def op(name, left, right):
    return Program(name, children=(left, right))


def task(name, features):
    return TaskSpec(name, "Predict the acquired response", features, (), ())


def learned_call(library, name, program, rows):
    spec = task(name, tuple(sorted(rows[0].features)))
    identifier, _ = library.learn(Candidate(name, program, "composition"), spec, rows)
    return next(atom for atom in library.callable_atoms(spec, limit=32)
                if atom.definition.source_concept_id == identifier and atom.name.endswith(":abstract:0:definition"))


def mechanisms():
    library = KnowledgeLibrary()
    square = op("mul", feature("x"), feature("x"))
    linear = op("mul", Program("const", value=2.), feature("z"))
    first = learned_call(library, "quadratic_component", square,
                         [Observation({"x": x}, x * x) for x in (-2., -1., 0., 1., 2.)])
    second = learned_call(library, "linear_component", linear,
                          [Observation({"z": z}, 2 * z) for z in (-2., -1., 0., 1., 2.)])
    return library, first, second


def process_rows(regime, coefficient, values=(-2., -1., 1., 2.)):
    return [Observation({"x": x, "z": z, "regime": regime}, x * x + coefficient * z)
            for x in values for z in values]


def test_new_component_law_preserves_the_other_learned_call_and_the_successful_old_domain():
    library, first, second = mechanisms()
    program = op("add", first, second)
    support = process_rows(-1., 2.)
    counterexamples = process_rows(1., 3.)
    evolution, budget = KnowledgeEvolution(), SearchBudget(6000)
    candidates = evolution.local_revisions(program, task("changed_process", ("x", "z", "regime")),
        counterexamples, budget, support_observations=support)
    assert candidates
    best = min(candidates, key=lambda candidate: loss(candidate.program, counterexamples, 0., .5))
    assert best.program.children[0] == first
    assert best.program.children[1].op == "if"
    assert best.program.children[1].children[1] == second
    assert loss(best.program, support, 0., .5) < 1e-12
    assert loss(best.program, process_rows(-.5, 2., (-1.7, .4, 1.3)), 0., .5) < 1e-12
    assert loss(best.program, process_rows(.7, 3., (-1.4, .3, 1.6)), 0., .5) < 1e-12
    assert loss(program, counterexamples, 0., .5) > 1.
    assert evolution.last_event["compute_cost"] == budget.spent <= 6000
    assert any(item["old_support_preserved"] and item["loss_after"] < 1e-12
               for item in evolution.last_event["tested"])
    learned_id, _ = library.learn(best, task("changed_process", ("x", "z", "regime")), support + counterexamples)
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    learned = next(concept for concept in restored.concepts if concept.concept_id == learned_id)
    assert first.definition.source_concept_id in learned.parents
    assert loss(learned.program, support + counterexamples, 0., .5) < 1e-12


def test_indistinguishable_old_and_new_conditions_cannot_erase_successful_mechanisms():
    _, first, second = mechanisms()
    program = op("add", first, second)
    support, contradictory = process_rows(0., 2.), process_rows(0., 3.)
    evolution = KnowledgeEvolution()
    candidates = evolution.local_revisions(program, task("conflict", ("x", "z", "regime")),
        contradictory, SearchBudget(6000), support_observations=support)
    assert not candidates
    assert evolution.last_event["condition_learning"]["condition"] is None
    small = SearchBudget(3)
    assert evolution.local_revisions(program, task("small", ("x", "z", "regime")),
        contradictory, small, support_observations=support) == []
    assert small.spent <= small.limit


def test_empirical_merge_shares_execution_and_new_task_counterexamples_restore_separate_definitions():
    library = KnowledgeLibrary()
    x = feature("x")
    power = x
    for _ in range(8):
        power = op("mul", power, x)
    first = learned_call(library, "nonlinear", op("add", power, x),
        [Observation({"x": value}, value ** 9 + value) for value in (-2., -1., 0., 1., 2.)])
    second = learned_call(library, "linear", op("mul", Program("const", value=2.), x),
        [Observation({"x": value}, 2 * value) for value in (-2., -1., 0., 1., 2.)])
    original = op("add", first, second)
    acquired = [Observation({"x": value}, value ** 9 + 3 * value) for value in (0., 1.) * 4]
    evolution, budget = KnowledgeEvolution(), SearchBudget(1000)
    before_library = json.dumps(library.to_dict(), sort_keys=True)
    merged = evolution.merge_equivalent(original, task("first_domain", ("x",)), acquired, budget)
    assert len(merged) == 1 and merged[0].program != original
    assert json.dumps(library.to_dict(), sort_keys=True) == before_library
    assert evolution.last_event["compute_cost"] == budget.spent <= 1000
    before, after = EvaluationContext(acquired[1]), EvaluationContext(acquired[1])
    assert before.evaluate(original) == after.evaluate(merged[0].program)
    assert before.call_evaluations == 2 and after.call_evaluations == 1 and after.cache_hits >= 1
    assert after.primitive_steps < before.primitive_steps
    # Outside the acquired range, the original distinct computations execute.
    outside = Observation({"x": 2.}, 2. ** 9 + 6.)
    assert merged[0].program.predict(outside) == original.predict(outside)

    # The merged result can itself become a learned callable definition. Its
    # original two source definitions survive both knowledge and evolution JSON.
    learned_merge = learned_call(library, "merged_model", merged[0].program, acquired)
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    restored = KnowledgeEvolution.from_dict(json.loads(json.dumps(evolution.to_dict())))
    current = Program.from_dict(json.loads(json.dumps(learned_merge.to_dict())))
    assert current.op == "call"
    fresh = [Observation({"x": value}, value ** 9 + 3 * value) for value in (.25, .5, .75)]
    assert loss(current, fresh, 0., .5) > .05
    split_budget = SearchBudget(100)
    split = restored.reopen_merges(current, task("second_domain", ("x",)), fresh, split_budget)
    assert split and split[0].program == original
    assert loss(split[0].program, fresh + acquired, 0., .5) == 0.
    assert split_budget.spent == 2 * len(fresh)
    assert restored.merges[0]["status"] == "counterexample_reopened"
    assert len(restored.merges[0]["counterexamples"]) == len(fresh)
    assert first.definition != second.definition


def test_unequal_current_measurements_do_not_merge_calls_or_destroy_definitions():
    _, first, second = mechanisms()
    program = op("add", first, second)
    evolution, budget = KnowledgeEvolution(), SearchBudget(100)
    assert evolution.merge_equivalent(program, task("distinguished", ("x", "z", "regime")),
                                     process_rows(0., 2.), budget) == []
    assert evolution.merges == []
    assert evolution.last_event["compute_cost"] == budget.spent <= budget.limit
