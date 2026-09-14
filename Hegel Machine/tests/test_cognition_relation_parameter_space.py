"""One surviving sampled radius does not exhaust the supported continuous cell."""

from dataclasses import replace
from random import Random

from hegel_machine.cognition.competing_object_experiments import assess_competing_object_probe, plan_competing_object_probe
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.program_experiment_methods import ProgramExperimentMethods
from hegel_machine.cognition.relation_revisions import propose_relation_revisions
from test_cognition_nested_relation_revisions import composition, mse


def trajectory(program, distances, seed):
    random, history, rows = Random(seed), [], []
    for distance in distances:
        fields = {"f0": 1., "f1": 0., "x0": 0., "x1": distance,
                  "q0": random.uniform(-2., 2.), "q1": random.uniform(-2., 2.)}
        row = Observation(fields, 0., tuple(history))
        rows.append(replace(row, target=program.predict(row)))
        history.append(fields)
    return rows


def test_validation_removes_two_samples_but_reopens_alternatives_inside_the_supported_cell():
    old, actual = composition(), composition(1.2)
    training = trajectory(actual, (.2, .7, 1.7, 1.9, .5, .7, 1.7, .2), 151)
    validation = trajectory(actual, (.2, 1., 1.4, 1.9, 1., .5, 1.4, .2), 157)
    spec = TaskSpec("continuous_relation", "", tuple(sorted(old.required_features())), (), ())
    initial = propose_relation_revisions(spec, training, SearchBudget(12000), [old])
    survivors = [item for item in initial if mse(item["candidate"].program, validation) < 1e-7]
    assert len(survivors) == 1
    paid = [*training, *validation]
    reopened = propose_relation_revisions(spec, paid, SearchBudget(12000), [old])
    assert len(reopened) >= 3
    assert all(mse(item["candidate"].program, paid) < 1e-7 for item in reopened)
    assert all(item["research_program"]["selection_steps"][-1]["radius_interval"] == [1., 1.4]
               for item in reopened)
    # The complete window/event computations still differ on an unobserved
    # setup inside that supported cell, so further experiments have a question.
    unobserved = trajectory(actual, (.2, 1.15, 1.25, .2), 163)
    predictions = [[item["candidate"].program.predict(row) for row in unobserved] for item in reopened]
    assert max(max(values) - min(values) for values in zip(*predictions)) > .1


def test_compact_question_retries_a_zero_response_method_within_its_generation_allowance():
    names = tuple(sorted(composition().required_features()))
    bounds = {name: ((1., 1.) if name == "f0" else (0., 0.) if name == "f1" else (-2., 2.))
              for name in names}
    spec = TaskSpec("relation_questions", "", names, (), (), feature_bounds=bounds)
    methods = ProgramExperimentMethods()

    def plan(low, high, serial):
        expression = Program("abs", children=(Program("sub", children=(
            Program("feature", name="x0"), Program("feature", name="x1"))),))
        predicate = Program("and", children=(
            Program("ge", children=(expression, Program("const", value=low))),
            Program("lt", children=(expression, Program("const", value=high)))))
        entries = [{"candidate": Candidate(str(index), composition(radius), "relation")}
                   for index, radius in enumerate((low, high))]
        return plan_competing_object_probe(spec, entries, SearchBudget(50000),
            serial=serial, methods=methods, questions=[{"question_id": "relation", "predicate": predicate,
                "history_depth": 3, "uncovered_phases": ["enter", "maintain", "exit"]}])

    initial = plan(1., 1.4, 2)
    history, pairs = [], []
    actual = composition(1.4)
    for point in initial["probe"].sequence:
        row = Observation(point, 0., tuple(history))
        row = replace(row, target=actual.predict(row))
        pairs.append((row, row))
        history.append(point)
    evidence = assess_competing_object_probe(initial, pairs, SearchBudget(5000))
    assert evidence["status"] == "distinguished"
    methods.observe(initial, evidence, command_spec=spec, task_id="source")
    assert methods.records
    # Compact recall preserves the x roles but centers the unrelated q roles:
    # it switches the condition while all full event/window responses stay zero.
    # The same allowance must still permit a useful new preparation.
    next_plan = plan(1.2, 1.4, 3)
    assert next_plan["probe"] is not None
    assert next_plan["proposal_source"] == "program_disagreement"
    assert next_plan["method_search_cost"] > 0
    assert 0 < next_plan["generation_cost"] <= 6000
    assert next_plan["predicted_separation"] > 1e-7
    assert next_plan["question_hypotheses"] != next_plan["hypotheses"]
    history = []
    for position, point in enumerate(next_plan["probe"].sequence):
        row = Observation(point, 0., tuple(history))
        for hypothesis in next_plan["hypotheses"]:
            assert hypothesis["predictions"][position] == Program.from_dict(hypothesis["program"]).predict(row)
        history.append(point)
