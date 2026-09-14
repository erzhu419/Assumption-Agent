"""Transfer scheduling with real fits, held-out values and charged budgets."""

import json

import pytest

from hegel_machine.cognition.abstractions import extract_templates, fit_template
from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec


def spec(task, names=("x",)):
    return TaskSpec(task, "predict observed output", names, (), ())


def const(value):
    return Program("const", value=value)


def feature(name):
    return Program("feature", name=name)


def add(left, right):
    return Program("add", children=(left, right))


def mul(left, right):
    return Program("mul", children=(left, right))


def affine_shelf(count=12):
    library = KnowledgeLibrary()
    for index in range(count):
        library.learn(Candidate(str(index), add(mul(const(index + 2), feature("x")), const(index + 1)), "refit"),
                      spec(f"source_{index}"))
    return library


def prior_enumeration(library, target_spec, observations, budget):
    """The previous insertion-order schedule, using the same current fitter.

    This comparator even receives the fitter's new exact-trial reuse. Only its
    ordering and lack of cross-source structure sharing are held at v2.
    """
    programs = []
    for template in library.abstracts:
        for mapping in template.mappings(target_spec.feature_names):
            programs.extend(fit_template(template, mapping, observations, budget))
    return min(programs, key=lambda program: loss(program, observations, initial_prediction(observations, .5), .5))


def best_candidate(candidates, rows):
    return min(candidates, key=lambda candidate: loss(candidate.program, rows, initial_prediction(rows, .5), .5))


def test_late_nonlinear_structure_beats_parameter_alias_enumeration_under_same_budget():
    library = affine_shelf()
    source = add(mul(const(2), Program("abs", children=(feature("x"),))), const(1))
    source_id, _ = library.learn(Candidate("absolute", source, "representation"), spec("absolute"))
    rows = [Observation({"u": u}, 4 * abs(u) + 1) for u in (-3., -2., -1., -.2, .2, 1., 2., 3.)]
    held_out = [Observation({"u": u}, 4 * abs(u) + 1) for u in (-2.7, -1.3, -.5, .5, 1.3, 2.7)]
    target = spec("new_parameter_and_role", ("u",))
    current_budget, prior_budget = SearchBudget(160), SearchBudget(160)
    candidates = library.propose(target, rows, current_budget, preserve_parameters=False)
    current = best_candidate(candidates, rows)
    previous = prior_enumeration(library, target, rows, prior_budget)

    assert current.memory_id == source_id
    assert loss(current.program, held_out, 0, .5) < 1e-12
    assert loss(previous, held_out, 0, .5) > 10
    assert current_budget.spent == 56 < prior_budget.spent == 160
    assert library.last_search["template_bindings"] == 14
    assert library.last_search["unique_fit_shapes"] == 3
    assert library.last_search["distinct_programs"] == 2


def history_rows(offset=0.0):
    rows = []
    for trajectory in range(3):
        history = []
        for step in range(8):
            u = ((step * 3 + trajectory * 7) % 11 - 5) * .6 + offset
            v = ((step * 7 + trajectory * 2) % 13 - 6) * .8 - offset
            target = 4 * u - 1.5 * history[-2]["v"] + 1 if len(history) >= 2 else 2 * u + .5
            rows.append(Observation({"u": u, "v": v}, target, tuple(history)))
            history.append({"u": u, "v": v, "__target__": target})
    return rows


def test_complete_trajectories_refit_renamed_history_and_startup_branches():
    library = affine_shelf()
    mature = add(mul(const(2), feature("x")), mul(const(3), Program("lag", name="z", lag=2)))
    startup = add(mul(const(7), feature("x")), const(-2))
    source = Program("if", children=(Program("available", lag=2), mature, startup))
    source_id, _ = library.learn(Candidate("history", source, "representation"), spec("history", ("x", "z")))
    rows, held_out = history_rows(), history_rows(.27)
    target = spec("renamed_trajectory", ("u", "v"))
    current_budget, prior_budget = SearchBudget(500), SearchBudget(500)
    candidates = library.propose(target, rows, current_budget, preserve_parameters=False)
    current = best_candidate(candidates, rows)
    previous = prior_enumeration(library, target, rows, prior_budget)

    assert current.memory_id == source_id
    assert current.program.minimum_history_depth == 0
    assert current.program.history_depth == 2
    assert loss(current.program, held_out, 0, .5) < 1e-12
    assert loss(previous, held_out, 0, .5) > 15
    assert current_budget.spent == 300 < prior_budget.spent <= prior_budget.limit
    ranked = library.last_search["ranked_shapes"]
    assert ranked[0]["mapping"] == {"role_0": "u", "role_1": "v"}
    assert ranked[0]["fitted_rows"] == len(rows)
    assert len(ranked[0]["trajectory_losses"]) == 3
    # Bare lag hypotheses also retain startup losses instead of being scored
    # only on the mature rows from which their parameters could be fitted.
    bare_lag = [item for item in ranked if item["history_depth"] == 2 and item["fitted_rows"] == 18]
    assert bare_lag and all(len(item["trajectory_losses"]) == 3 for item in bare_lag)


def test_shared_shape_preserves_sources_and_refits_again_when_data_change():
    library = affine_shelf(2)
    target = spec("renamed", ("u",))
    rows = [Observation({"u": u}, 7 * u + 2) for u in (-2., -1., 1., 2.)]
    first_budget = SearchBudget(100)
    first = library.propose(target, rows, first_budget, preserve_parameters=False)
    assert library.last_search["unique_fit_shapes"] == 1
    assert library.last_search["shared_fit_bindings"] == 1
    assert len({candidate.program for candidate in first}) == 1
    assert {candidate.memory_id for candidate in first} == {concept.concept_id for concept in library.concepts}
    assert first_budget.spent == 12

    changed_rows = [Observation({"u": u}, -3 * u + .4) for u in (-2., -1., 1., 2.)]
    second_budget = SearchBudget(100)
    second = library.propose(target, changed_rows, second_budget, preserve_parameters=False)
    assert second_budget.spent == first_budget.spent
    assert second[0].program != first[0].program
    assert second[0].program.predict(Observation({"u": 3.0}, -8.6)) == pytest.approx(-8.6, abs=1e-7)
    assert {candidate.candidate_id for candidate in first}.isdisjoint(candidate.candidate_id for candidate in second)
    assert all("renamed" not in concept.successful_tasks for concept in library.concepts)


def test_equal_sample_outputs_stay_distinct_and_tested_failure_changes_retrieval_order():
    library = KnowledgeLibrary()
    absolute_id, _ = library.learn(Candidate("absolute", Program("abs", children=(feature("x"),)), "representation"),
                                   spec("absolute"))
    square_id, _ = library.learn(Candidate("square", mul(feature("x"), feature("x")), "representation"), spec("square"))
    rows = [Observation({"x": x}, x * x) for x in (-1., 0., 1.)]
    target = spec("ambiguous_samples")
    candidates = library.propose(target, rows, SearchBudget(100))
    assert library.last_search["unique_fit_shapes"] == 2
    assert {candidate.memory_id for candidate in candidates} == {absolute_id, square_id}
    # On a fresh measurement x=2 these two structures cease to agree.
    fresh = Observation({"x": 2.0}, 4.0)
    absolute = next(candidate for candidate in candidates if candidate.memory_id == absolute_id)
    square = next(candidate for candidate in candidates if candidate.memory_id == square_id)
    assert square.program.predict(fresh) == pytest.approx(4.0, abs=1e-7)
    assert absolute.program.predict(fresh) == pytest.approx(2.0, abs=1e-7)
    library.record_transfer_result(absolute, target, accepted=False, loss=4.0, baseline_loss=1.0)
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    # Only one structural screen fits into this budget. The actual failed
    # transfer now puts the square first, despite its larger expression.
    shortlisted = restored.propose(spec("following"), rows, SearchBudget(12))
    assert {candidate.memory_id for candidate in shortlisted} == {square_id}
    assert restored.abstracts[0].transfer_history[-1]["accepted"] is False
    assert restored.concepts[0].successful_tasks == ["absolute"]


def test_excluded_law_remains_a_refittable_tool_without_revalidating_source():
    library = KnowledgeLibrary()
    source = add(mul(const(2), feature("x")), const(1))
    source_rows = [Observation({"x": x}, 2 * x + 1) for x in (-1., 0., 1.)]
    source_id, _ = library.learn(Candidate("source", source, "refit"), spec("source"), source_rows)
    target = spec("different_system")
    library.record_counterexample(source_id, target, 25.0, 0.0, Observation({"x": 0.0}, -4.0))
    assert library.retrieve(target) == []
    rows = [Observation({"x": x}, 3 * x - 4) for x in (-1., 0., 1.)]
    candidate = best_candidate(library.propose(target, rows, SearchBudget(100)), rows)
    library.record_transfer_result(candidate, target, accepted=True, loss=0.0, baseline_loss=6.0)
    child_id, created = library.learn(candidate, target, rows)
    assert created and child_id != source_id
    assert candidate.program.predict(Observation({"x": 3.0}, 5.0)) == pytest.approx(5.0, abs=1e-7)
    original = library.concepts[0]
    assert original.successful_tasks == ["source"]
    assert original.excluded_tasks == ["different_system"]
    assert library.abstracts[0].transfer_history[-1]["accepted"] is True
    assert library.retrieve(spec("source"))


def test_guarded_history_threshold_refits_without_reading_missing_startup_history():
    predicate = Program("ge", children=(Program("lag", name="x", lag=1), const(.2)))
    source = Program("if", children=(Program("available", lag=1), predicate, const(0)))
    template = extract_templates(source, "source")[0]
    rows = []
    for offset in (0.0, .4):
        history = []
        for u in (-1., 3., 1., 2., 0., 4.):
            u += offset
            target = float(history[-1]["u"] >= 1.7) if history else 0.0
            rows.append(Observation({"u": u}, target, tuple(history)))
            history.append({"u": u, "__target__": target})
    budget = SearchBudget(4000)
    details = {}
    programs = fit_template(template, {"role_0": "u"}, rows, budget, report=details)
    best = min(programs, key=lambda program: details["losses"][program])
    assert details["fitted_rows"] == len(rows)
    assert loss(best, rows, 0.0, .5) == 0.0
    assert best.predict(Observation({"u": 2.5}, 0.0)) == 0.0
    assert best.predict(Observation({"u": 0.0}, 1.0, ({"u": 2.5},))) == 1.0
    assert best.predict(Observation({"u": 0.0}, 0.0, ({"u": 1.0},))) == 0.0
    assert 0 < budget.spent < budget.limit
