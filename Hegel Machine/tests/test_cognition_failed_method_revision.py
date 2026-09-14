"""Parameter failures must change executable research and its later price."""

from copy import deepcopy
from dataclasses import replace
import json
import random

import pytest

from hegel_machine.cognition.action_policy import ActionOption
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.method_conditions import method_applicability, method_profile
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.research_methods import ResearchMethods


CONTEXT = "squared:numeric:single:model_residual"
CONDITION = {"response": "numeric", "controls": 1}


def problem(name="consumer", feature="x"):
    task = TaskSpec(name, "Predict the measured response", (feature,), (), (),
                    feature_bounds={feature: (-1., 1.)}, max_probe_length=24)
    program = Program("add", children=(Program("feature", name=feature), Program("if", children=(
        Program("available", lag=2), Program("lag", name=feature, lag=2), Program("const", value=0.)))))
    source = Candidate("source_fit", program, "transfer", memory_id="concept_1")
    rng, rows = random.Random(47), []
    for length in (12, 16, 20):
        past = []
        for _ in range(length):
            value = rng.uniform(-1., 1.)
            target = .3 * value - (.8 * past[-4][feature] if len(past) >= 4 else 0.)
            rows.append(Observation({feature: value}, target, tuple(past)))
            past.append({feature: value, "__target__": target})
    return task, source, rows


def profile(task, source, rows, trace=(), residual=1.):
    return method_profile(task, rows, source, trace, residual_loss=residual, baseline_loss=2.)


def step(action, before, gain=0., **kwargs):
    return {"action": action, "context": CONTEXT, "profile": deepcopy(before), "gain": gain,
            "compute_cost": 100, "observation_cost": 0, "execution_complete": True, **kwargs}


def failure(before, **kwargs):
    return step("refit_transfer", before, source_id="concept_1", candidate_count=2,
                residual_resolved=False, training_rows=before["observation_count"], **kwargs)


def exhausted(*, name="consumer", feature="x"):
    task, source, rows = problem(name, feature)
    before = profile(task, source, rows[:28])
    trace = [failure(before), step("refit", before)]
    return task, source, rows, trace, profile(task, source, rows, trace)


def test_actual_source_fits_fail_and_the_next_executable_step_changes_to_representation():
    task, source, rows = problem()
    library = KnowledgeLibrary()
    identifier, _ = library.learn(replace(source, memory_id=None), task)
    source = replace(source, memory_id=identifier)
    trace = []
    for acquired in (rows[:28], rows):
        before = profile(task, source, acquired, trace)
        budget = SearchBudget(500)
        proposals = library.propose(task, acquired, budget, source_concept_ids={identifier}, preserve_parameters=False)
        measured_loss = min(loss(candidate.program, acquired, 0., .5) for candidate in proposals)
        assert measured_loss > .01
        assert budget.spent > 0
        trace.append({**failure(before), "source_id": identifier, "candidate_count": len(proposals),
                      "compute_cost": budget.spent})
    current = profile(task, source, rows, trace)
    assert current["source_revision"]["failed_attempts"] == 2
    assert not method_applicability(None, current, "refit")["can_try"]
    options = [ActionOption("refit", 0, 96), ActionOption("task_value", 6, 6, (None, {})),
               ActionOption("representation", 0, 1954)]
    revision = library.methods.revise_plan(current, options, trace)
    assert revision["action"] == "representation"
    assert revision["excluded_actions"] == ["refit", "task_value"]
    assert revision["compute_cost"] == 1954
    assert library.methods.methods == []  # Choosing a trial does not invent a successful method.


def test_unfinished_or_empty_work_is_not_a_failed_parameter_trial():
    task, source, rows = problem()
    before = profile(task, source, rows)
    incomplete = [{**failure(before), "candidate_count": 0}, failure(before, execution_complete=False)]
    current = profile(task, source, rows, incomplete)
    assert "source_revision" not in current
    assert method_applicability(None, current, "refit")["can_try"]
    one = profile(task, source, rows, [failure(before)])
    assert one["source_revision"]["status"] == "parameter_repair_unresolved"
    assert method_applicability(None, one, "refit")["can_try"]


def test_normal_refit_feedback_counts_but_changed_controls_or_structure_reopen_a_trial():
    task, source, rows, trace, current = exhausted()
    assert current["data"]["max_history"] > trace[0]["profile"]["data"]["max_history"]
    assert not method_applicability(None, current, "refit")["can_try"]
    changed = replace(source, program=Program("feature", name="x"))
    assert "source_revision" not in profile(task, changed, rows, trace)
    narrower = deepcopy(trace)
    for row in narrower:
        row["profile"]["data"]["coverage"][0]["bins"] = [1, 2]
    assert "source_revision" not in profile(task, source, rows, narrower)
    assert "source_revision" not in profile(task, source, rows, trace, residual=0.)
    resolved = [*trace, {**failure(current), "residual_resolved": True}]
    assert "source_revision" not in profile(task, source, rows, resolved)


def test_structural_successor_keeps_current_distinguishing_and_boundary_experiments():
    *_, current = exhausted()
    book = ResearchMethods()
    for action in ("task_value", "representative"):
        for evidence in ({"candidate_disagreement": .3}, {"expected_risk_reduction": .2},
                         {"boundary_support": [{}], "unresolved_boundary_width": .1}):
            options = [ActionOption(action, 6, 6, (None, evidence)), ActionOption("representation", 0, 900)]
            revision = book.revise_plan(current, options)
            assert action not in revision["excluded_actions"]
        options = [ActionOption(action, 6, 6, (None, {})), ActionOption("representation", 0, 900)]
        assert action in book.revise_plan(current, options)["excluded_actions"]
    assert book.revise_plan(current, [ActionOption("task_value", 6, 6, (None, {}))]) is None


def test_failed_parameter_history_overrides_broad_method_means_and_prices_a_conditional_successor():
    task, source, rows, trace, current = exhausted()
    ordinary_profile = profile(task, source, rows)
    book = ResearchMethods()
    for action in ("refit", "task_value", "representative"):
        book.observe(CONTEXT, [step(action, ordinary_profile, .9)], condition=CONDITION, source_task="earlier")
    cheap = [ActionOption("refit", 0, 10), ActionOption("task_value", 6, 6), ActionOption("representative", 8, 8)]
    limits = dict(task_id="next", profile=current, observation_remaining=100, compute_remaining=5000)
    assert book.choose(CONTEXT, CONDITION, cheap, trace, **limits) is None
    assert book.opportunity(CONTEXT, CONDITION, trace=trace, **limits) is None
    identifier = book.observe(CONTEXT, [step("representation", current, .8, compute_cost=1800)],
        condition=CONDITION, source_task="successful_successor")
    renamed = exhausted(name="renamed", feature="r")[-1]
    restored = ResearchMethods(json.loads(json.dumps(book.to_dict())))
    quote = restored.opportunity(CONTEXT, CONDITION, task_id="new_problem", profile=renamed,
                                 observation_remaining=100, compute_remaining=5000)
    assert quote["method_id"] == identifier and quote["compute_cost"] == 1800
    assert quote["expected_gain"] == .8
    conditional = ResearchMethods({"methods": [deepcopy(restored.methods[-1])]})
    assert conditional.opportunity(CONTEXT, CONDITION, task_id="untried_parameters", profile=ordinary_profile,
                                   observation_remaining=100, compute_remaining=5000) is None
    restored.observe(CONTEXT, [step("representation", renamed, -.2, method_id=identifier)],
                     condition=CONDITION, source_task="failed_successor")
    assert restored.opportunity(CONTEXT, CONDITION, task_id="another", profile=renamed,
                                observation_remaining=100, compute_remaining=5000) is None


def test_required_neutral_measurement_is_learned_and_executed_before_its_dependent_fit():
    task, source, rows = problem()
    before = profile(task, source, rows[:28])
    after = profile(task, source, rows)
    trace = [step("refit", before, compute_cost=200),
             step("representative", before, observation_cost=20, compute_cost=20),
             step("representation", after, .8, compute_cost=1800, requires_actions=["representative"])]
    book = ResearchMethods()
    identifier = book.observe(CONTEXT, trace, condition=CONDITION, source_task="joint_source")
    method = book.methods[0]
    assert method["steps"] == ["representative", "representation"]
    assert method["mean_observation_cost"] == 20 and method["mean_compute_cost"] == 1820
    assert method["step_outcomes"][0]["gain"] == 0.
    restored = ResearchMethods(json.loads(json.dumps(book.to_dict())))
    options = [ActionOption("representative", 20, 20), ActionOption("representation", 0, 1800)]
    choice = restored.choose(CONTEXT, CONDITION, options, [], task_id="joint_target", profile=before,
                            observation_remaining=50, compute_remaining=5000)
    assert choice.name == "representative"
    execution = [step("representative", before, method_id=identifier, observation_cost=20, compute_cost=20)]
    restored.observe(CONTEXT, execution, condition=CONDITION, source_task="joint_target")
    choice = restored.choose(CONTEXT, CONDITION, options, execution, task_id="joint_target", profile=after,
                            observation_remaining=30, compute_remaining=4980)
    assert choice.name == "representation" and restored.last_choice["method_id"] == identifier


def test_saved_failure_trace_and_methods_make_identical_resume_decisions():
    task, source, rows, trace, current = exhausted()
    book = ResearchMethods()
    book.observe(CONTEXT, [step("representation", current, .8)], condition=CONDITION, source_task="earlier")
    restored_trace = json.loads(json.dumps(trace))
    restored = ResearchMethods(json.loads(json.dumps(book.to_dict())))
    restored_profile = profile(task, source, rows, restored_trace)
    assert restored_profile == current
    options = [ActionOption("refit", 0, 20), ActionOption("representation", 0, 100)]
    assert restored.revise_plan(restored_profile, options, restored_trace) == book.revise_plan(current, options, trace)
    limits = dict(task_id="next", observation_remaining=100, compute_remaining=5000)
    assert restored.opportunity(CONTEXT, CONDITION, profile=restored_profile, **limits) == book.opportunity(
        CONTEXT, CONDITION, profile=current, **limits)
