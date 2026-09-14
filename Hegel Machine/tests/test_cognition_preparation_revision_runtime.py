"""Actual shared questions validate revised preparation without replacing its source."""

import json

from hegel_machine.cognition.condition_questions import plan_condition_questions
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import SearchBudget, TaskSpec
from hegel_machine.cognition.program_disagreement_probes import propose_program_disagreement_probes
from hegel_machine.cognition.program_experiment_methods import ProgramExperimentMethods
from hegel_machine.cognition.competing_object_experiments import assess_competing_object_probe

from test_cognition_program_experiment_methods import _plan, _measure
from test_cognition_temporal_method_runtime import const, field, op, freeze, purchase


def apparatus(*, changed=False):
    control, sensor = ("drive", "meter") if changed else ("u", "reading")
    delay, gain, center = (3, -.7, .13) if changed else (1, .8, .21)
    history = op("if", Program("available", lag=delay),
                 Program("lag", name=control, lag=delay), const(0.))
    reading = op("add", op("add", op("mul", const(.4), field(control)),
                 op("mul", const(gain), history)), op("mul", const(.15), op("mul", history, history)))
    binding = {sensor: reading}
    predicate = op("and", op("ge", field(sensor), const(center - 1e-4)),
                   op("lt", field(sensor), const(center + 1e-4)))
    programs = [predicate, op("mul", const(2.), predicate), const(0.)]
    spec = TaskSpec("preparation", "Apply public controls.", (control,), (), (),
                    feature_bounds={control: (-1., 1.)}, max_probe_length=48)
    question = {"question_id": "activation", "predicate": predicate, "history_depth": 0,
                "uncovered_phases": ["active"], "dependent_hypothesis_ids": ["0", "1", "2"]}
    return spec, programs, binding, question


def learned():
    spec, programs, binding, _ = apparatus()
    proposals = propose_program_disagreement_probes(spec, programs, SearchBudget(6000), binding=binding, length=6)
    # This actual source experiment adjusts the current control and prepares
    # the earlier control; the latter becomes insufficient under changed gain.
    proposal = next(item for item in proposals if item["sequence_mode"] == "prefix" and
                    item["boundary"]["temporal_recipe"]["coordinate_dependency"] == 0)
    plan = _plan(proposal, programs, binding)
    actual = _measure(plan, programs[1], binding)
    assert actual["status"] == "distinguished"
    library = KnowledgeLibrary()
    library.program_experiment_methods.observe(plan, actual, task_id="source", command_spec=spec, binding=binding)
    return KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))


def plan_question(library, *, changed=True, binding=None, task_id="successor"):
    spec, programs, original_binding, question = apparatus(changed=changed)
    binding = original_binding if binding is None else binding
    plan = plan_condition_questions(spec, [question], SearchBudget(64000), binding=binding,
        methods=library.program_experiment_methods, response_programs=programs, task_id=task_id)
    if plan["forecast_status"] == "predicted_distinction":
        freeze(plan, programs, binding)
    return plan, spec, programs, binding


def paid_feedback(library, plan, spec, programs, binding, task_id, *, actual_binding=None):
    actual = purchase(plan["probe"].sequence, programs[1], actual_binding or binding)
    assessment = assess_competing_object_probe(plan, actual, SearchBudget(5000))
    action = library.program_experiment_methods.observe(plan, assessment,
        task_id=task_id, command_spec=spec, binding=binding)
    return assessment, action


def no_fresh(*args, **kwargs):
    raise AssertionError("revised preparation restarted fresh condition synthesis")


def test_shared_question_learns_child_only_from_measurement_and_restores_both_methods(monkeypatch):
    library = learned()
    parent = library.program_experiment_methods.to_dict()[0]
    monkeypatch.setattr("hegel_machine.cognition.condition_questions.propose_program_disagreement_probes", no_fresh)
    plan, spec, programs, binding = plan_question(library)
    assert plan["fresh_synthesis_skipped"]
    assert plan["proposal"]["parent_method_id"] == parent["method_id"]
    assert not plan["proposal"].get("method_id")
    assert library.program_experiment_methods.to_dict() == [parent]
    assessment, _ = paid_feedback(library, plan, spec, programs, binding, "successor")
    assert assessment["status"] == "distinguished"
    assert library.program_experiment_methods.records[0] == parent
    child = library.program_experiment_methods.to_dict()[1]
    assert child["parent_method_id"] == parent["method_id"] and len(child["support"]) == 1
    assert child["temporal_recipe"]["context_by_role"] != parent["temporal_recipe"]["context_by_role"]
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    later, spec, programs, binding = plan_question(library, task_id="later")
    assert later["proposal"]["method_id"] == child["method_id"]
    assessment, action = paid_feedback(library, later, spec, programs, binding, "later")
    assert assessment["status"] == "distinguished" and action["cross_task"]
    # Parent remains independently executable after child learning and restore.
    library.program_experiment_methods = ProgramExperimentMethods([library.program_experiment_methods.records[0]])
    returned, spec, programs, binding = plan_question(library, changed=False, task_id="return")
    assert returned["proposal"]["method_id"] == parent["method_id"]
    assert paid_feedback(library, returned, spec, programs, binding, "return")[0]["status"] == "distinguished"


def test_first_failed_revision_keeps_a_negative_child_and_preserves_its_parent():
    library = learned()
    parent = library.program_experiment_methods.to_dict()[0]
    plan, spec, programs, binding = plan_question(library)
    actual_binding = {"meter": op("add", const(.3), binding["meter"])}
    assessment, action = paid_feedback(library, plan, spec, programs, binding, "successor",
                                       actual_binding=actual_binding)
    assert assessment["calibration_status"] == "counterexample" and action is not None
    assert library.program_experiment_methods.records[0] == parent
    child = library.program_experiment_methods.records[1]
    assert child["parent_method_id"] == parent["method_id"]
    assert child["support"] == [] and child["status"] == "counterexample" and child["counterexamples"]
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    blocked, *_ = plan_question(library)
    assert blocked["forecast_status"] == "requires_paired_measurements"


def test_supported_child_failure_blocks_old_calibration_then_actual_recalibration_recovers():
    library = learned()
    parent = library.program_experiment_methods.to_dict()[0]
    plan, spec, programs, binding = plan_question(library)
    paid_feedback(library, plan, spec, programs, binding, "successor")
    plan, spec, programs, binding = plan_question(library, task_id="new_drift")
    child_id = plan["proposal"]["method_id"]
    corrected = {"meter": op("add", const(.3), binding["meter"])}
    assessment, action = paid_feedback(library, plan, spec, programs, binding, "new_drift", actual_binding=corrected)
    assert assessment["calibration_status"] == "counterexample" and action["method_id"] == child_id
    assert library.program_experiment_methods.records[0] == parent
    assert plan_question(library, task_id="new_drift")[0]["forecast_status"] == "requires_paired_measurements"
    recovered, spec, programs, binding = plan_question(library, binding=corrected, task_id="recalibrated")
    assert recovered["forecast_status"] == "predicted_distinction"
    assert paid_feedback(library, recovered, spec, programs, binding, "recalibrated")[0]["status"] == "distinguished"
