"""Temporal methods reach shared questions and real callable transfer decisions."""

import json

from hegel_machine.cognition.competing_object_experiments import _forecast, assess_competing_object_probe
from hegel_machine.cognition.condition_questions import plan_condition_questions
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.scale_bindings import bind_measurement_arguments
from hegel_machine.cognition.transfer_condition_research import TransferConditionResearch


def const(value):
    return Program("const", value=value)


def field(name):
    return Program("feature", name=name)


def op(name, *children):
    return Program(name, children=children)


def case(control="u", sensor="reading", *, depth=1, center=.231234, extra=False):
    names = (control, "unused") if extra else (control,)
    spec = TaskSpec("temporal_method", "Apply public controls.", names, (), (),
                    feature_bounds={name: (-1., 1.) for name in names}, max_probe_length=32)
    previous = op("if", Program("available", lag=depth),
                  Program("lag", name=control, lag=depth), const(0.))
    binding = {sensor: op("sub", field(control), previous)}
    predicate = op("and", op("ge", field(sensor), const(center - 1e-6)),
                   op("lt", field(sensor), const(center + 1e-6)))
    question = {"question_id": "activation", "predicate": predicate, "history_depth": 0,
                "uncovered_phases": ["active"], "dependent_hypothesis_ids": ["0", "1"]}
    programs = [op("if", predicate, const(7.), const(0.)), const(0.)]
    if extra:
        programs.append(field("unused"))
    return spec, binding, question, programs


def freeze(plan, programs, binding):
    plan["hypotheses"] = [{"hypothesis_id": str(index), "candidate_id": str(index),
        "program": program.to_dict(),
        "predictions": _forecast(bind_measurement_arguments(program, binding), plan["probe"].sequence)}
        for index, program in enumerate(programs)]
    plan["predicted_fields"] = {name: _forecast(program, plan["probe"].sequence)
                                for name, program in binding.items()}
    return plan


def purchase(sequence, truth, binding):
    commands, measured, pairs = [], [], []
    for point in sequence:
        command = Observation(point, 0., tuple(commands))
        values = {**point, **{name: program.predict(command) for name, program in binding.items()}}
        row = Observation(values, 0., tuple(measured))
        target = truth.predict(row)
        pairs.append((Observation(point, target, tuple(commands)), Observation(values, target, tuple(measured))))
        commands.append({**point, "__target__": target})
        measured.append({**values, "__target__": target})
    return pairs


def learned():
    spec, binding, question, programs = case(extra=True)
    library = KnowledgeLibrary()
    methods = library.program_experiment_methods
    plan = plan_condition_questions(spec, [question], SearchBudget(64000), binding=binding,
        methods=methods, response_programs=programs, task_id="source")
    assert plan["forecast_status"] == "predicted_distinction"
    freeze(plan, programs, binding)
    pairs = purchase(plan["probe"].sequence, programs[0], binding)
    measured = assess_competing_object_probe(plan, pairs, SearchBudget(5000))
    action = methods.observe(plan, measured, task_id="source", command_spec=spec, binding=binding)
    assert action and action["action"] == "learn_program_experiment_method"
    return KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))


def test_shared_temporal_question_rebinds_without_unrelated_response_controls_or_new_support():
    library = learned()
    spec, binding, question, programs = case("renamed", "meter", depth=2, center=-.37183)
    methods = library.program_experiment_methods
    before = methods.to_dict()
    plan = plan_condition_questions(spec, [question], SearchBudget(64000), binding=binding,
        methods=methods, response_programs=programs, task_id="successor")
    assert plan["proposal_source"] == "learned_method" and plan["fresh_synthesis_skipped"]
    assert methods.to_dict() == before
    freeze(plan, programs, binding)
    actual = purchase(plan["probe"].sequence, programs[0], binding)
    assessment = assess_competing_object_probe(plan, actual, SearchBudget(5000))
    assert assessment["status"] == "distinguished"
    action = methods.observe(plan, assessment, task_id="successor", command_spec=spec, binding=binding)
    assert action["action"] == "use_program_experiment_method" and action["cross_task"]


def test_measured_bad_dynamic_calibration_is_blocked_until_new_mapping_changes_the_question():
    methods = learned().program_experiment_methods
    spec, stale, question, programs = case()
    plan = plan_condition_questions(spec, [question], SearchBudget(64000), binding=stale,
        methods=methods, response_programs=programs, task_id="changed")
    assert plan["proposal_source"] == "learned_method"
    freeze(plan, programs, stale)
    current = {"reading": op("add", const(.2), stale["reading"])}
    assessment = assess_competing_object_probe(plan,
        purchase(plan["probe"].sequence, programs[0], current), SearchBudget(5000))
    assert assessment["calibration_status"] == "counterexample"
    action = methods.observe(plan, assessment, task_id="changed", command_spec=spec, binding=stale)
    assert action["action"] == "revise_program_experiment_method"
    blocked = plan_condition_questions(spec, [question], SearchBudget(64000), binding=stale,
        methods=methods, response_programs=programs, task_id="changed")
    assert blocked["forecast_status"] == "requires_paired_measurements"
    revised = plan_condition_questions(spec, [question], SearchBudget(64000), binding=current,
        methods=methods, response_programs=programs, task_id="changed")
    assert revised["proposal_source"] == "learned_method"
    freeze(revised, programs, current)
    assert assess_competing_object_probe(revised,
        purchase(revised["probe"].sequence, programs[0], current), SearchBudget(5000))["status"] == "distinguished"


def test_callable_transfer_uses_a_learned_temporal_method_after_an_inactive_pilot():
    library = learned()
    spec, binding, question, programs = case("renamed", "meter", depth=2)
    calls = []
    for gain in (1., 2.):
        body = op("if", question["predicate"].remap_features({"meter": "value"}), const(gain), const(0.))
        # The definition reads a lexical argument; its original body is retained.
        def parameterize(program):
            if program.op == "feature":
                return Program("arg", name=program.name)
            from dataclasses import replace
            return replace(program, children=tuple(parameterize(child) for child in program.children))
        definition = Definition(f"law_{gain}", ("value",), parameterize(body), f"source_{gain}")
        calls.append(Program("call", name=definition.definition_id, definition=definition,
                             children=(field("meter"),)))
    pilot = purchase(({"renamed": 0.},) * 4, programs[0], binding)
    plan = TransferConditionResearch(calls).plan(spec, calls[1], SearchBudget(64000),
        paid_rows=[row for _, row in pilot], paired_rows=pilot, binding=binding,
        methods=library.program_experiment_methods, task_id="call_successor")
    assert plan["forecast_status"] == "predicted_distinction"
    assert plan["proposal_source"] == "learned_method" and plan["fresh_synthesis_skipped"]
    actual = purchase(plan["probe"].sequence, calls[1], binding)
    assert assess_competing_object_probe(plan, actual, SearchBudget(5000))["status"] == "distinguished"
