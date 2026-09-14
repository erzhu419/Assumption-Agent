"""Paid procedures must transfer their operation, and learn from real failure."""

import json

import pytest

from hegel_machine.cognition.competing_object_experiments import assess_competing_object_probe
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.program_disagreement_probes import _forecast, propose_program_disagreement_probes
from hegel_machine.cognition.program_experiment_methods import ProgramExperimentMethods


def _f(name):
    return Program("feature", name=name)


def _c(value):
    return Program("const", value=value)


def _op(op, *children):
    return Program(op, children=children)


def _case(*, renamed=False, temporal=False):
    if renamed:
        bounds = {"knob_z": (0., 1.), "knob_x": (-1., 1.), "knob_y": (-1., 1.)}
        names = ("reading_b", "reading_a", "reading_c")
        binding = {names[0]: _op("add", _c(.1), _op("mul", _c(-1.2), _f("knob_y"))),
                   names[1]: _op("add", _c(.2), _op("mul", _c(.7), _f("knob_x"))),
                   names[2]: _f("knob_z")}
        thresholds = (.73000001, .73000002)
    else:
        bounds = {"a": (-1., 1.), "b": (-1., 1.), "gain": (0., 1.)}
        names = ("left", "right", "weight")
        binding = dict(zip(names, [_f("a"), _f("b"), _f("gain")]))
        thresholds = (.99990001, .99990002)
    programs = [_op("mul", _f(names[2]), _op("ge", _op("mul", _f(names[0]), _f(names[1])), _c(value)))
                for value in thresholds]
    if temporal:
        programs = [Program("event_delta", lag=1, children=(program,)) for program in programs]
    spec = TaskSpec("commands", "independent public controls", tuple(bounds), (), (), feature_bounds=bounds)
    return spec, programs, binding


def _plan(proposal, programs, binding):
    return {"probe": proposal["probe"],
            "proposal": {key: value for key, value in proposal.items() if key not in {"probe", "predictions"}},
            "hypotheses": [{"hypothesis_id": str(index), "candidate_id": str(index),
                "program": program.to_dict(), "predictions": predictions}
                for index, (program, predictions) in enumerate(zip(programs, proposal["predictions"]))],
            "predicted_fields": {name: _forecast(program, proposal["probe"].sequence, lambda units: True)
                                 for name, program in binding.items()}}


def _measure(plan, truth, binding):
    commands, measured, pairs = [], [], []
    for point in plan["probe"].sequence:
        command = Observation(point, 0., tuple(commands))
        fields = {name: program.predict(command) for name, program in binding.items()}
        row = Observation(fields, 0., tuple(measured))
        target = truth.predict(row)
        pairs.append((Observation(point, target, tuple(commands)), Observation(fields, target, tuple(measured))))
        commands.append({**point, "__target__": target})
        measured.append({**fields, "__target__": target})
    return assess_competing_object_probe(plan, pairs, SearchBudget(2000))


@pytest.mark.parametrize("temporal", [False, True])
def test_paid_joint_procedure_survives_json_renames_new_calibration_and_threshold(temporal):
    spec, programs, binding = _case(temporal=temporal)
    fresh = SearchBudget(6000)
    proposals = propose_program_disagreement_probes(spec, programs, fresh, binding=binding, serial=61)
    assert proposals and proposals[0]["boundary"]["solver_strategy"] == "joint_segment"
    source = _plan(proposals[0], programs, binding)
    actual = _measure(source, programs[0], binding)
    assert actual["status"] == "distinguished"
    library = KnowledgeLibrary()
    action = library.program_experiment_methods.observe(source, actual,
        task_id="source", command_spec=spec, binding=binding)
    assert action["action"] == "learn_program_experiment_method"
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    methods = restored.program_experiment_methods
    current_spec, current_programs, current_binding = _case(renamed=True, temporal=temporal)
    replay_budget = SearchBudget(3000)
    proposed = methods.propose(current_spec, current_programs, replay_budget,
        binding=current_binding, serial=91)
    current_fresh = SearchBudget(6000)
    assert propose_program_disagreement_probes(current_spec, current_programs, current_fresh,
        binding=current_binding, serial=91)
    assert proposed and 0 < replay_budget.spent < current_fresh.spent
    assert proposed[0]["boundary"]["thresholds"] == [.73000001, .73000002]
    assert proposed[0]["sequence_mode"] == proposals[0]["sequence_mode"]
    assert set(proposed[0]["probe"].sequence[0]) == set(current_spec.feature_bounds)
    # Calling a method and checking a forecast do not create empirical support.
    assert len(methods.records[0]["support"]) == 1
    current = _plan(proposed[0], current_programs, current_binding)
    assessment = _measure(current, current_programs[0], current_binding)
    action = methods.observe(current, assessment, task_id="successor",
        command_spec=current_spec, binding=current_binding)
    assert action["action"] == "use_program_experiment_method" and action["cross_task"]
    assert len(methods.records[0]["support"]) == 2


def test_actual_calibration_counterexample_restricts_only_the_failed_calibration():
    spec, programs, binding = _case()
    source = _plan(propose_program_disagreement_probes(spec, programs, SearchBudget(6000), binding=binding)[0],
                   programs, binding)
    methods = ProgramExperimentMethods()
    methods.observe(source, _measure(source, programs[0], binding), task_id="source", command_spec=spec, binding=binding)
    other = dict(methods.to_dict()[0], method_id="other_predicate",
                 predicate_pattern=_op("abs", _f("role_0")).to_dict())
    methods.records.append(other)
    current_spec, current_programs, current_binding = _case(renamed=True)
    proposed = methods.propose(current_spec, current_programs, SearchBudget(3000), binding=current_binding)[0]
    plan = _plan(proposed, current_programs, current_binding)
    changed_binding = {**current_binding, "reading_b": _op("add", _c(.4), current_binding["reading_b"])}
    assessment = _measure(plan, current_programs[0], changed_binding)
    assert assessment["calibration_status"] == "counterexample"
    action = methods.observe(plan, assessment, task_id="changed", command_spec=current_spec, binding=current_binding)
    assert action["action"] == "revise_program_experiment_method"
    assert methods.records[0]["status"] == "supported"
    assert methods.records[0]["failure_scopes"][0]["reason"] == "calibration_counterexample"
    assert methods.records[0]["counterexamples"][0]["field_losses"]["reading_b"] > 0.
    assert methods.records[1]["status"] == "supported"
    assert methods.propose(current_spec, current_programs, SearchBudget(3000), binding=current_binding) == []


def test_observed_model_set_failure_remains_useful_but_no_rows_do_not_teach():
    spec, programs, binding = _case()
    source = _plan(propose_program_disagreement_probes(spec, programs, SearchBudget(6000), binding=binding)[0],
                   programs, binding)
    methods = ProgramExperimentMethods()
    assert methods.observe(source, {"status": "no_observations", "observation_rows": 0},
        task_id="empty", command_spec=spec, binding=binding) is None
    methods.observe(source, _measure(source, programs[0], binding), task_id="source", command_spec=spec, binding=binding)
    proposal = methods.propose(spec, programs, SearchBudget(3000), binding=binding)[0]
    plan = _plan(proposal, programs, binding)
    assessment = _measure(plan, _op("add", _c(2.), programs[0]), binding)
    assert assessment["status"] == "all_refuted" and assessment["calibration_status"] == "supported_on_probe"
    action = methods.observe(plan, assessment, task_id="new_law", command_spec=spec, binding=binding)
    assert action["action"] == "use_program_experiment_method"
    assert action["outcome"] == "all_refuted" and methods.records[0]["status"] == "supported"


def test_method_declines_missing_current_calibration_and_insufficient_budget():
    spec, programs, binding = _case()
    source = _plan(propose_program_disagreement_probes(spec, programs, SearchBudget(6000), binding=binding)[0],
                   programs, binding)
    methods = ProgramExperimentMethods()
    methods.observe(source, _measure(source, programs[0], binding), task_id="source", command_spec=spec, binding=binding)
    assert methods.propose(spec, programs, SearchBudget(3000)) == []
    empty = SearchBudget(0)
    assert methods.propose(spec, programs, empty, binding=binding) == [] and empty.spent == 0


def test_single_boundary_coordinate_method_moves_to_the_current_renamed_threshold():
    spec = TaskSpec("source", "one public knob", ("old_knob",), (), (), feature_bounds={"old_knob": (-1., 1.)})
    binding = {"sensor": _f("old_knob")}
    condition = _op("ge", _f("sensor"), _c(.31))
    programs = [condition, _op("mul", _c(2.), condition)]
    proposed = propose_program_disagreement_probes(spec, programs, SearchBudget(6000), binding=binding)[0]
    source = _plan(proposed, programs, binding)
    methods = ProgramExperimentMethods()
    methods.observe(source, _measure(source, programs[0], binding), task_id="source", command_spec=spec, binding=binding)
    assert methods.records[0]["target_strategy"] in {"at", "above"}
    assert methods.records[0]["solver_strategy"] == "coordinate"
    current_spec = TaskSpec("current", "renamed knob", ("new_knob",), (), (), feature_bounds={"new_knob": (-2., 2.)})
    current_binding = {"new_sensor": _op("mul", _c(-1.5), _f("new_knob"))}
    current_condition = _op("ge", _f("new_sensor"), _c(.72))
    current_programs = [current_condition, _op("mul", _c(3.), current_condition)]
    replay = methods.propose(current_spec, current_programs, SearchBudget(3000), binding=current_binding)
    assert replay and replay[0]["boundary"]["thresholds"] == [.72]
    plan = _plan(replay[0], current_programs, current_binding)
    assessment = _measure(plan, current_programs[0], current_binding)
    assert assessment["status"] == "distinguished"
