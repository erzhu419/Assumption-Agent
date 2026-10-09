"""Instrument substitution and experiment solving retain real prefix time."""

import json

import pytest

from hegel_machine.cognition.language import Definition, MissingHistory, Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.scale_bindings import bind_measurement_arguments


def field(name):
    return Program("feature", name=name)


def const(value):
    return Program("const", value=value)


def binary(op, left, right):
    return Program(op, children=(left, right))


def history_rows(values, sensor):
    commands, measurements, command_history, sensor_history = [], [], [], []
    for value in values:
        command = Observation({"u": value}, 0., tuple(command_history))
        measured = {"reading": sensor.predict(command)}
        commands.append(command)
        measurements.append(Observation(measured, 0., tuple(sensor_history)))
        command_history.append({**command.features, "__target__": 0.})
        sensor_history.append({**measured, "__target__": 0.})
    return commands, measurements


def test_nested_lag_and_available_shift_without_changing_response_definition():
    lagged = Program("if", children=(Program("available", lag=2),
        Program("lag", name="u", lag=2), const(0.)))
    sensor = binary("add", binary("mul", const(2.), field("u")), lagged)
    body = Program("if", children=(Program("available", lag=1),
        binary("sub", Program("arg", name="now"), Program("arg", name="previous")), const(0.)))
    definition = Definition("response_difference", ("now", "previous"), body)
    original = Program("call", name=definition.definition_id, definition=definition,
        children=(field("reading"), Program("lag", name="reading", lag=1)))
    rebound = bind_measurement_arguments(original, {"reading": sensor})
    assert rebound.definition is definition
    commands, measured = history_rows((.3, -.2, .8, -.5, .1), sensor)
    assert [rebound.predict(row) for row in commands] == pytest.approx(
        [original.predict(row) for row in measured])
    assert rebound.predict(commands[0]) == 0.


def test_guarded_instrument_startup_does_not_create_an_observation_before_reset():
    sensor = Program("if", children=(Program("available", lag=1),
        Program("lag", name="u", lag=1), const(4.)))
    previous = Program("lag", name="reading", lag=1)
    rebound = bind_measurement_arguments(previous, {"reading": sensor})
    commands, measured = history_rows((.2, .6, -.4, .8), sensor)
    with pytest.raises(MissingHistory):
        rebound.predict(commands[0])
    assert [rebound.predict(row) for row in commands[1:]] == pytest.approx(
        [previous.predict(row) for row in measured[1:]])
    assert rebound.predict(commands[1]) == 4.


@pytest.mark.parametrize("operator", ("window_sum", "event_delta"))
def test_previous_temporal_instrument_uses_its_own_earlier_reset_boundary(operator):
    sensor = Program(operator, lag=3, children=(field("u"),))
    original = Program("if", children=(Program("available", lag=1),
        Program("lag", name="reading", lag=1), const(0.)))
    rebound = bind_measurement_arguments(original, {"reading": sensor})
    commands, measured = history_rows((.2, .6, -.4, .8, -.3), sensor)
    assert [rebound.predict(row) for row in commands] == pytest.approx(
        [original.predict(row) for row in measured])


def accumulated_measurements(values, decay, initial):
    """Observe a reset state by direct recurrence, independently of Program."""
    state, commands, measurements, command_history, measured_history = initial, [], [], [], []
    for value in values:
        state = decay * state + value
        command, measured = {"u": value}, {"reading": state}
        commands.append(Observation(command, 0., tuple(command_history)))
        measurements.append(Observation(measured, 0., tuple(measured_history)))
        command_history.append(command)
        measured_history.append(measured)
    return commands, measurements


@pytest.mark.parametrize('decay', [0., .5, -.7])
@pytest.mark.parametrize('initial', [0., 2.4])
def test_lagged_accumulated_reading_replays_only_its_actual_past_prefix(decay, initial):
    sensor = Program('accumulate', value=decay, children=(field('u'), const(initial)))
    for values in ((.3,), (-.2, .8), (.3, -.2, .8, -.5, .1, .9, -.4, .6)):
        commands, measured = accumulated_measurements(values, decay, initial)
        for lag in (1, 3):
            previous = Program('lag', name='reading', lag=lag)
            rebound = bind_measurement_arguments(previous, {'reading': sensor})
            for index, (command, reading) in enumerate(zip(commands, measured)):
                if index < lag:
                    with pytest.raises(MissingHistory):
                        rebound.predict(command)
                else:
                    assert rebound.predict(command) == pytest.approx(previous.predict(reading), abs=1e-12)


def test_lagged_state_inside_nested_temporal_response_survives_json_and_control_rename():
    sensor = Program('accumulate', value=.81, children=(field('u'), const(1.7)))
    previous = Program('if', children=(Program('available', lag=2),
        Program('lag', name='reading', lag=2), const(0.)))
    definition = Definition('measured_temporal_response', ('reading_now', 'reading_before'),
        Program('window_sum', lag=3, children=(Program('event_delta', lag=1, children=(
            binary('add', Program('arg', name='reading_now'),
                binary('mul', const(.3), Program('arg', name='reading_before'))),)),)))
    original = Program('call', name=definition.definition_id, definition=definition,
        children=(field('reading'), previous))
    bound = bind_measurement_arguments(original, {'reading': sensor})
    assert bound.definition is definition
    restored = Program.from_dict(json.loads(json.dumps(bound.to_dict()))).remap_features({'u': 'voltage'})
    for values in ((.8,), (.3, -.2, .8), tuple((index % 7 - 3) / 4. for index in range(43))):
        commands, measured = accumulated_measurements(values, .81, 1.7)
        renamed = [Observation({'voltage': row.features['u']}, row.target,
                     tuple({'voltage': past['u']} for past in row.history)) for row in commands]
        assert [restored.predict(row) for row in renamed] == pytest.approx(
            [original.predict(row) for row in measured], abs=1e-10)


def window(name, center):
    return binary("and", binary("ge", field(name), const(center - 1e-6)),
                  binary("lt", field(name), const(center + 1e-6)))


def claims(predicate):
    return [Program("if", children=(predicate, const(gain), const(0.))) for gain in (1., 2.)]


def measured_pairs(sequence, binding, truth):
    command_history, measured_history, pairs = [], [], []
    for point in sequence:
        raw = Observation(point, 0., tuple(command_history))
        fields = {name: program.predict(raw) for name, program in binding.items()}
        measured = Observation(fields, 0., tuple(measured_history))
        target = truth.predict(measured)
        pairs.append((Observation(point, target, tuple(command_history)),
                      Observation(fields, target, tuple(measured_history))))
        command_history.append({**point, "__target__": target})
        measured_history.append({**fields, "__target__": target})
    return pairs


@pytest.mark.parametrize("difference", (False, True))
def test_dynamic_prefix_solves_pure_delay_and_zero_steady_gain_without_inventing_history(difference):
    from hegel_machine.cognition.program_disagreement_probes import propose_program_disagreement_probes
    prior = Program("if", children=(Program("available", lag=1),
        Program("lag", name="u", lag=1), const(0.)))
    sensor = binary("sub", field("u"), prior) if difference else prior
    programs = claims(window("reading", .231234))
    spec = TaskSpec("dynamic_gate", "Apply public controls.", ("u",), (), (),
                    feature_bounds={"u": (-1., 1.)})
    budget = SearchBudget(6000)
    proposals = propose_program_disagreement_probes(spec, programs, budget, binding={"reading": sensor})
    assert proposals and budget.spent <= 6000
    proposal = proposals[0]
    assert proposal["boundary"]["solver_strategy"] == "temporal_prefix"
    pairs = measured_pairs(proposal["probe"].sequence, {"reading": sensor}, programs[1])
    assert any(measured.target == 2. and measured.history for _, measured in pairs)
    for program, predicted in zip(programs, proposal["predictions"]):
        assert predicted == pytest.approx([program.predict(measured) for _, measured in pairs])
    if difference:
        assert any(left["u"] != right["u"] for left, right in
                   zip(proposal["probe"].sequence, proposal["probe"].sequence[1:]))


def test_two_delayed_sensor_conditions_use_one_real_prefix_and_frozen_full_predictions():
    from hegel_machine.cognition.competing_object_experiments import (
        assess_competing_object_probe, plan_competing_object_probe,
    )
    from hegel_machine.cognition.condition_questions import plan_condition_questions
    previous = Program("if", children=(Program("available", lag=1),
        Program("lag", name="u", lag=1), const(0.)))
    binding = {"sensor_a": binary("add", field("u"), binary("mul", const(.5), previous)),
               "sensor_b": binary("sub", previous, field("u"))}
    predicate = binary("and", window("sensor_a", .2312), window("sensor_b", -.4117))
    programs = claims(predicate)
    spec = TaskSpec("joint_dynamic_gate", "Apply public controls.", ("u",), (), (),
                    feature_bounds={"u": (-1., 1.)})
    entries = [{"candidate": Candidate(str(index), program, "retained")}
               for index, program in enumerate(programs)]
    plan = plan_competing_object_probe(spec, entries, SearchBudget(64000), binding=binding)
    assert plan["forecast_status"] == "predicted_distinction"
    pairs = measured_pairs(plan["probe"].sequence, binding, programs[1])
    assessment = assess_competing_object_probe(plan, pairs, SearchBudget(1000))
    assert assessment["status"] == "distinguished"
    assert assessment["calibration_status"] == "supported_on_probe"
    assert any(row.target == 2. for _, row in pairs)
    assert plan["hypotheses"][1]["predictions"] == [row.target for _, row in pairs]
    question = {"question_id": "joint_sensor_condition", "predicate": predicate,
                "history_depth": 0, "dependent_hypothesis_ids": ["0", "1"],
                "uncovered_phases": ["active"]}
    shared = plan_condition_questions(spec, [question], SearchBudget(64000), binding=binding)
    assert shared["forecast_status"] == "predicted_distinction"
    assert any(row.target == 2. for _, row in measured_pairs(shared["probe"].sequence, binding, programs[1]))


def test_startup_only_instrument_extends_real_probe_to_its_available_boundary():
    from hegel_machine.cognition.competing_object_experiments import plan_competing_object_probe
    sensor = Program("available", lag=4)
    programs = [field("reading"), binary("mul", const(2.), field("reading"))]
    entries = [{"candidate": Candidate(str(index), program, "retained")}
               for index, program in enumerate(programs)]
    spec = TaskSpec("startup_instrument", "Apply public controls.", ("u",), (), (),
                    feature_bounds={"u": (-1., 1.)})
    plan = plan_competing_object_probe(spec, entries, SearchBudget(64000), binding={"reading": sensor})
    assert plan["forecast_status"] == "predicted_distinction"
    assert len(plan["probe"].sequence) >= 5
    assert plan["predicted_fields"]["reading"][:5] == [0., 0., 0., 0., 1.]
    assert plan["hypotheses"][1]["predictions"][:5] == [0., 0., 0., 0., 2.]
