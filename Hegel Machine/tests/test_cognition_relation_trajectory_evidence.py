"""A complete separating experiment cannot be diluted by unrelated trajectories."""

from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from hegel_machine.cognition.scale_bindings import bind_measurement_arguments


INTERVAL = {'fits': [{'radius': .8}], 'radius_interval': [.4, 1.]}


def measured_law():
    fields = ('f0', 'x0', 'q0', 'f1', 'x1', 'q1')
    relation = Program('neighbor_sum', children=(Program('const', value=.8),
        *(Program('arg', name=name) for name in fields)))
    flow = Program('mul', children=(Program('const', value=.0015), relation))
    response = Program('window_sum', lag=3, children=(
        Program('event_delta', lag=1, children=(flow,)),))
    definition = Definition('retained_response', fields, response, 'measured_source')
    return Program('call', name=definition.definition_id, definition=definition,
                   children=tuple(Program('feature', name=name) for name in fields))


def trajectory(distances, *, commands=False):
    history, flows, events, rows = [], [], [], []
    for distance in distances:
        actual = {'f0': 1., 'f1': 0., 'x0': 0., 'x1': distance, 'q0': .2, 'q1': 1.2}
        # Independent physical readout: symmetric exchange at radius .8,
        # followed by its reset event and the three most recent events.
        flow = .0015 * (actual['q1'] - actual['q0']) if distance < .8 else 0.
        events.append(0. if not flows else flow - flows[-1])
        target = sum(events[-3:])
        flows.append(flow)
        fields = ({name: value for name, value in actual.items() if not name.startswith('f')}
                  | {'selector_control': .5}) if commands else actual
        rows.append(Observation(fields, target, tuple(history)))
        history.append({**fields, '__target__': target})
    return rows


def mse(program, rows):
    return sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows)


def candidates(program, rows, experiments=()):
    research = RelationIntervalResearch(program, [], SearchBudget(10000), structure=deepcopy(INTERVAL))
    budget = SearchBudget(30000)
    answer = research.alternatives(rows, budget, tolerance=1e-7, experiments=experiments)
    assert budget.spent <= budget.limit
    return research, answer


def test_unrelated_good_trajectory_cannot_restore_a_radius_refuted_by_an_independent_probe():
    program = measured_law()
    counterexample = trajectory([1.2, .7, .7, .7, .7, 1.2, 1.2, 1.2, 1.2])
    unrelated = trajectory([1.2] * 140)
    assert len(counterexample) == 9 and len(unrelated) == 140
    _, initial = candidates(program, counterexample)
    assert initial
    assert all(mse(item['candidate'].program, counterexample) <= 1e-7 for item in initial)
    # The old global mean accepts the zero prediction after adding good rows.
    wrong = Program('const', value=0.)
    assert mse(wrong, counterexample) == pytest.approx(1.5e-6)
    assert mse(wrong, [*counterexample, *unrelated]) < 1e-7
    restored = Program.from_dict(json.loads(json.dumps(program.to_dict())))
    saved = json.loads(json.dumps([asdict(row) for row in counterexample]))
    retained_experiment = [Observation(row['features'], row['target'], tuple(row['history'])) for row in saved]
    _, after = candidates(restored, [*counterexample, *unrelated], [retained_experiment])
    assert after
    assert all(mse(item['candidate'].program, counterexample) <= 1e-7 for item in after)
    assert all(mse(item['candidate'].program, unrelated) <= 1e-7 for item in after)
    assert all(next(iter(item['parameter_values'].values())) > .7 for item in after)


def test_revised_input_binding_retests_the_same_paid_counterexample_in_its_new_complete_program():
    measured = measured_law()
    definition = measured.definition.to_dict()
    control = Program('feature', name='selector_control')
    cut = Program('const', value=.5)
    binding = {name: Program('feature', name=name) for name in ('x0', 'x1', 'q0', 'q1')}
    binding.update(f0=Program('ge', children=(cut, control)), f1=Program('ge', children=(control, cut)))
    wrong = bind_measurement_arguments(measured, binding)
    paid = [*trajectory([1.2, .7, .7, .7, .7, 1.2, 1.2, 1.2, 1.2], commands=True),
            *trajectory([1.2] * 140, commands=True)]
    research, rejected = candidates(wrong, paid, [paid[:9]])
    assert not rejected and research.status == 'refuted'
    corrected = {**binding, 'f1': Program('lt', children=(cut, control))}
    rebound = bind_measurement_arguments(measured, corrected)
    restored = Program.from_dict(json.loads(json.dumps(rebound.to_dict())))
    research, answers = candidates(restored, paid, [paid[:9]])
    assert answers
    assert all(mse(item['candidate'].program, paid[:9]) <= 1e-7 for item in answers)
    assert any(mse(item['candidate'].program, paid) < 1e-16 for item in answers)
    assert measured.definition.to_dict() == definition
    assert mse(wrong, paid[:9]) > 1e-7
