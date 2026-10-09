"""Instrument fitting keeps observationally ambiguous executable explanations."""

from dataclasses import replace
import json
from random import Random

import pytest

from hegel_machine.cognition.instrument_models import fit_instrument_model, instrument_binding
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec


COMMAND = TaskSpec('unknown_meter', 'public bounded input', ('u',), (), (),
                   feature_bounds={'u': (-1., 1.)})
MEASURED = replace(COMMAND, feature_names=('meter',))


class MeasuredMemory:
    def __init__(self, finite):
        self.finite = finite
        self.observation_cost = 0

    def acquire(self, controls):
        state, history, past_controls, result = 0., [], [], []
        for value in controls:
            state = .72 * state + value
            reading = (value + sum(.72 ** lag * past_controls[-lag]
                for lag in range(1, min(2, len(past_controls)) + 1))) if self.finite else state
            result.append((Observation({'u': value}, 7777., tuple(history)),
                           Observation({'meter': reading}, -3333.)))
            history.append({'u': value, '__target__': 9876.})
            past_controls.append(value)
        self.observation_cost += len(result)
        return result


def error(program, pairs):
    return sum((program.predict(command) - measured.features['meter']) ** 2
               for command, measured in pairs) / len(pairs)


@pytest.mark.parametrize('finite', [False, True])
def test_short_paid_trajectories_retain_both_finite_and_stateful_explanations(finite):
    laboratory = MeasuredMemory(finite)
    random = Random(157)
    training = [pair for _ in range(6)
                for pair in laboratory.acquire([random.uniform(-1., 1.) for _ in range(3)])]
    budget = SearchBudget(24000)
    model = fit_instrument_model(COMMAND, MEASURED, training, budget)
    field = model['fields']['meter']
    hypotheses = field['hypotheses']
    assert {'finite_history_polynomial', 'continuous_state'} <= {item['kind'] for item in hypotheses}
    assert all(item['status'] == 'training_candidate' for item in hypotheses)
    assert len({item['hypothesis_id'] for item in hypotheses}) == len(hypotheses)
    programs = [Program.from_dict(item['program']) for item in hypotheses]
    assert len(set(programs)) == len(programs)
    assert 'meter' not in instrument_binding(model)
    assert model['observed_history_depth'] == field['observed_history_depth'] == 2

    short = laboratory.acquire([.4, -.2, .8])
    assert all(error(program, short) < 1e-9 for program in programs)
    restored = json.loads(json.dumps(model))
    assert restored['fields']['meter']['hypotheses'] == hypotheses
    assert 'meter' not in instrument_binding(restored)
    tail = laboratory.acquire([1., 0., 0., 0., 0., 0., 0., 0.])
    losses = [(item['kind'], error(program, tail)) for item, program in zip(hypotheses, programs)]
    expected = 'finite_history_polynomial' if finite else 'continuous_state'
    assert min(value for kind, value in losses if kind == expected) < 1e-9
    assert all(value > 1e-4 for kind, value in losses if kind != expected)
    assert laboratory.observation_cost == 18 + 3 + 8
    assert model['compute_cost'] == budget.spent <= 24000


def test_static_fit_still_stops_without_speculative_temporal_candidates():
    pairs = [(Observation({'u': value}, 12345.), Observation({'meter': .4 + .7 * value}, -54321.))
             for value in (-.8, -.5, -.2, .1, .4, .7)]
    no_history = fit_instrument_model(COMMAND, MEASURED, pairs, SearchBudget(24000), max_history=0)
    current = fit_instrument_model(COMMAND, MEASURED, pairs, SearchBudget(24000))
    assert current == no_history
    assert len(current['fields']['meter']['hypotheses']) == 1
    assert error(instrument_binding(current)['meter'], pairs) < 1e-10


def test_zero_history_setting_does_not_invent_a_state_to_resolve_dynamic_error():
    laboratory = MeasuredMemory(False)
    pairs = [pair for values in ([.4, -.3, .8], [-.2, .7, -.4], [.1, .9, -.6])
             for pair in laboratory.acquire(values)]
    model = fit_instrument_model(COMMAND, MEASURED, pairs, SearchBudget(24000), max_history=0)
    assert 'meter' in model['unmodeled_fields']
    assert not instrument_binding(model)


def test_membership_fit_keeps_paid_equivalent_cuts_that_disagree_at_unseen_endpoints():
    from hegel_machine.cognition.models import Probe
    from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab

    with UnifiedObjectLab(apparatus_seed=9013) as laboratory:
        probe = laboratory.random_probe(183, length=48)
        batch = laboratory.acquire(probe, ('fine',))
        pairs = [(Observation(point, 98765.), measured)
                 for point, measured in zip(probe.sequence, batch.observations)]
        model = fit_instrument_model(laboratory.spec, laboratory.spec_for(('fine',)),
                                     pairs, SearchBudget(24000))
        choices = [(name, field['hypotheses']) for name, field in model['fields'].items()
                   if field['kind'] == 'interval_membership' and len(field['hypotheses']) > 1]
        assert choices
        point = probe.sequence[0]
        endpoint_probe = Probe('new_membership_endpoints', tuple(
            {**point, 'focus': value} for value in (0., .5, 1., 1.5, 2., 2.5, 3.)))
        observed = laboratory.acquire(endpoint_probe, ('fine',)).observations
        distinguishable = False
        for name, hypotheses in choices:
            programs = [Program.from_dict(item['program']) for item in hypotheses]
            assert all(max(abs(program.predict(command) - measured.features[name])
                           for command, measured in pairs) < 1e-10 for program in programs)
            predictions = [tuple(program.predict(Observation(control, 0.))
                                 for control in endpoint_probe.sequence) for program in programs]
            if len(set(predictions)) > 1:
                distinguishable = True
                actual = tuple(row.features[name] for row in observed)
                assert actual in predictions
                assert name not in instrument_binding(model)
        assert distinguishable
