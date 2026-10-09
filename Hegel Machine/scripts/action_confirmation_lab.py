"""Finite nested event-response laboratory for confirmation/recovery studies."""

from dataclasses import asdict
from random import Random

from hegel_machine.cognition.measurement_lab import MeasurementBatch
from hegel_machine.cognition.models import Observation, Probe, TaskSpec


class NestedResponseLab:
    channels = ()

    def __init__(self, radius):
        self.radius = radius
        bounds = {'f0': (1., 1.), 'f1': (0., 0.), 'q0': (-1., 1.),
                  'q1': (-1., 1.), 'x0': (-1., 1.), 'x1': (-1., 1.)}
        self.spec = TaskSpec('controlled_nested_response', 'Predict the controlled response',
            tuple(bounds), (), (), feature_bounds=bounds, max_probe_length=64)
        self.acquisitions = []
        self.observation_cost = 0

    @property
    def costs(self):
        return {'observation_cost': self.observation_cost, 'measurement_cost': 0,
                'queries': len(self.acquisitions)}

    def spec_for(self, channels=()):
        if channels:
            raise ValueError('This laboratory exposes response and controls only')
        return self.spec

    def random_probe(self, seed, *, length=24, name='sample'):
        random = Random(seed)
        return Probe(name, tuple({field: random.uniform(*limits)
            for field, limits in self.spec.feature_bounds.items()} for _ in range(length)))

    def acquire(self, probe, channels=()):
        self.spec_for(channels)
        rows, history, flows, combined = [], [], [], []
        for point in probe.sequence:
            flow = ((point['f0'] - point['f1']) * (point['q1'] - point['q0'])
                    if abs(point['x0'] - point['x1']) < self.radius else 0.)
            event = flow - flows[-1] if flows else 0.
            flows.append(flow)
            combined.append(1.3 * (point['q0'] + point['q1']) - .7 * event)
            target = sum(combined[-3:])
            rows.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, '__target__': target})
        self.observation_cost += len(rows)
        self.acquisitions.append({'probe': asdict(probe), 'channels': [],
            'observations': [asdict(row) for row in rows], 'observation_cost': len(rows),
            'measurement_cost': 0})
        return MeasurementBatch(rows, len(rows), 0, ())

    def evaluation_rows(self, *, seed, length=64, trajectories=4):
        independent = NestedResponseLab(self.radius)
        return [row for index in range(trajectories) for row in independent.acquire(
            independent.random_probe(seed + 10007 * index, length=length,
                                     name=f'evaluation_{index}')).observations]


def supplied_source_family(radius):
    """The same explicitly supplied source family as the nested-response test."""
    from hegel_machine.cognition.language import Program
    c = lambda value: Program('const', value=value)
    f = lambda name: Program('feature', name=name)
    relation = Program('neighbor_sum', children=(c(radius),
        *(f(name) for name in ('f0', 'x0', 'q0', 'f1', 'x1', 'q1'))))
    total = Program('object_sum', children=(f('q0'), f('q1')))
    event = Program('event_delta', lag=1, children=(relation,))
    combined = Program('add', children=(Program('mul', children=(c(1.3), total)),
                                        Program('mul', children=(c(-.7), event))))
    return Program('window_sum', lag=3, children=(combined,))
