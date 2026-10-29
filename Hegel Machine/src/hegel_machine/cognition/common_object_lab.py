"""Expose the existing anonymous object apparatus through ordinary Common I/O."""

from copy import deepcopy
from dataclasses import asdict

from .models import Observation, Probe


class CommonObjectLab:
    """A public adapter; role bindings still come from measured transitions."""

    def __init__(self, laboratory):
        self.laboratory = laboratory
        self.rows = 0
        self.queries = 0

    def public_spec(self):
        spec = self.laboratory.spec_for(('fine',))
        return {'controls': list(spec.feature_bounds),
            'feature_bounds': deepcopy(spec.feature_bounds),
            'max_probe_length': spec.max_probe_length,
            'observable_fields': list(spec.feature_names),
            'action_kinds': ['trajectory'], 'channels': ['fine']}

    @staticmethod
    def action_cost(action):
        return len(action['sequence'])

    def execute(self, action):
        if action.get('kind') != 'trajectory' or action.get('channels', ['fine']) != ['fine']:
            raise ValueError('Object relation research requires fine-bank trajectories')
        spec = self.laboratory.spec_for(('fine',))
        if not 0 < len(action['sequence']) <= spec.max_probe_length:
            raise ValueError('Trajectory exceeds the public object apparatus limit')
        probe = Probe(f'common_object:{self.queries}', tuple(action['sequence']))
        batch = self.laboratory.acquire(probe, ('fine',))
        first = self.rows
        self.rows += len(batch.observations)
        self.queries += 1
        return {'action': deepcopy(action), 'probe_id': probe.probe_id,
            'rows': [asdict(row) for row in batch.observations],
            'transitions': deepcopy(batch.transitions),
            'row_indices': list(range(first, self.rows)), 'observation_index': first,
            'observation_cost': batch.observation_cost, 'measurement_cost': batch.measurement_cost}

    @staticmethod
    def observation_rows(record):
        return [Observation(row['features'], row['target'], tuple(row.get('history', [])))
                for row in record['rows']]


def object_records_from_common(observations):
    """Project only purchased before/after observations, keeping resets paired."""
    records = []
    for record in observations:
        sequence = record['action']['sequence']
        for position, (controls, row, transition) in enumerate(zip(
                sequence, record['rows'], record['transitions'])):
            records.append({'controls': deepcopy(controls), **deepcopy(transition),
                'target': row['target'], 'history': deepcopy(sequence[:position]),
                'position': position, 'probe_id': record['probe_id']})
    return records


def measured_selector_values(model, records):
    """Use the same observed selector binding as the historical inducer."""
    selector = (model or {}).get('focus_control')
    selected = {}
    if selector:
        for field in model.get('focus_candidates', {}):
            active = next((row for row in records if row['before'].get(field) == 1.), None)
            if active is not None:
                selected[field] = active['controls'][selector]
    return selected
