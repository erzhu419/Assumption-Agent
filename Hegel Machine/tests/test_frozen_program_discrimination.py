"""A fresh distinguishing measurement follows its frozen public forecast."""

from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, Probe
from scripts import run_frozen_program_discrimination as study


def fixture(tmp_path, monkeypatch, *, only_old=False):
    output = tmp_path / 'run'
    public = {'controls': ['u'], 'feature_bounds': {'u': [-1., 1.]},
        'control_levels': {'u': [-1., .2, .8]}, 'action_kinds': ['trajectory'],
        'max_probe_length': 24}

    class PublicLab:
        def __init__(self, **_):
            self.costs = {'observation_cost': 0, 'measurement_cost': 0}
            self.actions = []

        def public_spec(self):
            return deepcopy(public)

        def action_cost(self, action):
            return len(action['sequence'])

        def execute(self, action):
            # Opening a target before the prospective file is committed would
            # make this a retrospective comparison rather than an experiment.
            frozen = json.loads((output / 'frozen_plan.json').read_text())
            assert frozen['selected_action'] == action
            assert [h['predictions'] for h in frozen['hypotheses']] == [[0., 0.], [.8, .8]]
            self.actions.append(deepcopy(action))
            self.costs['observation_cost'] += len(action['sequence'])
            history, rows = [], []
            for point in action['sequence']:
                rows.append(asdict(Observation(point, 2., tuple(history))))
                history.append(point)
            return {'action': deepcopy(action), 'rows': rows,
                'row_indices': list(range(len(rows))), 'observation_cost': len(rows),
                'measurement_cost': 0}

        def observation_rows(self, record):
            return [Observation(row['features'], row['target'], tuple(row['history']))
                    for row in record['rows']]

    lab = PublicLab()
    monkeypatch.setattr(study, 'AnonymousHistoryLab', lambda **_: lab)
    # Runtime copying is unchanged shared infrastructure; these controls target
    # forecast ordering, charging and scientific interpretation only.
    monkeypatch.setattr(study, 'clone_or_copy', lambda *args, **kwargs: None)

    def reference(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {'file': str(path), 'json_path': []}

    old = [{'u': -1.}, {'u': -1.}]
    registration = {'planning_compute_cap': 6000, 'assessment_compute_cap': 48,
        'activity_seconds': 30, 'serial': 17, 'observation_cap': 24,
        'mse_threshold': 1e-10, 'prior_costs_ref': {'file': 'retained_old_account.json'},
        'case_ref': reference('case.json', {'kind': 'fixture', 'seed': 0}),
        'paid_records_ref': reference('records.json', [{'action': {'sequence': old},
            'rows': [asdict(Observation(point, 2.)) for point in old]}]),
        'hypotheses': [{'hypothesis_id': name, 'candidate_id': name,
            'program_ref': reference(name + '.json', program.to_dict())}
            for name, program in [('old', Program('const', value=0.)),
                                  ('new', Program('feature', name='u'))]]}

    def planner(spec, programs, budget, *, serial, length, control_levels):
        assert spec.feature_names == ('u',)
        assert spec.feature_bounds == {'u': (-1., 1.)}
        assert control_levels == public['control_levels']
        assert serial == 17 and length == public['max_probe_length']
        assert budget.spend(5)
        values = [-1.] if only_old else [-1., .2, .8]
        return [{'probe': Probe(str(value), ({'u': value}, {'u': value})),
            'predictions': [[0., 0.], [value, value]],
            'predicted_separation': value ** 2, 'sequence_mode': 'maintain',
            'boundary': {'public_control_levels': deepcopy(control_levels)}} for value in values]

    monkeypatch.setattr(study, 'propose_program_disagreement_probes', planner)
    return registration, output, lab


def test_finite_public_grid_and_forecast_are_frozen_before_new_measurement(tmp_path, monkeypatch):
    registration, output, lab = fixture(tmp_path, monkeypatch)
    result = study.run(registration, output)
    assert result['status'] == 'completed'
    assert lab.actions == [{'kind': 'trajectory', 'channels': [],
        'sequence': [{'u': .8}, {'u': .8}]}]
    planning = json.loads((output / 'planning.json').read_text())
    assert planning['eligible_probe_ids'] == ['0.8', '0.2']
    assert result['costs']['planning_compute'] == 5
    assert result['costs']['assessment_compute'] == 4
    assert result['costs']['observation_cost'] == 2
    assert result['new_record_append_contract']['destination_record_index'] == 1
    assert result['new_record_append_contract']['destination_row_offset'] == 2


def test_only_already_purchased_probe_does_not_buy_or_assess_again(tmp_path, monkeypatch):
    registration, output, lab = fixture(tmp_path, monkeypatch, only_old=True)
    monkeypatch.setattr(study, 'assess_competing_object_probe',
        lambda *args, **kwargs: pytest.fail('old trajectory was assessed again'))
    result = study.run(registration, output)
    assert result['status'] == 'no_new_distinguishing_probe'
    assert lab.actions == [] and not (output / 'frozen_plan.json').exists()
    assert result['costs']['compute_cost'] == 5
    assert result['costs']['assessment_compute'] == result['costs']['observation_cost'] == 0


def test_relative_improvement_does_not_confirm_two_failed_frozen_hypotheses(tmp_path, monkeypatch):
    registration, output, _ = fixture(tmp_path, monkeypatch)
    result = study.run(registration, output)
    losses = [item['measured_loss'] for item in result['hypotheses']]
    assert losses[1] < losses[0] and min(losses) > registration['mse_threshold']
    assert result['assessment_status'] == 'all_refuted'
    assert result['next_research'] == 'revise_competing_structures_from_shared_counterexamples'
    assert not result['independent_confirmation'] and not result['parent_replaced']


def test_assessment_failure_preserves_purchased_evidence_and_consumed_cost(tmp_path, monkeypatch):
    registration, output, lab = fixture(tmp_path, monkeypatch)

    def interrupted(plan, pairs, budget, **kwargs):
        assert (output / 'observation.json').is_file()
        assert len(pairs) == 2 and budget.spend(1)
        raise RuntimeError('assessment interrupted after actual work')

    monkeypatch.setattr(study, 'assess_competing_object_probe', interrupted)
    with pytest.raises(RuntimeError, match='assessment interrupted'):
        study.run(registration, output)
    saved = json.loads((output / 'summary.json').read_text())
    observation = json.loads((output / 'observation.json').read_text())
    assert len(lab.actions) == 1 and len(observation['rows']) == 2
    assert saved['status'] == 'failed' and saved['costs']['compute_cost'] == 6
    assert saved['costs']['observation_cost'] == 2
    assert saved['costs']['fits'] == saved['costs']['model_calls'] == 0
    assert not saved['parent_replaced'] and not saved['independent_confirmation']
