"""The terminal review reuses success and measures only surviving competition."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from hegel_machine.cognition.checkpoint_storage import read_json_checkpoint
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from scripts import run_anonymous_revision_inquiry as evaluator
from scripts import run_independent_state_method_source as entry


def prepared(tmp_path, monkeypatch, *, resolve=False):
    workspace = tmp_path / 'source'
    workspace.mkdir()
    public = {'controls': ['u'], 'feature_bounds': {'u': [-1., 1.]},
        'control_levels': {'u': [i / 10 for i in range(-10, 11)]},
        'action_kinds': ['trajectory'], 'max_probe_length': 24}

    def threshold(value):
        return Program('ge', children=(Program('feature', name='u'), Program('const', value=value)))

    candidates, declared = [], []
    for index, boundary in enumerate((.35, .225, .475)):
        program = Program('state_replay', children=(Program('if', children=(
            Program('available', lag=1), threshold(.55), threshold(boundary))),))
        model = f'candidate{index}.py'
        (workspace / model).write_text(export_temporal_program(program, feature_names=('u',)))
        candidates.append({'id': f'candidate{index}', 'program': program.to_dict()})
        declared.append({'native_index': index, 'model': model})
    values = [(-.7, 0.), (.3, 0.), (.7, 1.)]
    if resolve:
        values.append((.4, 1.))
    values.extend([(.7, 1.)] * (512 - len(values)))
    records = [{'action': {'kind': 'trajectory', 'sequence': [{'u': value}], 'channels': []},
        'rows': [{'features': {'u': value}, 'target': target, 'history': []}],
        'row_indices': [index], 'observation_cost': 1, 'measurement_cost': 0}
        for index, (value, target) in enumerate(values)]
    case_ref = entry.save(tmp_path / 'case.json', {'kind': 'fixture_only', 'seed': 4})
    original = entry.save(tmp_path / 'original_request.json', {
        'new_compute_cap': 60000, 'new_observation_cap': 536,
        'new_python_seconds': 120., 'activity_seconds': 180., 'case_ref': case_ref})
    training_ref = entry.save(workspace / 'training.json', {
        'workspace': str(workspace), 'paid_observations': records})
    source_ref = entry.save(tmp_path / 'source_result.json', {
        'request_ref': original, 'training_ref': training_ref,
        'new_costs': {'compute_cost': 16123, 'observation_cost': 512,
                      'python_execution_seconds': 7.},
        'costs_after': {'compute_cost': 37008, 'observation_cost': 512,
                        'python_execution_seconds': 8., 'measurement_cost': 0},
        'activity_wall_seconds': 11.,
        'confirmation': {'observations': records, 'rows': 512, 'returncode': 0,
            'mse': 0., 'passed': True, 'mse_threshold': 1e-10,
            'submitted_candidate': {'model': 'candidate0.py'}}})
    request = {'source_result_ref': source_ref,
        'native_result_ref': entry.save(tmp_path / 'native.json', {'candidates': candidates}),
        'public_spec_ref': entry.save(tmp_path / 'public.json', public), 'candidates': declared}
    events = []

    def frozen_evaluation(directory, model, rows, *, timeout_seconds):
        assert model != '0.py', 'the already confirmed candidate must reuse its paid score'
        assert timeout_seconds <= 113.
        events.append(('evaluate', model))
        namespace = {}
        exec((Path(directory) / model).read_text(), namespace)
        # This fixture executes only the frozen predict function, never fit.
        predictions = [namespace['predict']({'features': row.features, 'history': row.history})
                       for row in rows]
        return {'returncode': 0, 'mse': sum((a - b.target) ** 2 for a, b in zip(predictions, rows)) / len(rows),
            'evaluation_rows': len(rows), 'predictions': predictions, 'python_execution_seconds': .01}

    class Lab:
        def __init__(self, kind, seed):
            assert (kind, seed) == ('fixture_only', 4)
            assert not resolve, 'settled competition must not construct a new apparatus'

        def public_spec(self):
            return public

        def restore_paid_account(self, observations, costs):
            self.offset = sum(len(record['rows']) for record in observations)
            assert self.offset == 512 and costs['observation_cost'] == 512

        def execute(self, action):
            assert events == [('evaluate', '1.py'), ('evaluate', '2.py')]
            assert action['sequence'] == [{'u': .4}]
            assert read_json_checkpoint(tmp_path / 'review/paid_review.json')['plan'] is not None
            events.append(('query', deepcopy(action)))
            row = Observation({'u': .4}, 1., ())
            return {'action': action, 'rows': [asdict(row)], 'row_indices': [self.offset],
                'observation_cost': 1, 'measurement_cost': 0}

        def observation_rows(self, record):
            return [Observation(row['features'], row['target'], tuple(row['history']))
                    for row in record['rows']]

    monkeypatch.setattr(evaluator, 'evaluate_terminal_model', frozen_evaluation)
    monkeypatch.setattr(entry, 'AnonymousHistoryLab', Lab)
    return request, workspace, events


def test_paid_review_reuses_success_scores_competitors_first_and_buys_only_shortest_legal_probe(tmp_path, monkeypatch):
    request, workspace, events = prepared(tmp_path, monkeypatch)
    before = (workspace / 'candidate0.py').read_bytes()
    result = entry.review_paid_confirmation(request, tmp_path / 'review')
    assert result['status'] == 'distinguished', result
    assert [score['reused'] for score in result['scores']] == [True, False, False]
    assert len(events) == 3 and events[-1][0] == 'query'
    assert result['compute_limit'] == 43877 and result['observation_limit'] == 24
    assert result['new_costs']['observation_cost'] == 1
    assert result['new_costs']['compute_cost'] >= 1024
    assert result['fits'] == result['model_calls'] == 0
    assert not result['confirmed_model_changed']
    assert (workspace / 'candidate0.py').read_bytes() == before
    assert read_json_checkpoint(request['source_result_ref']['file'])['confirmation']['mse'] == 0.


def test_paid_review_stops_without_query_when_existing_confirmation_resolves_competition(tmp_path, monkeypatch):
    request, _, events = prepared(tmp_path, monkeypatch, resolve=True)
    result = entry.review_paid_confirmation(request, tmp_path / 'review')
    assert result['status'] == 'competition_resolved_on_paid_scope', result
    assert events == [('evaluate', '1.py'), ('evaluate', '2.py')]
    assert result['new_costs']['compute_cost'] == 1024
    assert result['new_costs']['observation_cost'] == 0
    assert result['fits'] == result['model_calls'] == 0
