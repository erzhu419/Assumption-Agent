"""Frozen Common confirmation and fixed panels share labels without refitting."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from scripts import run_frozen_state_confirmation as study


class PublicFixtureLab:
    """A deterministic test instrument; no production apparatus is consulted."""
    confirmation_precondition = staticmethod(AnonymousHistoryLab.confirmation_precondition)
    observation_rows = AnonymousHistoryLab.observation_rows
    restore_paid_account = AnonymousHistoryLab.restore_paid_account

    def __init__(self, output, *, failed_fresh=False):
        self.output, self.failed_fresh = output, failed_fresh
        self._costs = {'observation_cost': 0, 'measurement_cost': 0, 'queries': 0}
        self._next_row_index = 0
        self.calls = []

    @property
    def costs(self):
        return dict(self._costs)

    def public_spec(self):
        return {'controls': ['u'], 'feature_bounds': {'u': [-1., 1.]},
            'control_levels': {'u': [-1., 0., 1.]}, 'max_probe_length': 4,
            'evaluation_max_trajectory_length': 4, 'action_kinds': ['trajectory']}

    @staticmethod
    def rows(sequence, *, zero=False):
        history, rows = [], []
        for point in sequence:
            rows.append(Observation(dict(point), 0. if zero else point['u'], tuple(history)))
            history.append(dict(point))
        return rows

    def assert_frozen(self):
        protocol = json.loads((self.output / 'frozen_protocol.json').read_text())
        if protocol.get('post_confirmation_panels') == []:
            assert protocol['holds'] == []
        else:
            assert len(protocol['holds']) == 7
            assert all(len(item['sequence']) == 4 for item in protocol['holds'])
            assert len({tuple(point['u'] for point in item['sequence'])
                        for item in protocol['holds']}) == 7
        assert protocol['fresh_row_identity_offset'] == 2
        assert not protocol['new_labels_available_at_freeze']
        assert all((self.output / 'workspace' / model).is_file()
                   for model in protocol['models'].values())

    def confirmation_actions(self, n, rng):
        self.assert_frozen()
        values = [1.] * n if self.failed_fresh else [rng.choice([-1., 0., 1.]) for _ in range(n)]
        return [{'kind': 'trajectory', 'channels': [],
                 'sequence': [{'u': value} for value in values]}]

    def execute(self, action):
        self.assert_frozen()
        self.calls.append(('fresh', deepcopy(action)))
        rows = self.rows(action['sequence'])
        first = self._next_row_index
        self._next_row_index += len(rows)
        self._costs['observation_cost'] += len(rows)
        self._costs['queries'] += 1
        return {'action': deepcopy(action), 'rows': [asdict(row) for row in rows],
            'observation_index': first, 'row_indices': list(range(first, first + len(rows))),
            'measurements': [], 'observation_cost': len(rows), 'measurement_cost': 0}

    def evaluation_rows(self, *, length, seed):
        self.assert_frozen()
        self.calls.append(('random', length))
        return [row for start in range(0, length, 4) for row in self.rows(
            [{'u': float(index % 3 - 1)} for index in range(min(4, length - start))],
            zero=self.failed_fresh)]

    def evaluation_sequences(self, sequences):
        self.assert_frozen()
        self.calls.append(('holds', deepcopy(sequences)))
        return [row for sequence in sequences for row in self.rows(sequence, zero=self.failed_fresh)]


def registration(tmp_path, *, failed_fresh=False):
    def ref(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {'file': str(path), 'json_path': []}

    sequence = [{'u': 0.}, {'u': 0. if failed_fresh else 1.}]
    record = {'action': {'kind': 'trajectory', 'channels': [], 'sequence': sequence},
        'rows': [asdict(row) for row in PublicFixtureLab.rows(sequence)],
        'row_indices': [0, 1], 'observation_index': 0, 'measurements': [],
        'observation_cost': 2, 'measurement_cost': 0}
    prior = {'paid_records_ref': ref('paid.json', [record]), 'append_record_refs': []}
    selected = Program('const', value=0.) if failed_fresh else Program('feature', name='u')
    source = tmp_path / 'selected_source.py'
    source.write_text(export_temporal_program(selected))
    return {'case_ref': ref('case.json', {'kind': 'fixture', 'seed': 1}),
        'paid_registration_ref': ref('prior.json', prior),
        'models': [{'id': 'selected', 'source_file': str(source)},
            {'id': 'old', 'program_ref': ref('old.json', Program('const', value=0.).to_dict())},
            {'id': 'residual', 'program_ref': ref('residual.json', Program('const', value=2.).to_dict())}],
        'selected_id': 'selected', 'confirmation_seed': 11, 'confirmation_rows': 4,
        'terminal_random_rows': 6, 'terminal_seed': 17,
        'mse_threshold': 1e-10, 'common_compute_cap': 20, 'comparison_compute_cap': 150,
        'execution_seconds': 20., 'activity_seconds': 60.,
        'prior_costs_ref': {'file': 'old_ledger.json', 'json_path': []}}


@pytest.mark.parametrize('failed_fresh', [False, True])
def test_real_common_finish_and_frozen_panels_preserve_gates_and_costs(tmp_path, monkeypatch, failed_fresh):
    plan = registration(tmp_path, failed_fresh=failed_fresh)
    output = tmp_path / 'run'
    lab = PublicFixtureLab(output, failed_fresh=failed_fresh)
    monkeypatch.setattr(study, 'AnonymousHistoryLab', lambda **kwargs: lab)
    source = Path(plan['models'][0]['source_file'])
    original = source.read_bytes()
    # Source snapshots have independent controls; preserve the actual submitted
    # source copy while avoiding hundreds of unrelated copies in this fixture.
    monkeypatch.setattr(study, 'clone_or_copy', lambda src, dst:
        Path(dst).write_bytes(Path(src).read_bytes()) if Path(src) == source else None)
    execute = study.CommonArchiveInquiryAgent._execute
    actions = []

    def called(self, action, arguments, laboratory):
        actions.append(action)
        return execute(self, action, arguments, laboratory)

    monkeypatch.setattr(study.CommonArchiveInquiryAgent, '_execute', called)
    terminal = study.evaluate_terminal_model
    scores = []

    def scored(workspace, model, rows, **kwargs):
        scores.append((model, len(rows)))
        return terminal(workspace, model, rows, **kwargs)

    monkeypatch.setattr(study, 'evaluate_terminal_model', scored)
    result = study.run(plan, output)
    assert result['status'] == 'completed'
    assert actions == ['finish']
    assert result['selected_passed_all'] is (not failed_fresh)
    assert result['common_model_adopted_in_isolated_workspace'] is (not failed_fresh)
    assert not result['parent_archive_replaced'] and not result['independent_task_transfer']
    assert scores[:2] == [('old.py', 4), ('residual.py', 4)]
    assert ('selected.py', 4) not in scores  # Common result is reused, not rescored.
    assert len(scores) == 8
    assert result['costs']['common']['compute_cost'] == 6  # Paid2 plus fresh4.
    assert result['costs']['fixed_comparison_compute'] == 110
    assert result['costs']['compute_cost'] == 116
    assert result['costs']['purchased_confirmation_rows'] == 4
    assert lab.costs['observation_cost'] == 4
    assert result['costs']['evaluator_response_rows'] == 34
    assert result['costs']['unique_new_response_rows'] == 38
    assert result['costs']['common']['model_calls'] == result['fits'] == 0
    finished = json.loads((output / 'common_finish.json').read_text())
    assert finished['fresh_confirmation']['observation_indices'] == [2, 3, 4, 5]
    assert source.read_bytes() == original
    assert (output / 'workspace' / 'selected.py').read_bytes() == original
    for name, count in [('terminal_random_material.json', 6), ('long_hold_material.json', 28)]:
        compact = json.loads((output / name).read_text())
        assert sum(len(item['targets']) for item in compact) == count
        assert all(set(item) == {'sequence', 'targets'} for item in compact)
    if failed_fresh:
        assert result['panels']['fresh_confirmation']['selected']['passed'] is False
        assert result['panels']['random48']['selected']['passed'] is True
        assert result['panels']['long_hold48']['selected']['passed'] is True
        assert result['next_step'] == 'localize_frozen_counterexamples_before_any_revision'
    assert json.loads((output / 'summary.json').read_text()) == json.loads(json.dumps(result))


@pytest.mark.parametrize('failed_fresh', [False, True])
def test_confirmation_only_uses_real_common_finish_without_extra_panels(tmp_path, monkeypatch, failed_fresh):
    plan = registration(tmp_path, failed_fresh=failed_fresh)
    plan.update(post_confirmation_panels=[], comparison_compute_cap=0, common_compute_cap=6)
    plan['models'] = plan['models'][:1]
    del plan['terminal_random_rows'], plan['terminal_seed']
    output = tmp_path / 'confirmation_only'
    lab = PublicFixtureLab(output, failed_fresh=failed_fresh)
    monkeypatch.setattr(study, 'AnonymousHistoryLab', lambda **kwargs: lab)
    source = Path(plan['models'][0]['source_file'])
    original = source.read_bytes()
    monkeypatch.setattr(study, 'clone_or_copy', lambda src, dst:
        Path(dst).write_bytes(Path(src).read_bytes()) if Path(src) == source else None)

    def unexpected(*args, **kwargs):
        pytest.fail('this registered unit contains only Common confirmation')

    monkeypatch.setattr(study, 'hold_panel', unexpected)
    monkeypatch.setattr(study, 'evaluate_terminal_model', unexpected)
    monkeypatch.setattr(lab, 'evaluation_rows', unexpected)
    monkeypatch.setattr(lab, 'evaluation_sequences', unexpected)
    result = study.run(plan, output)
    assert result['status'] == 'completed'
    assert result['selected_passed_confirmation'] is (not failed_fresh)
    assert result['selected_passed_all'] is (not failed_fresh)
    assert result['common_model_adopted_in_isolated_workspace'] is (not failed_fresh)
    assert result['post_confirmation_panels'] == []
    assert list(result['panels']) == ['fresh_confirmation']
    assert result['panel_materials'] == {}
    assert [kind for kind, _ in lab.calls] == ['fresh']
    assert result['costs']['compute_cost'] == 6  # Real paid2 precondition + fresh4.
    assert result['costs']['fixed_comparison_compute'] == 0
    assert result['costs']['unique_new_response_rows'] == 4
    assert result['costs']['purchased_confirmation_rows'] == 4
    assert result['costs']['evaluator_response_rows'] == 0
    assert result['costs']['common']['model_calls'] == result['fits'] == 0
    assert not result['parent_archive_replaced']
    assert not (output / 'terminal_random_material.json').exists()
    assert not (output / 'long_hold_material.json').exists()
    finished = json.loads((output / 'common_finish.json').read_text())
    assert finished['mse'] == 0.
    assert finished['fresh_confirmation']['observations']
    assert finished['fresh_confirmation']['observation_indices'] == [2, 3, 4, 5]
    assert finished['fresh_confirmation']['passed'] is (not failed_fresh)
    assert result['next_step'] == ('localize_frozen_counterexamples_before_any_revision'
        if failed_fresh else 'retain_confirmed_candidate_and_review_method_evidence')
    assert source.read_bytes() == original
    submitted = output / 'workspace' / result['submitted_candidate']['model']
    assert submitted.read_bytes() == original
    assert json.loads((output / 'summary.json').read_text()) == json.loads(json.dumps(result))


@pytest.mark.parametrize('changed', [None, 'model', 'material'])
def test_retained_hold_material_reuses_only_matching_baseline_scores(tmp_path, monkeypatch, changed):
    plan = registration(tmp_path)
    original_output = tmp_path / 'original'
    lab = PublicFixtureLab(original_output)
    monkeypatch.setattr(study, 'AnonymousHistoryLab', lambda **kwargs: lab)
    source = Path(plan['models'][0]['source_file'])
    monkeypatch.setattr(study, 'clone_or_copy', lambda src, dst:
        Path(dst).write_bytes(Path(src).read_bytes()) if Path(src) == source else None)
    original = study.run(plan, original_output)
    material_file = original_output / 'long_hold_material.json'
    original_material = material_file.read_bytes()
    plan['hold_material_ref'] = {'file': str(material_file), 'json_path': []}
    plan['hold_score_refs'] = {identifier: {
        'score_ref': {'file': str(original_output / 'summary.json'),
                      'json_path': ['panels', 'long_hold48', identifier]},
        'model_file': str(original_output / 'workspace' / (identifier + '.py'))}
        for identifier in ('old', 'residual')}
    if changed == 'model':
        # A baseline edited after the old run is a new prediction, even when
        # the task and number of held rows are unchanged.
        Path(plan['models'][1]['program_ref']['file']).write_text(
            json.dumps(Program('const', value=.5).to_dict()))
    elif changed == 'material':
        records = json.loads(original_material)
        records[0]['targets'][0] = 20.
        changed_file = tmp_path / 'different_material.json'
        changed_file.write_text(json.dumps(records))
        plan['hold_material_ref'] = {'file': str(changed_file), 'json_path': []}

    output = tmp_path / 'reuse'
    lab = PublicFixtureLab(output)
    terminal = study.evaluate_terminal_model
    scored = []

    def counted(workspace, model, rows, **kwargs):
        scored.append((model, len(rows)))
        assert all('__target__' not in point for row in rows for point in row.history)
        return terminal(workspace, model, rows, **kwargs)

    monkeypatch.setattr(study, 'evaluate_terminal_model', counted)
    result = study.run(plan, output)
    assert result['status'] == 'completed'
    assert [kind for kind, _ in lab.calls] == ['fresh', 'random']
    assert not (output / 'long_hold_material.json').exists()
    assert material_file.read_bytes() == original_material
    role = result['panel_materials']['long_hold48']
    assert role['role'] == 'retained_evaluation_regression'
    assert role['source_ref'] == plan['hold_material_ref']
    assert role['rows'] == 28 and role['new_response_rows'] == 0
    assert result['costs']['retained_evaluation_rows'] == 28
    assert result['costs']['evaluator_response_rows'] == 6
    assert result['costs']['unique_new_response_rows'] == 10
    assert result['costs']['compute_cost'] == {None: 60, 'model': 88, 'material': 116}[changed]
    assert ('selected.py', 28) in scored
    reused_ids = ({'old', 'residual'} if changed is None else
                  {'residual'} if changed == 'model' else set())
    for identifier in ('old', 'residual'):
        score = result['panels']['long_hold48'][identifier]
        assert ((identifier + '.py', 28) in scored) is (identifier not in reused_ids)
        if identifier in reused_ids:
            assert score['new_compute_cost'] == 0
            assert score['reused_from'] == plan['hold_score_refs'][identifier]['score_ref']
            assert score['predictions_ref'] == original['panels']['long_hold48'][identifier]['predictions_ref']
            assert not (output / 'predictions' / ('long_hold48_' + identifier + '.json')).exists()
        else:
            assert 'reused_from' not in score
