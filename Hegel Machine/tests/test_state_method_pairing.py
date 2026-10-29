"""Small real Common arms preserve the freeze-before-evaluation method contract."""

from pathlib import Path

from hegel_machine.cognition.checkpoint_storage import read_json_checkpoint
from hegel_machine.cognition.models import Observation
from scripts import run_state_method_pairing as study
from test_native_startup_revision import scene
from test_native_representation_routing import PublicOnlyLab


def paid_fixture(tmp_path):
    agent, _, arguments = scene(tmp_path / 'fixture')
    values = {'records': agent.observations, 'public': PublicOnlyLab().public_spec(),
        'programs': {item['id']: item['program'] for item in arguments['theories']},
        'counterexample': {'rows': [0]}, 'costs': {'compute_cost': 0},
        'case': {'kind': 'test_only', 'seed': 11}}
    reference = study.save(tmp_path / 'inputs.json', values)
    ref = lambda *path: {**reference, 'json_path': list(path)}
    paid = {'paid_records_ref': ref('records'), 'append_record_refs': [],
        'public_spec_ref': ref('public'), 'expected_rows': len(agent.rows),
        'parents': [{'candidate_id': name, 'program_ref': ref('programs', name)}
                    for name in values['programs']],
        'diagnostic_parent_id': 'old', 'validation_records': [3],
        'counterexample_ref': ref('counterexample'), 'prior_costs_ref': ref('costs'),
        'mse_threshold': 1e-10, 'research_compute_cap': 10000, 'max_candidates': 4}
    return study.save(tmp_path / 'paid.json', paid), ref('case'), values


def test_real_methods_complete_before_shared_evaluator_and_do_not_launch_successors(tmp_path, monkeypatch):
    paid, case, values = paid_fixture(tmp_path)
    prepared = study.prepare(paid, tmp_path / 'prepared', case_ref=case, evaluation_seed=401,
        research_compute_cap=10000, compute_cap=30000, evaluation_rows=20,
        approval_ref=paid)
    assert prepared['quote']['status'] == 'quoted'
    assert prepared['cells']['00']['selection_plan'].get('first_segment') is None
    selected = prepared['cells']['10']['selection_plan']['first_segment']
    assert selected['arguments']['native_representation_plan']['search_directions'] == ['discrete_state']
    directory = tmp_path / 'run'
    opened = []

    class Evaluator:
        def __init__(self, **_):
            manifest = read_json_checkpoint(directory / 'frozen_outputs.json')
            assert set(manifest['models']) == {'00', '10'}
            assert not manifest['evaluation_labels_opened']
            opened.append(True)

        def evaluation_rows(self, *, length, seed):
            assert length == 20 and seed == 401
            return [Observation(row['features'], row['target'], tuple(row['history']))
                    for record in values['records'] for row in record['rows']]

    monkeypatch.setattr(study, 'AnonymousHistoryLab', Evaluator)
    result = study.run(prepared, directory)
    assert opened == [True]
    assert result['evaluation_lifecycle']['all_outputs_frozen_before_labels']
    outcomes = {name: read_json_checkpoint(directory / name / 'outcome.json') for name in ('00', '10')}
    for name, outcome in outcomes.items():
        assert outcome['execution_complete']
        assert outcome['native_research_complete']
        assert outcome['registered_attempt_returned']
        assert outcome['comparison']['models_requested'] == outcome['comparison']['models_completed']
        assert outcome['costs']['observation_cost'] == outcome['costs']['model_calls'] == 0
        assert outcome['method_ref'] == prepared['cells'][name]['method_ref']
    assert outcomes['00']['actual_action_content'] != outcomes['10']['actual_action_content']
    assert outcomes['00']['data_ref'] == outcomes['10']['data_ref']
    assert not result['ordinary_use']['successors_started']
    assert not result['ordinary_use']['general_guard_supported']
    selection = read_json_checkpoint(directory / 'method_selection.json')
    assert selection['preferences'][0]['preference'] == 'baseline'
    assert selection['preferences'][0]['identified_effect'] == 'fixed_data_method_effect'
    assert not selection['unresolved_feedback_refs']
    assert read_json_checkpoint(directory / 'evaluation_material.json')['new_response_rows'] == 20


def test_preparation_only_reads_public_paid_material(tmp_path, monkeypatch):
    paid, case, _ = paid_fixture(tmp_path)
    def no_science(*args, **kwargs):
        raise AssertionError('static preparation cannot fit or open future labels')
    monkeypatch.setattr(study, 'AnonymousHistoryLab', no_science)
    monkeypatch.setattr('hegel_machine.cognition.state_search.closed_startup_claims', no_science)
    monkeypatch.setattr('hegel_machine.cognition.state_search.closed_state_claims', no_science)
    prepared = study.prepare(paid, tmp_path / 'prepared', case_ref=case, evaluation_seed=402,
        research_compute_cap=10000, compute_cap=30000, evaluation_rows=20)
    assert not prepared['execution_authorized']
    selection = read_json_checkpoint(Path(prepared['static_selection_ref']['file']))
    assert selection['scientific_fits'] == selection['new_observations'] == 0


def test_shared_paid_comparison_is_restored_before_method_profiles_and_not_repeated(tmp_path, monkeypatch):
    import pytest
    from hegel_machine.cognition.method_conditions import _method_observables

    paid, case, values = paid_fixture(tmp_path)
    registration = study.prepare(paid, tmp_path / 'prepared', case_ref=case, evaluation_seed=403,
        research_compute_cap=10000, compute_cap=30000, evaluation_rows=20, approval_ref=paid)
    source, source_lab, parents = study._scene(registration, tmp_path / 'source')
    source._execute('run', {'operator': 'research_choice',
        'models': [name + '.py' for name in parents],
        'research_question': source.knowledge['finite_native_author']['question']}, source_lab)
    source._prepare_program_state()
    registration['initial_scene_ref'] = study.save(source.workspace / 'training.json',
        source.checkpoint_result('researching'))
    for cell in registration['cells'].values():
        cell['start_state_ref'] = registration['initial_scene_ref']
    before = dict(source.costs)
    directory = tmp_path / 'run'
    opened = []

    class PublicRestoreLab(PublicOnlyLab):
        def __init__(self, **_):
            self._past_costs = {'observation_cost': 0, 'measurement_cost': 0, 'queries': 0}

        @property
        def costs(self):
            return self._past_costs

        def observation_rows(self, record):
            return [Observation(row['features'], row['target'], tuple(row['history']))
                    for row in record['rows']]

        def evaluation_rows(self, *, length, seed):
            manifest = read_json_checkpoint(directory / 'frozen_outputs.json')
            assert set(manifest['models']) == {'00', '10'}
            assert not manifest['evaluation_labels_opened']
            opened.append(True)
            return [row for record in values['records'] for row in self.observation_rows(record)]

    monkeypatch.setattr(study, 'AnonymousHistoryLab', PublicRestoreLab)
    result = study.run(registration, directory)
    assert opened == [True]
    profiles = []
    for name in ('00', '10'):
        outcome = read_json_checkpoint(directory / name / 'outcome.json')
        profile = study.read(outcome['profile_ref'])
        profiles.append(profile)
        assert profile['residual']['state'] == 'unresolved'
        assert profile['residual']['transition']['startup_first_divergences'] > 0
        assert profile['residual']['transition']['mature_first_divergences'] == 0
        assert outcome['loss'] == 0 and outcome['execution_complete']
        assert {key: outcome['costs_after'][key] - outcome['costs'][key]
                for key in before} == pytest.approx(before)
        state = read_json_checkpoint(directory / name / 'workspace/research_state.json')
        comparisons = [entry for entry in state['operator_results']
                       if entry.get('operator') == 'research_choice']
        assert comparisons[0]['evaluated_model']['mse'] == profile['residual']['loss'] > 0
        for comparison in comparisons[1:]:
            retained = [row for row in comparison['evaluations']
                        if row['path'] in {name + '.py' for name in parents}]
            assert all(row['paid_score_reuse']['status'] == 'reused' for row in retained)
    assert _method_observables(profiles[0]) == _method_observables(profiles[1])
    selection = read_json_checkpoint(directory / 'method_selection.json')
    assert selection['preferences'][0]['preference'] == 'baseline'
    assert not result['ordinary_use']['general_guard_supported']
