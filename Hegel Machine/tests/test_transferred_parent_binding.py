"""A transferred pool's measured winner replaces the diagnostic question seed."""

from copy import deepcopy

from scripts import run_independent_state_method_source as entry
from scripts import run_state_method_pairing as study
from scripts.run_paid_state_revision import bind_restored_state_author
from test_state_method_pairing import paid_fixture


def test_registered_parent_binding_keeps_theory_and_complete_pool(tmp_path):
    paid, case, values = paid_fixture(tmp_path)
    source = study.read(paid)
    source['diagnostic_parent_id'] = 'constant'
    study.save(paid['file'], source)
    prepared = study.prepare(paid, tmp_path / 'prepared', case_ref=case,
        evaluation_seed=501, research_compute_cap=10000, compute_cap=30000,
        evaluation_rows=20, approval_ref=paid)
    agent, lab, models = study._scene(prepared, tmp_path / 'scene')
    choice = agent._execute('run', {'operator': 'research_choice',
        'models': [name + '.py' for name in models],
        'research_question': prepared['question']}, lab)
    assert choice['models_completed'] == len(values['programs'])
    assert agent.evaluated_model['source_model'] == 'old.py'
    old_index, before = deepcopy(agent.theory_index), deepcopy(agent.costs)
    bound = bind_restored_state_author(agent, prepared['finite_native_author'])
    assert bound['actual_parent_binding']['binding_kind'] == 'existing_registered_program'
    assert bound['actual_parent_binding']['theory_id'] == 'old'
    assert agent.theory_index == old_index and agent.costs == before
    contract = agent.knowledge['finite_native_author']
    assert contract['question']['theory_target']['parent_theory_id'] == 'old'
    assert contract['question']['use_sites'][0]['model'] == 'old.py'
    assert set(contract['predictor_paths']) == {'old.py', 'constant.py'}
    assert contract['question']['theory_target']['role_signature'] == prepared[
        'question']['theory_target']['role_signature']


def test_new_source_refreshes_static_proposals_from_actual_shared_winner(tmp_path, monkeypatch):
    paid, case, values = paid_fixture(tmp_path)
    source = study.read(paid)
    prefix = study.save(tmp_path / 'prefix_result.json', {
        'paid_rows': source['expected_rows'], 'records_ref': source['paid_records_ref'],
        'costs': {'compute_cost': 0, 'observation_cost': source['expected_rows']}})
    study.save(tmp_path / 'public_spec.json', values['public'])
    request = {'case': values['case'], 'paid_prefix_ref': prefix,
        'parents': source['parents'], 'diagnostic_parent_id': 'constant',
        'prefix_rows': source['expected_rows'], 'evaluation_seed': 502, 'evaluation_rows': 20,
        'validation_records': source['validation_records'], 'research_compute_cap': 10000,
        'per_cell_compute_cap': 30000, 'per_cell_activity_seconds': 60,
        'per_cell_python_seconds': 30, 'source_activity_seconds': 120,
        'source_python_seconds': 30, 'source_comparison_compute_upper': 40,
        'mse_threshold': 1e-10, 'comparison_scope': 'both_existing_transferred_programs'}
    seen = []

    def pair(registration, output):
        question = registration['question']
        assert question['theory_target']['parent_theory_id'] == 'old'
        assert question['use_sites'][0]['model'] == 'old.py'
        static = study.read(registration['static_selection_ref'])
        assert static['actual_parent_binding']['theory_id'] == 'old'
        assert static['ordinary']['proposal']['arguments']['research_question'][
            'theory_target']['parent_theory_id'] == 'old'
        alternative = registration['cells']['10']['selection_plan']['first_segment']
        assert alternative['arguments']['research_question']['theory_target'][
            'parent_theory_id'] == 'old'
        assert registration['cells']['00']['scope']['pool_models'] == 2
        first_charge = static['previous_preparation_costs']['compute_cost']
        second_charge = static['new_compute_cost']
        assert registration['preparation_costs']['compute_cost'] == first_charge + second_charge
        assert registration['components']['projection_proposals']['compute'] == first_charge + second_charge
        assert registration['method_condition_compute_cap'] == 1000 - second_charge
        assert registration['compute_cap'] == 60000 + 1000 + first_charge
        assert registration['quote']['status'] == 'quoted'
        seen.append(True)
        return {'status': 'actual_source_binding_ready'}

    monkeypatch.setattr(entry, 'run_pair', pair)
    result = entry.run(request, tmp_path / 'source')
    assert seen == [True] and result['status'] == 'actual_source_binding_ready'
    assert result['actual_parent_binding']['theory_id'] == 'old'
    assert result['shared_comparison_costs']['compute_cost'] == 40
    assert study.read(result['initial_scene_ref'])['evaluated_model']['source_model'] == 'old.py'


def test_new_source_executes_registered_public_prefix_without_resampling(tmp_path, monkeypatch):
    from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab

    paid, _, values = paid_fixture(tmp_path)
    source = study.read(paid)
    actions = [{'kind': 'trajectory', 'channels': [],
        'sequence': [{'u': value} for value in sequence]}
        for sequence in ([-1.], [1.], [1., -1., 1., -1.])]
    actions_ref = study.save(tmp_path / 'registered_public_prefix.json', actions)
    request = {'case': {'kind': 'hysteresis', 'seed': 801},
        'prefix_actions_ref': actions_ref, 'prefix_rows': 6,
        'parents': source['parents'], 'diagnostic_parent_id': 'constant',
        'evaluation_seed': 503, 'evaluation_rows': 20, 'validation_records': [1],
        'research_compute_cap': 10000, 'per_cell_compute_cap': 30000,
        'per_cell_activity_seconds': 60, 'per_cell_python_seconds': 30,
        'source_activity_seconds': 120, 'source_python_seconds': 30,
        'source_comparison_compute_upper': 12, 'mse_threshold': 1e-10,
        'comparison_scope': 'both_existing_transferred_programs'}

    class RegisteredPrefixLab(AnonymousHistoryLab):
        def confirmation_actions(self, *args, **kwargs):
            raise AssertionError('the registered source prefix must not be replaced by a sample')

    def no_research(*args, **kwargs):
        raise AssertionError('the complete retained pool already explains this simple prefix')

    monkeypatch.setattr(entry, 'AnonymousHistoryLab', RegisteredPrefixLab)
    monkeypatch.setattr(entry, 'run_pair', no_research)
    result = entry.run(request, tmp_path / 'fixed_prefix')
    assert result['status'] == 'retained_models_explain_paid_prefix_no_research'
    prefix = study.read(result['prefix_costs_ref'])
    assert prefix['paid_rows'] == 6 and prefix['costs']['observation_cost'] == 6
    records = study.read(prefix['records_ref'])
    assert [record['action']['sequence'] for record in records] == [item['sequence'] for item in actions]
