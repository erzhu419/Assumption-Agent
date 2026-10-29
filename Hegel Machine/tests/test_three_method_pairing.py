"""Three frozen methods share one scene; completed paid arms remain references."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from scripts import run_state_method_pairing as state
from test_state_method_pairing import paid_fixture
from test_theory_consolidation import agent_scene
from test_v8_method_action_pairing import registration, runner


def three_methods():
    plan = registration()
    plan.update(comparison_kind='three_method_same_data', data_role='independent_successor',
        execution_authorized=True, approval_ref={'file': 'test_approval.json', 'json_path': []})
    plan['cells']['20'] = deepcopy(plan['cells']['10'])
    plan['cells']['20']['method_ref'] = {'id': 'conditional-method', 'version': 1}
    return plan


def factories(restored, evaluated, destination, expected_cells=('00', '10', '20')):
    def restore(name, contract, output):
        assert not (destination / 'shared_evaluation_preparation.json').exists()
        restored.append(name)
        agent, lab = agent_scene(output)
        lab.public_spec = lambda: {'action_kinds': ['trajectory'], 'controls': ['u'],
            'max_probe_length': 8, 'feature_bounds': {'u': (-1., 1.)}}
        agent.rows = [Observation({'u': 0.}, 1.)]
        agent.config = replace(agent.config, compute_budget=10000, observation_budget=0,
            execution_seconds=30)
        agent._save('old.py', export_temporal_program(Program('const', value=1.), feature_names=('u',)))
        agent._open_question({'id': 'q', 'question': 'Compare the retained synthetic model',
            'parent_id': 'task', 'answer_type': 'number', 'context_ref': {'domain': 'synthetic'}})
        profile = state.save(output / 'profile.json', {})
        return {'agent': agent, 'lab': lab, 'path': 'method.py', 'models': ['old.py'],
            'scope': contract['scope'], 'question_ref': {'id': 'q'}, 'profile_ref': profile,
            'origin_ref': profile, **{key: deepcopy(contract[key]) for key in
                ('start_state_ref', 'data_ref', 'split_ref', 'quality_contract_ref',
                 'resource_contract_ref', 'evaluation_ref')}}

    def prepare_evaluation():
        manifest = state.read({'file': str(destination / 'frozen_outputs.json'), 'json_path': []})
        assert set(manifest['models']) == set(expected_cells)
        assert not manifest['evaluation_labels_opened']
        return {'already_paid_material_ref': 'test_material.json'}

    def evaluate(name, frozen, evaluation_ref, account):
        assert (destination / 'shared_evaluation_preparation.json').is_file()
        evaluated.append(name)
        account.costs['compute_cost'] += 2
        return {'loss': 0., 'quality': {'threshold': 1e-10}, 'coverage': 2, 'completed': True}

    return {'restore_cell': restore, 'prepare_evaluation': prepare_evaluation,
        'evaluate_frozen': evaluate,
        'ordinary_use': lambda *_: pytest.fail('frozen successor cannot feed its labels back')}


def test_three_same_data_arms_freeze_before_shared_labels_and_form_all_pairs(tmp_path, monkeypatch):
    plan = three_methods()
    plan['quote'] = runner.quote_method_action_pairing(plan)
    assert plan['quote']['cells'] == ['00', '10', '20']
    monkeypatch.setattr(runner, 'execute_registered_method_unit',
        lambda *args, **kwargs: {'status': 'completed', 'model': 'old.py'})
    restored, evaluated = [], []
    result = runner.run_method_action_pairing(plan, tmp_path / 'study',
        **factories(restored, evaluated, tmp_path / 'study'))
    assert restored == evaluated == ['00', '10', '20']
    assert [Path(ref['file']).name for ref in result['pair_refs']] == [
        'pair_00_10.json', 'pair_00_20.json', 'pair_10_20.json']
    assert state.read(result['pair_refs'][2])['identified_effect'] == 'condition_vs_fixed_method'
    assert result['evaluation_lifecycle']['all_outputs_frozen_before_labels']
    assert not result['ordinary_use']['method_feedback_executed']
    changed = deepcopy(plan)
    changed['cells']['20']['data_ref']['file'] = 'other_scene.json'
    assert runner.quote_method_action_pairing(changed)['status'] == 'quote_incomplete'
    changed = deepcopy(plan)
    changed['D1']['legal'] = True
    assert runner.quote_method_action_pairing(changed)['status'] == 'quote_incomplete'


def test_retrospective_completion_only_executes_and_scores_missing_arm(tmp_path, monkeypatch):
    plan = three_methods()
    original = deepcopy(plan)
    original.pop('comparison_kind')
    original['cells']['10'] = original['cells'].pop('20')
    original['quote'] = runner.quote_method_action_pairing(original)
    old_registration = state.save(tmp_path / 'original_registration.json', original)
    monkeypatch.setattr(runner, 'execute_registered_method_unit',
        lambda *args, **kwargs: {'status': 'completed', 'model': 'old.py'})
    old = runner.run_method_action_pairing(original, tmp_path / 'old',
        **factories([], [], tmp_path / 'old', expected_cells=('00', '10')))
    original_files = {path: path.read_bytes() for path in (tmp_path / 'old').rglob('*') if path.is_file()}
    plan.update(retrospective_completion=True, retained_registration_ref=old_registration,
        retained_outcome_refs={'00': old['outcome_refs']['00'], '20': old['outcome_refs']['10']})
    plan['quote'] = runner.quote_method_action_pairing(plan)
    assert plan['quote']['status'] == 'quoted'
    restored, evaluated = [], []
    result = runner.run_method_action_pairing(plan, tmp_path / 'completed',
        **factories(restored, evaluated, tmp_path / 'completed'))
    assert restored == evaluated == ['10']
    assert result['status'] == 'retrospective_completed_method_contrast'
    lifecycle = result['evaluation_lifecycle']
    assert lifecycle['prior_labels_opened'] and lifecycle['new_outputs_frozen_before_label_reuse']
    assert not lifecycle['all_outputs_frozen_before_labels']
    for name in ('00', '20'):
        assert all(value == 0 for value in result['new_stage_costs'][name].values())
        assert result['comparison_costs'][name] == state.read(plan['retained_outcome_refs'][name])['costs']
    assert result['new_stage_costs']['10'] == result['comparison_costs']['10']
    assert all(not state.read(ref)['method_condition_label']['eligible_for_condition_learning']
               for ref in result['pair_refs'])
    assert all(path.read_bytes() == value for path, value in original_files.items())
    plan['cells']['20']['compute_cap'] += 1
    assert any('changed its original research contract' in message
        for message in runner.quote_method_action_pairing(plan)['missing'])


def test_revised_conditional_method_reuses_completed_original_and_fixed_arms(tmp_path, monkeypatch):
    original = three_methods()
    original['quote'] = runner.quote_method_action_pairing(original)
    old_registration = state.save(tmp_path / 'original_registration.json', original)
    monkeypatch.setattr(runner, 'execute_registered_method_unit',
        lambda *args, **kwargs: {'status': 'completed', 'model': 'old.py'})
    old = runner.run_method_action_pairing(original, tmp_path / 'old',
        **factories([], [], tmp_path / 'old'))
    original_files = {path: path.read_bytes() for path in (tmp_path / 'old').rglob('*') if path.is_file()}
    revised = deepcopy(original)
    revised['cells']['20']['method_ref'] = {'id': 'conditional-method-with-witness', 'version': 1}
    revised.update(retrospective_completion=True, retained_registration_ref=old_registration,
        retained_outcome_refs={name: old['outcome_refs'][name] for name in ('00', '10')})
    revised['quote'] = runner.quote_method_action_pairing(revised)
    assert revised['quote']['status'] == 'quoted'
    restored, evaluated = [], []
    result = runner.run_method_action_pairing(revised, tmp_path / 'revised',
        **factories(restored, evaluated, tmp_path / 'revised'))
    assert restored == evaluated == ['20']
    assert result['status'] == 'retrospective_completed_method_contrast'
    for name in ('00', '10'):
        assert all(value == 0 for value in result['new_stage_costs'][name].values())
        assert result['comparison_costs'][name] == state.read(old['outcome_refs'][name])['costs']
    assert result['new_stage_costs']['20'] == result['comparison_costs']['20']
    assert all(not state.read(ref)['method_condition_label']['eligible_for_condition_learning']
               for ref in result['pair_refs'])
    assert all(path.read_bytes() == content for path, content in original_files.items())


def test_preparation_uses_all_three_frozen_methods_and_one_evaluation_pool(tmp_path, monkeypatch):
    paid, case, values = paid_fixture(tmp_path)
    initial = state.prepare(paid, tmp_path / 'initial', case_ref=case, evaluation_seed=31,
        compute_cap=30000, research_compute_cap=10000, evaluation_rows=20)
    bundle = {'ordinary_source_ref': {'file': str(tmp_path / 'initial/preparation/ordinary.py')},
        'cells': {name: {'id': initial['cells'][name]['method_ref']['id'],
            'selection_plan': initial['cells'][name]['selection_plan'], 'source_refs': [paid]}
            for name in ('00', '10')}}
    bundle['cells']['20'] = {**deepcopy(bundle['cells']['10']), 'id': 'guarded_P1'}
    bundle_ref = state.save(tmp_path / 'methods.json', bundle)
    plan = state.prepare(paid, tmp_path / 'three', case_ref=case, evaluation_seed=32,
        compute_cap=30000, research_compute_cap=10000, evaluation_rows=20,
        method_pair_ref=bundle_ref, data_role='independent_successor', approval_ref=paid)
    assert plan['quote']['cells'] == ['00', '10', '20']
    assert plan['comparison_kind'] == 'three_method_same_data'
    assert plan['components']['Qeval_predictions'] == {'compute': 60, 'observations': 20}
    assert plan['compute_cap'] == 90000 + plan['method_condition_compute_cap'] + plan['preparation_costs']['compute_cost']
    assert len({json.dumps(cell['data_ref'], sort_keys=True) for cell in plan['cells'].values()}) == 1
    directory = tmp_path / 'real_run'

    class SharedEvaluator:
        def __init__(self, **_):
            manifest = state.read({'file': str(directory / 'frozen_outputs.json'), 'json_path': []})
            assert set(manifest['models']) == {'00', '10', '20'}

        def evaluation_rows(self, **_):
            return [Observation(row['features'], row['target'], tuple(row['history']))
                    for record in values['records'] for row in record['rows']]

    monkeypatch.setattr(state, 'AnonymousHistoryLab', SharedEvaluator)
    result = state.run(plan, directory)
    assert len(result['pair_refs']) == 3
    assert all(state.read(reference)['execution_complete'] for reference in result['outcome_refs'].values())
    assert state.read(result['outcome_refs']['20'])['method_ref']['id'] == 'guarded_P1'


def test_paid_evaluation_material_opens_only_in_evaluation_callback(tmp_path, monkeypatch):
    material_ref = {'file': str(tmp_path / 'paid_material.json'), 'json_path': []}
    plan = {'retrospective_completion': True, 'paid_evaluation_material_ref': material_ref,
        'evaluation': {'rows': 2}}
    callbacks = state.make_factories(plan, tmp_path / 'run')
    # The paid label file does not exist until after the factories are built.
    state.save(material_ref['file'], {'material': [{'sequence': [{'u': 1.}, {'u': 2.}], 'targets': [3., 4.]}]})
    monkeypatch.setattr(state, 'AnonymousHistoryLab', lambda **_: pytest.fail('paid labels cannot be bought again'))
    receipt = state.read(callbacks['prepare_evaluation']())
    assert receipt['new_response_rows'] == 0 and receipt['paid_original_ref'] == material_ref
    assert 'material' not in receipt
