"""Declared condition selection and frozen successors keep different data roles."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.checkpoint_storage import read_json_checkpoint
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from scripts import run_state_method_pairing as study
from scripts.run_method_action_pairing import run_method_action_pairing
from test_state_method_cohort import frozen_pair, source_fixture


@pytest.mark.parametrize('role', ['development', 'condition_validation', 'independent_successor'])
def test_registered_role_reaches_pair_and_successor_never_invokes_feedback(tmp_path, monkeypatch, role):
    paid, case, values = source_fixture(tmp_path / 'source', 0)
    bundle = frozen_pair(tmp_path, paid, case)
    registration = study.prepare(paid, tmp_path / 'prepared', case_ref=case,
        evaluation_seed=901, research_compute_cap=10000, compute_cap=30000,
        evaluation_rows=20, approval_ref=paid, method_pair_ref=bundle,
        data_role=role, method_condition_compute_cap=20000)
    assert registration['data_role'] == registration['quote']['data_role'] == role
    assert registration['components']['feedback_preservation']['compute'] == 20000
    assert registration['method_condition_compute_cap'] == 20000
    output = tmp_path / 'run'

    class Evaluator:
        def __init__(self, **_):
            frozen = read_json_checkpoint(output / 'frozen_outputs.json')
            assert set(frozen['models']) == {'00', '10'}
            assert not frozen['evaluation_labels_opened']

        def evaluation_rows(self, *, length, seed):
            assert (length, seed) == (20, 901)
            return [Observation(row['features'], row['target'], tuple(row['history']))
                    for record in values['records'] for row in record['rows']]

    monkeypatch.setattr(study, 'AnonymousHistoryLab', Evaluator)
    factories = study.make_factories(registration, output)
    original = factories['ordinary_use']
    calls = []

    def feedback(*args):
        assert role != 'independent_successor', 'frozen successor labels cannot select a method'
        calls.append(True)
        return original(*args)

    factories['ordinary_use'] = feedback
    result = run_method_action_pairing(registration, output, **factories)
    pair = study.read(result['pair_refs'][0])
    assert result['data_role'] == pair['data_role'] == role
    assert all(study.read(ref)['execution_complete'] for ref in result['outcome_refs'].values())
    if role == 'independent_successor':
        assert calls == []
        assert result['status'] == 'completed_independent_successor_pairing'
        assert result['ordinary_use']['consumed_pair_refs'] == []
        assert result['ordinary_use']['new_compute_cost'] == 0
        assert pair['evaluation_lifecycle']['role'] == 'independent_successor_after_freeze'
        assert pair['evaluation_lifecycle']['status'] == 'independent_successor_evaluation_after_freeze'
        assert result['ordinary_use']['successors_started']
        assert result['ordinary_use']['frozen_method_execution_completed']
        assert not result['ordinary_use']['method_feedback_executed']
        assert not (output / 'method_selection.json').exists()
    else:
        assert calls == [True]
        assert result['ordinary_use']['consumed_pair_refs'] == result['pair_refs']
        assert pair['evaluation_lifecycle']['role'] == 'development_after_freeze'
        if role == 'condition_validation':
            selected = study.read(result['ordinary_use']['selection_ref'])
            assert selected['local_support_refs'] == []


def test_frozen_guard_installation_preserves_selection_evidence_and_live_coverage(tmp_path):
    paid, case, _ = source_fixture(tmp_path / 'source', 0)
    bundle_ref = frozen_pair(tmp_path, paid, case)
    bundle = study.read(bundle_ref)
    challenger = bundle['cells']['10']
    body = deepcopy(challenger['selection_plan'])
    coordinates = [{'path': ['data', 'max_history'], 'kind': 'number'}]
    selected = {'general_guard_supported': True, 'coordinate_paths': coordinates,
        'measured_coverage': {'public_0': [0., 100.]}}
    selected_ref = study.save(tmp_path / 'frozen_condition_selection.json', selected)
    challenger.update(id='cohort_guard', selection_ref=selected_ref,
        selection_plan={'ordinary_method_path': 'ordinary.py',
            'condition': Program('feature', name='public_0').to_dict(),
            'condition_feature_paths': coordinates,
            'when_true': {key: value for key, value in body.items() if key != 'ordinary_method_path'},
            'when_false': {}})
    guarded_ref = study.save(tmp_path / 'guarded_pair.json', bundle)
    registration = study.prepare(paid, tmp_path / 'prepared', case_ref=case,
        evaluation_seed=902, research_compute_cap=10000, compute_cap=30000,
        evaluation_rows=20, approval_ref=paid, method_pair_ref=guarded_ref,
        data_role='independent_successor', method_condition_compute_cap=0)
    cell = registration['cells']['10']
    assert cell['selection_ref'] == selected_ref
    factory = study.make_factories(registration, tmp_path / 'run')['restore_cell']
    scene = factory('10', cell, tmp_path / 'run' / '10')
    agent = scene['agent']
    installed = study.read(scene['method_artifact_ref'])
    assert installed['selection_ref'] == selected_ref
    assert installed['scope']['general_guard_supported']
    agent._method_use_profile = study.read(scene['profile_ref'])
    memory = agent._program_memory(scene['path'], {})
    assert memory['method_condition_support'] == {'can_apply': True, 'outside_coverage': []}
    assert memory['method_condition_features']['public_0'] == 4.
    agent._method_use_profile['data']['max_history'] = 101
    memory = agent._program_memory(scene['path'], {})
    assert memory['method_condition_support'] == {'can_apply': False, 'outside_coverage': ['public_0']}
