"""Real shared-question generation versus a frozen initial question collection."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def replay_module():
    path = ROOT / 'scripts/replay_question_generation_ablation.py'
    specification = importlib.util.spec_from_file_location('question_generation_ablation', path)
    module = importlib.util.module_from_spec(specification)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / 'scripts'))
        specification.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def replay(replay_module, tmp_path_factory):
    if not replay_module.POINTER.exists():
        pytest.skip('The original historical source pointer is unavailable')
    pointer = json.loads(replay_module.POINTER.read_text())
    if not Path(pointer['archive']).exists():
        pytest.skip('The original historical archive is not mounted')
    output = tmp_path_factory.mktemp('question_generation') / 'receipt.json'
    return replay_module.replay_generation_ablation(output=output)


def test_real_original_source_gate_not_successor_answers_supplies_scoped_support(replay_module, replay):
    for source in replay['source_gates']:
        assert source['threshold'] == .002
        assert source['validation_loss'] < source['threshold']
        assert source['stop_reason'] in {'adequate_and_compact', 'knowledge_reused'}
        assert source['adaptive_development'] and source['independent_confirmation'] is False
        assert all('episodes[14]' not in ref['field'] for ref in source['source_evidence_refs'])
    payload, source = replay_module.load_source(replay_module.POINTER)
    failed = deepcopy(payload)
    failed['episodes'][0]['validation_loss'] = 1.
    with pytest.raises(ValueError, match='original scoped support gate'):
        replay_module.source_gate(failed, source, 0)


def test_on_generates_shared_question_then_really_consolidates_and_reuses_its_output(replay):
    on, frozen = replay['arms']['on'], replay['arms']['frozen']
    assert on['initial_question_ids'] == frozen['initial_question_ids']
    assert len(on['new_questions']) == 1 and frozen['new_questions'] == []
    created = on['new_questions'][0]
    assert created['origin']['kind'] == 'multi_episode_commonality'
    assert {ref['id'] for ref in created['origin']['source_refs']} == {'history:t01', 'history:t06'}
    assert on['source_phase']['selected_question_id'] == created['id']
    assert on['source_phase']['frozen_candidates'][0]['source_count'] == 2
    assert on['source_phase']['frozen_candidates'][0]['source_reconstruction'] == 'exact'
    assert len(on['automatic_recall']['generated_theory_ids']) == 2
    consumed = {identifier for group in on['selection']['groups'] for identifier in group['theory_ids']}
    assert set(on['automatic_recall']['generated_theory_ids']) <= consumed
    assert not replay['protocol']['new_question_or_formula_supplied_to_on']


def test_freezing_questions_retains_full_old_abilities_source_counterexample_and_parent(replay):
    assert replay['protocol']['same_low_level_tools_and_full_old_library']
    for arm in replay['arms'].values():
        assert arm['old_template_binding_count'] == 4 and arm['raw_old_proposals'] == 8
        assert len(arm['candidate_program_ids']) == 7
        assert arm['source_instances_preserved'] and arm['counterexample_retained']
        assert arm['new_observations'] == arm['new_model_calls'] == 0
    assert replay['arms']['frozen']['source_phase']['frozen_candidates'] == []
    assert replay['arms']['frozen']['source_phase']['selected_question_id'] == 'retained_t15_response'


def test_actual_parent_prediction_plan_and_full_costs_preserve_the_measured_null(replay):
    comparison = replay['comparison']
    assert all(comparison[key] for key in ('same_candidate_programs', 'same_parent_program',
        'same_suffix_predictions', 'same_experiment_choice'))
    for arm in replay['arms'].values():
        assert arm['selected_experiment_id'] == 'representative_0'
        assert arm['parent']['suffix_mse'] == pytest.approx(3.8466553728192677e-19)
        assert arm['parent_pending_query']['actions']
        assert arm['selected_action_executed'] is False and arm['action_benefit'].startswith('UNKNOWN')
        assert arm['costs']['total'] == sum(value for key, value in arm['costs'].items() if key != 'total')
        assert arm['costs']['total'] <= replay['protocol']['same_total_compute_budget']
    on, frozen = replay['arms']['on'], replay['arms']['frozen']
    assert comparison['on_minus_frozen_compute'] == on['costs']['total'] - frozen['costs']['total'] > 0
    assert frozen['costs']['target_route'] == 0  # Its retained question already ran in the first phase.


def test_successor_label_changes_do_not_change_generated_questions_candidates_or_selected_plan(
        replay_module, replay, tmp_path, monkeypatch):
    payload, source = replay_module.load_source(replay_module.POINTER)
    for index in replay_module.SUFFIX_INDICES:
        payload['episodes'][14]['evaluation']['prediction_examples'][index]['target'] += 123.
    monkeypatch.setattr(replay_module, 'load_source', lambda pointer: (payload, source))
    changed = replay_module.replay_generation_ablation(output=tmp_path / 'changed_suffix.json')
    for name in ('on', 'frozen'):
        before, after = replay['arms'][name], changed['arms'][name]
        assert after['new_questions'] == before['new_questions']
        assert after['candidate_program_ids'] == before['candidate_program_ids']
        assert after['selected_experiment_id'] == before['selected_experiment_id']
        assert after['parent']['suffix_predictions'] == before['parent']['suffix_predictions']
        assert after['parent']['suffix_mse'] != before['parent']['suffix_mse']


def test_window_history_with_no_declared_use_stays_an_explicit_unresolved_case(replay):
    boundary = replay['window_boundary']
    if boundary['status'] == 'UNKNOWN_source_unavailable':
        pytest.skip('The additional retained window source is unavailable')
    assert boundary['retained_question_ids'] == ['task']
    assert boundary['generated_question_count'] == 0
    assert boundary['status'].startswith('unresolved')
