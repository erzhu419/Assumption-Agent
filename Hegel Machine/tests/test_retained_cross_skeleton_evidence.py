"""Restore actual t07 labels while preserving chronology and missing rows."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def retained():
    specification = importlib.util.spec_from_file_location('retained_cross_skeleton_evidence',
        ROOT / 'scripts/retained_cross_skeleton_evidence.py')
    module = importlib.util.module_from_spec(specification)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / 'scripts'))
        specification.loader.exec_module(module)
    if not module.POINTER.is_file() or not Path(json.loads(module.POINTER.read_text())['archive']).is_file():
        pytest.skip('The retained historical source is not mounted')
    payload, source = module.load_source(module.POINTER)
    return module, payload, source, module.recover_t07_evidence(payload, source)


def test_fixed_training_and_postfreeze_split_matches_real_public_actions(retained):
    _, payload, _, result = retained
    assert len(result['training_rows']) == 8
    assert len(result['comparison_rows']) == 2
    assert [(row['action_index'], row['step']) for row in result['comparison_refs']] == [(65, 0), (65, 1)]
    for group in ('training', 'comparison'):
        for row, reference in zip(result[f'{group}_rows'], result[f'{group}_refs']):
            sequence = payload['episodes'][6]['actions'][reference['action_index']]['details']['sequence']
            assert row.features == sequence[reference['step']]
            assert [{name: point[name] for name in row.features} for point in row.history] == sequence[:reference['step']]
    assert result['comparison_refs'][0]['label_scored_in_prior_continuation_diagnostics']
    assert not result['comparison_refs'][1]['label_scored_in_prior_continuation_diagnostics']
    assert not any(row['new_blind_evidence'] for row in result['comparison_refs'])


def test_duplicate_counterexamples_preserve_both_refs_without_two_episodes_or_extra_cost(retained):
    _, payload, _, result = retained
    observation = payload['episodes'][6]['memory_events'][0]['observation']
    assert result['comparison_rows'][0].target == observation['history'][0]['__target__']
    assert result['comparison_rows'][1].target == observation['target']
    assert result['comparison_rows'][1].history[0]['__target__'] == result['comparison_rows'][0].target
    for reference in result['comparison_refs']:
        assert len(reference['source_refs']) == 2
        assert 'memory_events[0]' in reference['source_refs'][0]['field']
        assert 'memory_events[1]' in reference['source_refs'][1]['field']
        assert reference['original_action_observation_cost'] == 8
    assert result['duplicates']['repeated_label_references_merged'] == 2
    assert result['duplicates']['unique_recovered_labels'] == 10


def test_missing_paid_rows_remain_missing_and_original_action_costs_remain_whole(retained):
    _, _, _, result = retained
    assert result['coverage']['65']['recorded_rows'] == 8
    assert result['coverage']['65']['recovered_unique_rows'] == 2
    assert result['coverage']['65']['missing_steps'] == [2, 3, 4, 5, 6, 7]
    assert result['coverage']['67']['recorded_rows'] == 8
    assert result['coverage']['67']['recovered_unique_rows'] == 0
    assert result['coverage']['67']['missing_steps'] == list(range(8))
    assert result['historical_costs']['postfreeze_observation_cost'] == 16
    assert result['historical_costs']['new_observation_cost'] == 0
    assert not result['scope']['complete_original_postfreeze_rows_recovered']


def test_later_rows_use_only_the_saved_ordered_targets_and_actual_prefix_history(retained):
    _, payload, _, result = retained
    examples = payload['episodes'][6]['evaluation']['prediction_examples']
    assert len(result['later_rows']) == 4
    for step, row in enumerate(result['later_rows']):
        assert row.features == examples[step]['features']
        assert row.target == examples[step]['target']
        assert list(row.history) == [{**example['features'], '__target__': example['target']}
                                     for example in examples[:step]]
        assert result['later_refs'][step]['action_index'] is None
        assert result['later_refs'][step]['original_observation_cost'] is None
    assert result['scope']['original_final_evaluation_rows'] == 192
    assert not result['scope']['independent_new_confirmation']


def test_models_stored_predictions_and_aggregate_losses_do_not_supply_recovered_labels(retained):
    module, payload, source, original = retained
    changed = deepcopy(payload)
    episode = changed['episodes'][6]
    episode['program'] = {'not_a_program': True}
    episode['actions'][69]['details']['after_program'] = {'not_a_program': True}
    episode['evaluation']['loss'] = 1e10
    for example in episode['evaluation']['prediction_examples']:
        example['prediction'] = -1e10
    recovered = module.recover_t07_evidence(changed, source)
    for group in ('training', 'comparison', 'later'):
        assert recovered[f'{group}_rows'] == original[f'{group}_rows']
        assert recovered[f'{group}_refs'] == original[f'{group}_refs']


def test_conflicting_duplicate_labels_cannot_be_silently_selected(retained):
    module, payload, source, _ = retained
    changed = deepcopy(payload)
    changed['episodes'][6]['memory_events'][1]['observation']['history'][0]['__target__'] += .1
    with pytest.raises(ValueError, match='Repeated retained observations disagree'):
        module.recover_t07_evidence(changed, source)
