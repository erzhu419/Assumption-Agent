"""Purchased labels adjudicate saved alternatives without new forecasting."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.conditional_experiments import assess_frozen_structure_predictions
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation


def paid_scene(delta=1.):
    sequence = [{'u': -.4}, {'u': -.4}, {'u': .5}, {'u': .5},
                {'u': -.4}, {'u': -.4}, {'u': -.4}, {'u': -.4}]
    reference = [{'u': -.4} for _ in sequence]
    records, rows = [], []
    for controls, changed in [(reference, False), (sequence, True)]:
        first = len(rows)
        measured = [Observation(point, .2 + (delta if changed and i in (3, 4) else 0.),
            tuple(deepcopy(controls[:i]))) for i, point in enumerate(controls)]
        rows.extend(measured)
        records.append({'action': {'kind': 'trajectory', 'sequence': deepcopy(controls)},
            'row_indices': list(range(first, first + len(measured))),
            'rows': [{'features': dict(row.features), 'target': row.target,
                'history': list(deepcopy(row.history))} for row in measured]})
    target = {'hypothesis_refs': [{'file': 'paid.json', 'json_path': ['candidates', i]}
                                 for i in range(3)],
        'forecast_witnesses': {'history_width': {'hypothesis_indices': [0, 1],
            'readout_step': 3, 'delta_predictions': [1., 2.]}}}
    agent = SimpleNamespace(rows=rows, observations=records,
        config=SimpleNamespace(compute_budget=100), costs={'compute_cost': 11})
    return agent, target, [{'reference_record_index': 0, 'record_index': 1}], [
        deepcopy(record['action']) for record in records]


@pytest.mark.parametrize('observed,status,refuted,compatible,next_kind', [
    (1., 'predictions_distinguished', [1], [0], 'compare_surviving_fixed_candidates'),
    (2., 'predictions_distinguished', [0], [1], 'compare_surviving_fixed_candidates'),
    (4., 'both_predictions_refuted', [0, 1], [], 'jointly_refit_frozen_structures'),
    (1. + 5e-11, 'predictions_distinguished', [1], [0], 'compare_surviving_fixed_candidates'),
    (1. + 2e-10, 'both_predictions_refuted', [0, 1], [], 'jointly_refit_frozen_structures'),
])
def test_actual_delta_decides_frozen_instances_with_original_relation_tolerance(
        monkeypatch, observed, status, refuted, compatible, next_kind):
    agent, target, pairs, actions = paid_scene(observed)
    before = deepcopy((target, agent.observations))
    monkeypatch.setattr(Program, 'predict', lambda *a: pytest.fail('paid forecast repeated'))
    result = assess_frozen_structure_predictions(agent, target, pairs, planned_actions=actions)
    assert result['status'] == status and result['complete']
    assert result['refuted_candidate_refs'] == [target['hypothesis_refs'][i] for i in refuted]
    assert result['compatible_candidate_refs'] == [target['hypothesis_refs'][i] for i in compatible]
    assert result['structure_families_refuted'] is False
    assert result['next_research']['kind'] == next_kind
    assert result['compute_cost'] == 4 and agent.costs['compute_cost'] == 15
    witness, = result['witnesses']
    assert witness['observed_delta'] == pytest.approx(observed)
    assert witness['evidence'] == [3, 11] and witness['record_indices'] == [0, 1]
    assert all(check['atol'] == 1e-10 for check in witness['checks'])
    assert (target, agent.observations) == before


def test_complete_quote_is_required_before_any_relation_check(monkeypatch):
    agent, target, pairs, actions = paid_scene()
    monkeypatch.setattr('hegel_machine.cognition.inquiry_revision._check_local_relations',
        lambda *a: pytest.fail('partial/unfunded check executed'))
    result = assess_frozen_structure_predictions(agent, target, pairs,
        planned_actions=actions, max_compute=3)
    assert result['reason'] == 'complete_frozen_readout_check_unfunded'
    assert result['required_compute'] == 4 and result['compute_cost'] == 0
    assert not result['complete'] and not result['witnesses']
    assert agent.costs['compute_cost'] == 11


@pytest.mark.parametrize('change', ['sequence', 'reset_history', 'paid_row_binding', 'missing_record'])
def test_wrong_paid_context_is_not_used_to_refute_a_saved_prediction(change):
    agent, target, pairs, actions = paid_scene()
    if change == 'sequence':
        agent.observations[1]['action']['sequence'][3]['u'] = .4
    elif change == 'reset_history':
        agent.observations[1]['rows'][0]['history'] = [{'u': .5}]
    elif change == 'paid_row_binding':
        agent.observations[1]['row_indices'][3] = 3
    else:
        pairs[0]['record_index'] = 2
    result = assess_frozen_structure_predictions(agent, target, pairs, planned_actions=actions)
    assert not result['complete'] and result['status'] == 'inconclusive'
    assert not result['refuted_candidate_refs'] and result['compute_cost'] == 0
    assert agent.costs['compute_cost'] == 11


def test_two_structural_questions_keep_separate_readout_refutations():
    agent, target, pairs, actions = paid_scene()
    target['forecast_witnesses']['condition_position'] = {
        'hypothesis_indices': [0, 2], 'readout_step': 4, 'delta_predictions': [1., 3.]}
    result = assess_frozen_structure_predictions(agent, target, pairs, planned_actions=actions)
    assert result['complete'] and result['status'] == 'predictions_distinguished'
    assert result['compute_cost'] == 8 and agent.costs['compute_cost'] == 19
    assert result['evidence'] == [3, 11, 4, 12]
    assert result['refuted_candidate_refs'] == [target['hypothesis_refs'][1], target['hypothesis_refs'][2]]
    assert result['compatible_candidate_refs'] == [target['hypothesis_refs'][0]]
    assert [item['readout_step'] for item in result['witnesses']] == [3, 4]
    assert result['structure_families_refuted'] is False


def test_one_frozen_match_does_not_hide_the_same_candidate_other_refutation():
    agent, target, pairs, actions = paid_scene()
    target['forecast_witnesses']['condition_position'] = {
        'hypothesis_indices': [0, 2], 'readout_step': 4, 'delta_predictions': [.5, 1.]}
    result = assess_frozen_structure_predictions(agent, target, pairs, planned_actions=actions)
    assert result['complete']
    assert result['refuted_candidate_refs'] == [target['hypothesis_refs'][0], target['hypothesis_refs'][1]]
    assert result['compatible_candidate_refs'] == [target['hypothesis_refs'][2]]
    assert result['witnesses'][0]['checks'][0]['status'] == 'satisfied'
    assert result['witnesses'][1]['checks'][0]['status'] == 'violated'
    assert result['structure_families_refuted'] is False
