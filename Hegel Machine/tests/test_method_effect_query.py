"""Query the real retained paired replay; never launch method practices."""

import json
from pathlib import Path

import pytest

from hegel_machine.cognition.method_effect_query import EFFECT_KIND, evaluate_method_effect
from test_evidence_procedure_execution import prepared


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / 'docs/evidence/frozen_theory_ablation_20261020.json'


def query(path=RECEIPT, **fields):
    return {'op': 'MethodEffect', 'effect_kind': EFFECT_KIND,
        'comparison_ref': {'file': str(path), 'json_path': []},
        'left_arm': 'remove', 'right_arm': 'keep', **fields}


def test_real_frozen_prior_receipt_reports_null_suffix_effect_and_full_extra_cost_without_running(tmp_path):
    before = RECEIPT.read_bytes()
    result = evaluate_method_effect(query(), tmp_path)
    assert result['status'] == 'answered'
    value = result['value']
    assert value['effect_kind'] == EFFECT_KIND
    assert value['left']['suffix_mse'] == value['right']['suffix_mse'] == pytest.approx(3.8466553728192677e-19)
    assert value['left']['compute_cost'] == 75 and value['right']['compute_cost'] == 103
    assert value['right_minus_left'] == {'suffix_mse': 0., 'compute_cost': 28}
    assert value['action_benefit']['status'] == 'scoped_unknown'
    assert result['scope']['fit_rows'] == [0, 1] and result['scope']['suffix_rows'] == [2, 3]
    assert result['scope']['literal_method_source_comparison'] is False
    assert result['independent_confirmation'] is False and result['new_method_executions'] == 0
    assert result['compute_cost'] == 0 and RECEIPT.read_bytes() == before
    assert list(tmp_path.iterdir()) == []
    assert 'program_catalog' not in result and 'suffix_predictions' not in value['left']


@pytest.mark.parametrize('requested_contract', [
    {'effect_kind': 'literal_method_source_effect'}, {'outcome': 'action_benefit'},
])
def test_unexecuted_action_or_unrecorded_literal_method_effect_is_unknown(tmp_path, requested_contract):
    result = evaluate_method_effect(query(**requested_contract), tmp_path)
    assert result['status'] == 'scoped_unknown' and 'value' not in result
    assert result['new_method_executions'] == 0


@pytest.mark.parametrize('missing', ['start', 'material', 'execution', 'outcome', 'cost', 'split', 'baseline'])
def test_missing_pairing_or_executed_result_stays_unknown_instead_of_zero_effect(tmp_path, missing):
    receipt = json.loads(RECEIPT.read_text())
    if missing == 'start':
        receipt['protocol'].pop('retained_incumbent_ref')
    elif missing == 'material':
        receipt['arms']['keep']['evaluation']['row_refs'] = []
    elif missing == 'execution':
        receipt.pop('executed_in_harness')
    elif missing == 'outcome':
        receipt['arms']['keep']['native_parent_choice'].pop('suffix_mse')
    elif missing == 'cost':
        receipt['arms']['keep']['accounted_compute'].pop('native_suffix_evaluation')
    elif missing == 'split':
        receipt['arms']['keep']['native_parent_choice']['suffix_row_indices'] = [0, 1]
    else:
        receipt['arms']['keep']['old_template_binding_count'] = 0
    path = tmp_path / 'incomplete.json'
    path.write_text(json.dumps(receipt))
    result = evaluate_method_effect(query(path), tmp_path)
    assert result['status'] == 'scoped_unknown' and 'value' not in result
    assert result['new_method_executions'] == 0


def test_two_unpaired_practice_mse_records_are_not_method_effect_evidence(tmp_path):
    path = tmp_path / 'practices.json'
    path.write_text(json.dumps({'practice_trials': [
        {'case': 'same', 'method_source': 'old', 'fresh_confirmation': {'passed': True, 'mse': .2},
         'confirmation_seed': 101, 'costs': {'compute_cost': 20}},
        {'case': 'same', 'method_source': 'new', 'fresh_confirmation': {'passed': True, 'mse': .1},
         'confirmation_seed': 102, 'costs': {'compute_cost': 30}}]}))
    result = evaluate_method_effect(query(path), tmp_path)
    assert result['status'] == 'scoped_unknown' and result['detail'] == 'no_supported_paired_execution_contract'


def test_actual_record_references_and_difference_direction_follow_requested_arms(tmp_path):
    result = evaluate_method_effect(query(left_arm='keep', right_arm='remove'), tmp_path)
    assert result['status'] == 'answered'
    assert result['value']['right_minus_left']['compute_cost'] == -28
    assert result['value']['left']['configuration_ref']['json_path'] == ['arms', 'keep']
    assert result['scope']['source_costs_ref']['json_path'] == ['historical_costs_shared_in_both_arms']


def test_common_step_reads_actual_paired_effect_then_withdraws_answer_when_receipt_disappears(tmp_path):
    agent, lab = prepared(tmp_path)
    agent.research_mission = {'purpose': 'method_learning', 'mission_id': 'retained_prior_effect'}
    receipt = tmp_path / 'retained_method_comparison.json'
    receipt.write_bytes(RECEIPT.read_bytes())
    question = {'id': 'frozen_prior_effect',
        'question': 'What suffix prediction and compute effect did this frozen prior configuration have?',
        'parent_id': 'task', 'depends_on': [], 'kind': 'method_effect', 'answer_type': 'relation',
        'answer_resolution_ref': {'kind': 'retained_paired_configuration_effect'},
        'query_semantics': {'mode': 'model_query', 'query_expression': query(receipt)},
        'use_sites': []}
    costs = dict(agent.costs)

    def step(research_question):
        return agent._execute('run', {'operator': 'research_step',
            'research_question': research_question}, lab)

    first = step(question)
    assert first['status'] == 'answer_ready_no_parent_use'
    assert first['answer']['value']['right_minus_left'] == {'suffix_mse': 0., 'compute_cost': 28}
    assert first['answer']['source_ref']['file'] == str(receipt)
    assert first['parent_revision_completed'] is False
    assert step({'id': question['id']})['answer'] == first['answer']
    assert agent.costs == costs

    receipt.unlink()
    missing = step({'id': question['id']})
    assert missing['status'] == 'scoped_unknown'
    assert missing['detail'] == 'retained_comparison_unavailable'
    current = next(item for item in agent.questions if item['id'] == question['id'])
    assert 'semantic_answer' not in current
    assert current['query_execution']['answer']['status'] == 'scoped_unknown'
    assert agent.costs == costs and agent.model is None and not agent.confirmations
