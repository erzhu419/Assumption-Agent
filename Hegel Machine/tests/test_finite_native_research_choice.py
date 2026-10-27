"""Existing action policy consumes a whole native unit, not another 50k slice."""
from copy import deepcopy
from dataclasses import replace

import pytest

from hegel_machine.cognition import finite_native_author as author
from hegel_machine.cognition import structural_inquiry as native
from hegel_machine.cognition.checkpoint_storage import write_json_checkpoint
from hegel_machine.cognition.python_research_routing import matching_refit, revision_action
from test_finite_native_author import ChoiceLab, declaration
from test_native_complete_research_unit import paid_scene


def pending_choice(tmp_path, *, refit_attempted=False):
    agent, question, latest, arguments = paid_scene(tmp_path)
    agent.config = replace(agent.config, observation_budget=10_000)
    old = 'archive/source/_evaluated_model/model.py'
    source = 'def predict(row): return 0.\n'
    agent._save(old, source)
    retained = {'model': old, 'support': 'new_paid_rows_unscored', 'mse': None}
    spec = {**ChoiceLab().public_spec(), 'channels': []}
    repair = revision_action(agent, retained, spec)
    if refit_attempted:
        copied = repair['next_action']['arguments']['path'] + '/_evaluated_model/model.py'
        agent._save(copied, source)
        unavailable = {'operator': 'refit', 'status': 'refit_unavailable', 'path': None,
            'source_model': old, 'tested_model': copied,
            'validation': {'returncode': 0, 'fit_available': False},
            'paid_rows_at_refit': len(agent.rows), 'paid_records_at_refit': len(agent.observations),
            'research_question_id': question['id'], 'question_meaning_version': 1}
        agent.operator_results.append(unavailable)
        agent.method_results.append({'proposal': deepcopy(repair['next_action']),
            'feedback': deepcopy(unavailable), 'evidence_after': {'rows': len(agent.rows)}})
        assert matching_refit(agent, old) is unavailable
    return agent, question, latest, retained, spec, repair


def test_completed_unavailable_refit_yields_to_payable_whole_native_unit(tmp_path, monkeypatch):
    agent, _, _, retained, spec, previous_repair = pending_choice(tmp_path, refit_attempted=True)
    before = deepcopy(agent.costs)
    monkeypatch.setattr(agent, '_execute',
        lambda *a, **k: pytest.fail('action selection must not execute a fit, prediction or purchase'))
    # The published older advice is still an actual refit proposal. Its saved
    # same-scope unavailable result must displace it, not a fabricated gain.
    choice = agent._python_research_decision(retained, spec, repair=previous_repair)
    assert choice['recommendation']['operator'] == 'native_condition_research'
    assert choice['next_action']['arguments']['max_compute'] == 446856
    unit = choice['research_unit']
    assert unit['minimum_compute'] == unit['search_compute'] + unit['completion_reserve']['compute_cost']
    assert unit['completion_reserve']['retained_pool_models'] == 4020
    assert unit['completion_reserve']['comparison_compute_cost'] > unit['search_compute']
    assert choice['decision']['options'] == [{'name': 'native_condition_research',
        'observation_cost': 0, 'compute_cost': unit['minimum_compute']}]
    assert agent.costs == before


def test_new_cheaper_repair_can_win_without_preferring_native_identity(tmp_path):
    agent, _, _, retained, spec, repair = pending_choice(tmp_path)
    choice = agent._python_research_decision(retained, spec, repair=repair)
    assert choice['recommendation']['operator'] == 'refit_response'
    assert choice['next_action'] == repair['next_action']
    offered = {item['name']: item for item in choice['decision']['options']}
    assert offered['refit_response']['compute_cost'] < offered['native_condition_research']['compute_cost']
    assert choice['decision']['reason'] == 'untried_action'


def test_qualified_retained_response_finishes_before_native_research(tmp_path, monkeypatch):
    agent, _, _, retained, spec, _ = pending_choice(tmp_path)
    retained = {**retained, 'support': 'paid_evaluation', 'mse': 0.}
    monkeypatch.setattr(native, 'native_condition_method',
        lambda *a, **k: pytest.fail('a supported retained response does not need a new native quote'))
    choice = agent._python_research_decision(retained, spec)
    assert choice['next_action'] == {'action': 'finish', 'arguments': {'model': retained['model']}}
    assert [item['name'] for item in choice['decision']['options']] == ['retain_response']


@pytest.mark.parametrize('blocked', ('attempted', 'unaffordable'))
def test_attempted_or_unfunded_whole_unit_never_falls_back_to_short_native(tmp_path, blocked):
    agent, question, latest, retained, spec, repair = pending_choice(tmp_path, refit_attempted=True)
    entry = native.native_condition_method(agent)['current_executions'][0]
    if blocked == 'attempted':
        agent.method_results.append({'proposal': deepcopy(entry['resume_action']),
            'feedback': {'operator': 'native_condition_research', 'research_question_id': question['id'],
                'question_meaning_version': 1, 'native_condition_research': deepcopy(latest)},
            'evidence_after': {'rows': len(agent.rows)}})
    else:
        minimum = entry['complete_research_unit']['minimum_compute']
        agent.config = replace(agent.config, compute_budget=minimum - 1)
        assert agent._remaining_compute() > 50_000
    choice = agent._python_research_decision(retained, spec, repair=repair)
    assert choice['recommendation'] is None and 'next_action' not in choice
    assert not any(option['name'] == 'native_condition_research' for option in choice['decision']['options'])


@pytest.mark.parametrize('funded', (True, False))
def test_finite_new_data_compares_strong_pool_then_consumes_whole_current_quote(tmp_path, monkeypatch, funded):
    monkeypatch.chdir(tmp_path)
    agent, question, latest, arguments = paid_scene(tmp_path)
    contract = declaration()
    contract['question']['id'] = question['id']
    contract['max_compute'] = 50_000
    contract['max_candidates'] = 64
    contract['predictor_paths'] = ['old.py']
    agent._save('old.py', 'def predict(row): return 0.\n')
    question.update(deepcopy(contract['question']))
    agent.knowledge['finite_native_author'] = deepcopy(contract)
    latest = deepcopy(latest)
    latest.update(observed_record_count=9, fit_evidence=list(range(126)), fit_record_indices=list(range(9)))
    question['native_condition_research'] = [deepcopy(latest)]
    agent.operator_results[0]['native_condition_research'] = deepcopy(latest)
    agent.observations = [{'row_indices': list(item['row_indices']), 'rows': [
        {'features': deepcopy(agent.rows[index].features), 'target': agent.rows[index].target,
         'history': list(agent.rows[index].history)} for index in item['row_indices']]}
        for item in agent.observations]
    current = native.native_condition_method(agent)['current_executions'][0]
    if not funded:
        agent.config = replace(agent.config,
            compute_budget=current['complete_research_unit']['minimum_compute'] - 1)
    decision = native.native_condition_method(agent)
    write_json_checkpoint(tmp_path / 'method_decision.json', {'research_options': [decision]})
    state = {'questions': deepcopy(agent.questions), 'operator_results': deepcopy(agent.operator_results),
        'method_results': [], 'costs': deepcopy(agent.costs),
        'limits': {'compute_budget': agent.config.compute_budget, 'execution_seconds': 1800.}}
    memory = {'finite_native_author': contract, 'research_state': state}
    spec = ChoiceLab().public_spec()
    first = author.select_actions(spec, agent.observations, memory)
    assert first['action'] == 'run' and first['arguments']['operator'] == 'research_choice'
    assert first['arguments']['models'] == ['old.py']
    # This is the existing completed-comparison envelope, not invented losses
    # or a new evaluation. The selector consumes its exact paid cutoff.
    state['method_results'].append({'proposal': deepcopy(first), 'feedback': {
        'operator': 'research_choice', 'research_question_id': question['id'],
        'question_meaning_version': 1, 'paid_rows': 145, 'evaluations': [{'path': 'old.py'}]},
        'evidence_after': {'rows': 145}})
    before = deepcopy(state)
    selected = author.select_actions(spec, agent.observations, memory)
    if funded:
        offered = decision['current_executions'][0]['resume_action']
        assert selected['action'] == 'run'
        assert selected['arguments']['operator'] == 'native_condition_research'
        assert selected['arguments']['max_compute'] == offered['arguments']['max_compute'] > 50_000
        assert 'matched_pairs' not in selected['arguments']
        assert selected['arguments']['max_candidates'] == arguments['max_candidates']
    else:
        assert selected['stop']['request']['reason'] == 'complete_native_research_unit_unfunded'
        assert 'action' not in selected
    assert state == before and agent.costs['model_calls'] == agent.costs['observation_cost'] == 0


def test_selected_complete_quote_reaches_actual_native_fit_full_comparison_and_retained_parent(tmp_path):
    from test_native_condition_inquiry import (
        COMPOSITION, PaidConditionalLab, buy_fixture, execute, prepared,
    )

    agent, lab = prepared(tmp_path), PaidConditionalLab()
    buy_fixture(agent, lab)
    agent._save('old.py', 'def predict(row): return 0.\n')
    agent.knowledge['finite_native_author'] = {'predictor_paths': ['old.py']}
    first = execute(agent, lab, 'run', {'operator': 'native_condition_research',
        'question': COMPOSITION, 'max_compute': 2600,
        'programs': [{'op': 'const', 'value': 0.}],
        'matched_pairs': [{'reference_record_index': 0, 'record_index': 1}],
        'max_candidates': 4})['native_condition_research']
    assert first['status'] == 'search_incomplete'
    agent._active_question_id = COMPOSITION['id']
    advice = agent._python_research_decision(None, lab.public_spec())
    assert advice['recommendation']['operator'] == 'native_condition_research'
    assert advice['research_unit']['minimum_compute'] > advice['research_unit']['search_compute']
    paid_before, actions_before = lab.paid, deepcopy(lab.actions)
    old_text = (agent.workspace / 'old.py').read_bytes()
    chosen = advice['next_action']
    second = execute(agent, lab, chosen['action'], chosen['arguments'])['native_condition_research']
    assert second['status'] == 'candidates_require_evaluation'
    assert second['searched_candidates'] > 0
    assert any(stage['stage'].startswith('joint_') and stage.get('completed_fits', 0) > 0
               for stage in second['search_stages'])
    assert second['completion_reserve']['completion_funded']
    models = list(dict.fromkeys(['old.py',
        *(item['model'] for item in first['exported_models']),
        *(item['model'] for item in second['exported_models'])]))
    compared = execute(agent, lab, 'run', {'operator': 'research_choice', 'models': models,
        'research_question': {'id': COMPOSITION['id']}})
    assert not compared.get('truncated') and 'error' not in compared
    assert [item['path'] for item in compared['evaluations']] == models
    retained = agent._evaluated_summary()
    assert retained['source_model'] in models and retained['source_model'] != 'old.py'
    assert retained['mse'] < next(item['mse'] for item in compared['evaluations'] if item['path'] == 'old.py')
    assert retained['evaluated_rows'] == paid_before == 80
    assert lab.paid == paid_before and lab.actions == actions_before
    assert (agent.workspace / 'old.py').read_bytes() == old_text
    assert agent.costs['model_calls'] == 0 and agent.model is None
