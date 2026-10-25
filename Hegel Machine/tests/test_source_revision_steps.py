"""Whole recorded comparisons remain distinct from intermediate retentions."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.research_episodes import extract_episodes


@pytest.fixture
def recorded_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(Program, 'predict', lambda *a, **k: pytest.fail('saved metadata only'))
    programs = [Program('const', value=value).to_dict() for value in (2., 1.5, 1., 3.)]
    names = [f'native_{index}.py' for index in range(4)]
    question = {'id': 'response', 'meaning_version': 1,
        'context_ref': {'source_id': 'one_physical_source'},
        'native_condition_research': [{'round_index': 0, 'fit_record_indices': [0],
            'fit_evidence': [0, 1, 2], 'observed_record_count': 1,
            'candidates': [{'model': name, 'program': program} for name, program in zip(names, programs)]}]}

    def summary(index, mse):
        return {'model': f'_evaluated_model/{names[index]}', 'source_model': names[index],
                'mse': mse, 'evaluated_rows': 3, 'evaluation_scope': 'paid_evaluation'}

    def score(index, mse, incumbent, retained):
        candidate = {'model': names[index], 'mse': mse, 'evaluated_rows': 3,
                     'evaluation_scope': 'paid_evaluation'}
        return {'path': names[index], 'returncode': 0, 'mse': mse, 'loss_kind': 'mse',
                'paid_residuals': {'evaluated_rows': 3, 'examples': []},
                'retention_decision': {'status': 'candidate_retained' if retained else 'incumbent_retained',
                    'same_paid_rows': True, 'candidate': candidate, 'incumbent': incumbent,
                    'scope': 'paid comparison, not independent confirmation'}}

    # The initial parent's current score is saved last, after two actual
    # adoptions. A failed old archive entry is still in the declared full pool.
    evaluations = [
        {'path': 'failed_old_archive.py', 'returncode': 1, 'error': 'retained original failure'},
        score(3, 9., summary(0, 4.), False),
        score(1, 2.25, summary(0, 4.), True),
        score(2, 1., summary(1, 2.25), True),
        score(0, 4., summary(2, 1.), False)]
    arguments = {'operator': 'research_choice', 'models': [row['path'] for row in evaluations]}
    feedback = {'operator': 'research_choice', 'research_question_id': 'response',
        'question_meaning_version': 1, 'paid_rows': 3, 'evaluations': evaluations,
        'evaluated_model': summary(2, 1.), 'compute_cost': 15, 'python_execution_seconds': .5,
        'program_action': {'action': 'run', 'arguments': deepcopy(arguments)}}
    return SimpleNamespace(workspace=tmp_path, questions=[question], revisions=[], relation_research={},
        theory_index={}, events=[{'action': 'run', 'arguments': arguments, 'feedback': feedback}],
        observations=[{'row_indices': [0, 1, 2], 'rows': [
            {'features': {'u': float(index)}, 'target': 0., 'history': []} for index in range(3)]}],
        config=SimpleNamespace(confirmation_mse_threshold=1e-10), costs={'compute_cost': 15})


def source_episode(agent):
    return next(row for row in extract_episodes(agent)['episodes'] if row.get('source_revision_steps'))


def test_complete_step_uses_actual_initial_parent_and_final_winner_keeps_intermediate_and_failed_pool(recorded_comparison):
    agent = recorded_comparison
    original = deepcopy((agent.questions, agent.events, agent.observations, agent.costs))
    episode = source_episode(agent)
    step, = episode['source_revision_steps']
    assert step['status'] == 'completed' and step['retained_outcome'] == 'parent_changed'
    assert step['before_model'] == 'native_0.py' and step['after_model'] == 'native_2.py'
    assert step['before_ref']['json_path'][-3:] == ['candidates', 0, 'program']
    assert step['source_ref']['json_path'][-3:] == ['candidates', 2, 'program']
    assert step['source_comparison']['before_evaluation_ref']['json_path'][-1] == 4
    assert step['source_comparison']['after_evaluation_ref']['json_path'][-1] == 3
    assert len(step['member_retention_refs']) == 2
    assert len([change for change in episode['actual_changes'] if change.get('before_ref')]) == 2
    assert step['comparison_pool_ref']['json_path'] == ['events', 0, 'feedback', 'evaluations']
    assert step['scope']['declared_model_count'] == step['scope']['evaluated_model_count'] == 5
    assert step['scope']['same_paid_rows'] and not step['scope']['independent_confirmation']
    assert [ref['json_path'][-1] for ref in step['cost_refs']] == ['compute_cost', 'python_execution_seconds']
    assert step['cost_scope'] == 'whole_research_choice_not_additive_per_revision'
    assert 'before_program' not in step and 'after_program' not in step and 'observations' not in step
    assert original == (agent.questions, agent.events, agent.observations, agent.costs)


def test_complete_repeated_comparison_is_a_negative_step_not_cached_old_adoption(recorded_comparison):
    agent = recorded_comparison
    first = agent.events[0]
    repeated = deepcopy(first)
    feedback = repeated['feedback']
    final = deepcopy(feedback['evaluated_model'])
    # The summary's old adoption note is deliberately stale, while every
    # decision in this second actual action retains the current parent.
    feedback['evaluated_model']['selection'] = deepcopy(first['feedback']['evaluations'][3]['retention_decision'])
    for row in feedback['evaluations'][1:]:
        row['retention_decision']['incumbent'] = deepcopy(final)
        row['retention_decision']['status'] = 'incumbent_retained'
    agent.events.append(repeated)
    episode = source_episode(agent)
    positive, negative = episode['source_revision_steps']
    assert positive['retained_outcome'] == 'parent_changed'
    assert negative['status'] == 'completed' and negative['retained_outcome'] == 'parent_unchanged'
    assert negative['before_ref'] == negative['source_ref']
    assert negative['member_retention_refs'] == []
    assert negative['action_ref']['json_path'] == ['events', 1]
    assert len([change for change in episode['actual_changes'] if change.get('before_ref')]) == 2
    assert episode['source_id'] == 'one_physical_source'


@pytest.mark.parametrize('missing,reason', [
    ('roster', 'saved_comparison_roster_incomplete_or_truncated'),
    ('truncated', 'saved_comparison_roster_incomplete_or_truncated'),
    ('chain', 'saved_comparison_retention_chain_inconsistent'),
    ('terminal', 'terminal_model_does_not_match_actual_retention_chain'),
    ('program', 'unique_saved_initial_and_terminal_Programs_required'),
    ('ambiguous_program', 'unique_saved_initial_and_terminal_Programs_required'),
    ('initial_score', 'finite_saved_initial_incumbent_score_required'),
])
def test_incomplete_or_inconsistent_whole_step_stays_unknown_without_replacing_old_ledger(
        recorded_comparison, missing, reason):
    agent = recorded_comparison
    feedback = agent.events[0]['feedback']
    if missing == 'roster':
        feedback['evaluations'].pop()
    elif missing == 'truncated':
        feedback['evaluations'][2]['truncated'] = True
    elif missing == 'chain':
        feedback['evaluations'][3]['retention_decision']['incumbent']['mse'] = 99.
    elif missing == 'terminal':
        feedback['evaluated_model']['mse'] = 99.
    elif missing == 'program':
        agent.questions[0]['native_condition_research'][0]['candidates'].pop(0)
    elif missing == 'ambiguous_program':
        candidates = agent.questions[0]['native_condition_research'][0]['candidates']
        candidates.append(deepcopy(candidates[0]))
    else:
        feedback['evaluations'][1]['retention_decision']['incumbent'].pop('mse')
    episode = source_episode(agent)
    step, = episode['source_revision_steps']
    assert step['status'] == 'UNKNOWN' and step['reason'] == reason
    assert 'before_ref' not in step and 'source_ref' not in step
    assert step['comparison_pool_ref']['json_path'] == ['events', 0, 'feedback', 'evaluations']
    assert len(step['cost_refs']) == 2
    assert 'actual_changes' in episode


def test_adjacent_route_does_not_become_a_claimed_source_producer(recorded_comparison):
    agent = recorded_comparison
    agent.events.insert(0, {'action': 'run', 'arguments': {'operator': 'research_route'},
        'feedback': {'operator': 'research_route', 'research_question_id': 'response',
                     'frozen_recall': {'work': [{'theory_ids': ['unrelated_export']}]}}})
    step, = source_episode(agent)['source_revision_steps']
    assert step['status'] == 'completed'
    assert step['producer_link']['status'] == 'UNKNOWN'
    assert step['action_ref']['json_path'] == ['events', 1]
    assert all(ref['json_path'][:2] == ['events', 1] for ref in step['cost_refs'])


def test_missing_original_criterion_is_not_replaced_by_a_default(recorded_comparison):
    agent = recorded_comparison
    del agent.config.confirmation_mse_threshold
    step, = source_episode(agent)['source_revision_steps']
    assert step['status'] == 'completed'
    assert step['source_comparison'] == {'status': 'UNKNOWN', 'reason': 'original_paid_mse_criterion_required'}


@pytest.mark.parametrize('operator', ['refit', 'response'])
def test_actual_preceding_validated_append_uses_existing_complete_choice_contract(recorded_comparison, operator):
    agent = recorded_comparison
    event = agent.events[0]
    event['arguments']['models'].pop()
    event['feedback']['program_action']['arguments']['models'].pop()
    validated = ({'operator': 'refit', 'status': 'refit_supported_on_validation', 'path': 'native_0.py'}
                 if operator == 'refit' else {'operator': 'response',
                    'candidates': [{'path': 'native_0.py', 'validation_mse': 1e-12}]})
    agent.events.insert(0, {'action': 'run', 'feedback': validated})
    step, = source_episode(agent)['source_revision_steps']
    assert step['status'] == 'completed'
    assert step['scope']['evaluated_model_count'] == 5


def test_unproven_tail_and_later_validation_cannot_complete_an_earlier_pool(recorded_comparison):
    agent = recorded_comparison
    event = agent.events[0]
    event['arguments']['models'].pop()
    event['feedback']['program_action']['arguments']['models'].pop()
    agent.events.append({'action': 'run', 'feedback': {'operator': 'refit',
        'status': 'refit_supported_on_validation', 'path': 'native_0.py'}})
    step, = source_episode(agent)['source_revision_steps']
    assert step['status'] == 'UNKNOWN'
    assert step['reason'] == 'saved_comparison_roster_incomplete_or_truncated'


def frozen_terminal_export(agent):
    question = agent.questions[0]
    candidates = question['native_condition_research'][0]['candidates']
    candidate = candidates.pop(2)
    target = {'role_signature': {'u': {'role': 'control', 'unit': 'V', 'time': 'current'}},
              'output_unit': 'V', 'context_ref': question['context_ref']}
    question['theory_target'] = deepcopy(target)
    identifier = 'frozen_recall:response@1:old_relation:binding:0:1:fit:1'
    agent.theory_index[identifier] = {**deepcopy(target), 'version': 1,
        'program': candidate['program'], 'consolidation_ref': {'candidate_id': 'old_relation'}}
    question['revision_candidate_exports'] = [{'model': 'native_2.py',
        'theory_ref': {'id': identifier, 'version': 1},
        'program_source_ref': {'file': 'research_state.json',
            'json_path': ['theory_index', identifier, 'program'], 'version': 1},
        'transfer_ref': {'file': 'shared/theory_library.json', 'json_path': ['transfers', 4]}}]
    return identifier


def recorded_route(identifier):
    return {'action': 'run', 'arguments': {'operator': 'research_route'}, 'feedback': {
        'operator': 'research_route', 'research_question_id': 'response', 'question_meaning_version': 1,
        'new_compute_cost': 100, 'frozen_recall': {'new_compute_cost': 90,
            'work': [{'candidate_id': 'old_relation', 'theory_ids': [identifier], 'new_compute_cost': 80}]}}}


def test_exact_terminal_export_links_unique_past_work_with_nested_cost_scopes(recorded_comparison):
    agent = recorded_comparison
    identifier = frozen_terminal_export(agent)
    agent.events.insert(0, recorded_route(identifier))
    step, = source_episode(agent)['source_revision_steps']
    link = step['producer_link']
    assert step['status'] == 'completed' and link['status'] == 'recorded'
    assert link['route_ref']['json_path'] == ['events', 0]
    assert link['work_ref']['json_path'] == ['events', 0, 'feedback', 'frozen_recall', 'work', 0]
    assert link['returned_theory_ref']['json_path'][-2:] == ['theory_ids', 0]
    assert link['theory_ref'] == {'id': identifier, 'version': 1}
    assert link['transfer_ref']['json_path'] == ['transfers', 4]
    assert link['cost_scopes']['nested_costs_not_additive']
    assert link['cost_scopes']['route_total_ref']['json_path'] == ['events', 0, 'feedback', 'new_compute_cost']
    assert link['cost_scopes']['work_subset_ref']['json_path'][-2:] == [0, 'new_compute_cost']
    assert all(ref['json_path'][:2] == ['events', 1] for ref in step['cost_refs'])


@pytest.mark.parametrize('mismatch', ['question', 'meaning', 'constructor', 'ambiguous', 'cached', 'future'])
def test_unproven_route_cannot_receive_terminal_export_attribution(recorded_comparison, mismatch):
    agent = recorded_comparison
    identifier = frozen_terminal_export(agent)
    route = recorded_route(identifier)
    if mismatch == 'question':
        route['feedback']['research_question_id'] = 'different_question'
    elif mismatch == 'meaning':
        route['feedback']['question_meaning_version'] = 2
    elif mismatch == 'constructor':
        route['feedback']['frozen_recall']['work'][0]['candidate_id'] = 'different_constructor'
    elif mismatch == 'cached':
        route['feedback']['frozen_recall']['cache_hit'] = True
    if mismatch == 'future':
        agent.events.append(route)
    else:
        agent.events.insert(0, route)
        if mismatch == 'ambiguous':
            agent.events.insert(0, deepcopy(route))
    step, = source_episode(agent)['source_revision_steps']
    assert step['status'] == 'completed' and step['producer_link']['status'] == 'UNKNOWN'
    assert step['producer_link']['transfer_ref']['json_path'] == ['transfers', 4]


def test_episode_keeps_real_unavailable_and_unfinished_action_refs_in_original_order(recorded_comparison):
    agent = recorded_comparison
    agent.events.extend([
        {'action': 'run', 'arguments': {'operator': 'refit'}, 'feedback': {
            'operator': 'refit', 'research_question_id': 'response', 'question_meaning_version': 1,
            'status': 'refit_unavailable', 'compute_cost': 3}},
        {'action': 'run', 'arguments': {'operator': 'native_condition_research'}, 'feedback': {
            'operator': 'native_condition_research', 'research_question_id': 'response',
            'question_meaning_version': 1, 'status': 'search_incomplete', 'compute_cost': 12}}])
    events = deepcopy(agent.events)
    episode = source_episode(agent)
    assert episode['research_action_refs'] == [
        {'file': 'research_context.json', 'json_path': ['events', index]} for index in range(3)]
    assert len(episode['source_revision_steps']) == 1
    assert agent.events == events
