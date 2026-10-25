"""Historical progress is read from actual execution, not cumulative ref counts."""
from copy import deepcopy
import json
from types import SimpleNamespace

from hegel_machine.cognition.method_effect_query import retained_research_progress


def scene():
    key = 'relation_transfer:frozen_recall:rev@1:old_owner'
    historical = {'id': 'rev', 'kind': 'program_revision', 'meaning_version': 1,
        'theory_target': {'parent_theory_id': 'old_parent', 'paid_evidence_scope': {'rows': 97}},
        'frozen_recall_execution': {'result': {'execution_work_groups': [
            {'candidate_ids': ['alias', 'old_owner'], 'execution_key': key}]}}}
    current = {'id': 'rev', 'kind': 'program_revision', 'meaning_version': 2,
        'theory_target': {'parent_theory_id': 'winner'}, 'meaning_history': [historical]}
    execution = {'request_basis': {'candidate_id': 'old_owner', 'candidate_version': 1,
        'parent': {'id': 'old_parent', 'version': 1}, 'components': [{'id': 'c', 'version': 3}]},
        'cursor': {'layer': 1, 'offset': 2,
            'pending_fits': [{'indices': [1, 2], 'position': {'layer': 1, 'offset': 2}}],
            'fitting': {'structures': [{'program': {'large': 'do not copy'}}],
                'grammar_cursor': {'positions': {'relation': 3}, 'next_operation': 2,
                    'frontier': [{'program': 'not a progress field'}],
                    'thresholds_by_term': [[1, 2, 3]]}}},
        'allowance': 8184, 'search_completed': False,
        'result': {'new_compute_cost': 7602, 'minimum_compute_for_progress': 8185,
            'retained_execution_refs': [key], 'generated_theories': [
                {'theory_id': 'instance', 'version': 2, 'status': 'refuted',
                    'binding': {'fitting': {'fitting_method': 'stable_lstsq',
                        'fit_reuse_ref': {'file': 'old.json', 'json_path': ['fit']},
                        'parameters': [1, 2, 3], 'search_completed': True}}}]}}
    return SimpleNamespace(questions=[current], theory_consolidation={'executions': {key: execution}},
        events=[], method_results=[], operator_results=[],
        costs={'compute_cost': 100}, evaluated_model={'model': 'winner.py'},
        knowledge={'finite_native_author': {'predictor_paths': ['a.py', 'b.py']}},
        candidate_paths=['b.py', 'c.py'])


def test_progress_reads_history_shared_owner_and_pending_without_mutation():
    agent = scene()
    original = deepcopy(vars(agent))
    result = retained_research_progress(agent)
    assert vars(agent) == original
    assert [q['meaning_version'] for q in result['questions']] == [1, 2]
    execution = result['executions'][0]
    assert execution['operative'] is True
    assert execution['parent'] == {'id': 'old_parent', 'version': 1}
    assert execution['cursor']['pending_fits'] == [{'indices': [1, 2], 'position': {'layer': 1, 'offset': 2}}]
    assert execution['latest_result']['new_compute_cost'] == 7602
    assert execution['generated_refs'][0]['fitting']['fit_reuse_ref']['file'] == 'old.json'
    assert result['current_parent'] == {'model': 'winner.py'}
    assert result['full_pool_count'] == 3


def test_progress_counts_only_event_suffix_and_keeps_nested_costs_separate():
    agent = scene()
    agent.events = [{'feedback': {'frozen_recall': {'new_compute_cost': 10000}}},
        {'action': 'run', 'feedback': {'operator': 'research_route',
            'research_track_question_ref': {'question_id': 'rev', 'meaning_version': 1},
            'frozen_recall': {'new_compute_cost': 900, 'work': [
                {'candidate_id': 'old_owner', 'shared_candidate_ids': ['old_owner', 'alias'],
                    'theory_ids': ['instance'], 'new_compute_cost': 700}]}}}]
    agent.method_results = [{'proposal': {'action': 'run', 'arguments': {'operator': 'old'}}},
        {'proposal': {'action': 'run', 'arguments': {'operator': 'research_choice'}}, 'feedback': {}}]
    result = retained_research_progress(agent, event_start=1, method_start=1)
    assert len(result['new_events']) == len(result['new_methods']) == 1
    assert result['new_events'][0]['research_track_question_ref']['meaning_version'] == 1
    assert result['new_event_compute_sums']['frozen_recall_total'] == 900
    assert result['new_event_compute_sums']['frozen_recall_work_only'] == 700
    assert result['new_event_compute_sums']['are_additive_to_each_other'] is False
    assert result['costs'] == {'compute_cost': 100}


def test_progress_full_pool_includes_native_and_validated_paths_with_existing_rules():
    agent = scene()
    agent.questions.append({'id': 'native', 'native_condition_research': [
        {'candidates': [{'model': 'native.py'}, {'model': 'b.py'}],
         'exported_models': [{'model': 'native_export.py'}]}]})
    agent.operator_results = [
        {'operator': 'refit', 'status': 'refit_supported_on_validation', 'path': 'b.py'},
        {'operator': 'response', 'candidates': [
            {'path': 'validated.py', 'validation_mse': 0.},
            {'path': 'failed_validation.py', 'validation_mse': 1.}]}]
    result = retained_research_progress(agent)
    assert result['full_pool_count'] == 6
    assert 'current_native_banks' in result['full_pool_scope']


def test_progress_excludes_repeated_programs_and_model_rosters_and_handles_missing_selection():
    agent = scene()
    agent.events = [{'action': 'query', 'feedback': {'selection': None,
        'program_action': {'action': 'run', 'arguments': {'operator': 'research_choice',
            'models': ['a.py'] * 2000}},
        'evaluations': [{'mse': .2}, {'mse': .1}, {'error': 'failed'}]}}]
    result = retained_research_progress(agent)
    serialized = json.dumps(result)
    assert 'do not copy' not in serialized and 'not a progress field' not in serialized
    assert 'thresholds_by_term' not in serialized and 'parameters' not in serialized
    assert len(serialized) < 6000
    event = result['new_events'][0]
    assert event['selected_experiment'] is None
    assert event['program_action']['models_count'] == 2000
    assert event['comparison'] == {'evaluations': 3, 'finite_nonnegative_mse_count': 2,
                                   'minimum_recorded_mse': .1}
