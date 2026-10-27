"""Price the actual finite search gap without fitting or purchasing evidence."""

from copy import deepcopy
from dataclasses import replace

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition import structural_inquiry as native


def paid_scene(tmp_path):
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm='code_react', source_task=False,
        knowledge={'files': {}, 'finite_native_author': {
            'predictor_paths': [f'archive/paid_{index:04d}.py' for index in range(4020)]}},
        config=InquiryLLMConfig(compute_budget=5_000_000, execution_seconds=1800.,
            confirmation_rows=48, max_model_calls=0, observation_budget=0))
    agent.rows = [Observation({'u': index / 145}, 0., tuple({'u': 0.} for _ in range(11)))
                  for index in range(145)]
    agent.observations = [{'row_indices': list(range(index * 14, (index + 1) * 14))}
                          for index in range(9)] + [{'row_indices': list(range(126, 145))}]
    question = agent._open_question({'id': 'paid_joint', 'question': 'Resolve the paid counterexample',
        'parent_id': 'task', 'theory_target': {'revision_fitting': {}}})
    result = {'round_index': 0, 'status': 'search_incomplete', 'compute_cost': 49_000,
        'fit_evidence': list(range(145)), 'fit_record_indices': list(range(10)),
        'observed_record_count': 10, 'candidates': [], 'exported_models': [],
        'search_stages': [{'stage': 'matched_evidence', 'compute_cost': 4896},
            {'stage': 'joint_conditional_response', 'compute_cost': 0,
             'blocked_work': {'additional_compute': 1000}}],
        'search_allowance_estimate': {'conditional_joint_attempt': 441670},
        'completion_reserve': {'search_allowance': 50_000}}
    arguments = {'operator': 'native_condition_research', 'question': {'id': question['id']},
        'matched_pairs': [{'reference_record_index': 8, 'record_index': 9}],
        'max_candidates': 64, 'max_compute': 50_000}
    question['native_condition_research'] = [deepcopy(result)]
    agent.operator_results = [{'operator': 'native_condition_research',
        'research_question_id': question['id'], 'arguments': deepcopy(arguments),
        'native_condition_research': result}]
    agent._active_question_id = question['id']
    return agent, question, result, arguments


def test_whole_search_keeps_full_strong_pool_exports_and_finish_funded(tmp_path):
    agent, question, result, arguments = paid_scene(tmp_path)
    before = deepcopy(agent.costs)
    reserve = native._native_completion_reserve(agent, question, 64)
    offer = native._native_compute_continuation(agent, question, result, arguments, reserve)
    assert offer['status'] == 'available'
    assert offer['action']['arguments']['max_compute'] == 446856
    unit = offer['research_unit']
    assert unit['minimum_compute'] == 446856 + reserve['compute_cost']
    assert reserve['retained_pool_models'] == 4020
    assert reserve['comparison_compute_cost'] == 145 * (4020 + 64)
    assert reserve['finish_compute_cost'] == 145 + 48
    assert unit['resumption'] == 'bounded_restart_with_retained_programs_not_an_internal_fit_cursor'
    assert unit['estimate_is_completion_guarantee'] is False
    assert agent.costs == before and len(agent.knowledge['finite_native_author']['predictor_paths']) == 4020


def test_extra_compute_above_old_fragment_does_not_make_a_whole_unit_payable(tmp_path):
    agent, question, result, arguments = paid_scene(tmp_path)
    reserve = native._native_completion_reserve(agent, question, 64)
    minimum = 446856 + reserve['compute_cost']
    agent.config = replace(agent.config, compute_budget=minimum - 1)
    offer = native._native_compute_continuation(agent, question, result, arguments, reserve)
    assert offer['available_compute_after_completion_reserve'] > 50_000
    assert offer['status'] == 'shared_compute_insufficient'
    assert 'action' not in offer


def test_original_expired_execution_window_blocks_even_an_affordable_search(tmp_path):
    agent, question, result, arguments = paid_scene(tmp_path)
    agent.costs['python_execution_seconds'] = agent.config.execution_seconds
    reserve = native._native_completion_reserve(agent, question, 64)
    offer = native._native_compute_continuation(agent, question, result, arguments, reserve)
    assert offer['status'] == 'execution_time_exhausted'
    assert 'action' not in offer
    assert agent.config.execution_seconds == 1800.


def test_published_same_bank_offer_does_not_require_an_unavailable_refit(tmp_path):
    agent, question, _, arguments = paid_scene(tmp_path)
    before = deepcopy(agent.costs)
    entry = native.native_condition_method(agent)['current_executions'][0]
    assert entry['complete_research_unit']['work_reason'] == 'search_incomplete'
    assert entry['resume_action'] == entry['compute_continuation']['action']
    assert entry['resume_action']['arguments']['max_compute'] == 446856
    assert entry['resume_action']['arguments']['matched_pairs'] == arguments['matched_pairs']
    assert agent.costs == before and agent.method_results == []


def test_new_paid_evidence_refreshes_joint_quote_without_old_pair_or_short_ceiling(tmp_path):
    agent, question, _, _ = paid_scene(tmp_path)
    agent.rows.extend(Observation({'u': 1.}, 0., ()) for _ in range(24))
    agent.observations.append({'row_indices': list(range(145, 169))})
    entry = native.native_condition_method(agent)['current_executions'][0]
    unit = entry['complete_research_unit']
    assert unit['work_reason'] == 'new_paid_evidence' and unit['quoted_paid_rows'] == 169
    assert unit['previous_paid_rows'] == entry['paid_rows'] == 145
    assert entry['current_paid_rows'] == 169
    assert entry['resume_action']['arguments']['max_compute'] > 50_000
    assert 'matched_pairs' not in entry['resume_action']['arguments']
    assert entry['resume_action']['arguments']['question'] == {'id': question['id']}


def test_finite_joint_purchase_returns_to_selection_but_boundary_update_stays_inline(tmp_path, monkeypatch):
    agent, question, result, arguments = paid_scene(tmp_path)
    question['native_condition_research'] = []
    queries = [{'purpose': purpose, 'matched_action_pairs': [], 'query_action': {
        'action': 'query', 'arguments': {'actions': [{'kind': 'trajectory', 'sequence': [{'u': .5}]}]}}}
        for purpose in ('matched_evidence', 'conditional_boundary')]
    outcome = {**deepcopy(result), 'status': 'matched_evidence_without_candidate',
        'candidates': [], 'evidence': [], 'proposed_experiments': queries,
        'finite_prior': 'fixture', 'research_decision': {'kind': 'structural_mismatch'},
        'remaining_boundary': {}}
    monkeypatch.setattr(native, 'research_native_conditions', lambda *a, **k: outcome)
    monkeypatch.setattr(native, '_register_native_theories', lambda *a: {'status': 'no_candidates'})
    monkeypatch.setattr(agent, '_return_structural_answer', lambda *a: None)
    laboratory = type('PublicOnlyLab', (), {'public_spec': lambda self: {'controls': ['u']}})()
    returned = native.execute_native_condition_research(agent, arguments, laboratory, question)
    proposed = returned['native_condition_research']['proposed_experiments']
    assert 'after_observation' not in proposed[0]['query_action']['arguments']
    assert proposed[0]['research_continuation']['max_compute'] == 50_000
    boundary = proposed[1]['query_action']['arguments']['after_observation'][0]
    assert boundary['boundary_update'] == {'round_index': 0, 'proposal_index': 1}
    assert agent.costs['observation_cost'] == agent.costs['model_calls'] == 0
