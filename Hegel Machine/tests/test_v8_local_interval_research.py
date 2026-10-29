"""A measured local question can select a new scan without a global solution."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.finite_native_author import (
    _query_segment_problem, research_segment_options, select_research_segment,
)
from hegel_machine.cognition.native_condition_research import (
    _pair, _spec, propose_local_conditional_scans,
)
from hegel_machine.cognition.research_programs import propose_research_action_revisions


def local_scene():
    public = {'controls': ['u'], 'reset': True, 'action_kinds': ['trajectory'],
        'feature_bounds': {'u': [-1., 1.]}, 'max_probe_length': 12,
        'control_levels': {'u': [-1., -.4, 0., .1, .2, .3, .4, .5, 1.]}}
    sequence = [{'u': -.4} for _ in range(12)]
    record = {'action': {'kind': 'trajectory', 'channels': [], 'sequence': sequence},
        'row_indices': list(range(12)), 'rows': [{'features': dict(point),
            'history': deepcopy(sequence[:index]), 'target': 0.}
            for index, point in enumerate(sequence)]}
    local = {'status': 'conditional', 'control': 'u', 'background': {'u': -.4},
        'conditional_gain': .04, 'conditional_loss': 0., 'threshold_interval': [0., .5],
        'reference_record_index': 0, 'record_index': 1}
    round_ = {'round_index': 9, 'status': 'candidates_require_evaluation',
        'evidence': [local], 'observed_record_count': 1, 'fit_evidence': list(range(12)),
        'candidates': [], 'exported_models': []}
    question = {'id': 'q', 'question': 'Explain the remaining public failure',
        'meaning_version': 1, 'context_ref': {'public': 'lab'},
        'native_condition_research': [round_]}
    continuation = {'operator': 'native_condition_research',
        'research_question': {'id': 'q'}, 'max_candidates': 2, 'max_compute': 7}
    source = {'file': 'research_state.json', 'json_path': ['questions', 0,
        'native_condition_research', 9]}
    return public, [record], round_, question, continuation, source


def offered_query():
    public, records, round_, question, continuation, source = local_scene()
    generated, = propose_local_conditional_scans(public, records, round_,
        continuation=continuation, source_ref=source)
    option = {**generated, 'family': 'Q', 'proposal': deepcopy(generated['query_action']),
        'complete_quote': {'status': 'available', 'compute_cost': 100, 'observation_cost': 12,
            'paid_rows': 12, 'retained_pool_models': 1, 'max_exported_models': 2,
            'remaining_execution_seconds': 20.}}
    callback = deepcopy(continuation)
    callback.update(max_compute=50, matched_pairs=[
        {'reference_record_index': 1, 'record_index': 2}])
    option['proposal']['arguments']['after_observation'] = [callback]
    return public, records, round_, question, continuation, source, option


def test_local_scan_reuses_the_actual_reference_and_original_matched_consumer():
    public, records, round_, question, continuation, source, option = offered_query()
    before = deepcopy((records, round_, continuation))
    assert 'predicted_separation' not in option
    assert option['local_evidence_ref']['json_path'][-2:] == ['evidence', 0]
    assert 'max_compute' not in option['research_continuation']
    assert option['proposal']['arguments']['actions'][0] == records[0]['action']
    scan = option['proposal']['arguments']['actions'][1]['sequence']
    assert len(scan) == 12 and scan[0] == {'u': -.4}
    planned = {'rows': [{'features': point, 'history': scan[:index]}
                        for index, point in enumerate(scan)]}
    matched = _pair(_spec(public), [records[0], planned], 0, 1)
    assert matched['matched_rows'] == 12
    assert _query_segment_problem(option, public, records, question,
        {'question': question}, [], 12) is None
    assert (records, round_, continuation) == before


@pytest.mark.parametrize('change', ['interval', 'no_support', 'short_scan', 'single_point_consumer'])
def test_local_target_does_not_borrow_a_forecast_or_wrong_analysis_contract(change):
    public, records, round_, question, continuation, source, option = offered_query()
    if change == 'interval':
        option['local_conditional_interval']['threshold_interval'] = [.1, .2]
    elif change == 'no_support':
        round_['evidence'][0]['conditional_gain'] = 0.
        option['predicted_separation'] = 1.  # Cannot override the actual source.
    elif change == 'short_scan':
        option['proposal']['arguments']['actions'][1]['sequence'] = \
            option['proposal']['arguments']['actions'][1]['sequence'][:7]
    else:
        option['proposal']['arguments']['after_observation'][0]['boundary_update'] = {}
    assert _query_segment_problem(option, public, records, question,
        {'question': question}, [], 12) == 'actual_local_interval_and_matched_consumer_required'


def test_existing_finite_generator_can_choose_local_scan_and_keeps_shared_closure():
    public, records, round_, question, continuation, source, option = offered_query()
    state = {'questions': [question], 'operator_results': [], 'method_results': [],
        'limits': {'compute_budget': 1000, 'observation_budget': 24}, 'costs': {}}
    memory = {'finite_native_author': {'question': question, 'predictor_paths': ['old.py']},
        'research_state': state, 'method_decision': {'research_options': [option]}}
    inventory = research_segment_options(public, records, memory, state=state)
    assert len(inventory['options']) == 1
    ordinary = {'action': 'query', 'arguments': {'actions': [{'kind': 'trajectory',
        'sequence': [{'u': .4} for _ in range(12)]}]}}
    candidates = propose_research_action_revisions(ordinary, inventory)['candidates']
    assert len(candidates) == 1
    plan = candidates[0]['research_selection_plan']
    selected = select_research_segment(public, records, memory, plan=plan,
        ordinary_select_actions=lambda *_: ordinary)
    assert selected['arguments']['actions'][1]['sequence'][1] == {'u': 0.}
    common = {'action': 'run', 'arguments': {'operator': 'research_choice', 'models': ['old.py']}}
    assert select_research_segment(public, records, memory, plan=plan,
        ordinary_select_actions=lambda *_: common) == common


def test_existing_query_callback_remaps_reused_reference_to_actual_record(monkeypatch):
    from hegel_machine.cognition.online_inquiry_llm import InquiryLLMAgent
    from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent

    agent = object.__new__(TemporalOperatorAgent)
    agent.questions, agent.operator_results = [], []
    agent.costs = {'compute_cost': 0, 'python_execution_seconds': 0.}
    received = []
    def consume(action, arguments, laboratory):
        received.append(deepcopy(arguments))
        return {'operator': arguments['operator'], 'status': 'fixture_consumer_received'}
    agent._execute = consume
    callback = {'operator': 'native_condition_research', 'max_compute': 50,
        'matched_pairs': [{'reference_record_index': 10, 'record_index': 11}]}
    monkeypatch.setattr(InquiryLLMAgent, '_query', lambda *_: {
        'after_observation': [callback], 'record_indices': [3, 10],
        'reused_record_indices': [3], 'record_index_map': {'10': 3, '11': 10}})
    result = agent._query({}, None)
    assert received[0]['matched_pairs'] == [{'reference_record_index': 3, 'record_index': 10}]
    assert result['after_observation_results'][0]['status'] == 'fixture_consumer_received'
    assert callback['matched_pairs'] == [{'reference_record_index': 10, 'record_index': 11}]
