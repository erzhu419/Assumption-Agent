"""A re-read of an old experiment must not review later acquired evidence."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.inquiry_revision import (
    analyze_paid_evidence, analyze_revision, readout_record_references,
)
from hegel_machine.cognition.code_provider_context import prepare_code_context
from test_common_archive_online_research import CHILD, PARENT, prepared
from test_inquiry_revision import PublicLab, action, make_agent, revision, write


READOUT = """def readout(result, spec, memory):
    rows = [i for record in result['observations'] for i in record['row_indices']]
    return {'status': 'refuted', 'answer': {'seen_scope': result['analysis_scope']},
            'evidence': rows}
"""


def test_readout_exposes_sequential_controls_and_history_instead_of_pulse_labels():
    from test_native_condition_research import record
    controls = [{'drive': 1., 'switch': 0.}, {'drive': 0., 'switch': 0.},
                {'drive': 0., 'switch': 1.}, *[{'drive': 0., 'switch': 0.}] * 9]
    full = record(controls, 100)
    suffix = deepcopy(full)
    suffix['rows'] = suffix['rows'][3:]
    suffix['row_indices'] = suffix['row_indices'][3:]
    suffix['observation_index'] = 103
    before = deepcopy((full, suffix))
    first, second = readout_record_references([full, suffix], [66, 67])
    expected = [{'steps': [0, 0], 'controls': {'drive': 1., 'switch': 0.}},
                {'steps': [1, 1], 'controls': {'drive': 0., 'switch': 0.}},
                {'steps': [2, 2], 'controls': {'drive': 0., 'switch': 1.}},
                {'steps': [3, 11], 'controls': {'drive': 0., 'switch': 0.}}]
    assert first['control_runs'] == second['control_runs'] == expected
    assert first['initial_history_length'] == 0
    assert second['initial_history_length'] == 3
    assert first['response_row_ranges'] == [[100, 111]]
    assert second['response_row_ranges'] == [[103, 111]]
    assert (full, suffix) == before


def test_old_revision_keeps_its_rows_forecasts_and_unreviewed_later_evidence(tmp_path):
    agent, lab = make_agent(tmp_path), PublicLab()
    write(agent, 'model.py', 'def predict(row): return 0.\n', lab)
    write(agent, 'readout.py', READOUT, lab)
    agent._execute('query', {'actions': [action(1)]}, lab)
    acquired = agent._execute('query', {'actions': [action(2, 3)],
        'revision': revision(['model.py'], readout='readout.py')}, lab)['revision']
    old_forecasts = deepcopy(acquired['forecasts'])
    agent._execute('query', {'actions': [action(4, 5, 6)]}, lab)
    question = {'id': 'experiment', 'status': 'open', 'acquired_evidence': [1, 2, 3, 4, 5]}
    updated = analyze_revision(agent, agent.revisions[0], lab, question)
    scope = updated['returned_answer']['analysis_scope']
    assert scope['record_indices'] == [1]
    assert scope['response_row_ranges'] == [[1, 2]]
    assert scope['cited_response_row_ranges'] == [[1, 2]]
    assert scope['response_row_count'] == 2
    assert scope['paid_rows_at_analysis'] == 6 and scope['newer_paid_row_count'] == 3
    assert scope['forecast_evidence_cutoff'] == {'rows': 1, 'records': 1}
    assert scope['forecast_results'] == [{'model': 'model.py', 'prospective_mse': 6.5,
                                        'scored_rows': 2, 'complete_experiment': True}]
    assert updated['returned_answer']['answer']['seen_scope'] == updated['analysis_scope']
    assert updated['forecasts'] == old_forecasts
    assert question['reviewed_through_observation'] == 2
    assert question['analysis_scope'] == scope
    assert updated['analysis_attempts'][0]['analysis_scope']['paid_rows_at_analysis'] == 3
    assert updated['analysis_attempts'][1]['analysis_scope']['paid_rows_at_analysis'] == 6
    assert lab.paid == 6 and agent.costs['model_calls'] == 0


def test_actual_query_record_analysis_has_its_own_rows_without_old_forecasts(tmp_path):
    agent, lab = make_agent(tmp_path), PublicLab()
    write(agent, 'model.py', 'def predict(row): return 0.\n', lab)
    write(agent, 'readout.py', READOUT, lab)
    agent._execute('query', {'actions': [action(1, 2)],
        'revision': revision(['model.py'], readout='readout.py')}, lab)
    agent._execute('query', {'actions': [action(3, 4)]}, lab)
    question = {'id': 'latest', 'status': 'open'}
    updated = analyze_paid_evidence(agent, 'readout.py', lab, question, record_indices=[1])
    scope = updated['returned_answer']['analysis_scope']
    assert scope['kind'] == 'paid_evidence_analysis' and scope['revision_index'] == 1
    assert scope['record_indices'] == [1] and scope['response_row_ranges'] == [[2, 3]]
    assert scope['response_row_count'] == 2 and scope['newer_paid_row_count'] == 0
    assert scope['forecast_results'] == [] and scope['forecast_evidence_cutoff'] is None
    assert updated['forecasts'] == [] and 'observations' not in updated
    assert updated['returned_answer']['evidence'] == [2, 3]
    assert question['reviewed_through_observation'] == 3
    assert lab.paid == 4 and agent.costs['model_calls'] == 0


def test_selected_disjoint_records_do_not_claim_the_intervening_rows(tmp_path):
    agent, lab = make_agent(tmp_path), PublicLab()
    write(agent, 'readout.py', READOUT, lab)
    agent._execute('query', {'actions': [action(1, 2), action(3, 4), action(5, 6)]}, lab)
    result = analyze_paid_evidence(agent, 'readout.py', lab, record_indices=[0, 2])
    assert result['analysis_scope']['response_row_ranges'] == [[0, 1], [4, 5]]
    assert result['analysis_scope']['response_row_count'] == 4


def checked_readout(evidence):
    return f"""def readout(result, spec, memory):
    return {{'status': 'supported', 'answer': 'The cited responses agree.',
            'evidence': {evidence!r}, 'checks': [{{'id': 'paired_response',
                'terms': [{{'row': {evidence[0]}, 'coefficient': 1}},
                          {{'row': {evidence[1]}, 'coefficient': -1}}],
                'relation': 'eq', 'expected': 0., 'atol': 1e-9}}]}}
"""


@pytest.mark.parametrize('evidence', [[0, 1], [0, 2]])
def test_selected_analysis_rejects_old_or_partly_outside_paid_evidence(tmp_path, evidence):
    agent, lab = prepared(tmp_path, 'hegel'), PublicLab()
    agent._execute('query', {'actions': [action(7, 7)], 'question': PARENT}, lab)
    agent._execute('query', {'actions': [action(7, 7)], 'question': CHILD}, lab)
    write(agent, 'readout.py', checked_readout(evidence), lab)
    costs = deepcopy(agent.costs)

    result = agent._execute('run', {'operator': 'analyze_evidence', 'path': 'readout.py',
        'record_indices': [1], 'research_question': CHILD}, lab)

    record = result['revision']
    assert 'selected paid observation records' in record['readout_error']
    assert record['status'] == 'observed' and 'returned_answer' not in record
    assert 'check_results' not in record and 'evidence_status' not in record
    assert record['analysis_attempts'][0]['authored_answer']['evidence'] == evidence
    assert record['analysis_scope']['response_row_ranges'] == [[2, 3]]
    assert 'answer' not in result['question']
    parent = next(q for q in agent.questions if q['id'] == PARENT['id'])
    assert not parent.get('child_answers')
    assert agent.costs['compute_cost'] == costs['compute_cost']
    assert agent.costs['observation_cost'] == costs['observation_cost'] == lab.paid == 4


@pytest.mark.parametrize('selection,evidence,ranges', [
    ([1], [2, 3], [[2, 3]]),
    ([0, 1], [0, 2], [[0, 3]]),
])
def test_selected_analysis_checks_current_or_explicitly_mixed_record_rows(
        tmp_path, selection, evidence, ranges):
    agent, lab = prepared(tmp_path, 'hegel'), PublicLab()
    agent._execute('query', {'actions': [action(7, 7)], 'question': PARENT}, lab)
    agent._execute('query', {'actions': [action(7, 7)], 'question': CHILD}, lab)
    write(agent, 'readout.py', checked_readout(evidence), lab)

    result = agent._execute('run', {'operator': 'analyze_evidence', 'path': 'readout.py',
        'record_indices': selection, 'research_question': CHILD}, lab)

    record = result['revision']
    assert record['status'] == 'answered' and record['evidence_status'] == 'supported'
    assert record['analysis_scope']['response_row_ranges'] == ranges
    assert [w['row'] for w in record['check_results'][0]['witnesses']] == evidence
    parent = next(q for q in agent.questions if q['id'] == PARENT['id'])
    assert parent['child_answers'][-1]['evidence'] == evidence
    assert parent['child_answers'][-1]['evidence_status'] == 'supported'
    assert lab.paid == 4 and agent.costs['model_calls'] == 0


def test_prospective_readout_can_compare_paid_prefix_with_its_new_experiment(tmp_path):
    agent, lab = make_agent(tmp_path), PublicLab()
    write(agent, 'model.py', 'def predict(row): return 0.\n', lab)
    write(agent, 'readout.py', checked_readout([0, 2]), lab)
    agent._execute('query', {'actions': [action(7, 7)]}, lab)

    result = agent._execute('query', {'actions': [action(7, 7)],
        'revision': revision(['model.py'], readout='readout.py')}, lab)['revision']

    assert result['status'] == 'answered' and result['evidence_status'] == 'supported'
    assert result['record_indices'] == [1]
    assert result['returned_answer']['evidence'] == [0, 2]
    assert result['analysis_scope']['response_row_ranges'] == [[2, 3]]
    assert result['evidence_cutoff'] == {'rows': 2, 'records': 1}
    assert result['forecasts'][0]['prospective_mse'] == 49.
    assert lab.paid == 4


@pytest.mark.parametrize('selection', [[], [1], [0, 0]])
def test_record_selection_error_does_not_create_a_revision_or_purchase(tmp_path, selection):
    agent, lab = make_agent(tmp_path), PublicLab()
    agent._execute('query', {'actions': [action(1, 2)]}, lab)
    with pytest.raises(ValueError, match='existing paid observation records'):
        analyze_paid_evidence(agent, 'readout.py', lab, record_indices=selection)
    assert agent.revisions == [] and lab.paid == 2


def test_common_runtime_forwards_selected_paid_record_to_existing_readout(tmp_path):
    agent, lab = prepared(tmp_path, 'hegel'), PublicLab()
    write(agent, 'readout.py', READOUT, lab)
    agent._execute('query', {'actions': [action(1, 2), action(3, 4)]}, lab)
    result = agent._execute('run', {
        'operator': 'analyze_evidence', 'path': 'readout.py', 'record_indices': [1],
        'research_question': CHILD}, lab)
    record = result['revision']
    assert record['record_indices'] == [1]
    assert record['returned_answer']['evidence'] == [2, 3]
    assert record['analysis_scope']['response_row_ranges'] == [[2, 3]]
    assert record['forecasts'] == []
    assert result['question']['analysis_scope']['record_indices'] == [1]
    assert lab.paid == 4


def test_live_and_compact_context_keep_old_scope_and_later_unreviewed_rows(tmp_path):
    agent, lab = prepared(tmp_path, 'hegel'), PublicLab()
    write(agent, 'readout.py', READOUT, lab)
    agent._execute('query', {'actions': [action(1, 2)], 'question': CHILD,
        'revision': revision(['old.py'], readout='readout.py')}, lab)
    agent._execute('query', {'actions': [action(3, 4)], 'question': CHILD}, lab)
    agent._execute('run', {'operator': 'analyze_revision', 'revision_index': 0,
                           'path': 'readout.py'}, lab)
    request = agent._request(lab)
    compact = prepare_code_context(tmp_path, request, schema={}, instructions='test')
    for context in (request, compact):
        online = context['online_research']
        question = next(q for q in online['questions'] if q['id'] == CHILD['id'])
        assert question['analysis_scope']['response_row_ranges'] == [[0, 1]]
        assert question['unreviewed_evidence'] == [2, 3]
        saved = online['recent_revisions'][0]
        assert saved['record_indices'] == [0] and saved['response_row_ranges'] == [[0, 1]]
        assert saved['newer_paid_row_count'] == 2
        assert saved['forecast_results'][0]['model'] == 'old.py'
        latest = online['recent_paid_records'][-1]
        assert latest['record_index'] == 1 and latest['response_row_range'] == [2, 3]
        assert online['current_paid_rows'] == 4 and online['paid_record_count'] == 2


def test_analyzing_latest_selected_record_leaves_uncited_earlier_rows_unreviewed(tmp_path):
    agent, lab = prepared(tmp_path, 'hegel'), PublicLab()
    write(agent, 'readout.py', READOUT, lab)
    agent._execute('query', {'actions': [action(1, 2), action(3, 4)], 'question': CHILD}, lab)
    agent._execute('run', {'operator': 'analyze_evidence', 'path': 'readout.py',
                           'record_indices': [1], 'research_question': CHILD}, lab)
    question = next(q for q in agent._online_research()['questions'] if q['id'] == CHILD['id'])
    assert question['unreviewed_evidence'] == [0, 1]
    assert question['evidence'] == [2, 3]


def test_revised_readout_preserves_prior_cited_rows_and_legacy_returned_answer(tmp_path):
    agent, lab = prepared(tmp_path, 'hegel'), PublicLab()
    write(agent, 'readout.py', READOUT, lab)
    agent._execute('query', {'actions': [action(1, 2), action(3, 4), action(5, 6)],
                           'question': CHILD}, lab)
    agent._execute('run', {'operator': 'analyze_evidence', 'path': 'readout.py',
                           'record_indices': [2], 'research_question': CHILD}, lab)
    # Existing returned answers still identify reviewed evidence when an older
    # saved record has no attempt-level authored answer.
    agent.revisions[0].pop('analysis_attempts')
    for index in (0, 2):
        write(agent, 'readout.py', f"def readout(*args): return {{'status':'inconclusive', "
              f"'answer':'Local reading', 'evidence':[{index}]}}\n", lab)
        if index == 0:
            agent._execute('run', {'operator': 'analyze_evidence', 'path': 'readout.py',
                'record_indices': [0, 1], 'research_question': CHILD}, lab)
        else:
            agent._execute('run', {'operator': 'analyze_revision', 'path': 'readout.py',
                                   'revision_index': 1}, lab)
    question = next(q for q in agent._online_research()['questions'] if q['id'] == CHILD['id'])
    assert question['evidence'] == [2]
    assert question['unreviewed_evidence'] == [1, 3]
    assert [a['authored_answer']['evidence'] for a in agent.revisions[1]['analysis_attempts']] == [[0], [2]]
