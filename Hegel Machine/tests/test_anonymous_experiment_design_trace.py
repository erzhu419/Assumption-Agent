"""Report scientific sequence without equating tool usage or paid fit with success."""

from copy import deepcopy
import json

import pytest

from scripts.run_anonymous_online_research import experimental_design_trace


def example():
    action = {'kind': 'trajectory', 'sequence': [{'u': .5}], 'channels': []}
    record = {'action': action, 'row_indices': [72], 'rows': [{'target': 2.}]}
    revision = {'index': 0, 'target': 'response', 'status': 'answered',
        'proposal': {'id': 'r0', 'affected_object': 'representation',
            'models': ['old.py', 'new.py'], 'readout': 'readout.py', 'comparison_index': 0},
        'actions': [action], 'evidence_cutoff': {'rows': 72, 'records': 24},
        'prior_fit': [{'model': model, 'returncode': 0, 'mse': 0.} for model in ['old.py', 'new.py']],
        'forecasts': [{'model': model, 'returncode': 0, 'predictions': [prediction],
            'prospective_mse': error, 'scored_rows': 1, 'complete_experiment': True,
            'source': {'files': {'huge.py': 'scientific source must not enter summary'}}}
            for model, prediction, error in [('old.py', 1., 1.), ('new.py', 2., 0.)]],
        'observations': [record], 'analysis_is_agent_authored': True,
        'analysis_attempts': [{'path': 'readout.py'}],
        'authored_answer': {'status': 'supported', 'answer': {'surviving': ['new.py']}, 'evidence': [72]}}
    comparison = {'index': 0, 'models': ['old.py', 'new.py'], 'mse_threshold': 1e-10,
        'eligible_models': ['old.py', 'new.py'], 'prior_fit': revision['prior_fit'],
        'comparisons': [{'id': 'q0', 'actions': [action], 'row_count': 1, 'predicted_separation': .25}],
        'selected_experiment': {'id': 'q0', 'observation_cost': 1, 'predicted_separation': .25}}
    confirmation = {'passed': True, 'mse': 0., 'rows': 96, 'observation_indices': list(range(73, 169))}
    events = [
        {'step': 0, 'action': 'run', 'arguments': {'operator': 'compare_experiments'},
         'feedback': {'comparison': comparison}},
        {'step': 1, 'action': 'query', 'arguments': {'actions': [action]},
         'feedback': {'revision': revision, 'observations': [record]}},
        {'step': 2, 'action': 'finish', 'arguments': {'model': 'new.py'},
         'feedback': {'fresh_confirmation': confirmation}}]
    return {'status': 'completed', 'model': 'new.py', 'fresh_confirmation': confirmation,
        'events': events, 'method_results': deepcopy(events),
        'revision_proposals': [revision], 'experiment_comparisons': [comparison]}


def test_supported_rivals_separate_before_submission_without_duplicate_or_source_payload():
    result = example()
    trace = experimental_design_trace(result)
    assert trace['complete_discrimination']
    assert len(trace['revisions']) == len(trace['comparisons']) == 1
    event = trace['revisions'][0]
    assert event['supported_models'] == ['old.py', 'new.py']
    assert event['measured_refuted_models'] == ['old.py']
    assert event['measured_supported_models'] == ['new.py']
    assert event['before_first_submission'] and not event['after_framework_confirmation']
    assert event['disagreement_witness']['values'] == {'old.py': 1., 'new.py': 2.}
    assert 'scientific source must not enter summary' not in json.dumps(trace)
    assert trace['explicit_queries'][0]['paid_rows'] == 1


@pytest.mark.parametrize('change', ['underfit', 'no_readout', 'unconsumed_evidence', 'final_failure'])
def test_discrimination_needs_supported_rivals_consumed_evidence_and_later_acceptance(change):
    result = example()
    revision = result['revision_proposals'][0]
    if change == 'underfit':
        revision['prior_fit'][0]['mse'] = .1
    elif change == 'no_readout':
        revision.pop('analysis_is_agent_authored')
    elif change == 'unconsumed_evidence':
        revision['authored_answer']['evidence'] = [0]
    else:
        result['fresh_confirmation']['passed'] = False
        result['status'] = 'confirmation_failed'
        result['model'] = None
    trace = experimental_design_trace(result)
    assert not trace['complete_discrimination']


def test_refuting_both_hypotheses_preserves_the_valid_later_revision_outcome():
    result = example()
    result['revision_proposals'][0]['forecasts'][1]['prospective_mse'] = .25
    trace = experimental_design_trace(result)
    event = trace['revisions'][0]
    assert event['measured_outcome'] == 'hypothesis_set_refuted'
    assert event['followed_by_accepted_model']
    assert event['authored_readout']
    assert not trace['complete_discrimination']


def test_framework_confirmation_counterexample_does_not_become_pre_submission_design():
    result = example()
    result['events'].insert(0, {'step': 0, 'action': 'finish', 'arguments': {'model': 'old.py'},
        'feedback': {'fresh_confirmation': {'passed': False, 'mse': .1,
            'observation_indices': list(range(72, 168))}}})
    trace = experimental_design_trace(result)
    assert not trace['complete_discrimination']
    assert not trace['revisions'][0]['before_first_submission']
    assert trace['revisions'][0]['after_framework_confirmation']


def test_explicit_queries_remain_visible_without_formal_comparison():
    result = example()
    result['revision_proposals'] = []
    result['experiment_comparisons'] = []
    result['events'] = result['events'][1:]
    del result['events'][0]['feedback']['revision']
    trace = experimental_design_trace(result)
    assert not trace['complete_discrimination']
    assert trace['explicit_queries'][0]['classification'] == 'unclassified_explicit_query'
    assert trace['explicit_queries'][0]['paid_rows'] == 1


def test_frozen_actions_shorthand_joins_execution_evidence_once_before_submission():
    action = {'kind': 'trajectory', 'sequence': [{'u': .5}] * 24, 'channels': []}
    records = [{'action': action, 'row_indices': list(range(start, start + 24))}
               for start in range(72, 264, 24)]
    path = 'shared/consolidated_method.py'
    question = {'id': 'long_memory', 'question': 'Can long trajectories distinguish the rivals?'}
    confirmation = {'passed': True, 'mse': 0., 'rows': 96}
    result = {'model': 'model.py', 'fresh_confirmation': confirmation,
        'events': [
            {'step': -1, 'action': 'query', 'arguments': {'actions': [action]},
             'feedback': {'observations': [records[0]]},
             'usage': {'backend': 'common_paid_prefix'}},
            {'step': 0, 'action': 'query', 'arguments': {'path': path},
             'feedback': {'observations': records, 'observation_cost': 192, 'executed_path': path},
             'usage': {'backend': 'frozen_program_execution'}},
            {'step': 1, 'action': 'query', 'arguments': {'path': path},
             'feedback': {'fresh_confirmation': confirmation, 'method_submission': 'model.py'}}],
        'method_results': [
            {'step': 0, 'path': path, 'proposal': {'actions': [action] * 8, 'research_question': question},
             'feedback': {'observation_cost': 192, 'executed_path': path}},
            {'step': 1, 'path': path, 'proposal': {'model': 'model.py'},
             'feedback': {'fresh_confirmation': confirmation}}]}

    trace = experimental_design_trace(result)

    assert len(trace['explicit_queries']) == 1
    query = trace['explicit_queries'][0]
    assert query['paid_rows'] == 192
    assert len(query['actions']) == 8
    assert query['before_first_submission']
    assert query['step'] == 0
    assert query['classification'] == 'unclassified_explicit_query'
    assert trace['first_submission']['step'] == 1
    assert not trace['complete_discrimination']
    assert trace['interpretation'].startswith('complete_discrimination requires')


def test_nested_live_procedure_and_method_only_records_keep_the_same_chronology():
    result = example()
    original = experimental_design_trace(result)
    result['events'] = [{'step': 1, 'action': 'run', 'arguments': {'operator': 'procedure'},
                         'feedback': {'steps': deepcopy(result['events'])}}]
    assert experimental_design_trace(result) == original
    result['events'] = []
    assert experimental_design_trace(result) == original


def test_author_readout_after_the_final_submission_cannot_establish_the_sequence():
    result = example()
    acquisition = result['events'][1]
    acquisition['feedback']['revision'] = deepcopy(result['revision_proposals'][0])
    acquisition['feedback']['revision'].pop('analysis_is_agent_authored')
    result['events'].append({'step': 3, 'action': 'run',
        'arguments': {'operator': 'analyze_revision', 'revision_index': 0},
        'feedback': {'revision': result['revision_proposals'][0]}})
    assert not experimental_design_trace(result)['complete_discrimination']


def test_readout_after_failed_submission_is_not_pre_submission_discrimination():
    result = example()
    revision = result['revision_proposals'][0]
    result['events'][1]['feedback']['revision'] = deepcopy(revision)
    result['events'][1]['feedback']['revision'].pop('analysis_is_agent_authored')
    result['events'].insert(2, {'step': 2, 'action': 'finish', 'arguments': {'model': 'old.py'},
        'feedback': {'fresh_confirmation': {'passed': False, 'mse': .1}}})
    result['events'].insert(3, {'step': 3, 'action': 'run',
        'arguments': {'operator': 'analyze_revision', 'revision_index': 0},
        'feedback': {'revision': revision}})
    trace = experimental_design_trace(result)
    assert trace['revisions'][0]['before_first_submission']
    assert trace['revisions'][0]['followed_by_accepted_model']
    assert not trace['revisions'][0]['readout_before_first_submission']
    assert not trace['complete_discrimination']


def test_child_deferred_model_leaves_parent_free_to_design_experiment_before_real_submission():
    result = example()
    result['events'].insert(0, {'step': 0, 'action': 'run',
        'arguments': {'operator': 'procedure', 'path': 'child.py', 'confirm': True},
        'feedback': {'status': 'submission_proposed', 'steps': [
            {'step': 0, 'proposal': {'model': 'child_candidate.py'},
             'feedback': {'status': 'confirmation_not_authorized', 'next_action': {
                 'action': 'finish', 'arguments': {'model': 'child_candidate.py'}}}}]}})
    trace = experimental_design_trace(result)
    assert trace['complete_discrimination']
    assert trace['first_submission']['model'] == 'new.py'
    assert trace['first_submission']['position'] == 3
    assert trace['deferred_model_proposals'] == [{'position': 0, 'step': 0,
        'model': 'child_candidate.py', 'status': 'confirmation_not_authorized'}]
    assert not trace['revisions'][0]['after_framework_confirmation']


def test_existing_frozen_online_method_produces_the_reported_chain(tmp_path):
    from hegel_machine.cognition.anonymous_method_learning import AnonymousPreparedMethodAgent
    from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
    from test_anonymous_method_online_tools import METHOD
    from test_common_archive_observable_research import MeasuredLab, PARENT
    from test_common_archive_online_research import READOUT
    from test_inquiry_revision import action

    agent = AnonymousPreparedMethodAgent(None, tmp_path, arm='hegel', source_task=False,
        config=InquiryLLMConfig(observation_budget=20, confirmation_rows=3,
                               execution_seconds=30., compute_budget=1000))
    lab = MeasuredLab()
    agent._execute('query', {'actions': [action(0., channels=['meter'])]}, lab)
    for name, slope in [('a.py', 2), ('b.py', 3)]:
        agent._execute('write', {'path': name, 'code': f'def predict(row): return {slope}*row["features"]["u"]+1.\n'}, lab)
    agent._execute('write', {'path': 'parent.py', 'code': PARENT}, lab)
    agent._execute('write', {'path': 'readout.py', 'code': READOUT}, lab)
    agent.bind_method(METHOD)
    result = agent.execute_method(lab, max_steps=4)
    trace = experimental_design_trace(result)
    assert trace['complete_discrimination']
    assert trace['revisions'][0]['target'] == 'meter'
    assert trace['revisions'][0]['prior_fit'][0]['mse'] == 0.
    assert trace['revisions'][0]['measured_outcome'] == 'supported_and_refuted'
    assert len(trace['revisions']) == 1
