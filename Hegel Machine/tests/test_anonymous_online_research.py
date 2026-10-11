"""Paid-boundary replay checks; scripted providers are not discovery evidence."""

from copy import deepcopy
import json
from random import Random

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from scripts.run_anonymous_method_revision import COUNTEREXAMPLE
from scripts.run_anonymous_online_research import (
    RESEARCH_TASK, prepare_paid_prefix, protocol, research_trace, run_online,
    representation_protocol, conditional_protocol)
from test_anonymous_method_learning import memoryless_source


def saved_trial(path):
    lab = AnonymousHistoryLab(kind=COUNTEREXAMPLE['kind'], seed=COUNTEREXAMPLE['seed'])
    prefix = lab.confirmation_actions(72, Random(COUNTEREXAMPLE['prefix_seed']))
    first = lab.confirmation_actions(96, Random(COUNTEREXAMPLE['confirmation_seed']))
    records = [lab.execute(action) for action in [*prefix, *first]]
    later = lab.execute({'kind': 'trajectory', 'sequence': [{'u': 0.0}], 'channels': []})
    later['rows'][0]['target'] = 987654321
    path.write_text(json.dumps({'paid_observations': [*records, later],
        'model': 'REJECTED_SECRET_MODEL.py',
        'events': [{'future_diagnosis': 'DO_NOT_IMPORT_THIS_TRACE'}]}))
    return json.loads(path.read_text())['paid_observations'][:7]


def test_restore_actual_boundaries_without_physics_and_continue_confirmation_stream(tmp_path, monkeypatch):
    source = tmp_path / 'trial.json'
    records = saved_trial(source)
    prepare = prepare_paid_prefix(source)
    def cannot_buy(*args, **kwargs):
        raise AssertionError('restoration must not buy or regenerate observations')
    monkeypatch.setattr(AnonymousHistoryLab, 'execute', cannot_buy)
    streams = {}
    for count in (72, 168):
        for arm in ('code_react', 'hegel'):
            agent = CommonArchiveInquiryAgent(None, tmp_path / f'{count}_{arm}', arm=arm,
                confirmation_seed=COUNTEREXAMPLE['confirmation_seed'], source_task=False)
            lab = AnonymousHistoryLab(kind=COUNTEREXAMPLE['kind'], seed=COUNTEREXAMPLE['seed'])
            account = prepare(agent, lab, {'paid_prefix_rows': count})
            assert agent.observations == records[:3 if count == 72 else 7]
            assert len(agent.rows) == agent.costs['observation_cost'] == lab.costs['observation_cost'] == count
            assert agent.costs['model_calls'] == agent.costs['compute_cost'] == 0
            assert lab.costs['queries'] == (3 if count == 72 else 7)
            assert account['historical_prefix_cost']['observation_cost'] == count
            assert account['prefix_origin']['new_purchases'] == 0
            assert agent.events[-1]['feedback']['observation_cost'] == 0
            assert RESEARCH_TASK in agent.common_interface
            public = json.dumps(agent.observations) + json.dumps(agent.events)
            assert '987654321' not in public
            assert 'REJECTED_SECRET_MODEL' not in public
            assert 'DO_NOT_IMPORT_THIS_TRACE' not in public
            expected_rng = Random(COUNTEREXAMPLE['confirmation_seed'])
            if count == 168:
                lab.confirmation_actions(96, expected_rng)
            next_actions = lab.confirmation_actions(96, agent.confirmation_rng)
            assert next_actions == lab.confirmation_actions(96, expected_rng)
            streams[count, arm] = next_actions
    assert streams[72, 'code_react'] == streams[72, 'hegel']
    assert streams[168, 'code_react'] == streams[168, 'hegel']
    assert streams[72, 'hegel'] != streams[168, 'hegel']


def test_four_online_trials_keep_same_archive_and_separate_inherited_costs(tmp_path, memoryless_source, monkeypatch):
    source = tmp_path / 'trial.json'
    records = saved_trial(source)
    plan = protocol(source, memoryless_source)
    plan['limits']['max_model_calls'] = 1
    requests = []
    class InspectProvider:
        def next_action(self, request):
            requests.append(deepcopy(request))
            return {'action': 'run', 'arguments': {'code': "print('fixture only')"},
                    'usage': {'backend': 'scripted_interface_test', 'total_tokens': 1}}
    def cannot_buy(*args, **kwargs):
        raise AssertionError('this fixture has no acquisition action')
    monkeypatch.setattr(AnonymousHistoryLab, 'execute', cannot_buy)

    output = tmp_path / 'online'
    summary = run_online(plan, output, provider_factory=lambda _: InspectProvider())

    assert len(summary['trials']) == len(requests) == 4
    assert summary['evidence_kind'] == 'scripted_interface_check'
    assert all(request['session_knowledge'] == requests[0]['session_knowledge'] for request in requests)
    assert [len(request['observations']) for request in requests] == [72, 72, 168, 168]
    assert all(RESEARCH_TASK in request['task_interface'] for request in requests)
    for trial, request in zip(summary['trials'], requests):
        count = len(request['observations'])
        assert trial['historical_prefix_cost']['observation_cost'] == count
        assert trial['costs']['observation_cost'] == count
        assert trial['additional_cost']['observation_cost'] == 0
        assert trial['additional_cost']['model_calls'] == trial['additional_cost']['reported_tokens'] == 1
        assert trial['terminal']['status'] == 'no_submitted_model'
        assert '987654321' not in json.dumps(request)
        assert 'REJECTED_SECRET_MODEL' not in json.dumps(request)
        assert 'DO_NOT_IMPORT_THIS_TRACE' not in json.dumps(request)
    assert (output / 'source_at_run/scripts/run_anonymous_online_research.py').is_file()
    assert records[3]['rows'] == requests[2]['laboratory_observations'][3]['rows']


def test_representation_pairs_buy_short_prefixes_with_equal_tools_and_no_candidate_menu(
        tmp_path, memoryless_source):
    import pytest
    plan = representation_protocol(memoryless_source)
    plan['limits']['max_model_calls'] = 1
    requests = []
    class InspectProvider:
        def next_action(self, request):
            requests.append(deepcopy(request))
            return {'action': 'run', 'arguments': {'code': "print('fixture only')"},
                    'usage': {'backend': 'scripted_interface_test', 'total_tokens': 1}}
    summary = run_online(plan, tmp_path / 'pairs', provider_factory=lambda _: InspectProvider())
    assert len(summary['trials']) == len(requests) == 8
    assert summary['independent_parameter_pairs'] == 2
    assert summary['evidence_kind'] == 'scripted_interface_check'
    for request, trial in zip(requests, summary['trials']):
        assert len(request['observations']) == trial['costs']['observation_cost'] == 72
        assert max(len(row['history']) for row in request['observations']) == 2
        assert len(request['laboratory_observations']) == 24
        assert trial['costs']['measurement_cost'] == 0
        assert trial['post_prefix_cost']['observation_cost'] == 0
        assert trial['post_prefix_cost']['model_calls'] == 1
        assert 'historical_prefix_cost' not in trial
        assert request['capabilities']['max_probe_length'] == 24
        assert request['capabilities'] == requests[0]['capabilities']
        assert request['session_knowledge'] == requests[0]['session_knowledge']
        assert RESEARCH_TASK in request['task_interface']
        public = json.dumps(request)
        for private in ('finite2', 'AmbiguousInstrumentLab', 'apparatus_seed', 'parameter_pair'):
            assert private not in public
    for start in (0, 4):
        for request in requests[start:start + 4]:
            assert [row['features'] for row in request['observations']] == [
                row['features'] for row in requests[start]['observations']]
            assert [row['target'] for row in request['observations']] == pytest.approx(
                [row['target'] for row in requests[start]['observations']], abs=1e-12)
    assert requests[0]['observations'] != requests[4]['observations']


def test_conditional_cohort_starts_fresh_with_shared_tools_and_no_rescue_science(
        tmp_path, memoryless_source):
    plan = conditional_protocol(memoryless_source)
    old = representation_protocol(memoryless_source)
    assert {case['seed'] for case in plan['cases']} == {71221031, 71222040}
    assert not ({case['seed'] for case in old['cases']}
                & {case['seed'] for case in plan['cases']})
    assert [case['arms'] for case in plan['cases']] == [
        ['code_react', 'hegel'], ['hegel', 'code_react']]
    assert all(case['kind'] == 'recurrent' for case in plan['cases'])
    assert plan['limits'] == dict(max_model_calls=96, token_budget=10_000_000,
        execution_seconds=600., observation_budget=576, confirmation_rows=96,
        compute_budget=250_000)
    assert plan['provider']['timeout_seconds'] == 600.
    assert plan['execution_workers'] == 2
    assert plan['shared_research_task'] == RESEARCH_TASK
    assert not {'source_trial', 'source_workspace', 'resume_study'} & plan.keys()
    assert plan['source_archive'] == str(memoryless_source.resolve())
    for module in ('conditional_research', 'observable_research'):
        assert f'src/hegel_machine/cognition/{module}.py' in plan['runtime_snapshot']

    plan['limits']['max_model_calls'] = 1
    requests = {}
    class InspectProvider:
        def __init__(self, workspace):
            self.key = (workspace.parent.name, workspace.name)
        def next_action(self, request):
            requests[self.key] = deepcopy(request)
            return {'action': 'run', 'arguments': {'code': "print('fixture only')"},
                    'usage': {'backend': 'scripted_interface_test', 'total_tokens': 1}}

    output = tmp_path / 'conditional'
    summary = run_online(plan, output, provider_factory=InspectProvider)
    assert summary['cohort'] == 'conditional_components_new_parameters'
    assert summary['independent_parameter_sets'] == 2
    assert 'independent_parameter_pairs' not in summary
    assert len(summary['trials']) == len(requests) == 4
    assert summary['evidence_kind'] == 'scripted_interface_check'
    for case in plan['cases']:
        code, hegel = [requests[case['id'], arm] for arm in ('code_react', 'hegel')]
        assert code['observations'] == hegel['observations']
        assert code['laboratory_observations'] == hegel['laboratory_observations']
        assert code['session_knowledge'] == hegel['session_knowledge']
        assert code['capabilities'] == hegel['capabilities']
        for request in (code, hegel):
            assert request['conditional_models'] == request['observable_models'] == {}
            assert request['model_dependencies'] == []
            assert request['research_phase'] == 'continuation'
            assert len(request['observations']) == 72
            assert len(request['laboratory_observations']) == 24
            assert all(not record['measurements'] for record in request['laboratory_observations'])
            assert request['capabilities']['max_probe_length'] == 24
            assert RESEARCH_TASK in request['task_interface']
            assert "evaluate {component:'chosen_id'" in request['task_interface']
            assert 'bind_models' in request['task_interface']
            assert 'no manual manifest' in request['task_interface']
            public = json.dumps(request)
            for private in ('apparatus_seed', 'parameter_pair', 'AmbiguousInstrumentLab',
                            'observable_dependency_rescue', '71221031', '71222040'):
                assert private not in public
    assert requests['case_000', 'hegel']['observations'] != requests['case_001', 'hegel']['observations']
    assert all(trial['costs']['observation_cost'] == 72
               and trial['costs']['measurement_cost'] == 0
               and trial['post_prefix_cost']['observation_cost'] == 0
               and trial['terminal']['status'] == 'no_submitted_model'
               for trial in summary['trials'])
    for module in ('conditional_research', 'observable_research'):
        assert (output / f'source_at_run/src/hegel_machine/cognition/{module}.py').is_file()


def test_trace_reports_preacquisition_disagreement_and_return_without_success_verdict():
    revision = {'index': 0, 'proposal': {'id': 'r0', 'affected_object': 'predictor'},
        'status': 'answered', 'evidence_cutoff': {'rows': 72, 'records': 3},
        'observations': [{'rows': [{'target': 1.}]}],
        'prior_fit': [{'model': 'a.py', 'mse': 0., 'returncode': 0},
                      {'model': 'b.py', 'mse': 0., 'returncode': 0}],
        'forecasts': [
            {'model': 'a.py', 'returncode': 0, 'predictions': [0., 1.], 'prospective_mse': 0.5},
            {'model': 'b.py', 'returncode': 0, 'predictions': [1., 1.], 'prospective_mse': 0.}],
        'authored_answer': {'status': 'refuted', 'answer': 'Authored finding', 'evidence': [72]}}
    result = {'revision_proposals': [revision],
        'online_questions': [{'id': 'child', 'parent_id': 'parent', 'revision_index': 0}],
        'events': [
            {'step': 3, 'action': 'query', 'feedback': {'revision': revision}},
            {'step': 4, 'action': 'write', 'arguments': {'path': 'new.py', 'code': 'new scientific source'}}]}
    trace = research_trace(result)['prospective_revisions'][0]
    assert trace['prior_fit'] == revision['prior_fit']
    assert trace['predictions_differ'] and trace['distinct_prediction_vectors'] == 2
    assert trace['forecasts'][0]['prospective_mse'] == 0.5
    assert trace['authored_answer']['status'] == 'refuted'
    assert trace['question']['parent_id'] == 'parent'
    assert trace['next_action_after_acquisition'] == {
        'step': 4, 'action': 'write', 'arguments': {'path': 'new.py'}}
    assert 'success' not in trace
    assert research_trace({'events': [], 'revision_proposals': []})['prospective_revisions'] == []


def test_delayed_readout_keeps_actual_decision_order_when_same_question_is_revised_again():
    acquired = {'index': 0, 'question_id': 'child',
        'proposal': {'id': 'r0', 'affected_object': 'predictor'},
        'observations': [{'rows': [{'target': 1.}]}], 'status': 'observed'}
    answer = {'status': 'refuted', 'answer': 'Old explanation fails', 'evidence': [72]}
    analyzed = {**deepcopy(acquired), 'status': 'answered', 'authored_answer': answer,
        'analysis_attempts': [{'path': 'readout.py', 'execution': {'returncode': 0},
                              'authored_answer': answer}]}
    revised = {'index': 1, 'question_id': 'child',
        'proposal': {'id': 'r1', 'affected_object': 'predictor'},
        'observations': [{'rows': [{'target': 0.}]}], 'status': 'answered'}
    result = {'revision_proposals': [analyzed, revised],
        'online_questions': [{'id': 'child', 'parent_id': 'parent', 'revision_index': 1}],
        'events': [
            {'step': 3, 'action': 'query', 'feedback': {'revision': acquired}},
            {'step': 4, 'action': 'write', 'arguments': {'path': 'readout.py', 'code': 'authored readout'}},
            {'step': 5, 'action': 'run', 'arguments': {'operator': 'analyze_revision', 'revision_index': 0},
             'feedback': {'revision': analyzed}},
            {'step': 6, 'action': 'evaluate', 'arguments': {'model': 'revised.py',
                                                         'research_question': {'id': 'parent'}}},
            {'step': 7, 'action': 'query', 'feedback': {'revision': revised}}]}
    first, second = research_trace(result)['prospective_revisions']
    assert first['question_id'] == second['question_id'] == 'child'
    assert first['question']['parent_id'] == 'parent'
    assert first['acquisition_step'] == 3
    assert first['next_action_after_acquisition']['step'] == 4
    assert first['analysis_events'] == [{'step': 5, 'procedure_step': None, 'attempt': 0,
        'readout': 'readout.py', 'execution': {'returncode': 0}, 'error': None,
        'authored_answer': answer, 'next_model_decision': {'step': 6, 'action': 'evaluate',
            'arguments': {'model': 'revised.py', 'research_question': {'id': 'parent'}}}}]
    assert second['acquisition_step'] == 7
    assert second['next_action_after_acquisition'] is None
