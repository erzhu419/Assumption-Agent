"""The new method carrier uses ordinary Common, not the native DAG replayer."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import finite_native_author as author
from hegel_machine.cognition import theory_consolidation as library
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.research_programs import export_research_selection_program
from hegel_machine.cognition.temporal_paired_method import execute_procedure
from test_inquiry_revision import PublicLab, action, make_agent


def registered(agent, *, plan=None, max_steps=1):
    body = author.export_finite_native_author() + '''
def select_actions(public_spec, observations, memory):
    return {'action':'query','arguments':{'actions':[
        {'kind':'trajectory','sequence':[{'u':0.}]}]}}
'''
    agent._save('ordinary.py', body)
    plan = {**(plan or {}), 'ordinary_method_path': 'ordinary.py'}
    return library.install_python_research_method(agent, candidate_id='local:P1',
        source=export_research_selection_program(plan), selection_plan=plan,
        source_refs=[{'file': 'paid.json', 'json_path': [0]}], max_steps=max_steps)


def test_T12_python_and_native_carriers_have_explicit_different_consumers(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, cls=CommonArchiveInquiryAgent)
    candidate = registered(agent)
    called = []
    monkeypatch.setattr(library, '_transfer_research_program',
        lambda *args: called.append('native') or {'status': 'native_replayed'})
    python_result = library.transfer_theory(agent, {'candidate_id': candidate['id']})
    assert python_result['execute_action']['arguments']['operator'] == 'procedure'
    assert called == []
    agent.theory_consolidation['candidates']['old'] = {'kind': 'research_program'}
    library._save(agent)
    assert library.transfer_theory(agent, {'candidate_id': 'old'})['status'] == 'native_replayed'
    assert called == ['native']
    assert candidate['source_ids'] == [] and not candidate['world_support_transferred']
    assert agent.knowledge['manifest']['research_programs'][0]['candidate_kind'] == 'python_research_method'


def test_T13_ordinary_common_invocation_consumes_actual_changed_action_and_returns_feedback(
        tmp_path, monkeypatch):
    agent, lab = make_agent(tmp_path, cls=CommonArchiveInquiryAgent, compute=10000), PublicLab()
    question = {'id': 'task', 'meaning_version': 1,
        'question': 'Which measured public action distinguishes the retained hypotheses?'}
    agent.knowledge['finite_native_author'] = {'question': question, 'predictor_paths': []}
    proposal = {'action': 'query', 'research_question': question, 'arguments': {'actions': [action(.5)],
        'after_observation': [{'operator': 'fixture_boundary_readout', 'records': 'acquired'}]}}
    option = {'family': 'Q', 'proposal': proposal, 'question_ref': question,
        'predicted_separation': 1., 'proposal_ref': {'file': 'original_proposal.json'},
        'complete_quote': {'status': 'available', 'compute_cost': 1, 'observation_cost': 1,
            'paid_rows': 0, 'retained_pool_models': 0, 'max_exported_models': 0,
            'remaining_execution_seconds': 30.}}
    monkeypatch.setattr(author, 'publish_research_options', lambda *args: [deepcopy(option)])
    candidate = registered(agent, plan={'first_segment': author.research_action_semantics(proposal)})
    received, original = [], agent._execute
    def execute(kind, arguments, laboratory):
        if kind == 'run' and arguments.get('operator') == 'fixture_boundary_readout':
            received.extend(arguments['records'])
            return {'status': 'completed', 'operator': 'fixture_boundary_readout', 'compute_cost': 0}
        return original(kind, arguments, laboratory)
    monkeypatch.setattr(agent, '_execute', execute)
    result = execute_procedure(agent, lab, candidate['execution']['path'], max_steps=1,
        allow_confirmation=False)
    assert result['status'] != 'method_error', result['steps']
    assert lab.actions[0]['sequence'] == [{'u': .5}]
    assert received == [0]
    group = agent.theory_consolidation['method_feedback'][candidate['id']]
    assert len(group['outcome_refs']) == 1
    record = json.loads((tmp_path / group['outcome_refs'][0]['file']).read_text())
    assert record['execution_carrier'] == 'python_select_actions'
    assert record['origin_ref']['json_path'] == ['method_results', 0]
    assert record['costs']['observation_cost'] == 1
    assert any(q.get('kind') == 'method_use' for q in agent.questions)


def test_install_retains_existing_pool_and_does_not_borrow_native_support(tmp_path):
    agent = make_agent(tmp_path, cls=CommonArchiveInquiryAgent)
    agent.theory_consolidation.setdefault('candidates', {})['old'] = {
        'kind': 'research_program', 'id': 'old', 'version': 2, 'source_ids': ['a', 'b']}
    before = deepcopy(agent.theory_consolidation['candidates']['old'])
    candidate = registered(agent)
    assert agent.theory_consolidation['candidates']['old'] == before
    assert candidate['paired_support_refs'] == []
    assert not candidate['scope']['general_guard_supported']
    assert registered(agent) == candidate
    with pytest.raises(ValueError, match='new candidate id'):
        library.install_python_research_method(agent, candidate_id=candidate['id'],
            source=export_research_selection_program(candidate['selection_plan']),
            selection_plan=candidate['selection_plan'], source_refs=[])


def test_partial_role_common_route_does_not_invent_roles_for_scalar_u(tmp_path):
    agent, lab = make_agent(tmp_path, cls=CommonArchiveInquiryAgent), PublicLab()
    result = agent._execute('run', {'operator': 'partial_role_experiment',
        'allocation': {'compute': 100}}, lab)
    assert result['status'] == 'UNKNOWN'
    assert 'model.focus_control' in result['required']
    assert agent.costs['compute_cost'] == result['compute_cost'] == 0
    assert lab.paid == 0


def test_native_dispatch_timing_excludes_nested_external_wait_and_retains_failed_call(tmp_path, monkeypatch):
    from hegel_machine.cognition import common_archive_inquiry as common
    from scripts.run_method_action_pairing import measured_execution_timing

    agent = make_agent(tmp_path, cls=CommonArchiveInquiryAgent)
    stamps = iter([10., 15.])
    monkeypatch.setattr(common, 'monotonic', lambda: next(stamps))
    def interrupted(*args):
        from hegel_machine.cognition.proposers import _EXECUTION_TIMING
        _EXECUTION_TIMING.get()['direct_external_python_seconds'] += 1.
        raise ValueError('retained execution failure')
    monkeypatch.setattr(agent, '_execute_budgeted_action', interrupted)
    with measured_execution_timing() as meter:
        with pytest.raises(ValueError, match='retained execution failure'):
            agent._execute('run', {'operator': 'native_condition_research'}, PublicLab())
        assert meter['native_dispatch_seconds'] == 5.
        assert meter['native_host_seconds'] == 4.
        assert meter['direct_external_python_seconds'] == 1.
        assert meter['native_calls'] == 1


def test_T22_common_method_question_closes_finite_no_alternative_without_installing(tmp_path):
    agent = make_agent(tmp_path, 'hegel', cls=CommonArchiveInquiryAgent, compute=10000)
    original = {'action': 'query', 'arguments': {'actions': [action(.5)]}}
    agent._save('actual_call.json', json.dumps({'method_results': [{
        'path': 'ordinary.py', 'proposal': original, 'feedback': {'status': 'completed'},
        'question_ref': {'question_id': 'task', 'meaning_version': 1}}]}))
    agent._save('ordinary.py', author.export_finite_native_author())
    agent._save('whole_return.json', json.dumps({'execution_complete': True,
        'gain': .2, 'observation_cost': 1, 'compute_cost': 5}))
    agent._save('public_profile.json', json.dumps({'public_phase': 'research'}))
    agent.theory_consolidation['candidates'] = {}
    method = agent._open_question({'id': 'choose_method', 'kind': 'method_use',
        'question': 'Which complete research action should serve the unresolved parent?',
        'parent_id': 'task', 'answer_type': 'research_action',
        'method_use_plan': {'candidate_id': 'old_method', 'binding': {}},
        'mission': {'purpose': 'method_learning', 'allocation': {'compute': 1000}}})
    source = {'file': 'actual_call.json', 'json_path': ['method_results', 0]}
    result = agent._execute('run', {'operator': 'research_route',
        'research_question': {'id': method['id']}, 'allocation': {'compute': 1000},
        'method_action_revision': {'ordinary_method_path': 'ordinary.py',
            'ordinary_proposal_ref': {**source, 'json_path': [*source['json_path'], 'proposal']},
            'segments': [{'root_result_ref': source, 'decision_start_ref': source}],
            'segment_return_refs': [{'segment_index': 0,
                'actual_return_ref': {'file': 'whole_return.json', 'json_path': []},
                'profile_ref': {'file': 'public_profile.json', 'json_path': []},
                'source_task_ref': {'file': 'actual_call.json', 'json_path': []}}]}}, PublicLab())
    assert result['status'] == 'complete_no_distinct_legal_alternative'
    assert result['method_action_research']['installed_candidate'] is None
    assert result['method_action_research']['proposals']['baseline']['id'] == 'P0'
    assert agent.theory_consolidation['candidates'] == {}
    assert agent.costs['observation_cost'] == 0
    assert result['next_action'] is None
    observed, = result['method_action_research']['sequence_observations']
    assert observed['status'] == 'measured_segment'
    question = next(item for item in agent.questions if item['id'] == method['id'])
    learned, = question['method_sequence_experience']['methods']
    assert learned['step_outcomes'] == []
    assert learned['sequence_outcomes'][0]['gain'] == .2


def test_unvalidated_general_guard_cannot_be_registered_as_supported(tmp_path):
    agent = make_agent(tmp_path, cls=CommonArchiveInquiryAgent)
    agent._save('ordinary.py', author.export_finite_native_author())
    plan = {'ordinary_method_path': 'ordinary.py', 'condition': {'op': 'const', 'value': 1.},
        'when_true': {}, 'when_false': {}}
    with pytest.raises(ValueError, match='group-validation'):
        library.install_python_research_method(agent, candidate_id='unsupported_guard',
            source=export_research_selection_program(plan), selection_plan=plan, source_refs=[])


def test_actual_ordinary_return_reopens_method_selection_without_native_fallthrough_or_fake_pair(
        tmp_path, monkeypatch):
    from hegel_machine.cognition.method_feedback import begin_method_feedback, record_python_method_feedback
    from hegel_machine.cognition.models import Observation

    agent, lab = make_agent(tmp_path, 'hegel', cls=CommonArchiveInquiryAgent, compute=10000), PublicLab()
    candidate = registered(agent)
    agent.rows = [Observation({'u': 0.}, 0.)]
    mission = {'purpose': 'method_learning', 'allocation': {'compute': 1000}}
    agent._open_question({'id': 'task', 'question': 'Research public response', 'mission': mission})
    owner = agent._open_question({'id': 'method_root', 'kind': 'method_use', 'parent_id': 'task',
        'question': 'Which action policy helps the unresolved parent?', 'mission': mission,
        'method_use_plan': {'candidate_id': candidate['id'], 'binding': {}}})
    candidate['method_question_ref'] = {'question_id': owner['id'], 'meaning_version': 1}
    agent._active_question_id = 'task'
    context = begin_method_feedback(agent, lab, candidate=candidate)
    binding = {'candidate_id': candidate['id'], 'candidate_version': candidate['version'],
        'origin_ref': {'file': 'actual_invocation.json', 'json_path': []}}
    monkeypatch.setattr(author, 'publish_research_options', lambda *args: [])
    def forbidden(*args, **kwargs):
        raise AssertionError('Python action feedback must not fall through to native-only consolidation')
    monkeypatch.setattr(library, 'consolidate_research', forbidden)
    returned = {'status': 'time_exhausted', 'execution_complete': False,
        'costs': {'compute_cost': 8}, 'cost_refs': [{'file': 'original_bill.json'}]}
    first = record_python_method_feedback(agent, candidate, binding, returned, context)
    def review(identifier):
        return agent._execute('run', {'operator': 'research_route',
            'research_question': {'id': identifier}, 'allocation': {'compute': 1000}}, lab)
    before = review(owner['id'])
    witness, = before['method_action_research']['ordinary_returns']
    assert witness['delivery'] == 'resource_non_delivery' and witness['costs']['compute_cost'] == 8
    assert before['method_action_research']['relative_labels_created'] == 0
    assert before['method_action_research']['selected_research_action'] == 'ordinary_research'
    assert before['next_action'] is None
    assert review(owner['id'])['cache_hit'] is True
    followup = first['method_question_refs'][0]['question_id']
    assert review(followup)['method_action_research']['ordinary_returns'][0]['feedback_ref'] == first['feedback_ref']
    second = record_python_method_feedback(agent, candidate, binding,
        {**returned, 'status': 'completed', 'execution_complete': True}, context)
    assert second['feedback_ref'] == first['feedback_ref']
    after = review(owner['id'])
    assert after['cache_hit'] is False
    assert after['method_action_research']['ordinary_returns'][0]['delivery'] == 'delivered'
    assert len(agent.theory_consolidation['method_feedback'][candidate['id']]['outcome_refs']) == 1
    assert lab.paid == 0


def test_two_steps_of_one_ordinary_procedure_update_one_invocation_origin(tmp_path, monkeypatch):
    agent, lab = make_agent(tmp_path, 'hegel', cls=CommonArchiveInquiryAgent, compute=10000), PublicLab()
    lab.public_spec = lambda: {'controls': ['u'], 'action_kinds': ['trajectory'],
        'max_probe_length': 12, 'feature_bounds': {'u': [-1., 1.]}}
    question = {'id': 'task', 'meaning_version': 1, 'question': 'Distinguish public actions.'}
    agent.knowledge['finite_native_author'] = {'question': question, 'predictor_paths': []}
    proposals = [{'action': 'query', 'research_question': question,
        'arguments': {'actions': [action(value)], 'after_observation': [
            {'operator': 'fixture_boundary_readout', 'records': 'acquired'}]}} for value in (.5, .6)]
    def options(*args):
        return [{'family': 'Q', 'proposal': proposal, 'question_ref': question,
            'predicted_separation': 1., 'proposal_ref': {'file': 'original.json', 'json_path': [index]},
            'complete_quote': {'status': 'available', 'compute_cost': 1, 'observation_cost': 1,
                'paid_rows': len(agent.rows), 'retained_pool_models': 0, 'max_exported_models': 0,
                'remaining_execution_seconds': 30.}} for index, proposal in enumerate(proposals)]
    monkeypatch.setattr(author, 'publish_research_options', options)
    candidate = registered(agent, max_steps=2, plan={
        'first_segment': author.research_action_semantics(proposals[0]),
        'second_segment': author.research_action_semantics(proposals[1])})
    original = agent._execute
    def execute(kind, arguments, laboratory):
        if kind == 'run' and arguments.get('operator') == 'fixture_boundary_readout':
            return {'status': 'completed', 'operator': 'fixture_boundary_readout'}
        return original(kind, arguments, laboratory)
    monkeypatch.setattr(agent, '_execute', execute)
    result = agent._execute('run', {'operator': 'procedure', 'path': candidate['execution']['path'],
        'max_steps': 2, 'confirm': False}, lab)
    assert result['status'] == 'method_step_limit'
    assert [item['sequence'][0]['u'] for item in lab.actions] == [.5, .6]
    refs = agent.theory_consolidation['method_feedback'][candidate['id']]['outcome_refs']
    assert len(refs) == 1
    record = json.loads((tmp_path / refs[0]['file']).read_text())
    assert record['origin_ref']['json_path'] == ['method_results', 0]
    assert record['costs']['observation_cost'] == 2
    assert record['invocation_binding']['paid_cutoff'] == {'rows': 0, 'records': 0}
    agent._save('predictor.py', "def predict(row): return row['features']['u']\n")
    comparison = agent._execute('run', {'operator': 'research_choice', 'models': ['predictor.py'],
        'research_question': {'id': 'task'}}, lab)
    agent._save('actual_comparison.json', json.dumps(comparison))
    agent._save('actual_delivery.json', json.dumps({'status': 'completed', 'execution_complete': True,
        'costs': deepcopy(agent.costs)}))
    comparison_ref = {'file': 'actual_comparison.json', 'json_path': []}
    finalized = agent._execute('run', {'operator': 'method_action_feedback',
        'candidate_id': candidate['id'], 'binding': record['invocation_binding'],
        'result_ref': {'file': 'actual_delivery.json', 'json_path': []},
        'comparison_ref': comparison_ref}, lab)
    updated = json.loads((tmp_path / finalized['feedback_ref']['file']).read_text())
    assert updated['execution_complete'] is True
    assert updated['complete_comparison_ref'] == comparison_ref
    assert len(agent.theory_consolidation['method_feedback'][candidate['id']]['outcome_refs']) == 1


@pytest.mark.parametrize('changed_body', [False, True])
def test_paid_pair_consumes_exact_frozen_body_and_preserves_its_identity(tmp_path, changed_body):
    from hegel_machine.cognition.method_action_research import research_method_actions
    from test_v8_paired_method_conditions import pair

    agent = make_agent(tmp_path, 'hegel', cls=CommonArchiveInquiryAgent, compute=10000)
    candidate = registered(agent, plan={'first_segment': author.research_action_semantics(
        {'action': 'query', 'arguments': {'actions': [action(.5)]}})})
    def save(name, value):
        agent._save(name, json.dumps(value))
        return {'file': name, 'json_path': []}
    source = save('source.json', {'method_results': [{'path': 'ordinary.py',
        'proposal': {'action': 'query', 'arguments': {'actions': [action(0.)]}},
        'feedback': {'status': 'completed'}}]})
    root = {**source, 'json_path': ['method_results', 0]}
    tested_ref = save('tested_body.json', candidate)
    actual_body = deepcopy(candidate)
    if changed_body:
        actual_body['selection_plan']['first_segment'] = {'action': 'different'}
    actual_ref = save('actual_body.json', actual_body)
    controlled = pair(0)
    profile_ref = save('profile.json', controlled['baseline']['profile'])
    contracts = {name: save(name + '.json', value)
                 for name, value in controlled['baseline']['basis'].items()}
    sides = {}
    for side in ('baseline', 'challenger'):
        outcome = {key: value for key, value in controlled[side].items() if key not in {'basis', 'profile'}}
        outcome.update(profile_ref=profile_ref, method_artifact_ref=actual_ref,
            method_ref={'id': candidate['id'], 'version': candidate['version']},
            **{name + '_ref': reference for name, reference in contracts.items()})
        sides[side + '_ref'] = save(side + '.json', outcome)
    pair_ref = save('pair.json', {**sides, 'data_role': 'development',
        'physical_source_ref': {'file': 'one_source'},
        'baseline_action_content': {'operator': 'refit'},
        'challenger_action_content': {'action': 'query'}})
    question = agent._open_question({'id': 'method_problem', 'kind': 'method_use',
        'question': 'Which real research step should serve this parent?', 'parent_id': 'task',
        'method_use_plan': {'candidate_id': 'old', 'binding': {}}})
    result = research_method_actions(agent, question, {'ordinary_method_path': 'ordinary.py',
        'ordinary_proposal_ref': {**root, 'json_path': [*root['json_path'], 'proposal']},
        'segments': [{'root_result_ref': root, 'decision_start_ref': root}],
        'tested_candidate_ref': tested_ref, 'paired_candidate_id': candidate['id'], 'pair_refs': [pair_ref]},
        public_spec=PublicLab().public_spec(), compute_limit=1000)
    if changed_body:
        assert result['reason'] == 'paired_body_content_mismatch'
        assert result['installed_candidate'] is None
    else:
        assert result['status'] == 'local_method_action_revision_selected'
        assert result['installed_candidate'] == candidate
        assert result['next_action']['arguments']['research_question'] == {'id': 'task', 'meaning_version': 1}
        assert agent.theory_consolidation['candidates'][candidate['id']] == candidate
        assert agent.theory_consolidation['method_feedback'][candidate['id']]['paired_selections']
