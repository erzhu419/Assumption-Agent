"""PR2 through Common/provider=None; toy formulas are interface regressions only.

PublicLab is synthetic. Routing must not acquire even its simulated observations
or turn finite candidate forecasts into independent scientific support.
"""

from copy import deepcopy

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.question_continuation import pending_query_arguments
from test_inquiry_revision import PublicLab, action


CONTEXT = {'scene': 'routing_fixture', 'output_unit': 'V', 'reset': 'per_trajectory'}
X = Program('feature', name='u')
SQUARE = Program('mul', children=(X, X))
ROLES = {'u': {'role': 'drive', 'unit': 'V', 'time': 'current'}}


def scene(tmp_path):
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm='hegel', source_task=False,
        config=InquiryLLMConfig(observation_budget=50, confirmation_rows=2,
            compute_budget=50000, execution_seconds=30., max_model_calls=0))
    lab = PublicLab()
    agent._execute('query', {'actions': [action(0., 1.)]}, lab)
    return agent, lab


def readout(level, index):
    return {'op': 'Readout', 'record_index': index, 'step': 0,
            'context_ref': deepcopy(CONTEXT), 'units': 'V', 'target': 'response',
            'binding': {'action': action(level), 'features': {'u': level}, 'history': []}}


def theory(identifier, program=X, **fields):
    return {'id': identifier, 'program': program.to_dict(),
            'role_signature': deepcopy(ROLES), 'output_unit': 'V',
            'context_ref': deepcopy(CONTEXT), 'declared_scope_ref': 'bounded_public_drive',
            'measured_coverage_ref': [], 'premise_refs': [], 'parameter_paths': [],
            'support': {'status': 'candidate', 'evidence': [], 'context_ref': deepcopy(CONTEXT)}, **fields}


def future_question(agent, identifier='future_response'):
    return {'id': identifier, 'question': 'Which candidate response applies to the unmeasured public input?',
            'parent_id': 'task', 'depends_on': [], 'kind': 'value', 'answer_type': 'number',
            'context_ref': deepcopy(CONTEXT), 'answer_resolution_ref': {'atol': 0.},
            'query_semantics': {'mode': 'model_query',
                'query_expression': readout(3., len(agent.observations))},
            'theory_target': theory('target'),
            'use_sites': [{'kind': 'choose_parent_experiment', 'parent_id': 'task'}]}


def probe(agent, level=2., identifier='distinguishing_probe'):
    return {'id': identifier, 'actions': [action(level)],
            'query': readout(level, len(agent.observations)),
            'tolerance': 0., 'observation_cost': 1}


def route(agent, lab, *, question=None, theories=None, experiments=None, mission=None):
    arguments = {'operator': 'research_route'}
    if question is not None:
        arguments['research_question'] = deepcopy(question)
    if theories is not None:
        arguments['theories'] = deepcopy(theories)
    if experiments is not None:
        arguments['experiments'] = deepcopy(experiments)
    if mission is not None:
        arguments['mission'] = deepcopy(mission)
    return agent._execute('run', arguments, lab)


def parent(agent):
    return next(q for q in agent.questions if q['id'] == 'task')


def native_round(program):
    return {'round_index': 0, 'status': 'search_incomplete', 'fit_evidence': [0, 1],
            'exported_models': [], 'candidates': [{'program': program.to_dict(),
                'research_program': {'graph': {'nodes': ['retained_graph_only']}}}]}


def assert_no_new_measurements(agent, lab, paid_before, actions_before):
    assert agent.costs['observation_cost'] == lab.paid == paid_before
    assert lab.actions == actions_before
    assert agent.costs['model_calls'] == 0 and not agent.confirmations and agent.model is None


def test_explicit_finite_question_pool_keeps_components_without_giving_them_a_vote(tmp_path):
    agent, lab = scene(tmp_path)
    current = future_question(agent)
    current['candidate_theory_ids'] = ['one', 'two']
    result = route(agent, lab, question=current,
        theories=[theory('one', X), theory('two', SQUARE),
                  theory('component', Program('lag', name='u', lag=3))],
        experiments=[probe(agent)])
    assert result['selection']['status'] != 'incomplete_predictions'
    assert {match['source_id'] for match in result['matching']['matches']} == {'one', 'two'}
    assert set(agent.theory_index) == {'one', 'two', 'component'}
    assert 'component' in agent.theory_index


def test_changing_finite_question_pool_invalidates_cached_comparison(tmp_path):
    agent, lab = scene(tmp_path)
    current = future_question(agent)
    current['candidate_theory_ids'] = ['one', 'two']
    route(agent, lab, question=current, theories=[theory('one', X), theory('two', SQUARE)],
          experiments=[probe(agent)])
    assert route(agent, lab, question=current)['cache_hit'] is True
    current['candidate_theory_ids'] = ['one']
    changed = route(agent, lab, question=current)
    assert changed['cache_hit'] is False
    assert len(changed['matching']['matches']) == 1


def test_missing_requested_candidate_leaves_question_unknown_without_dropping_it(tmp_path):
    agent, lab = scene(tmp_path)
    current = future_question(agent)
    current['candidate_theory_ids'] = ['one', 'unavailable']
    result = route(agent, lab, question=current, theories=[theory('one')],
                   experiments=[probe(agent)])
    assert result['status'] == 'scoped_unknown'
    assert result['missing_theory_ids'] == ['unavailable']
    assert set(agent.theory_index) == {'one'}


def test_native_and_relation_source_refs_reach_the_real_route_without_copying_banks(tmp_path):
    agent, lab = scene(tmp_path)
    native = agent._open_question({'id': 'native_bank', 'question': 'Retained native fixture',
        'native_condition_research': [native_round(X)]})
    agent._open_question({'id': 'relation_bank', 'question': 'Retained relation fixture'})
    agent.relation_research['relation_bank'] = {'round_index': 0, 'candidates': [
        {'program': SQUARE.to_dict(), 'graph': {'nodes': ['retained_relation_graph_only']}}]}
    before_banks = deepcopy((native, agent.relation_research))
    declarations = []
    for identifier, kind, owner in [('linear', 'native_candidate', 'native_bank'),
                                     ('quadratic', 'relation_candidate', 'relation_bank')]:
        declaration = theory(identifier)
        declaration.pop('program')
        declaration['source_ref'] = {'kind': kind, 'question_id': owner, 'round_index': 0,
                                     'candidate_index': 0, 'meaning_version': 1}
        declarations.append(declaration)
    paid, actions = lab.paid, deepcopy(lab.actions)
    result = route(agent, lab, question=future_question(agent), theories=declarations,
                   experiments=[probe(agent)])
    assert result['status'] == 'selected'
    assert [group['task_answer'] for group in result['selection']['groups']] == [3., 9.]
    assert all('program' not in entry and 'graph' not in entry
               for entry in agent.theory_index.values())
    assert all('source_ref' in entry for entry in agent.theory_index.values())
    assert (native, agent.relation_research) == before_banks
    assert_no_new_measurements(agent, lab, paid, actions)


def test_future_question_selects_legal_structural_alternatives_and_prepares_parent_only(tmp_path):
    agent, lab = scene(tmp_path)
    paid, actions = lab.paid, deepcopy(lab.actions)
    panel = [probe(agent)]
    result = route(agent, lab, question=future_question(agent),
                   theories=[theory('linear'), theory('quadratic', SQUARE)], experiments=panel)
    assert result['answer_execution']['status'] == 'scoped_unknown'
    assert result['answer_execution']['reason'] == 'unknown_due_to_missing_evidence'
    assert result['status'] == 'selected' and result['selection']['scores'][0]['G'] == 1.
    assert result['parent_use']['status'] == 'parent_query_prepared'
    assert pending_query_arguments(agent, 'task')['actions'] == panel[0]['actions']
    assert {tuple(group['theory_ids']) for group in result['selection']['groups']} == {('linear',), ('quadratic',)}
    assert result['selection']['support_updated'] is False and result['selection']['evidence'] == []
    assert_no_new_measurements(agent, lab, paid, actions)


def test_T28_same_route_reuses_completed_compute_and_does_not_run_python_again(tmp_path, monkeypatch):
    agent, lab = scene(tmp_path)
    route(agent, lab, question=future_question(agent), theories=[theory('linear'), theory('quadratic', SQUARE)],
          experiments=[probe(agent)])
    costs, pending = deepcopy(agent.costs), deepcopy(parent(agent)['pending_query'])

    def no_new_python(*args, **kwargs):
        raise AssertionError('An unchanged completed route must not rerun Python')

    monkeypatch.setattr(agent, '_python', no_new_python)
    result = route(agent, lab, question={'id': 'future_response'})
    assert result['cache_hit'] is True and result['new_compute_cost'] == 0
    assert agent.costs == costs and parent(agent)['pending_query'] == pending


@pytest.mark.parametrize('changed', ['plan_source', 'use_sites'])
def test_ordinary_PR1_pending_rejects_changed_plan_or_use_before_purchase(tmp_path, changed):
    from test_evidence_procedure_execution import prepared
    from test_question_continuation import paired_question, parent_plan

    agent, lab = prepared(tmp_path)
    owner = parent_plan(agent, lab)
    agent._execute('run', {'operator': 'research_step',
                          'research_question': paired_question(agent)}, lab)
    child = next(q for q in agent.questions if q['id'] == 'paired')
    meaning, answer = child['meaning_version'], deepcopy(child['semantic_answer'])
    old_pending = deepcopy(owner['pending_query'])
    original = (agent.workspace / 'shared/next.py').read_text()
    if changed == 'plan_source':
        agent._execute('write', {'path': 'shared/next.py',
            'code': original.replace("value=child['semantic_answer']['value']",
                                     "value=child['semantic_answer']['value'] + 3.")}, lab)
        expected_level = 5.
    else:
        agent._execute('write', {'path': 'shared/other_next.py', 'code': original}, lab)
        agent._open_question({'id': 'paired', 'use_sites': [
            {'kind': 'choose_parent_experiment', 'parent_id': 'parent', 'path': 'shared/other_next.py'}]})
        expected_level = 2.
    child = next(q for q in agent.questions if q['id'] == 'paired')
    # This is a changed consumer, not a changed measured answer or question.
    assert child['meaning_version'] == meaning and child['semantic_answer'] == answer
    paid, actions = lab.paid, deepcopy(lab.actions)
    with pytest.raises(ValueError, match='rebuilding'):
        agent._execute('query', {'pending_question': 'parent'}, lab)
    assert lab.paid == paid and lab.actions == actions
    assert owner['pending_query'] == old_pending
    rebuilt = agent._execute('run', {'operator': 'research_step',
                                    'research_question': {'id': 'paired'}}, lab)
    assert rebuilt['status'] == 'answer_applied'
    assert pending_query_arguments(agent, 'parent')['actions'] == [action(expected_level)]
    assert lab.paid == paid and lab.actions == actions and agent.costs['model_calls'] == 0


def test_changed_recursive_ancestor_meaning_invalidates_route_with_unchanged_direct_answer(tmp_path):
    agent, lab = scene(tmp_path)
    agent._open_question({'id': 'ancestor', 'question': 'Grounded ancestor fixture',
        'context_ref': {'scene': 'original'}, 'status': 'supported',
        'semantic_answer': {'status': 'answered', 'value': 1., 'version': 1}})
    agent._open_question({'id': 'direct_premise', 'question': 'Derived premise fixture',
        'status': 'supported', 'semantic_answer': {'status': 'answered', 'value': 1., 'version': 1},
        'support_paths': [[{'question_id': 'ancestor'}]]})
    question = future_question(agent)
    question.update(depends_on=['direct_premise'], support_paths=[[{'question_id': 'direct_premise'}]])
    selected = route(agent, lab, question=question,
        theories=[theory('linear'), theory('quadratic', SQUARE)], experiments=[probe(agent)])
    assert selected['status'] == 'selected'
    direct = next(q for q in agent.questions if q['id'] == 'direct_premise')
    unchanged_direct = deepcopy({key: direct[key] for key in ('meaning_version', 'semantic_answer', 'status')})
    agent._open_question({'id': 'ancestor', 'context_ref': {'scene': 'changed'}})
    assert {key: direct[key] for key in unchanged_direct} == unchanged_direct
    paid, actions = lab.paid, deepcopy(lab.actions)
    with pytest.raises(ValueError, match='rebuilding'):
        agent._execute('query', {'pending_question': 'task'}, lab)
    rerouted = route(agent, lab, question={'id': 'future_response'})
    assert rerouted['cache_hit'] is False
    assert rerouted['answer_execution']['status'] == 'scoped_unknown'
    assert rerouted['selection'] is None
    assert_no_new_measurements(agent, lab, paid, actions)


@pytest.mark.parametrize('changed', ['theory_index', 'experiment', 'dependency', 'query'])
def test_changed_exact_input_rejects_old_pending_then_allows_explicit_reselection(tmp_path, changed):
    agent, lab = scene(tmp_path)
    question = future_question(agent)
    dependency = agent._open_question({'id': 'calibration', 'question': 'Existing calibration',
        'status': 'supported', 'semantic_answer': {'status': 'answered', 'value': 1., 'version': 1}})
    question['depends_on'] = ['calibration']
    route(agent, lab, question=question, theories=[theory('linear'), theory('quadratic', SQUARE)],
          experiments=[probe(agent)])
    old = deepcopy(parent(agent)['pending_query_source'])
    declarations = None
    if changed == 'theory_index':
        declarations = [theory('quadratic', Program('add', children=(SQUARE, Program('const', value=1.))))]
        # Registration is itself a current index change; no helper selection is
        # substituted for the actual Common route exercised above and below.
        from hegel_machine.cognition.question_routing import register_theories
        register_theories(agent, declarations)
    elif changed == 'experiment':
        current = next(q for q in agent.questions if q['id'] == 'future_response')
        current['resolution_experiments'] = [probe(agent, 4., 'revised_probe')]
    elif changed == 'query':
        agent._open_question({'id': 'future_response', 'query_semantics': {
            'mode': 'model_query', 'query_expression': readout(4., len(agent.observations))}})
    else:
        dependency['semantic_answer']['version'] += 1
    paid = lab.paid
    with pytest.raises(ValueError, match='rebuilding'):
        agent._execute('query', {'pending_question': 'task'}, lab)
    assert lab.paid == paid
    result = route(agent, lab, question={'id': 'future_response'},
                   experiments=[probe(agent)] if changed == 'query' else None)
    assert result['cache_hit'] is False and result['status'] == 'selected'
    updated = parent(agent)['pending_query_source']
    if changed == 'query':
        assert updated['meaning_version'] > old['meaning_version']
    else:
        assert updated['routing_version'] > old['routing_version']
    assert pending_query_arguments(agent, 'task')['actions'] == (
        [action(4.)] if changed == 'experiment' else [action(2.)])
    assert lab.paid == paid and agent.costs['model_calls'] == 0


@pytest.mark.parametrize('changed', ['output_unit', 'input_unit', 'role'])
def test_unbound_units_or_roles_do_not_enter_selection_but_legal_other_structure_can(tmp_path, changed):
    agent, lab = scene(tmp_path)
    incompatible = theory('quadratic', SQUARE)
    if changed == 'output_unit':
        incompatible['output_unit'] = 'mV'
    elif changed == 'input_unit':
        incompatible['role_signature']['u']['unit'] = 'mV'
    else:
        incompatible['role_signature']['u']['role'] = 'another_physical_quantity'
    result = route(agent, lab, question=future_question(agent),
                   theories=[theory('linear'), incompatible], experiments=[probe(agent)])
    assert result['selection']['selected_experiment'] is None
    assert 'pending_query' not in parent(agent)
    assert any(item['id'] == 'quadratic' for item in result['binding_work'])
    assert all('quadratic' not in group['theory_ids'] for group in result['selection']['groups'])
    updated = route(agent, lab, question={'id': 'future_response'}, theories=[theory('quadratic', SQUARE)])
    assert updated['status'] == 'selected' and updated['selection']['scores'][0]['G'] == 1.
    assert agent.theory_index['quadratic']['version'] == 2
    assert agent.theory_index['quadratic']['history'][0]['role_signature'] == incompatible['role_signature']
    assert agent.costs['model_calls'] == 0


def test_hundred_clone_sources_leave_route_G_unchanged_and_all_ids_retained(tmp_path):
    agent, lab = scene(tmp_path)
    base = route(agent, lab, question=future_question(agent),
                 theories=[theory('linear'), theory('quadratic', SQUARE)], experiments=[probe(agent)])
    clones = [theory(f'clone_{index}') for index in range(100)]
    copied = route(agent, lab, question={'id': 'future_response'}, theories=clones)
    assert copied['selection']['scores'] == base['selection']['scores']
    assert copied['selection']['compute_cost'] == base['selection']['compute_cost']
    group = next(group for group in copied['selection']['groups'] if 'linear' in group['theory_ids'])
    assert set(group['theory_ids']) == {'linear', *(f'clone_{index}' for index in range(100))}
    assert len(agent.theory_index) == 102
    assert all(entry['support']['status'] == 'candidate' and entry['support']['evidence'] == []
               for entry in agent.theory_index.values())


def test_independent_theory_budget_routes_computation_and_task_support_keeps_unknown_theory(tmp_path):
    agent, lab = scene(tmp_path)
    record = agent.observations[0]
    measured = {'op': 'Readout', 'record_index': 0, 'step': 0, 'units': 'V',
                'context_ref': deepcopy(CONTEXT), 'binding': {'action': deepcopy(record['action']),
                    'features': deepcopy(record['rows'][0]['features']), 'history': []}}
    question = future_question(agent, 'theory_readout')
    question.update(queue='theory', use_sites=[], query_semantics={
        'mode': 'model_query', 'query_expression': measured})
    mission = {'mission_id': 'offline_theory', 'purpose': 'theory_learning', 'phase': 'theory', 'allocation': 10}
    agent._open_question(question)
    result = route(agent, lab, mission=mission)
    assert agent.research_agenda['selected_queue'] == 'theory'
    assert agent.research_agenda['selected_question_id'] == 'theory_readout'
    assert result['answer_execution']['status'] == 'answer_ready_no_parent_use'
    unknown = agent._open_question({'id': 'unknown_mechanism', 'question': 'An unresolved mechanism',
        'kind': 'structural_relation', 'queue': 'theory', 'answer_type': 'program',
        'mission': mission, 'use_sites': []})
    paid = lab.paid
    finish = {'action': 'finish', 'arguments': {'model': 'existing.py'}}
    result = route(agent, lab, mission={'purpose': 'task_prediction', 'task_supported': True,
                                      'supported_action': finish})
    assert result['status'] == 'task_supported' and result['selected_action'] == finish
    assert result['theory_status'] == 'unresolved'
    assert 'semantic_answer' not in unknown and unknown in agent.questions
    assert lab.paid == paid and agent.costs['model_calls'] == 0 and not agent.confirmations


def test_source_ref_meaning_version_does_not_alias_new_meaning_round_zero(tmp_path):
    agent, lab = scene(tmp_path)
    owner = agent._open_question({'id': 'native_source', 'question': 'Old scoped native question',
        'kind': 'value', 'answer_type': 'number', 'context_ref': {'scene': 'old'},
        'native_condition_research': [native_round(X)]})
    retained = theory('old_native')
    retained.pop('program')
    retained['source_ref'] = {'kind': 'native_candidate', 'question_id': owner['id'],
        'round_index': 0, 'candidate_index': 0}
    route(agent, lab, question=future_question(agent),
          theories=[retained, theory('quadratic', SQUARE)], experiments=[probe(agent)])
    agent._open_question({'id': owner['id'], 'context_ref': {'scene': 'new'}})
    owner['native_condition_research'] = [native_round(Program('const', value=100.))]
    assert owner['meaning_version'] == 2
    result = route(agent, lab, question={'id': 'future_response'},
                   experiments=[probe(agent, 4., 'after_meaning_change')])
    assert result['cache_hit'] is False
    group = next(group for group in result['selection']['groups'] if 'old_native' in group['theory_ids'])
    assert group['task_answer'] == 3.  # Kept old meaning/round, never new 100.
    assert 'meaning_version' not in retained['source_ref']
    assert agent.theory_index['old_native']['source_ref']['meaning_version'] == 1
    assert agent.costs['model_calls'] == 0


def test_two_step_route_uses_the_actual_first_outcome_and_only_prepares_second_purchase(tmp_path, monkeypatch):
    agent, lab = scene(tmp_path)
    first = probe(agent, 0., 'conditional_probe')
    second_index = len(agent.observations) + 1
    first['branches'] = [
        {'when': {'value': 0., 'atol': 0.}, 'actions': [action(2.)],
         'query': readout(2., second_index)},
        {'when': {'value': 10., 'atol': 0.}, 'actions': [action(5.)],
         'query': readout(5., second_index)}]
    first['observation_cost'] = 2
    result = route(agent, lab, question=future_question(agent),
                   theories=[theory('linear'), theory('quadratic', SQUARE)], experiments=[first])
    assert result['status'] == 'selected'
    assert pending_query_arguments(agent, 'task')['actions'] == [action(0.)]
    paid, actions = lab.paid, deepcopy(lab.actions)
    original_execute = lab.execute

    def surprising_first_outcome(selected):
        record = original_execute(selected)
        # Both finite candidates predict 0. The synthetic apparatus returns a
        # counterexample, so their predicted branch must not choose the next act.
        record['rows'][0]['target'] = 10.
        return record

    monkeypatch.setattr(lab, 'execute', surprising_first_outcome)
    acquired = agent._execute('query', {'pending_question': 'task'}, lab)
    continuation = acquired['conditional_continuation']
    assert continuation['status'] == 'second_step_prepared'
    assert continuation['first_outcome']['value'] == 10.
    assert continuation['first_outcome']['evidence'] == [paid]
    assert continuation['purchased_second_step'] is False
    assert parent(agent)['pending_query_source']['stage'] == 2
    assert pending_query_arguments(agent, 'task')['actions'] == [action(5.)]
    assert agent.observations[-1]['rows'][0]['target'] == 10.
    assert lab.paid == paid + 1 and lab.actions == actions + [action(0.)]
    assert len(agent.observations) == second_index
    assert agent.costs['model_calls'] == 0 and not agent.confirmations


def test_declared_family_refit_executes_checked_region_but_keeps_old_index_until_reregistered(tmp_path):
    from test_evidence_procedure_execution import prepared
    from test_executable_evidence_binding import return_region

    agent, lab = prepared(tmp_path)
    return_region(agent, lab, .9, 1.1)
    old_program = Program('mul', children=(Program('const', value=.25), X))
    target_program = Program('mul', children=(Program('const', value=1.), X))
    source = theory('retained_family', old_program, family_id='fixture_linear', parameter_paths=[[0]])
    source['refit_action'] = {'action': 'run', 'arguments': {'operator': 'refit', 'model': 'parent.py',
        'research_inputs': {'slope': 'slope_question'}, 'validation_records': [2],
        'path': 'shared/routing_refit'}}
    question = future_question(agent, 'refit_family_question')
    question['theory_target'] = theory('target', target_program, family_id='fixture_linear', parameter_paths=[[0]])
    paid, actions = lab.paid, deepcopy(lab.actions)
    result = route(agent, lab, question=question, theories=[source])
    work, = result['executed_reuse_work']
    assert work['source_id'] == source['id']
    assert work['feedback']['validation']['mse'] == 0.
    binding = work['feedback']['validation']['research_bindings'][0]
    assert binding['actual_parameters']['slope'] == 1.
    assert binding['inputs']['slope']['region']['lower'] == .9
    assert agent.theory_index[source['id']]['program'] == old_program.to_dict()
    assert agent.theory_index[source['id']]['version'] == 1
    updated = deepcopy(source)
    updated['program'] = target_program.to_dict()
    again = route(agent, lab, question={'id': question['id']}, theories=[updated])
    assert again['executed_reuse_work'] == []
    assert agent.theory_index[source['id']]['version'] == 2
    assert agent.theory_index[source['id']]['history'][0]['program'] == old_program.to_dict()
    assert_no_new_measurements(agent, lab, paid, actions)
