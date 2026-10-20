"""Synthetic integration checks; these are not autonomous scientific results."""
from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.code_provider_context import prepare_code_context
from test_common_archive_online_research import ConfirmationLab
from test_inquiry_revision import action


PARENT = '''def bind_research(inputs):
    global p
    r = inputs['slope']['region']
    p = (r['lower'] + r['upper']) / 2
def get_parameters(): return {'slope': p}
def fit(rows): pass
def predict(row): return p * row['features']['u']
'''


def setup(tmp_path, arm='hegel'):
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm=arm, source_task=False,
        config=InquiryLLMConfig(observation_budget=40, confirmation_rows=3,
                               execution_seconds=60., compute_budget=10000))
    lab = ConfirmationLab()
    agent._execute('query', {'actions': [action(-1, 1), action(-.5, .5)]}, lab)
    agent._execute('write', {'path': 'parent.py', 'code': PARENT}, lab)
    return agent, lab


def return_region(agent, lab, lower, upper, *, checked=True, question='slope_question'):
    answer = {'status': 'supported', 'answer': 'A working interval for this background',
        'parameter_region': {'lower': lower, 'upper': upper},
        'scope': {'controls': 'sampled background', 'assumption': 'local linear model'},
        'evidence': [0, 1]}
    if checked:
        answer['checks'] = [{'id': 'ordered', 'terms': [{'row': 1, 'coefficient': 1},
            {'row': 0, 'coefficient': -1}], 'relation': 'gt', 'expected': 0.,
            'atol': 0., 'scope': {'controls': 'paired public inputs'}}]
    agent._execute('write', {'path': 'answer.py', 'code':
        f'def readout(result,spec,memory): return {answer!r}\n'}, lab)
    return agent._execute('run', {'operator': 'analyze_evidence', 'path': 'answer.py',
        'research_question': {'id': question, 'question': 'Which parameter region is useful?',
                             'parent_id': 'task', 'depends_on': []}}, lab)


def test_child_update_changes_actual_parent_inputs_refit_and_new_confirmation(tmp_path):
    agent, lab = setup(tmp_path)
    return_region(agent, lab, 0., 1.)
    first = agent._execute('evaluate', {'model': 'parent.py',
        'research_inputs': {'slope': 'slope_question'}}, lab)
    assert first['mse'] == pytest.approx(.15625)
    old = first['evaluated_model']['model']
    old_inputs = (tmp_path / '_evaluated_model/model_research_inputs.json').read_text()
    return_region(agent, lab, .9, 1.1)
    assert agent.research_candidates['parent.py']['status'] == 'needs_rebuild'
    assert agent._request(lab)['research_repairs'][0]['next_action']['action'] == 'evaluate'
    # Original source unchanged, only the actually injected child value differs.
    second = agent._execute('evaluate', {'model': 'parent.py'}, lab)
    assert second['mse'] == 0.
    assert second['research_input_update']['inputs']['slope']['version'] == 2
    assert second['research_bindings'][0]['actual_parameters'] == {'slope': 1.}
    assert (tmp_path / 'parent.py').read_text() == PARENT
    # Retention is allowed to replace its incumbent; the previous result remains
    # explicit history and was not rewritten when the child answer changed.
    assert json.loads(old_inputs)['parent.py']['slope']['version'] == 1
    q = next(q for q in agent.questions if q['id'] == 'slope_question')
    assert q['answer_history'][0]['version'] == 1
    completed = agent._execute('finish', {'model': 'parent.py'}, lab)
    assert completed['fresh_confirmation']['passed']
    assert lab.paid == 7  # Four initial rows, three genuine fresh rows.
    assert completed['fresh_confirmation']['research_bindings'][0]['actual_parameters']['slope'] == 1.


def test_refuted_child_blocks_current_rebuild_but_preserves_old_frozen_candidate(tmp_path):
    agent, lab = setup(tmp_path)
    return_region(agent, lab, .9, 1.1)
    first = agent._execute('evaluate', {'model': 'parent.py',
        'research_inputs': {'slope': 'slope_question'}}, lab)
    old = first['evaluated_model']['model']
    bound_before = (tmp_path / '_evaluated_model/model_research_inputs.json').read_text()
    bad = {'status': 'supported', 'answer': 'Prose cannot erase the counterexample',
        'evidence': [1], 'checks': [{'id': 'zero', 'terms': [{'row': 1, 'coefficient': 1}],
            'relation': 'eq', 'expected': 0., 'scope': {'input': 1}}]}
    agent._execute('write', {'path': 'answer.py', 'code':
        f'def readout(result,spec,memory): return {bad!r}\n'}, lab)
    agent._execute('run', {'operator': 'analyze_evidence', 'path': 'answer.py',
        'research_question': {'id': 'slope_question'}}, lab)
    blocked = agent._execute('evaluate', {'model': 'parent.py'}, lab)
    assert blocked['status'] == 'dependency_unresolved'
    assert (tmp_path / '_evaluated_model/model_research_inputs.json').read_text() == bound_before
    older = agent._execute('evaluate', {'model': old}, lab)
    assert older['mse'] == 0.


def test_unrelated_parent_stays_current_and_unchecked_region_cannot_be_bound(tmp_path):
    agent, lab = setup(tmp_path)
    return_region(agent, lab, .9, 1.1, checked=False)
    with pytest.raises(ValueError, match='satisfied executable checks'):
        agent._execute('run', {'operator': 'bind_research', 'model': 'parent.py',
                              'inputs': {'slope': 'slope_question'}}, lab)
    return_region(agent, lab, .9, 1.1)
    agent._execute('evaluate', {'model': 'parent.py',
                              'research_inputs': {'slope': 'slope_question'}}, lab)
    before = deepcopy(agent.research_candidates['parent.py'])
    return_region(agent, lab, 2., 4., question='other_parameter')
    assert agent.research_candidates['parent.py'] == before


def test_flat_code_can_explicitly_rebind_same_evidence_with_same_tools(tmp_path):
    agent, lab = setup(tmp_path, 'code_react')
    return_region(agent, lab, 0., 1.)
    agent._execute('evaluate', {'model': 'parent.py',
                              'research_inputs': {'slope': 'slope_question'}}, lab)
    return_region(agent, lab, .9, 1.1)
    unchanged = agent._execute('evaluate', {'model': 'parent.py'}, lab)
    assert unchanged['mse'] > 0.
    assert unchanged['research_input_update']['status'] == 'old_dependency_version'
    agent._execute('run', {'operator': 'bind_research', 'model': 'parent.py',
                          'inputs': {'slope': 'slope_question'}}, lab)
    assert agent._execute('evaluate', {'model': 'parent.py'}, lab)['mse'] == 0.


def test_current_prompt_keeps_computed_evidence_and_old_files_remain_readable(tmp_path):
    files = {f'scratch_{i}.py': 'print("old work")\n' * 300 for i in range(20)}
    files['parent.py'] = PARENT
    request = {'files': files, 'events': [{'step': i, 'action': 'run', 'arguments': {
        'path': f'scratch_{i}.py'}, 'feedback': {'stdout': 'saved'}} for i in range(20)],
        'online_research': {'questions': [{'id': 'child', 'evidence_status': 'refuted'}]},
        'research_repairs': [{'model': 'parent.py', 'status': 'needs_rebuild'}]}
    context = prepare_code_context(tmp_path, request, schema={}, instructions='')
    assert context['files']['parent.py'] == PARENT
    assert context['files']['scratch_0.py'] == {'full_content_file': 'scratch_0.py'}
    assert len(context['events']) == 8 and context['event_history']['preview_start'] == 12
    assert context['online_research']['questions'][0]['evidence_status'] == 'refuted'
    assert json.loads((tmp_path / 'research_context.json').read_text()) == request


def test_new_paid_evidence_does_not_leave_current_evaluation_status(tmp_path):
    agent, lab = setup(tmp_path)
    agent._execute('evaluate', {'model': 'plain.py',
        'code': "def predict(row): return row['features']['u']\n"}, lab)
    agent._execute('query', {'actions': [action(2.)]}, lab)
    before = deepcopy(agent.research_candidates['plain.py']['current_evaluation'])
    costs, events, observations = deepcopy(agent.costs), deepcopy(agent.events), lab.paid
    request = agent._request(lab)
    state = request['research_repairs'][0]
    assert state['status'] == 'new_paid_rows_unscored'
    assert state['paid_rows'] == 4 and state['current_evaluation']['mse'] == 0.
    assert state['current_paid_rows'] == len(agent.rows) == 5
    assert state['evaluation_provenance']['status'] == 'last_recorded_evaluation'
    assert state['evaluation_provenance']['paid_rows'] == 4
    assert state['evaluation_provenance']['current_paid_rows'] == 5
    assert state['next_action_status'] == 'proposed_not_executed'
    assert agent.research_candidates['plain.py']['current_evaluation'] == before
    assert agent.costs == costs and agent.events == events and lab.paid == observations
    context = prepare_code_context(tmp_path / 'provider', request, schema={}, instructions='')
    assert context['research_repairs'][0]['next_action_status'] == 'proposed_not_executed'
    assert context['research_repairs'][0]['evaluation_provenance'] == state['evaluation_provenance']
    assert agent._execute('evaluate', {'model': 'plain.py'}, lab)['mse'] == 0.
    assert not agent._request(lab)['research_repairs']


def test_updated_callable_child_rebuilds_actual_parent_composition(tmp_path):
    from test_common_archive_observable_research import setup as measured_setup, CHILD, PARENT as COMPOSITION
    agent, lab = measured_setup(tmp_path, 'hegel')
    child = deepcopy(agent.observable_models['meter'])
    parent = agent._execute('evaluate', {'model': 'parent.py', 'code': COMPOSITION,
        'bindings': {'latent': child['model']}}, lab)
    assert parent['mse'] < 1e-25
    # A new observation refutes the old dependency while preserving all old data.
    lab.offset = 4.
    agent._execute('query', {'actions': [action(.3, channels=['meter'])]}, lab)
    blocked = agent._execute('evaluate', {'model': 'parent.py'}, lab)
    assert blocked['status'] == 'dependency_unresolved'
    assert 'meter' in blocked['error']
    # An authored replacement explains the observed piecewise relation. This
    # toy fixture supplies the formula; it checks actual composition, not discovery.
    revised = "def predict(row):\n    u=row['features']['u']\n    return 2*u+(4 if u==.3 else 1)\n"
    agent._execute('evaluate', {'target': 'meter', 'model': 'replacement.py', 'code': revised}, lab)
    result = agent._execute('evaluate', {'model': 'parent.py'}, lab)
    assert result['mse'] < 1e-25
    assert result['research_input_update']['components']['latent']['version'] == 1
    assert result['model_dependencies'][0]['model'] != child['model']


def test_corrected_representation_closes_computed_child_and_returns_parent(tmp_path):
    agent, lab = setup(tmp_path)
    collapsed = ('def represent(row): return [0.]\n'
                 'def predict_features(features): return 0.\n')
    agent._execute('write', {'path': 'view.py', 'code': collapsed}, lab)
    bad = agent._execute('evaluate', {'model': 'view.py'}, lab)
    assert bad['representation_diagnostic']['empirical_mse_floor'] > 0
    question = next(q for q in agent.questions if q['id'] == 'representation:view.py')
    assert question['status'] == 'open'
    assert agent._active_question_id == question['id']
    agent._execute('write', {'path': 'view.py', 'code':
        'def represent(row): return [row["features"]["u"]]\n'
        'def predict_features(features): return features[0]\n'}, lab)
    good = agent._execute('evaluate', {'model': 'view.py'}, lab)
    assert good['mse'] == 0
    assert question['status'] == 'resolved_on_paid_rows'
    assert agent._active_question_id == 'task'
    assert len(question['representation_assessments']) == 2
    assert question['representation_assessments'][0]['witnesses']
    assert question['representation_assessments'][1]['empirical_mse_floor'] == 0


def test_refit_training_collision_routes_even_when_validation_has_single_input(tmp_path):
    agent, lab = setup(tmp_path)
    agent._execute('query', {'actions': [action(.75)]}, lab)
    agent._execute('write', {'path': 'collapsed.py', 'code':
        'def represent(row): return [0.]\n'
        'def fit_features(features,targets): pass\n'
        'def predict_features(features): return 0.\n'}, lab)
    result = agent._execute('run', {'operator': 'refit', 'model': 'collapsed.py',
                                  'validation_records': [2]}, lab)
    assert result['validation']['representation_diagnostic']['empirical_mse_floor'] == 0
    assert result['validation']['training_representation_diagnostic']['empirical_mse_floor'] > 0
    assert agent.research_candidates['collapsed.py']['status'] == 'representation_conflict'
    question = next(q for q in agent.questions if q['id'] == 'representation:collapsed.py')
    assert question['representation_diagnostic']['scope'] == 'selected_fitting_rows'


def test_unexposed_new_predictor_does_not_claim_representation_diagnosis_resolved(tmp_path):
    agent, lab = setup(tmp_path)
    agent._execute('write', {'path': 'view.py', 'code':
        'def represent(row): return [0.]\n'
        'def predict_features(features): return 0.\n'}, lab)
    agent._execute('evaluate', {'model': 'view.py'}, lab)
    question = next(q for q in agent.questions if q['id'] == 'representation:view.py')
    agent._execute('write', {'path': 'view.py', 'code':
        'def predict(row): return row["features"]["u"]\n'}, lab)
    result = agent._execute('evaluate', {'model': 'view.py'}, lab)
    assert result['mse'] == 0
    assert question['status'] == 'open'
    assert len(question['representation_assessments']) == 1


def test_finish_paid_evaluation_excludes_new_confirmation_rows_but_refit_uses_current_evidence(tmp_path):
    agent, lab = setup(tmp_path)
    agent._execute('write', {'path': 'constant.py', 'code':
        'def fit(rows): pass\ndef predict(row): return 0.\n'}, lab)
    count = len(agent.rows)
    finished = agent._execute('finish', {'model': 'constant.py'}, lab)
    assert not finished['fresh_confirmation']['passed']
    assert finished['submitted_candidate']['paid_rows'] == count
    assert len(agent.rows) == count + agent.config.confirmation_rows
    candidate = agent.research_candidates['constant.py']
    assert candidate['paid_rows'] == candidate['current_evaluation']['paid_rows'] == count
    assert candidate['current_evaluation']['mse'] == finished['mse']
    assert candidate['status'] == 'new_paid_rows_unscored'

    evaluated = agent._execute('evaluate', {'model': 'constant.py'}, lab)
    assert candidate['paid_rows'] == candidate['current_evaluation']['paid_rows'] == len(agent.rows)
    assert candidate['current_evaluation']['mse'] == evaluated['mse']
    assert candidate['status'] == 'evaluated'
    refitted = agent._execute('run', {'operator': 'refit', 'model': 'constant.py',
                                    'validation_records': [2]}, lab)
    assert refitted['validation']['returncode'] == 0
    assert candidate['paid_rows'] == candidate['current_evaluation']['paid_rows'] == len(agent.rows)
    assert candidate['current_evaluation']['evaluation_scope'] == 'paid_trajectory_holdout'


@pytest.mark.parametrize('later_evaluation', [False, True])
def test_restore_recovers_old_finish_scope_only_when_it_is_the_latest_evaluation(tmp_path, later_evaluation):
    from test_common_archive_resume import checkpoint, agent_at, execute, SEED
    from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab

    old, lab, _ = checkpoint(tmp_path)
    if later_evaluation:
        # A real subsequent fit can have exactly the same zero error as the old
        # submission. Scalar equality alone cannot identify the score's scope.
        execute(old, lab, 'write', {'path': 'model.py', 'code': '''import json
def key(row): return json.dumps([row['features'], row.get('history', [])], sort_keys=True)
def fit(rows):
    global fitted
    fitted = {key(row): row['target'] for row in rows}
def predict(row): return fitted[key(row)]
'''})
        executed = execute(old, lab, 'evaluate', {'model': 'model.py'})
        assert executed['mse'] == old.confirmations[-1]['submitted_candidate']['mse'] == 0.
    previous = old.result('model_call_budget_exhausted')
    stored = previous['research_candidates']['model.py']
    # Recreate the actual old checkpoint defect without changing recorded work.
    stored['paid_rows'] = len(old.rows)
    stored['current_evaluation'].pop('paid_rows')
    history, evaluation = deepcopy(previous['events']), deepcopy(stored['current_evaluation'])
    resumed, target = agent_at(tmp_path / 'resumed'), TemporalInquiryLab(task_index=1)
    resumed.restore_research(previous, old.workspace, target, confirmation_seed=SEED)
    resumed._request(target)
    current = resumed.research_candidates['model.py']
    expected = len(old.rows) if later_evaluation else old.confirmations[-1]['submitted_candidate']['paid_rows']
    assert current['paid_rows'] == expected
    assert current['status'] == ('evaluated' if later_evaluation else 'new_paid_rows_unscored')
    assert {key: current['current_evaluation'][key] for key in evaluation} == evaluation
    assert resumed.events == history
    assert previous['research_candidates']['model.py']['paid_rows'] == len(old.rows)


def return_local_checks(agent, lab, *, expected=0., revision_index=None):
    answer = {'status': 'supported', 'answer': 'A local matched contrast', 'evidence': [0, 1],
        'checks': [{'id': 'difference', 'terms': [{'row': 1, 'coefficient': 1},
            {'row': 0, 'coefficient': -1}], 'relation': 'eq', 'expected': expected,
            'atol': 1e-9, 'scope': {'controls': [-1., 1.], 'history': 'actual recorded prefixes'}}]}
    agent._execute('write', {'path': 'local.py', 'code':
        f'def readout(result,spec,memory): return {answer!r}\n'}, lab)
    arguments = ({'operator': 'analyze_evidence', 'research_question': {
        'id': 'local', 'question': 'Does this contrast vanish?', 'parent_id': 'task'}}
        if revision_index is None else {'operator': 'analyze_revision', 'revision_index': revision_index})
    return agent._execute('run', {**arguments, 'path': 'local.py'}, lab)['revision']


def test_historical_local_checks_survive_later_notes_and_keep_measured_counterexample(tmp_path):
    agent, lab = setup(tmp_path)
    record = return_local_checks(agent, lab)
    assert record['evidence_status'] == 'refuted'
    selected = {'revision_index': record['index'], 'check_ids': ['difference']}
    agent._execute('write', {'path': 'notes.py', 'code':
        "def readout(result,spec,memory): return {'status':'inconclusive',"
        "'answer':'No new checked conclusion', 'evidence':[0,1]}\n"}, lab)
    agent._execute('run', {'operator': 'analyze_evidence', 'path': 'notes.py',
                          'research_question': {'id': 'local'}}, lab)
    result = agent._execute('run', {'operator': 'bind_research', 'model': 'parent.py',
                                  'inputs': {'contrast': selected}}, lab)
    bound = result['inputs']['contrast']
    assert bound['kind'] == 'local_relations' and bound['version'] == 1
    check = bound['checks'][0]
    assert check['status'] == 'violated' and check['expected'] == 0.
    assert check['value'] == 2.  # A rejected expectation is not a fitting target.
    assert check['scope'] == record['check_results'][0]['scope']
    assert list(check['witnesses'][0]['history']) == [{'u': -1.}]
    assert bound['evidence'] == [0, 1]
    assert agent.research_input_bindings['parent.py']['contrast'] == selected


@pytest.mark.parametrize('arm', ['hegel', 'code_react'])
def test_reanalysis_updates_only_declared_local_relation_consumers(tmp_path, arm):
    from hegel_machine.cognition.executable_evidence import before_candidate

    agent, lab = setup(tmp_path, arm)
    record = return_local_checks(agent, lab)
    inputs = {'contrast': {'revision_index': record['index']}}
    agent._execute('run', {'operator': 'bind_research', 'model': 'parent.py', 'inputs': inputs}, lab)
    frozen = deepcopy(agent.research_candidates['parent.py']['inputs'])
    agent.research_candidates['unrelated.py'] = {'status': 'evaluated'}
    return_local_checks(agent, lab, expected=2., revision_index=record['index'])
    updated = before_candidate(agent, 'parent.py')
    if arm == 'hegel':
        assert updated['inputs']['contrast']['version'] == 2
        assert updated['inputs']['contrast']['checks'][0]['status'] == 'satisfied'
    else:
        assert updated['status'] == 'old_dependency_version'
        assert agent.research_candidates['parent.py']['inputs'] == frozen
        rebound = agent._execute('run', {'operator': 'bind_research', 'model': 'parent.py',
                                        'inputs': inputs}, lab)
        assert rebound['inputs']['contrast']['version'] == 2
    assert frozen['contrast']['version'] == 1
    assert frozen['contrast']['checks'][0]['status'] == 'violated'
    assert agent.research_candidates['unrelated.py'] == {'status': 'evaluated'}


def test_local_binding_requires_selected_executed_checks_and_not_stale_failed_readout(tmp_path):
    from hegel_machine.cognition.executable_evidence import before_candidate

    agent, lab = setup(tmp_path)
    record = return_local_checks(agent, lab)
    base = {'operator': 'bind_research', 'model': 'parent.py'}
    for selection, error in [({'revision_index': -1}, 'revision_index'),
                             ({'revision_index': record['index'], 'check_ids': ['missing']}, 'check_ids'),
                             ({'revision_index': record['index'], 'check_ids': []}, 'check_ids')]:
        with pytest.raises(ValueError, match=error):
            agent._execute('run', {**base, 'inputs': {'contrast': selection}}, lab)
    agent._execute('run', {**base, 'inputs': {'contrast': {'revision_index': record['index']}}}, lab)
    prior_inputs = (tmp_path / 'model_research_inputs.json').read_text()
    agent._execute('write', {'path': 'local.py', 'code':
        'def readout(result,spec,memory): raise ValueError("analysis unfinished")\n'}, lab)
    agent._execute('run', {'operator': 'analyze_revision', 'revision_index': record['index'],
                          'path': 'local.py'}, lab)
    with pytest.raises(ValueError, match='executed numerical response checks'):
        before_candidate(agent, 'parent.py')
    assert (tmp_path / 'model_research_inputs.json').read_text() == prior_inputs
