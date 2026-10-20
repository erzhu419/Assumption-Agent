"""Existing measured components returned through the PR1 continuation path.

Local scripted laboratories check actual calls and dependencies, not discovery
or scientific transfer. No external model or real experiment is invoked.
"""

from copy import deepcopy
import json

from hegel_machine.cognition.code_agent import retain_evaluated_python_model
from test_common_archive_observable_research import setup, PARENT
from test_executable_evidence_binding import return_region
from test_inquiry_revision import action


def prepared(tmp_path, *, refit=False):
    agent, lab = setup(tmp_path, 'hegel')
    use = {'kind': 'supply_component', 'model': 'parent.py', 'slot': 'latent'}
    if refit:
        agent._execute('query', {'actions': [action(.7, .9, channels=['meter'])]}, lab)
        use['refit'] = {'validation_records': [1], 'path': 'shared/component_refit'}
    question = agent._open_question({'id': 'observable:meter', 'kind': 'applicability',
        'context_ref': {'target': 'meter', 'coordinates': 'authored_public_control_history'},
        'answer_type': 'program', 'answer_resolution_ref': {'evaluator': 'paid_observable_model'},
        'query_semantics': {'mode': 'solution_relation'}, 'use_sites': [use]})
    # Existing native evaluation returns the retained model in this meaning.
    agent._execute('evaluate', {'target': 'meter', 'model': agent.observable_models['meter']['model'],
        'research_question': {'id': question['id']}}, lab)
    parent = 'auxiliary = lambda row: 0.\n' + PARENT
    if refit:
        parent += '''
def fit(rows):
    global scale
    inputs = [auxiliary(row) for row in rows]
    scale = sum((row['target']-row['features']['u'])*value for row,value in zip(rows,inputs))/sum(value*value for value in inputs)
def predict(row): return scale*auxiliary(row)+row['features']['u']
'''
    agent._execute('write', {'path': 'parent.py', 'code': parent}, lab)
    return agent, lab, question


def step(agent, lab):
    return agent._execute('run', {'operator': 'research_step',
        'research_question': {'id': 'observable:meter'}}, lab)


def test_returned_component_enters_native_parent_predictions_and_is_not_repeated(tmp_path):
    agent, lab, question = prepared(tmp_path)
    old = agent._execute('evaluate', {'model': 'parent.py'}, lab)
    destination = tmp_path / 'archive/before_component/_evaluated_model'
    saved = retain_evaluated_python_model(tmp_path, old['evaluated_model']['model'], [], old['mse'], destination=destination)
    old_frozen = str((destination / saved['model']).relative_to(tmp_path))
    source = (tmp_path / 'parent.py').read_text()
    paid = lab.paid
    result = step(agent, lab)
    feedback = result['uses'][0]['feedback']
    assert result['status'] == 'answer_applied' and result['parent_revision_completed']
    assert old['mse'] > .1 and feedback['mse'] < 1e-25
    assert feedback['model_dependencies'][0]['calls'] == 2
    assert result['answer']['value']['component_id'] == 'meter'
    from hegel_machine.cognition.research_episodes import extract_episodes
    episode = next(item for item in extract_episodes(agent)['episodes']
                   if item['episode_id'] == f'{question["id"]}@meaning{question["meaning_version"]}')
    assert episode['adoption']['occurred']
    assert any(ref['json_path'][-2:] == ['parent_use_history', 0]
               for ref in episode['adoption']['execution_refs'])
    bound = agent.research_candidates['parent.py']['components']['latent']
    assert bound['question_ref'] == {'question_id': question['id'],
        'meaning_version': question['meaning_version'], 'answer_version': result['answer']['version']}
    assert (tmp_path / 'parent.py').read_text() == source
    costs = deepcopy(agent.costs)
    assert step(agent, lab)['status'] == 'reused_answer'
    assert agent.costs == costs and lab.paid == paid
    assert agent._execute('evaluate', {'model': old_frozen}, lab)['mse'] == old['mse']
    assert not agent.confirmations and agent.costs['model_calls'] == 0


def test_declared_refit_reuses_existing_binding_and_original_validation_records(tmp_path):
    agent, lab, _ = prepared(tmp_path, refit=True)
    paid = lab.paid
    result = step(agent, lab)
    feedback = result['uses'][0]['feedback']
    assert result['parent_revision_completed'] and feedback['validation']['mse'] < 1e-25
    assert feedback['validation']['model_dependencies'][0]['calls'] > 0
    assert feedback['split']['validation_records'] == [1]
    assert lab.paid == paid and agent.costs['model_calls'] == 0


def test_changed_component_question_meaning_expires_only_its_actual_consumer(tmp_path):
    agent, lab, question = prepared(tmp_path)
    first = step(agent, lab)['uses'][0]['feedback']
    old = first['evaluated_model']['model']
    agent._execute('evaluate', {'model': 'unrelated.py', 'code': 'def predict(row): return 0.\n'}, lab)
    unrelated = deepcopy(agent.research_candidates['unrelated.py'])
    agent._open_question({'id': question['id'], 'context_ref': {'target': 'another_measurement'}})
    blocked = agent._execute('evaluate', {'model': 'parent.py'}, lab)
    assert blocked['status'] == 'dependency_unresolved' and question['id'] in blocked['error']
    assert agent.research_candidates['parent.py']['status'] == 'needs_rebuild'
    assert 'current_evaluation' not in agent.research_candidates['parent.py']
    assert agent.research_candidates['unrelated.py'] == unrelated
    assert step(agent, lab)['status'] == 'scoped_unknown'
    assert agent._execute('evaluate', {'model': old}, lab)['mse'] < 1e-25


def test_refuted_or_unregistered_component_cannot_be_supplied_as_a_supported_answer(tmp_path):
    agent, lab, question = prepared(tmp_path)
    step(agent, lab)
    agent.observable_models['meter']['support'] = 'refuted_on_paid_rows'
    blocked = step(agent, lab)
    assert blocked['status'] == 'scoped_unknown'
    assert agent._execute('evaluate', {'model': 'parent.py'}, lab)['status'] == 'dependency_unresolved'
    # A copied return without an actual registry entry cannot create evidence.
    question['observable_target'] = 'unregistered_target'
    assert step(agent, lab)['status'] == 'scoped_unknown'
    assert 'applied_question_use' not in question


def test_new_component_version_is_rebound_and_old_frozen_parent_is_preserved(tmp_path):
    agent, lab, question = prepared(tmp_path)
    initial = step(agent, lab)['uses'][0]['feedback']
    old = initial['evaluated_model']['model']
    lab.offset = 4.
    agent._execute('query', {'actions': [action(.3, channels=['meter'])]}, lab)
    assert step(agent, lab)['status'] == 'scoped_unknown'
    revised = "def predict(row):\n    u=row['features']['u']\n    return 2*u+(4 if u==.3 else 1)\n"
    agent._execute('evaluate', {'target': 'meter', 'model': 'replacement.py', 'code': revised,
        'research_question': {'id': question['id']}}, lab)
    assert agent.observable_models['meter']['version'] == 1
    current = step(agent, lab)
    assert current['parent_revision_completed']
    assert current['uses'][0]['feedback']['mse'] < 1e-25
    assert current['answer']['value']['component_version'] == 1
    assert agent._execute('evaluate', {'model': old}, lab)['mse'] < 1e-25


def test_or_support_paths_preserve_component_until_all_justifications_fail(tmp_path):
    agent, lab, question = prepared(tmp_path)
    return_region(agent, lab, .9, 1.1, question='calibration_a')
    return_region(agent, lab, .9, 1.1, question='calibration_b')
    question['support_paths'] = [[{'question_id': 'calibration_a', 'answer_version': 1}],
                                 [{'question_id': 'calibration_b', 'answer_version': 1}]]
    assert step(agent, lab)['parent_revision_completed']
    before = deepcopy(agent.research_candidates['parent.py'])
    return_region(agent, lab, 2., 3., question='calibration_a')
    assert step(agent, lab)['status'] == 'reused_answer'
    assert agent.research_candidates['parent.py'] == before
    return_region(agent, lab, 2., 3., question='calibration_b')
    assert agent._execute('evaluate', {'model': 'parent.py'}, lab)['status'] == 'dependency_unresolved'
    assert step(agent, lab)['reason'] == 'premise_unknown'


def test_explicitly_removed_slot_binding_is_not_hidden_by_the_same_answer_cache(tmp_path):
    agent, lab, _ = prepared(tmp_path)
    step(agent, lab)
    path = tmp_path / 'model_bindings.json'
    bindings = json.loads(path.read_text())
    bindings['parent.py'] = {}
    path.write_text(json.dumps(bindings))
    assert step(agent, lab)['parent_revision_completed']
    assert json.loads(path.read_text())['parent.py']['latent'] == agent.observable_models['meter']['model']


def test_conditional_component_uses_the_authored_coordinates_and_preserves_another_bound_slot(tmp_path):
    from test_conditional_research import setup as conditional_setup, PARENT as COMPOSED
    from test_common_archive_observable_research import CHILD
    agent, lab = conditional_setup(tmp_path)
    agent._execute('evaluate', {'target': 'meter', 'model': 'meter.py', 'code': CHILD}, lab)
    question = agent._open_question({'id': 'conditional:response_relation', 'kind': 'applicability',
        'context_ref': {'coordinates': 'authored_measured_u_and_observed'}, 'answer_type': 'program',
        'answer_resolution_ref': {'evaluator': 'paid_conditional_view'},
        'query_semantics': {'mode': 'solution_relation'},
        'use_sites': [{'kind': 'supply_component', 'model': 'parent.py', 'slot': 'response_relation'}]})
    agent._execute('evaluate', {'component': 'response_relation',
        'model': agent.conditional_models['response_relation']['model'],
        'research_question': {'id': question['id']}}, lab)
    agent._execute('write', {'path': 'parent.py', 'code': COMPOSED}, lab)
    meter = agent.observable_models['meter']['model']
    (tmp_path / 'model_bindings.json').write_text(json.dumps({'parent.py': {'meter': meter}}))
    paid = lab.paid
    result = agent._execute('run', {'operator': 'research_step',
        'research_question': {'id': question['id']}}, lab)
    feedback = result['uses'][0]['feedback']
    assert result['parent_revision_completed'] and feedback['mse'] < 1e-25
    assert {entry['slot']: entry['calls'] for entry in feedback['model_dependencies']} == {
        'meter': 2, 'response_relation': 2}
    assert json.loads((tmp_path / 'model_bindings.json').read_text())['parent.py']['meter'] == meter
    assert result['answer']['value']['component_kind'] == 'conditional'
    assert lab.paid == paid and not agent.confirmations


def test_merely_binding_without_an_actual_parent_call_is_not_completed_or_cached(tmp_path):
    agent, lab, question = prepared(tmp_path)
    agent._execute('write', {'path': 'parent.py', 'code':
        'def bind_models(models): pass\ndef predict(row): return 0.\n'}, lab)
    result = step(agent, lab)
    assert result['status'] == 'author_work_request' and not result['parent_revision_completed']
    assert result['uses'][0]['feedback']['model_dependencies'][0]['calls'] == 0
    assert 'applied_question_use' not in question
    assert not agent.confirmations
