"""A typed new question recalls optional frozen work without choosing its id."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.question_routing import resolved_theories
from hegel_machine.cognition.theory_consolidation import LIBRARY_PATH
from test_question_routing import CONTEXT, SQUARE, X, future_question, probe, route, theory
from test_theory_consolidation import (
    MISSION, affine, agent_scene, assert_no_new_science, consolidate,
)
from test_revision_relations import binary, constant, feature, relation, revision


def frozen_scene(tmp_path):
    source, lab = agent_scene(tmp_path / 'source')
    result = consolidate(source, lab)
    target_path = tmp_path / 'target'
    (target_path / 'shared').mkdir(parents=True)
    (target_path / LIBRARY_PATH).write_text((source.workspace / result['library_path']).read_text())
    target, target_lab = agent_scene(target_path)
    return target, target_lab


def test_new_typed_question_automatically_recalls_then_consumes_a_frozen_family_and_caches(tmp_path, monkeypatch):
    agent, lab = frozen_scene(tmp_path)
    result = route(agent, lab, question=future_question(agent), theories=[theory('old_square', SQUARE)],
                   experiments=[probe(agent)], mission=deepcopy(MISSION))
    recalled = result['frozen_recall']
    identifier, = recalled['generated_theory_ids']
    assert recalled['new_compute_cost'] > 0
    assert result['status'] == 'selected'
    assert sorted(group['task_answer'] for group in result['selection']['groups']) == [7., 9.]
    assert Program.from_dict(agent.theory_index[identifier]['program']) == affine()
    assert 'old_square' in agent.theory_index
    assert agent.theory_index[identifier]['support']['status'] == 'candidate'
    assert not recalled['support_updated'] and not recalled['independent_confirmation']
    spent = agent.costs['compute_cost']
    writes = []
    save = agent._save

    def record_save(path, content):
        if path == LIBRARY_PATH:
            writes.append(path)
        return save(path, content)

    monkeypatch.setattr(agent, '_save', record_save)
    repeated = route(agent, lab, question={'id': 'future_response'})
    assert repeated['cache_hit'] and repeated['frozen_recall']['cache_hit']
    assert repeated['new_compute_cost'] == 0 and agent.costs['compute_cost'] == spent
    assert writes == []
    assert_no_new_science(agent, lab)


def test_missing_binding_stays_author_work_and_small_budget_is_not_silently_extended(tmp_path):
    agent, lab = frozen_scene(tmp_path)
    question = future_question(agent)
    question['theory_target'].pop('role_signature')
    missing = route(agent, lab, question=question, theories=[theory('old_square', SQUARE)],
                    experiments=[probe(agent)], mission=deepcopy(MISSION))
    assert missing['frozen_recall']['author_work'][0]['reason'] == 'typed_target_binding_required'
    assert missing['frozen_recall']['generated_theory_ids'] == []
    question = future_question(agent, 'funded_target')
    limited = route(agent, lab, question=question,
                    mission={'purpose': 'theory_learning', 'mission_id': 'one_unit_target_recall',
                             'theory_learning_allocation': {'compute': 1}})
    assert limited['frozen_recall']['status'] == 'UNKNOWN'
    assert limited['frozen_recall']['new_compute_cost'] == 1
    assert limited['frozen_recall']['generated_theory_ids'] == []
    funded = route(agent, lab, question={'id': 'funded_target'}, mission=deepcopy(MISSION))
    assert funded['frozen_recall']['generated_theory_ids']
    assert_no_new_science(agent, lab)


def test_actual_scoped_counterexample_blocks_automatic_recall_even_if_checks_are_then_omitted(tmp_path):
    agent, lab = frozen_scene(tmp_path)
    question = future_question(agent)
    question['theory_target']['evidence_checks'] = [{
        'source_ref': {'file': 'retained_fixture_rows.json', 'json_path': [0]},
        'context_ref': deepcopy(CONTEXT), 'observation': {'features': {'u': 0.}, 'target': 99., 'history': []}}]
    failed = route(agent, lab, question=question, theories=[theory('old_square', SQUARE)],
                   experiments=[probe(agent)], mission=deepcopy(MISSION))
    assert failed['frozen_recall']['work'][0]['status'] == 'refuted'
    assert failed['frozen_recall']['generated_theory_ids'] == []
    again = route(agent, lab, question=future_question(agent))
    assert again['frozen_recall']['work'][0]['status'] == 'refuted'
    assert again['frozen_recall']['generated_theory_ids'] == []
    assert any(item.get('scope_refutation') for item in resolved_theories(agent))
    assert 'old_square' in agent.theory_index
    assert_no_new_science(agent, lab)


def test_cross_skeleton_object_uses_the_existing_finite_components_without_manual_object_selection(tmp_path):
    left = binary('add', binary('mul', Program('abs', children=(feature(),)), constant(2)), constant(1))
    right = binary('add', binary('mul', binary('sub', feature(), constant(3)), constant(2)), constant(1))
    frozen = relation([revision('source_a', left), revision('source_b', right)])
    frozen['version'] = 1
    agent, lab = agent_scene(tmp_path)
    agent.theory_consolidation = {'candidates': {frozen['id']: frozen}, 'executions': {}}
    agent._save(LIBRARY_PATH, json.dumps(agent.theory_consolidation))
    result = route(agent, lab, question=future_question(agent), theories=[theory('old_coordinate', X)],
                   experiments=[probe(agent)], mission=deepcopy(MISSION))
    identifier, = result['frozen_recall']['generated_theory_ids']
    program = Program.from_dict(agent.theory_index[identifier]['program'])
    assert program.predict(Observation({'u': 4.}, 0.)) == 9.
    assert 'old_coordinate' in agent.theory_index
    assert sorted(group['task_answer'] for group in result['selection']['groups']) == [3., 7.]
    assert agent.theory_consolidation['candidates'][frozen['id']]['source_reconstruction'] == frozen['source_reconstruction']
    assert_no_new_science(agent, lab)


def test_explicit_comparison_pool_keeps_its_declared_votes_after_automatic_recall(tmp_path):
    agent, lab = frozen_scene(tmp_path)
    question = future_question(agent)
    question['candidate_theory_ids'] = ['old_square']
    result = route(agent, lab, question=question, theories=[theory('old_square', SQUARE)],
                   experiments=[probe(agent)], mission=deepcopy(MISSION))
    assert result['frozen_recall']['generated_theory_ids']
    assert len(result['selection']['groups']) == 1
    assert result['selection']['groups'][0]['theory_ids'] == ['old_square']
    assert_no_new_science(agent, lab)


def test_unbound_fixed_context_variable_remains_author_work(tmp_path):
    fixed = Program('feature', name='v')
    left = binary('add', Program('abs', children=(feature(),)), fixed)
    right = binary('add', binary('sub', feature(), constant(3)), fixed)
    roles = {'u': {'role': 'drive', 'unit': 'V', 'time': 'current'},
             'v': {'role': 'reference', 'unit': 'V', 'time': 'current'}}
    frozen = relation([revision('a', left, role_signature=roles),
                       revision('b', right, role_signature=roles)])
    frozen['version'] = 1
    agent, lab = agent_scene(tmp_path)
    agent.theory_consolidation = {'candidates': {frozen['id']: frozen}, 'executions': {}}
    agent._save(LIBRARY_PATH, json.dumps(agent.theory_consolidation))
    result = route(agent, lab, question=future_question(agent), theories=[theory('old_coordinate', X)],
                   experiments=[probe(agent)], mission=deepcopy(MISSION))
    assert not result['frozen_recall']['generated_theory_ids']
    assert result['frozen_recall']['author_work'][0]['reason'] == 'fixed_context_role_binding_required'
    assert set(agent.theory_index) == {'old_coordinate'}
    assert_no_new_science(agent, lab)


@pytest.mark.parametrize('mission', [None, {'purpose': 'task'}])
def test_ordinary_new_task_uses_its_existing_compute_for_frozen_recall(tmp_path, mission):
    agent, lab = frozen_scene(tmp_path)
    result = route(agent, lab, question=future_question(agent), theories=[theory('old_square', SQUARE)],
                   experiments=[probe(agent)], mission=mission)
    assert result['frozen_recall']['generated_theory_ids']
    assert sorted(group['task_answer'] for group in result['selection']['groups']) == [7., 9.]
    assert_no_new_science(agent, lab)


def test_ordinary_task_explicit_recall_allocation_is_honored(tmp_path):
    agent, lab = frozen_scene(tmp_path)
    result = agent._execute('run', {'operator': 'research_route',
        'research_question': future_question(agent), 'theories': [theory('old_square', SQUARE)],
        'experiments': [probe(agent)], 'mission': {'purpose': 'task'}, 'allocation': {'compute': 1}}, lab)
    assert result['frozen_recall']['status'] == 'UNKNOWN'
    assert result['frozen_recall']['new_compute_cost'] == 1
    assert result['frozen_recall']['generated_theory_ids'] == []
    assert_no_new_science(agent, lab)


def test_answered_task_reuses_the_evidence_and_parent_use_before_constructing_new_candidates(tmp_path):
    agent, lab = frozen_scene(tmp_path)
    question = future_question(agent)
    expression = question['query_semantics']['query_expression']
    public_action = deepcopy(expression['binding']['action'])
    agent.observations = [{'action': public_action, 'row_indices': [0],
        'rows': [{'features': {'u': 3.}, 'target': 9., 'history': []}]}]
    agent.rows = [Observation({'u': 3.}, 9.)]
    agent._save('shared/answered_use.py', '''import json
def select_actions(spec, observations, memory):
    state=json.load(open('research_state.json'))
    question=next(q for q in state['questions'] if q['id']=='future_response')
    assert question['semantic_answer']['value']==9.0
    return {'stop':'The retained answer completes this finite task.'}
''')
    question['use_sites'] = [{'kind': 'choose_parent_experiment', 'parent_id': 'task',
                              'path': 'shared/answered_use.py'}]
    answered = agent._execute('run', {'operator': 'research_step', 'research_question': question}, lab)
    assert answered['answer']['status'] == 'answered'
    spent = deepcopy(agent.costs)
    library = (agent.workspace / LIBRARY_PATH).read_text()
    result = route(agent, lab, question={'id': question['id']}, mission={'purpose': 'task'})
    assert result['status'] == 'reused_answer'
    assert result['frozen_recall']['status'] == 'not_needed_answer_available'
    assert result['new_compute_cost'] == 0 and agent.costs == spent
    assert agent.theory_index == {}
    assert (agent.workspace / LIBRARY_PATH).read_text() == library
    assert lab.paid == 0 and lab.actions == [] and agent.costs['model_calls'] == 0
