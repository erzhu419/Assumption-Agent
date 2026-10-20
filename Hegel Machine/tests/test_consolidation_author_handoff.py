"""Bounded induction can finish while a concrete author obligation remains."""

from copy import deepcopy
import json
from pathlib import Path

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.python_method_learning import PythonArchiveMethodLearner
from hegel_machine.cognition.question_routing import register_theories
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from hegel_machine.cognition.temporal_paired_practice import candidate_execution_summary, practice_execution_summary
from hegel_machine.cognition.theory_consolidation import LIBRARY_PATH
from test_executed_evidence_notebook import AccountedPublicLab
from test_question_routing import CONTEXT, SQUARE, theory
from test_theory_consolidation import affine, agent_scene, assert_no_new_science, sources


PROCEDURE = 'shared/request_shared_structure.py'


def prepared(path, *, mixed=False):
    agent, lab = agent_scene(path)
    declarations = sources()
    if mixed:
        declarations.append(theory('source_c', SQUARE,
            context_ref={**CONTEXT, 'source_id': 'source_c'}, context_family_ref='declared_drive_theme'))
    else:
        declarations[1]['program'] = SQUARE.to_dict()
        declarations[1]['parameter_paths'] = []
    for index, item in enumerate(declarations):
        filename = f'shared/source_fixture_{index}.json'
        # Synthetic retained source records; these are not acquired in this task.
        target = 0. if item['program'] == SQUARE.to_dict() else (1. if index == 0 else -3.)
        agent._save(filename, json.dumps({'program': item['program'],
            'rows': [{'features': {'u': 0.}, 'target': target, 'history': []}]}))
        item['support'].update(status='supported', evidence=[{'file': filename, 'json_path': ['rows', 0]}])
    register_theories(agent, declarations)
    return agent, lab, declarations


def mission(compute):
    return {'purpose': 'theory_learning', 'mission_id': 'author_handoff_fixture',
            'theory_learning_allocation': {'compute': compute}}


def route(agent, lab, **arguments):
    return agent._execute('run', {'operator': 'research_route', **arguments}, lab)


def request_question(agent):
    return next(q for q in agent.questions if isinstance(q.get('origin'), dict)
                and q['origin']['kind'] == 'multi_episode_commonality')


def dereference(agent, reference):
    value = json.loads((agent.workspace / reference['file']).read_text())
    for key in reference['json_path']:
        value = value[key]
    return value


def test_common_procedure_returns_the_concrete_author_request_then_retrieves_it_at_zero_budget(tmp_path):
    agent, lab, _ = prepared(tmp_path)
    agent._save(PROCEDURE, f'''def select_actions(spec, observations, memory):
    return {{'action':'run','arguments':{{'operator':'research_route','mission':{mission(10)!r}}}}}
''')
    first = agent._execute('run', {'operator': 'procedure', 'path': PROCEDURE, 'max_steps': 5}, lab)
    assert first['status'] == 'author_work_request' and len(first['steps']) == 1
    feedback = first['steps'][0]['feedback']
    assert feedback['bounded_status'] == 'completed' and feedback['new_compute_cost'] == 10
    assert feedback['frozen_candidates'] == [] and len(feedback['author_work']) == 2
    question = request_question(agent)
    assert 'completed_agenda_basis' not in question
    request = question['pending_author_work']
    assert request['required_contract']['source_refs'] and request['agenda_basis'] == question['agenda_basis']
    assert dereference(agent, request['author_work_ref']) == feedback['author_work']
    assert dereference(agent, request['bounded_result_ref'])['status'] == 'completed'
    exposed = next(item for item in agent._request(lab)['online_research']['questions'] if item['id'] == question['id'])
    assert exposed['pending_author_work'] == request
    before = deepcopy(agent.costs)
    repeated = route(agent, lab)
    assert repeated['status'] == 'author_work_request' and repeated['cache_hit']
    assert repeated['new_compute_cost'] == 0 and agent.costs == before
    assert repeated['required_contract'] == request['required_contract']
    assert repeated['author_work_ref'] == request['author_work_ref']
    assert_no_new_science(agent, lab)


def test_restoration_preserves_the_request_and_a_source_version_change_reopens_induction(tmp_path):
    agent, lab, declarations = prepared(tmp_path / 'original')
    route(agent, lab, mission=mission(10))
    original_request = deepcopy(request_question(agent)['pending_author_work'])
    previous = agent.result('author_work_request')
    agent._prepare_program_state()
    restored, no_acquisition_lab = agent_scene(tmp_path / 'restored')
    current_lab = AccountedPublicLab()
    current_lab.execute = no_acquisition_lab.execute
    restored.restore_practice_research(previous, agent.workspace, current_lab, confirmation_seed=81)
    before = deepcopy(restored.costs)
    repeated = route(restored, current_lab)
    assert repeated['author_work_ref'] == original_request['author_work_ref']
    assert repeated['required_contract'] == original_request['required_contract']
    assert repeated['new_compute_cost'] == 0 and restored.costs == before
    changed = deepcopy(declarations[1])
    changed.update(program=affine(4., -3.).to_dict(), parameter_paths=[[0, 0], [1]])
    rebuilt = route(restored, current_lab, theories=[changed], mission=mission(20000))
    assert rebuilt['status'] == 'completed' and rebuilt['new_compute_cost'] > 0
    assert len(rebuilt['frozen_candidates']) == 1
    assert 'pending_author_work' not in request_question(restored)
    assert restored.theory_index['source_b']['version'] == 2
    assert restored.theory_index['source_b']['history'][0]['program'] == SQUARE.to_dict()
    assert_no_new_science(restored, current_lab)


def test_author_supplied_query_survives_automatic_rescheduling_and_is_actually_executed(tmp_path):
    agent, lab, _ = prepared(tmp_path, mixed=True)
    initial = route(agent, lab, mission=mission(20000))
    assert initial['status'] == 'author_work_request' and len(initial['author_work']) == 1
    question = request_question(agent)
    expression = {'op': 'FindSharedStructure', 'typed_program_refs': [
        {'kind': 'theory', 'id': identifier, 'version': 1}
        for identifier in ('source_a', 'source_b')],
        'mapping_budget': {'compute': 20000, 'max_groups': 8}}
    authored = agent._open_question({'id': question['id'], 'answer_type': 'program',
        'answer_resolution_ref': {'contract': 'ordered_source_reconstruction_only'},
        'query_semantics': {'mode': 'model_query', 'query_expression': expression},
        'use_sites': []})
    version = authored['meaning_version']
    result = route(agent, lab)
    current = request_question(agent)
    assert current['meaning_version'] == version and current['answer_type'] == 'program'
    assert current['query_semantics']['query_expression'] == expression
    assert current['query_execution']['answer']['status'] == 'answered'
    assert current['semantic_answer']['scope']['typed_program_refs'] == expression['typed_program_refs']
    assert result['answer_execution']['answer']['value'] == current['semantic_answer']['value']
    assert next(entry for entry in agent.research_agenda['theory_agenda']
                if entry['question_id'] == current['id'])['work_kind'] == 'query'
    # The author's finite query does not remove the third source or its original obligation.
    assert set(agent.theory_index) == {'source_a', 'source_b', 'source_c'}
    original = dereference(agent, initial['bounded_result_ref'])
    assert original['author_work'] == initial['author_work']
    assert len(agent.theory_consolidation['candidates']) == 1
    before = deepcopy(agent.costs)
    route(agent, lab)
    assert request_question(agent)['meaning_version'] == version and agent.costs == before
    assert_no_new_science(agent, lab)


def test_mixed_frozen_candidates_and_unpaired_author_obligations_both_remain_available(tmp_path):
    agent, lab, _ = prepared(tmp_path, mixed=True)
    first = route(agent, lab, mission=mission(20000))
    assert first['status'] == 'author_work_request' and first['bounded_status'] == 'completed'
    frozen, = first['frozen_candidates']
    assert len(first['author_work']) == 1
    question = request_question(agent)
    saved = dereference(agent, question['pending_author_work']['bounded_result_ref'])
    assert saved['frozen_candidates'] == first['frozen_candidates']
    before = deepcopy(agent.costs)
    repeated = route(agent, lab)
    assert repeated['status'] == 'author_work_request' and repeated['new_compute_cost'] == 0
    assert agent.costs == before
    assert frozen['candidate_id'] in agent.theory_consolidation['candidates']
    assert all(item['exact'] for item in agent.theory_consolidation['candidates'][frozen['candidate_id']]['source_reconstruction'])
    assert_no_new_science(agent, lab)


def test_outer_paid_practice_summary_and_latest_trial_refresh_retain_the_small_request(tmp_path):
    directory = 'trials/local_author'
    agent, lab, _ = prepared(tmp_path / 'learner' / directory)
    source = f'''def select_actions(spec, observations, memory):
    return {{'action':'run','arguments':{{'operator':'research_route','mission':{mission(10)!r}}}}}
'''
    agent._save(PROCEDURE, source)
    agent._execute('run', {'operator': 'procedure', 'path': PROCEDURE, 'max_steps': 5}, lab)
    result = agent.result('author_work_request')
    (agent.workspace / 'result.json').write_text(json.dumps(result))
    summary = practice_execution_summary(result, directory)
    request = summary['latest_author_work_request']
    assert request['question_id'] == request_question(agent)['id']
    assert request['source_ref'] == {'file': f'{directory}/result.json',
                                   'json_path': ['method_results', 0, 'feedback']}
    assert request['required_contract'] and request['author_work_ref']
    assert 'author_work' not in request and 'frozen_candidates' not in request
    assert summary['research_steps'][0]['author_work_request'] == request
    assert candidate_execution_summary(summary)['latest_author_work_request'] == request
    learner = PythonArchiveMethodLearner(None, tmp_path / 'learner',
        knowledge={'files': {METHOD_PATH: source}}, cases={'fixture': {}},
        lab_factory=lambda _: lab, prepare_trial=None, config=InquiryLLMConfig(max_model_calls=0))
    old_summary = deepcopy(summary)
    old_summary.pop('latest_author_work_request')
    old_summary['research_steps'][0].pop('author_work_request')
    trial = {'case': 'fixture', 'directory': directory, 'status': 'author_work_request',
             'costs': deepcopy(agent.costs), 'method_source': source, 'execution_summary': old_summary}
    learner.practice_trials.append(trial)
    before = deepcopy(learner.costs)
    view = learner._paid_practice_view(trial)
    assert view['execution_summary']['latest_author_work_request'] == request
    assert learner._paid_practice_view(trial) == view and learner.costs == before
    assert_no_new_science(agent, lab)
