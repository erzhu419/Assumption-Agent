"""FindSharedStructure delegates to the existing finite typed induction."""

from copy import deepcopy
import json

from hegel_machine.cognition.abstractions import AbstractTemplate
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.question_object_queries import evaluate_shared_structure
from hegel_machine.cognition.question_routing import register_theories
from hegel_machine.cognition.theory_consolidation import LIBRARY_PATH
from test_theory_consolidation import MISSION, affine, agent_scene, assert_no_new_science, sources
from test_question_routing import SQUARE, theory


def shared_query(agent, **fields):
    return {'op': 'FindSharedStructure', 'typed_program_refs': [
        {'kind': 'theory', 'id': identifier, 'version': agent.theory_index[identifier]['version']}
        for identifier in ('source_a', 'source_b')],
        'mapping_budget': {'compute': 20000, 'max_groups': 8}, **fields}


def dereference(agent, reference):
    value = json.loads((agent.workspace / reference['file']).read_text())
    for part in reference['json_path']:
        value = value[part]
    return value


def test_query_returns_executable_library_and_inverse_refs_with_exact_source_reconstruction(tmp_path):
    agent, lab = agent_scene(tmp_path)
    originals = sources()
    register_theories(agent, originals)
    before = agent.costs['compute_cost']
    answer = evaluate_shared_structure(agent, {'id': 'shared'}, shared_query(agent), lab, MISSION)
    assert answer['status'] == 'answered'
    assert answer['compute_cost'] == agent.costs['compute_cost'] - before > 0
    assert answer['source_reconstruction_only'] and not answer['scientific_support']
    assert not answer['independent_confirmation'] and answer['evidence'] == []
    abstraction, = answer['value']['abstractions']
    template = AbstractTemplate.from_dict(dereference(agent, abstraction['abstract_program_ref']))
    instances = dereference(agent, abstraction['inverse_bindings_ref'])
    for instance in instances:
        identifier = instance['episode_id'].split(':')[-1]
        restored = template.instantiate(instance['mapping'], instance['parameters'])
        assert restored == Program.from_dict(agent.theory_index[identifier]['program'])
    assert all(item['exact'] for item in dereference(agent, abstraction['source_reconstruction_ref']))
    assert all('program' not in entry and 'template' not in entry for entry in answer['value']['abstractions'])
    assert_no_new_science(agent, lab)


def test_unchanged_query_reuses_induction_without_cost_or_library_writes(tmp_path, monkeypatch):
    agent, lab = agent_scene(tmp_path)
    register_theories(agent, sources())
    question, query = {'id': 'shared'}, shared_query(agent)
    first = evaluate_shared_structure(agent, question, query, lab, MISSION)
    spent = deepcopy(agent.costs)
    writes = []
    save = agent._save

    def record(path, content):
        if path == LIBRARY_PATH:
            writes.append(path)
        return save(path, content)

    monkeypatch.setattr(agent, '_save', record)
    repeated = evaluate_shared_structure(agent, question, query, lab, MISSION)
    assert repeated['status'] == 'answered' and repeated['value'] == first['value']
    assert repeated['cache_hit'] and repeated['compute_cost'] == 0
    assert agent.costs == spent and writes == []
    assert_no_new_science(agent, lab)


def test_stale_source_version_and_missing_typed_contract_do_not_invent_an_abstraction(tmp_path):
    agent, lab = agent_scene(tmp_path)
    originals = sources()
    register_theories(agent, originals)
    query = shared_query(agent)
    changed = deepcopy(originals[0])
    changed['program'] = affine(5., 7.).to_dict()
    register_theories(agent, [changed])
    stale = evaluate_shared_structure(agent, {'id': 'shared'}, query, lab, MISSION)
    assert stale['status'] == 'scoped_unknown' and stale['detail'] == 'typed_source_version_unavailable'
    assert stale['compute_cost'] == 0 and not agent.theory_consolidation
    changed.pop('role_signature')
    register_theories(agent, [changed])
    incomplete = evaluate_shared_structure(agent, {'id': 'shared'}, shared_query(agent), lab, MISSION)
    assert incomplete['status'] == 'author_work_request'
    assert incomplete['reason'] == 'source_role_unit_time_and_context_required'
    assert incomplete['compute_cost'] == 0 and not agent.theory_consolidation
    assert_no_new_science(agent, lab)


def test_budget_truncation_is_unknown_and_new_induction_keeps_its_theory_funding_requirement(tmp_path):
    agent, lab = agent_scene(tmp_path)
    register_theories(agent, sources())
    query = shared_query(agent)
    deferred = evaluate_shared_structure(agent, {'id': 'shared'}, query, lab, {'purpose': 'task'})
    assert deferred['status'] == 'research_deferred' and deferred['reason'] == 'theory_allocation_missing'
    assert deferred['compute_cost'] == 0 and not agent.theory_consolidation
    query['mapping_budget']['compute'] = 0
    incomplete = evaluate_shared_structure(agent, {'id': 'shared'}, query, lab, MISSION)
    assert incomplete['status'] == 'scoped_unknown' and incomplete['budget_exhausted']
    assert incomplete['partial_result']['abstractions'] == [] and incomplete['compute_cost'] == 0
    assert 'value' not in incomplete
    assert_no_new_science(agent, lab)


def test_common_research_step_constructs_the_shared_answer_and_reuses_it_without_support_or_cost(tmp_path):
    agent, lab = agent_scene(tmp_path)
    register_theories(agent, sources())
    # Declare the existing funded theory mission through the normal route;
    # there is no supported source agenda yet, so this does not build an object.
    agent._execute('run', {'operator': 'research_route', 'mission': deepcopy(MISSION)}, lab)
    assert not agent.theory_consolidation
    question = {'id': 'shared_query', 'question': 'What executable structure is shared by these typed sources?',
        'kind': 'structural_relation', 'answer_type': 'program',
        'answer_resolution_ref': {'contract': 'ordered_source_reconstruction_only'},
        'query_semantics': {'mode': 'model_query', 'query_expression': shared_query(agent)},
        'use_sites': []}
    before = agent.costs['compute_cost']
    first = agent._execute('run', {'operator': 'research_step', 'research_question': question}, lab)
    assert first['status'] == 'answer_ready_no_parent_use'
    assert first['answer']['status'] == 'answered' and first['answer']['value']['kind'] == 'shared_structure'
    stored = next(item for item in agent.questions if item['id'] == question['id'])
    computed = stored['query_execution']['answer']
    assert computed['compute_cost'] == agent.costs['compute_cost'] - before > 0
    assert computed['source_reconstruction_only'] and not computed['scientific_support']
    assert not computed['independent_confirmation'] and computed['evidence'] == []
    spent = deepcopy(agent.costs)
    snapshot = (agent.workspace / LIBRARY_PATH).read_text()
    second = agent._execute('run', {'operator': 'research_step', 'research_question': {'id': question['id']}}, lab)
    assert second['answer'] == first['answer'] and stored['query_execution']['answer']['cache_hit']
    assert agent.costs == spent and (agent.workspace / LIBRARY_PATH).read_text() == snapshot
    assert_no_new_science(agent, lab)


def test_finite_shared_answer_does_not_erase_unpaired_source_author_work(tmp_path):
    agent, lab = agent_scene(tmp_path)
    agent.research_mission = deepcopy(MISSION)
    originals = sources()
    orphan = theory('unpaired_source', SQUARE,
        context_ref={**originals[0]['context_ref'], 'source_id': 'unpaired_source'},
        context_family_ref='declared_drive_theme')
    register_theories(agent, [*originals, orphan])
    query = shared_query(agent)
    query['typed_program_refs'].append({'kind': 'theory', 'id': orphan['id'], 'version': 1})
    question = {'id': 'partial_shared', 'question': 'Find reconstructible objects in the finite supplied sources',
        'kind': 'structural_relation', 'answer_type': 'program',
        'answer_resolution_ref': {'contract': 'finite_source_reconstruction_only'},
        'query_semantics': {'mode': 'model_query', 'query_expression': query}, 'use_sites': []}
    result = agent._execute('run', {'operator': 'research_step', 'research_question': question}, lab)
    assert result['status'] == 'answer_ready_no_parent_use'
    assert result['answer']['value']['abstractions']
    stored = next(q for q in agent.questions if q['id'] == question['id'])
    pending = deepcopy(stored['pending_author_work'])
    assert pending['status'] == 'author_work_request'
    work = dereference(agent, pending['author_work_ref'])
    assert any('unpaired_source' in item['episode_id'] for item in work)
    before = deepcopy(agent.costs)
    repeated = agent._execute('run', {'operator': 'research_step',
                                    'research_question': {'id': question['id']}}, lab)
    assert repeated['answer'] == result['answer']
    assert stored['pending_author_work'] == pending and agent.costs == before
    assert_no_new_science(agent, lab)
