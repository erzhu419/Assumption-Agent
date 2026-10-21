"""A declared revision objective spends its existing allowance on joint edits."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.question_object_queries import evaluate_shared_structure
from hegel_machine.cognition.question_routing import register_theories
from hegel_machine.cognition.revision_relations import consolidate_revision_relations
from hegel_machine.cognition.theory_consolidation import LIBRARY_PATH
from test_question_routing import CONTEXT, theory
from test_revision_relations import edit_sources, guarded_sources
from test_theory_consolidation import MISSION, agent_scene, assert_no_new_science, sources


REVISION = 'program_revision_relation'


def revision_scene(tmp_path, originals=None):
    agent, lab = agent_scene(tmp_path)
    originals = originals or edit_sources()
    records, declarations, references, examples = [], [], [], {}
    for index, original in enumerate(originals):
        identifier = f'edit_source_{index}'
        reference = {'file': 'actual_edits.json', 'json_path': ['episodes', index], 'action_cutoff': 0}
        records.append({'task_id': identifier, 'program': original['after_program'],
            'actions': [{'action': 'revise', 'compute_cost': 2, 'details': {
                'before_program': original['before_program'], 'after_program': original['after_program']}}],
            'questions': [], 'status': 'source_failed_outside_local_scope'})
        declarations.append(theory(identifier, Program.from_dict(original['after_program']),
            context_ref={**deepcopy(CONTEXT), 'source_id': identifier}, context_family_ref='local_edits',
            episode_ref={'question_id': identifier, 'meaning_version': 1},
            program_source_ref={**reference,
                'json_path': ['episodes', index, 'actions', 0, 'details', 'after_program']}))
        references.append(reference)
        examples[identifier] = deepcopy(original.get('counterexamples', []))
    agent._save('actual_edits.json', json.dumps({'episodes': records}))
    register_theories(agent, declarations)
    query = {'op': 'FindSharedStructure', 'object_type': REVISION,
        'typed_program_refs': [{'kind': 'theory', 'id': item['id'], 'version': 1} for item in declarations],
        'episode_sources': references, 'counterexamples': examples,
        'mapping_budget': {'compute': 10000, 'max_groups': 8}}
    return agent, lab, query


def test_joint_request_uses_its_exact_engine_allowance_and_reuses_the_result(tmp_path):
    # The same ordinary engine determines the price of this controlled edit;
    # the Common query must not spend part of that allowance on after contexts.
    expected = consolidate_revision_relations(edit_sources(), object_type=REVISION)
    assert expected['status'] == 'completed'
    allowance = expected['actual_compute_cost']
    mixed = consolidate_revision_relations(edit_sources(), allocation={'compute': allowance})
    assert mixed['budget_exhausted'] and not any(item['kind'] == REVISION for item in mixed['candidates'])
    agent, lab, query = revision_scene(tmp_path)
    query['mapping_budget']['compute'] = allowance
    mission = {'mission_id': 'joint_fixture', 'purpose': 'theory_learning',
               'allocation': {'compute': allowance}}
    agent.research_mission = mission
    question = {'id': 'joint', 'meaning_version': 1}
    answer = evaluate_shared_structure(agent, question, query, lab, mission)
    assert answer['status'] == 'answered' and answer['compute_cost'] == allowance
    assert agent.costs['compute_cost'] == mission['resource_usage']['compute'] == allowance
    abstraction, = answer['value']['abstractions']
    assert abstraction['kind'] == REVISION
    frozen = agent.theory_consolidation['candidates'][abstraction['library_ref']['json_path'][-1]]
    assert all(item['before_exact'] and item['after_exact'] for item in frozen['source_reconstruction'])
    assert not answer['scientific_support'] and not frozen['active_library']
    snapshot = (agent.workspace / LIBRARY_PATH).read_text()
    repeated = evaluate_shared_structure(agent, question, query, lab, mission)
    assert repeated['cache_hit'] and repeated['compute_cost'] == 0
    assert repeated['value'] == answer['value']
    assert (agent.workspace / LIBRARY_PATH).read_text() == snapshot
    assert agent.costs['compute_cost'] == allowance
    assert_no_new_science(agent, lab)


def test_direct_consolidation_cache_distinguishes_requested_kind_and_keeps_old_objects(tmp_path):
    agent, lab, query = revision_scene(tmp_path)
    arguments = {'operator': 'theory_consolidation', 'theme': 'same_sources',
        'source_theory_ids': [item['id'] for item in query['typed_program_refs']],
        'episode_sources': query['episode_sources'], 'counterexamples': query['counterexamples'],
        'mission': deepcopy(MISSION), 'allocation': query['mapping_budget']}
    generic = agent._execute('run', arguments, lab)
    assert generic['status'] == 'completed'
    assert {item['kind'] for item in generic['frozen_candidates']} == {'program_context_relation', REVISION}
    old_ids = set(agent.theory_consolidation['candidates'])
    before = agent.costs['compute_cost']
    requested = agent._execute('run', {**arguments, 'object_type': REVISION}, lab)
    assert not requested['cache_hit'] and requested['new_compute_cost'] > 0
    assert requested['new_compute_cost'] == agent.costs['compute_cost'] - before
    assert {item['kind'] for item in requested['frozen_candidates']} == {REVISION}
    assert old_ids <= set(agent.theory_consolidation['candidates'])
    repeated = agent._execute('run', {**arguments, 'object_type': REVISION}, lab)
    assert repeated['cache_hit'] and repeated['new_compute_cost'] == 0
    assert agent.theory_consolidation['executions']['same_sources']['request_basis']['object_type'] == REVISION
    assert_no_new_science(agent, lab)


def test_common_query_changed_object_type_uses_a_new_meaning_version(tmp_path):
    agent, lab, query = revision_scene(tmp_path)
    agent.research_mission = deepcopy(MISSION)
    generic = deepcopy(query)
    generic.pop('object_type')
    question = {'id': 'shared', 'question': 'Which executable structure do these recorded edits share?',
        'kind': 'structural_relation', 'answer_type': 'program', 'use_sites': [],
        'answer_resolution_ref': {'contract': 'ordered_source_reconstruction_only'},
        'query_semantics': {'mode': 'model_query', 'query_expression': generic}}
    first = agent._execute('run', {'operator': 'research_step', 'research_question': question}, lab)
    assert first['answer']['status'] == 'answered'
    changed = deepcopy(question)
    changed['query_semantics']['query_expression'] = query
    second = agent._execute('run', {'operator': 'research_step', 'research_question': changed}, lab)
    stored = next(item for item in agent.questions if item['id'] == question['id'])
    assert stored['meaning_version'] == 2
    assert second['answer']['status'] == 'answered'
    assert {item['kind'] for item in second['answer']['value']['abstractions']} == {REVISION}
    basis = agent.theory_consolidation['executions']['FindSharedStructure:shared@2']['request_basis']
    assert basis['question_meaning'] == {'id': 'shared', 'meaning_version': 2}
    assert_no_new_science(agent, lab)


@pytest.mark.parametrize('mission', [{'purpose': 'task'},
    {'purpose': 'theory_learning', 'allocation': {'compute': 4}}])
def test_revision_query_respects_the_existing_funding_and_compute_caps(tmp_path, mission):
    agent, lab, query = revision_scene(tmp_path)
    mission = deepcopy(mission)
    answer = evaluate_shared_structure(agent, {'id': 'joint'}, query, lab, mission)
    if mission['purpose'] == 'task':
        assert answer['status'] == 'research_deferred' and answer['reason'] == 'theory_allocation_missing'
        assert answer['compute_cost'] == 0
    else:
        assert answer['status'] == 'scoped_unknown' and answer['budget_exhausted']
        assert answer['partial_result']['abstractions'] == []
        assert 0 < answer['compute_cost'] <= 4
        assert mission['resource_usage']['compute'] == answer['compute_cost']
    assert agent.costs['compute_cost'] == answer['compute_cost']
    assert_no_new_science(agent, lab)


def test_missing_actual_revisions_and_unlinked_parents_are_concrete_unknowns(tmp_path):
    agent, lab = agent_scene(tmp_path / 'no_revisions')
    register_theories(agent, sources())
    query = {'op': 'FindSharedStructure', 'object_type': REVISION,
        'typed_program_refs': [{'kind': 'theory', 'id': identifier, 'version': 1}
                               for identifier in ('source_a', 'source_b')],
        'mapping_budget': {'compute': 10000}}
    missing = evaluate_shared_structure(agent, {'id': 'joint'}, query, lab, MISSION)
    assert missing['status'] == 'scoped_unknown'
    assert missing['excluded'][0]['reason'] == 'two_actual_revision_sources_required'
    assert missing['partial_result']['abstractions'] == []
    assert_no_new_science(agent, lab)
    agent, lab, query = revision_scene(tmp_path / 'unlinked_parents', guarded_sources())
    unlinked = evaluate_shared_structure(agent, {'id': 'joint'}, query, lab, MISSION)
    assert unlinked['status'] == 'scoped_unknown'
    assert unlinked['partial_result']['abstractions'] == []
    assert any(item['reason'] == 'no_shared_parent_subprogram_in_revision' for item in unlinked['excluded'])
    assert not agent.theory_consolidation['candidates']
    assert_no_new_science(agent, lab)


def test_joint_engine_requires_the_actual_parent_reference():
    records = edit_sources()
    records[0].pop('before_ref')
    result = consolidate_revision_relations(records, object_type=REVISION)
    assert result['status'] == 'UNKNOWN' and result['candidates'] == []
    assert any(item['reason'] == 'actual_before_ref_required' for item in result['excluded'])
