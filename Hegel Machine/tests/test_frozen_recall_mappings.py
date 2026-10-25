"""Finite typed mappings remain competing candidates, never an authored answer."""

from copy import deepcopy
import json

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.theory_consolidation import LIBRARY_PATH, recall_frozen_theories
from test_frozen_theory_recall import frozen_scene
from test_question_routing import CONTEXT, ROLES, SQUARE, future_question, probe, route, theory
from test_theory_consolidation import agent_scene, assert_no_new_science, consolidate


def two_role_question(agent, identifier='both_drives'):
    question = future_question(agent, identifier)
    question['theory_target']['role_signature'] = {
        'u': deepcopy(ROLES['u']), 'v': deepcopy(ROLES['u'])}
    binding = question['query_semantics']['query_expression']['binding']
    binding.update(features={'u': 3., 'v': 5.},
                   action={'kind': 'trajectory', 'sequence': [{'u': 3., 'v': 5.}]})
    return question


def test_common_consumes_both_legal_mappings_without_removing_the_old_program(tmp_path, monkeypatch):
    agent, lab = frozen_scene(tmp_path)
    original = deepcopy(agent.theory_consolidation['candidates'])
    experiment = probe(agent)
    experiment['actions'] = [{'kind': 'trajectory', 'sequence': [{'u': 2., 'v': 4.}]}]
    experiment['query']['binding'].update(features={'u': 2., 'v': 4.},
                                         action=deepcopy(experiment['actions'][0]))

    result = route(agent, lab, question=two_role_question(agent),
        theories=[theory('old_square', SQUARE)], experiments=[experiment], mission={'purpose': 'task'})

    recall = result['frozen_recall']
    assert recall['status'] == 'completed'
    assert [item['mapping'] for item in recall['work']] == [{'role_0': 'u'}, {'role_0': 'v'}]
    identifiers = recall['generated_theory_ids']
    assert len(identifiers) == 2 and 'old_square' in agent.theory_index
    assert sorted(group['task_answer'] for group in result['selection']['groups']) == [7., 9., 11.]
    assert set(identifiers).issubset({identifier for group in result['selection']['groups']
                                    for identifier in group['theory_ids']})
    assert agent.theory_consolidation['candidates'] == original
    for identifier in identifiers:
        support = agent.theory_index[identifier]['support']
        assert support['status'] == 'candidate' and support['evidence'] == []
    assert not recall['support_updated'] and not recall['independent_confirmation']
    # Actual enumeration, pair checks and source-parameter instantiation:
    # one candidate + two * (one mapping + one role + five Program nodes).
    assert recall['new_compute_cost'] == 15
    spent = deepcopy(agent.costs)
    writes = []
    save = agent._save

    def track_save(path, content):
        if path == LIBRARY_PATH:
            writes.append(path)
        return save(path, content)

    monkeypatch.setattr(agent, '_save', track_save)
    repeated = route(agent, lab, question={'id': 'both_drives'}, mission={'purpose': 'task'})
    assert repeated['cache_hit'] and repeated['frozen_recall']['cache_hit']
    assert repeated['new_compute_cost'] == 0 and agent.costs == spent and writes == []
    assert_no_new_science(agent, lab)


def test_each_role_unit_and_time_contract_is_checked_before_registration(tmp_path):
    agent, lab = frozen_scene(tmp_path)
    question = two_role_question(agent)
    target = question['theory_target']['role_signature']
    target['v']['unit'] = 'mV'
    target['w'] = {**ROLES['u'], 'time': 'previous'}
    target['z'] = {**ROLES['u'], 'role': 'reference'}
    current = agent._open_question(question)

    result = recall_frozen_theories(agent, current, {})

    assert [item['mapping'] for item in result['work']] == [{'role_0': 'u'}]
    assert {item['mapping']['role_0'] for item in result['excluded']} == {'v', 'w', 'z'}
    assert all(item['reason'] == 'role_unit_time_contract_mismatch' for item in result['excluded'])
    assert len(result['generated_theory_ids']) == len(agent.theory_index) == 1
    assert_no_new_science(agent, lab)


def test_two_source_roles_are_injectively_rebound_in_both_orders(tmp_path):
    agent, lab = agent_scene(tmp_path)
    roles = {'u': deepcopy(ROLES['u']), 'v': deepcopy(ROLES['u'])}
    declarations = []
    for identifier, coefficient in [('left', 2.), ('right', 4.)]:
        program = Program('add', children=(Program('mul', children=(
            Program('const', value=coefficient), Program('feature', name='u'))),
            Program('feature', name='v')))
        declarations.append(theory(identifier, program, role_signature=roles,
            context_ref={**CONTEXT, 'source_id': identifier}, context_family_ref='two_drives',
            parameter_paths=[[0, 0]]))
    consolidate(agent, lab, theories=declarations, source_theory_ids=['left', 'right'])
    current = agent._open_question(two_role_question(agent))

    result = recall_frozen_theories(agent, current, {})

    assert [item['mapping'] for item in result['work']] == [
        {'role_0': 'u', 'role_1': 'v'}, {'role_0': 'v', 'role_1': 'u'}]
    predictions = [Program.from_dict(agent.theory_index[identifier]['program']).predict(
        Observation({'u': 3., 'v': 5.}, 0.)) for identifier in result['generated_theory_ids']]
    assert predictions == [11., 13.]
    assert_no_new_science(agent, lab)


def test_budget_cutoff_is_unknown_after_one_mapping_then_resumes_with_stable_ids_and_caches(tmp_path):
    agent, lab = frozen_scene(tmp_path)
    current = agent._open_question(two_role_question(agent))
    before = agent.costs['compute_cost']

    partial = recall_frozen_theories(agent, current, {'allocation': {'compute': 8}})

    assert partial['status'] == 'UNKNOWN' and partial['new_compute_cost'] == 8
    first, = partial['generated_theory_ids']
    assert [item['mapping'] for item in partial['work']] == [{'role_0': 'u'}]
    repeated = recall_frozen_theories(agent, current, {'allocation': {'compute': 8}})
    # An incomplete owner remains available in the persistent coverage round.
    # The unchanged short call pays its metadata/pattern work again; it is not
    # a completed-search cache hit and does not invent the second mapping.
    assert not repeated['cache_hit'] and repeated['new_compute_cost'] == 8
    assert repeated['generated_theory_ids'] == [first] and not repeated['search_completed']
    assert repeated['object_cursor'] == partial['object_cursor']
    assert agent.costs['compute_cost'] == before + 16
    resumed = recall_frozen_theories(agent, current, {'allocation': {'compute': 15}})
    assert resumed['status'] == 'completed'
    assert resumed['generated_theory_ids'] == [first, first + ':binding:1']
    assert resumed['new_compute_cost'] == 15
    assert len(agent.theory_index) == 2
    assert_no_new_science(agent, lab)


def test_a_failed_mapping_is_retained_while_another_mapping_remains_available(tmp_path):
    agent, lab = frozen_scene(tmp_path)
    question = two_role_question(agent)
    question['theory_target']['evidence_checks'] = [{
        'source_ref': {'file': 'retained_fixture_rows.json', 'json_path': [0]},
        'context_ref': deepcopy(CONTEXT),
        'observation': {'features': {'u': 0., 'v': 2.}, 'target': 5., 'history': []}}]
    current = agent._open_question(question)

    first = recall_frozen_theories(agent, current, {})

    assert [(item['mapping'], item['status']) for item in first['work']] == [
        ({'role_0': 'u'}, 'refuted'), ({'role_0': 'v'}, 'candidate_registered')]
    assert len(first['generated_theory_ids']) == 1
    rejected = first['work'][0]['theory_ids'][0]
    assert agent.theory_index[rejected]['compatible'] is False
    checks = deepcopy(agent.theory_consolidation['transfers'][0]['checks'])
    current['theory_target'].pop('evidence_checks')
    omitted = recall_frozen_theories(agent, current, {})
    assert omitted['work'][0]['status'] == 'refuted'
    assert omitted['generated_theory_ids'] == first['generated_theory_ids']
    assert agent.theory_consolidation['transfers'][0]['checks'] == checks
    frozen = next(iter(agent.theory_consolidation['candidates'].values()))
    assert all(item['exact'] for item in frozen['source_reconstruction'])
    assert_no_new_science(agent, lab)


def test_the_new_mapping_method_reopens_the_actual_form_of_an_old_unique_mapping_cache(tmp_path, monkeypatch):
    from hegel_machine.cognition import theory_consolidation
    agent, lab = frozen_scene(tmp_path)
    current = agent._open_question(two_role_question(agent))
    candidates = agent.theory_consolidation['candidates']
    current['frozen_recall_execution'] = {'allowance': 50000, 'basis': {
        'target': deepcopy(current['theory_target']), 'checks': [],
        'candidates': [{'id': key, 'version': item['version']} for key, item in candidates.items()],
        'components': []}, 'result': {'status': 'author_work_request',
        'generated_theory_ids': [], 'new_compute_cost': 3,
        'author_work': [{'reason': 'unique_typed_role_mapping_required'}]}}

    retained = deepcopy(current['frozen_recall_execution'])

    def unsupported_equivalence(*args, **kw):
        raise AssertionError('An unknown old library path cannot prove cross-library equivalence')

    monkeypatch.setattr(theory_consolidation, '_same_library_candidate', unsupported_equivalence)
    result = recall_frozen_theories(agent, current, {})

    assert not result['cache_hit'] and len(result['generated_theory_ids']) == 2
    assert retained['basis'].get('library_path') is None
    assert retained['result']['author_work'] == [{'reason': 'unique_typed_role_mapping_required'}]
    assert current['frozen_recall_execution']['basis']['method_ref'].endswith('AbstractTemplate.mappings')
    assert len(json.loads((agent.workspace / LIBRARY_PATH).read_text())['transfers']) == 2
    assert_no_new_science(agent, lab)
