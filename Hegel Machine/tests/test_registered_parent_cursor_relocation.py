"""A restored195 duty retains its exact constructor and each object's own progress."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from hegel_machine.cognition import registered_source_scene as scenes
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.theory_consolidation import (
    LIBRARY_PATH, _relation_request_basis, _shared_relation_progress,
    _recall_work_groups,
)


QUESTION = {'question_id': 'parent_revision:task:native_public_response', 'meaning_version': 1}
PREFIX = f'relation_transfer:frozen_recall:{QUESTION["question_id"]}@1:'
OLD_PREFIX = 'relation_transfer:frozen_recall:native_public_response@1:'
PARENT = 'frozen_recall:native_public_response@1:consolidated:0:relation:revision:0:binding:1:3:fit:1'
OWNERS = ('consolidated:0:relation:revision:0', 'consolidated:0:relation:revision:1')
NEW20 = 'consolidated:19:relation:revision:0'


def constructor(origin, frozen):
    return {
        'program_ref': {'file': str(origin / 'research_state.json'),
            'json_path': ['theory_index', PARENT, 'program'], 'version': 1},
        'pattern_ref': {'file': str(frozen),
            'json_path': ['candidates', OWNERS[0], 'definition', 'body']},
        'holes_ref': {'file': str(frozen), 'json_path': ['candidates', OWNERS[0], 'holes']},
        'transfer_ref': {'file': str(origin / LIBRARY_PATH), 'json_path': ['transfers', 10]},
        'components_ref': {'file': str(origin / LIBRARY_PATH),
            'json_path': ['transfers', 10, 'binding', 'components']},
    }


@pytest.fixture
def restored_scene(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('own195 constructor reuse must use restored RAM, without source IO or prediction')

    monkeypatch.setattr(scenes, 'load_source_values', forbidden)
    monkeypatch.setattr(Program, 'predict', forbidden)
    origins = (tmp_path / 'attempt_03/remove', tmp_path / 'attempt_02/remove')
    workspace = tmp_path / 'next/remove'
    frozen = tmp_path / 'frozen/shared/theory_library.json'
    program = Program('add', children=(Program('feature', name='u'),
        Program('const', value=.25))).to_dict()
    bound = {'hole_0': {'theory_id': 'retained_component', 'version': 1}}
    parent = {'id': PARENT, 'version': 1, 'program': program,
        'program_source_ref': constructor(workspace, frozen)['program_ref'],
        '_contract_source': {'source_ref': constructor(workspace, frozen),
            '_pattern': deepcopy(program), '_holes': {'hole_0': {'unit': '1'}},
            '_bound_holes': deepcopy(bound)}}
    components = [{'id': 'retained_component', 'version': 1,
        'program': Program('feature', name='u').to_dict()}]
    arguments = {'parent_theory_id': PARENT, 'component_theory_ids': ['retained_component'],
        'role_signature': {'u': {'role': 'input', 'unit': '1', 'time': 'current'}},
        'output_unit': '1', 'context_ref': {'id': 'retained97'}, 'evidence_checks': [],
        'fitting': {'operations': ['refit'], 'risk_quantile': .5,
            'max_history': 4, 'max_structures': 64}, 'allocation': {'comparison_rows': 97}}
    candidates = {key: {'id': key, 'version': 1, 'kind': 'typed_program_family'}
                  for key in (*OWNERS, NEW20)}
    executions = {}
    for index, (identifier, origin) in enumerate(zip(OWNERS, origins)):
        old_parent = {**parent, '_contract_source': {
            **deepcopy(parent['_contract_source']), 'source_ref': constructor(origin, frozen)}}
        executions[PREFIX + identifier] = {
            'request_basis': _relation_request_basis(old_parent, components,
                arguments, candidates[identifier], LIBRARY_PATH),
            'cursor': {'layer': index + 1, 'offset': index + 2,
                'pending_fits': [{'position': {'layer': index, 'offset': 1}}]},
            'search_completed': False, 'result': {}, 'allowance': 217350}
    # Real histories carry old65 and current195 searches at the same time.
    for identifier in (*OWNERS, NEW20):
        executions[OLD_PREFIX + identifier] = {'cursor': {'layer': 65, 'offset': 65}}
    transfers = [{} for _ in range(11)]
    transfers[10] = {'binding': {'components': deepcopy(bound)}}
    agent = SimpleNamespace(workspace=workspace,
        theory_consolidation={'candidates': candidates, 'executions': executions,
            'transfers': transfers},
        costs={'compute_cost': 2051398, 'python_execution_seconds': 861.6881658016937})
    scene = {'agent': agent, 'source': origins[-1],
        'previous': {'theory_index': {PARENT: {'program': deepcopy(program)}}}}
    retained = {'_retained_parent': parent, '_retained_components': components,
        'component_theory_ids': ['retained_component']}
    return scene, retained, arguments, origins


def test_relocated_constructor_matches_each_original195_cursor_without_IO(restored_scene):
    scene, retained, arguments, origins = restored_scene
    agent = scene['agent']
    before = deepcopy((agent.theory_consolidation, agent.costs, retained))
    actual = scenes._registered_retained_parent_inputs(scene, retained, QUESTION)
    for identifier, origin in zip(OWNERS, origins):
        candidate = agent.theory_consolidation['candidates'][identifier]
        key = PREFIX + identifier
        old = agent.theory_consolidation['executions'][key]
        without_relocation = _relation_request_basis(retained['_retained_parent'],
            retained['_retained_components'], arguments, candidate, LIBRARY_PATH)
        assert _shared_relation_progress(agent, arguments, candidate, without_relocation,
            key, agent.theory_consolidation['executions']) == (None, [])
        parent = actual[identifier]['_retained_parent']
        source = parent['_contract_source']['source_ref']
        original = old['request_basis']['parent']['contract_source']
        assert source == original
        assert parent['program_source_ref'] == original['program_ref']
        assert source['program_ref']['file'] == str(origin / 'research_state.json')
        # Exactly the three local addresses relocate. Frozen pattern and unit
        # declarations keep their original external identities.
        current = retained['_retained_parent']['_contract_source']['source_ref']
        assert {key for key in source if source[key] != current[key]} == {
            'program_ref', 'transfer_ref', 'components_ref'}
        basis = _relation_request_basis(parent, retained['_retained_components'],
            arguments, candidate, LIBRARY_PATH)
        matched, references = _shared_relation_progress(agent, arguments, candidate,
            basis, key, agent.theory_consolidation['executions'])
        assert references == [key]
        assert matched['cursor'] == old['cursor']
        assert matched['cursor'] != agent.theory_consolidation['executions'][
            OLD_PREFIX + identifier]['cursor']
    assert (agent.theory_consolidation, agent.costs, retained) == before


def test_unvisited_new20_does_not_borrow_old65_progress(restored_scene):
    scene, retained, arguments, _ = restored_scene
    actual = scenes._registered_retained_parent_inputs(scene, retained, QUESTION)
    assert actual[NEW20] is retained
    agent = scene['agent']
    candidate = agent.theory_consolidation['candidates'][NEW20]
    basis = _relation_request_basis(actual[NEW20]['_retained_parent'],
        retained['_retained_components'], arguments, candidate, LIBRARY_PATH)
    assert _shared_relation_progress(agent, arguments, candidate, basis,
        PREFIX + NEW20, agent.theory_consolidation['executions']) == (None, [])


def test_first195_binding_needs_no_source_or_constructor():
    parent = {'id': PARENT, 'version': 1, 'program': Program('feature', name='u').to_dict()}
    agent = SimpleNamespace(theory_consolidation={'candidates': {'old': {}, 'new': {}},
        'executions': {OLD_PREFIX + 'old': {'cursor': {'layer': 65, 'offset': 65}}}})
    retained = {'_retained_parent': parent, '_retained_components': [],
        'component_theory_ids': []}
    result = scenes._registered_retained_parent_inputs({'agent': agent}, retained, QUESTION)
    assert result == {'old': retained, 'new': retained}
    assert result['old'] is retained and result['new'] is retained


@pytest.mark.parametrize('change,reason', [
    ('version', 'different parent version'),
    ('program', 'Program differs'),
    ('binding', 'binding changed'),
    ('hole_reference', 'differs beyond local relocation'),
])
def test_a_different_parent_or_contract_cannot_reuse_saved_progress(restored_scene, change, reason):
    scene, retained, _, _ = restored_scene
    changed = deepcopy(retained)
    parent = changed['_retained_parent']
    if change == 'version':
        parent['version'] += 1
    elif change == 'program':
        parent['program'] = Program('const', value=-1.).to_dict()
    elif change == 'binding':
        parent['_contract_source']['_bound_holes']['hole_0']['version'] += 1
    else:
        parent['_contract_source']['source_ref']['holes_ref']['json_path'][-1] = 'different_units'
    with pytest.raises(ValueError, match=reason):
        scenes._registered_retained_parent_inputs(scene, changed, QUESTION)


@pytest.mark.parametrize('alias_offset', [0, 2])
def test_alias_group_keeps_original_shape_and_furthest_paid195_work(restored_scene, alias_offset):
    scene, retained, arguments, origins = restored_scene
    agent, owner = scene['agent'], OWNERS[0]
    alias = 'consolidated:0:relation:revision:3'
    definition = {'body': retained['_retained_parent']['program'], 'arguments': []}
    library = agent.theory_consolidation
    library['candidates'][owner].update(kind='program_revision_relation', definition=definition)
    library['candidates'][alias] = {**deepcopy(library['candidates'][owner]), 'id': alias}
    library['executions'][PREFIX + owner]['cursor'] = {'layer': 0, 'offset': 1, 'pending_fits': []}
    partial = deepcopy(library['executions'][PREFIX + owner])
    partial['request_basis']['candidate_id'] = alias
    partial['request_basis']['parent']['contract_source'] = constructor(
        origins[1], origins[0].parent.parent / 'frozen/shared/theory_library.json')
    # Keep the exact shared frozen definition addresses; only the local origin
    # differs in this saved alias, just as in successive completed scenes.
    for field in ('pattern_ref', 'holes_ref'):
        partial['request_basis']['parent']['contract_source'][field] = deepcopy(
            library['executions'][PREFIX + owner]['request_basis']['parent']['contract_source'][field])
    partial['cursor'] = {'layer': 0, 'offset': alias_offset, 'pending_fits': []}
    library['executions'][PREFIX + alias] = partial
    common = {key: retained for key in library['candidates']}
    stock = _recall_work_groups(agent, library['candidates'], library, common,
        prefix=PREFIX.removeprefix('relation_transfer:'))
    assert [owner, alias] in stock
    before = deepcopy(library['executions'])
    actual = scenes._registered_retained_parent_inputs(scene, retained, QUESTION)
    assert _recall_work_groups(agent, library['candidates'], library, actual,
        prefix=PREFIX.removeprefix('relation_transfer:')) == stock
    assert actual[owner] is actual[alias]
    group = next(group for group in stock if owner in group)
    candidate = library['candidates'][owner]
    basis = _relation_request_basis(actual[owner]['_retained_parent'],
        retained['_retained_components'], arguments, candidate, LIBRARY_PATH)
    result, refs = _shared_relation_progress(agent, {
        **arguments, '_execution_candidate_ids': group,
        '_execution_candidates': library['candidates']}, candidate, basis, PREFIX + owner,
        library['executions'])
    chosen = alias if alias_offset > 1 else owner
    assert result['execution_key'] == PREFIX + chosen
    assert result['cursor']['offset'] == max(1, alias_offset)
    assert PREFIX + chosen in refs
    assert library['executions'] == before
    record = next(x for x in scene['registered_parent_input_groups'] if owner in x['candidate_ids'])
    assert record['selected_execution_key'] == PREFIX + chosen
    assert record['retained_prior_positions'][alias] == {'layer': 0, 'offset': alias_offset}
    assert actual[NEW20] is retained


@pytest.fixture
def rejected_owner_scene(restored_scene):
    scene, retained, arguments, _ = restored_scene
    scene['source'].mkdir(parents=True)
    scene['starting_costs'] = deepcopy(scene['agent'].costs)
    expected = {'candidate_id': OWNERS[0], 'search_position': {'layer': 1, 'offset': 1}}
    previous = {'action': 'frozen_recall_on_registered_parent_duty', 'unit_index': 1,
        'research_question_ref': deepcopy(QUESTION), 'registered_unit': deepcopy(expected),
        'feedback': {'object_cursor': {'next_candidate_id': OWNERS[1]},
            'complete_research_visits': [{'status': 'complete_visit_executed',
                'candidate_id': OWNERS[0], 'search_position': {'layer': 0, 'offset': 1}}]}}
    actions = [previous, {'action': 'research_choice', 'truncated': False},
        {'action': 'execution_failure', 'exception': 'ValueError',
            'reason': 'actual complete unit differs from the registered persistent-order plan'}]
    path = scene['source'] / 'registered_actions.json'
    path.write_text(json.dumps(actions))
    registration = {'parent_research': {'complete_unit_compute_limits': {
        'remove': [deepcopy(expected)], 'keep': [deepcopy(expected)]}}}
    question = {'frozen_recall_execution': {'cursor': {'next_candidate_id': OWNERS[1]},
        'basis': {'meaning_version': 1}, 'result': {'search_completed': False}}}
    return scene, registration, question, actions, path


@pytest.mark.parametrize('arm', ['remove', 'keep'])
def test_rejected_same_owner_wrong_position_restores_only_outer_order(rejected_owner_scene, arm):
    scene, registration, question, _, path = rejected_owner_scene
    scene['registered_arm'] = arm
    before = deepcopy((scene['agent'].theory_consolidation, scene['agent'].costs))
    original_bytes = path.read_bytes()
    receipt = scenes._reconcile_registered_parent_order(scene, registration, question, QUESTION)
    assert receipt['before'] == OWNERS[1] and receipt['after'] == OWNERS[0]
    assert receipt['expected_position'] == {'layer': 1, 'offset': 1}
    assert receipt['rejected_actual_position'] == {'layer': 0, 'offset': 1}
    assert question['frozen_recall_execution']['cursor']['next_candidate_id'] == OWNERS[0]
    assert (scene['agent'].theory_consolidation, scene['agent'].costs) == before
    assert path.read_bytes() == original_bytes
    assert scenes._reconcile_registered_parent_order(scene, registration, question, QUESTION)[
        'status'] == 'registered_fair_order_preserved'


@pytest.mark.parametrize('deviation', ['unknown_pointer', 'different_failure', 'wrong_actual_owner',
                                      'already_completed_position', 'different_registered_owner', 'fees'])
def test_unknown_outer_cursor_or_fee_difference_is_not_reset(rejected_owner_scene, deviation):
    scene, registration, question, actions, path = rejected_owner_scene
    if deviation == 'unknown_pointer':
        question['frozen_recall_execution']['cursor']['next_candidate_id'] = NEW20
    elif deviation == 'different_failure':
        actions[-1]['reason'] = 'another failure'
    elif deviation == 'wrong_actual_owner':
        actions[0]['feedback']['complete_research_visits'][0]['candidate_id'] = NEW20
    elif deviation == 'already_completed_position':
        actions[0]['feedback']['complete_research_visits'][0]['search_position'] = {'layer': 1, 'offset': 1}
    elif deviation == 'different_registered_owner':
        registration['parent_research']['complete_unit_compute_limits']['remove'][0]['candidate_id'] = NEW20
    else:
        scene['agent'].costs['compute_cost'] -= 1
    path.write_text(json.dumps(actions))
    before = deepcopy(question)
    with pytest.raises(ValueError, match='registered'):
        scenes._reconcile_registered_parent_order(scene, registration, question, QUESTION)
    assert question == before
