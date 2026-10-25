"""Saved per-object parent constructors survive actual transfer routing."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import question_routing as routing
from hegel_machine.cognition import revision_relations as revisions
from hegel_machine.cognition import theory_consolidation as consolidation
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.question_routing import register_theories
from test_cross_object_fit_reference import objects
from test_question_routing import CONTEXT, ROLES, theory
from test_revision_relations import binary, constant, feature
from test_theory_consolidation import agent_scene


def test_saved_constructor_reaches_old_transfer_without_assigning_its_cursor_to_new_object(
        tmp_path, monkeypatch):
    agent, _ = agent_scene(tmp_path)
    old, new = objects()
    agent.theory_consolidation = {'candidates': {old['id']: old, new['id']: new}}
    consolidation._save(agent)
    program = binary('mul', constant(5.), feature())
    declaration = theory('native_parent', program)
    register_theories(agent, [declaration])
    constructor_file = tmp_path / 'original_constructor.json'
    constructor_file.write_text(json.dumps({'before_pattern': old['before_pattern'],
        'holes': old['holes'], 'components': {}, 'program': program.to_dict()}))
    reference = lambda *path: {'file': str(constructor_file), 'json_path': list(path)}
    source = {'program_ref': reference('program'), 'pattern_ref': reference('before_pattern'),
        'holes_ref': reference('holes'), 'components_ref': reference('components')}
    parent = {**declaration, 'version': 1, 'program_source_ref': reference('program'),
        '_contract_source': {'source_ref': deepcopy(source), '_pattern': old['before_pattern'],
            '_holes': old['holes'], '_bound_holes': {}}}
    pending = {'layer': 3, 'offset': 8, 'pending_fits': [
        {'indices': [0, 0], 'position': {'layer': 3, 'offset': index}} for index in range(5, 9)]}
    seen = []

    def propose(candidate, supplied, components, allocation, **kwargs):
        seen.append((candidate['id'], deepcopy(kwargs.get('cursor'))))
        assert supplied['_contract_source']['source_ref'] == source
        assert supplied['_contract_source']['_pattern'] == old['before_pattern']
        assert supplied['_contract_source']['_holes'] == old['holes']
        assert supplied['program_source_ref'] == source['program_ref']
        cursor = pending if candidate['id'] == old['id'] else {
            'layer': 0, 'offset': 0, 'pending_fits': []}
        return {'status': 'UNKNOWN', 'proposals': [], 'excluded': [],
            'actual_compute_cost': 1, 'budget_exhausted': True,
            'search_completed': False, 'cursor': deepcopy(cursor)}

    monkeypatch.setattr(revisions, 'propose_revision', propose)
    monkeypatch.setattr(Program, 'predict', lambda *a, **k: pytest.fail('routing does not predict'))
    arguments = {'candidate_id': old['id'], 'theory_id': f'frozen_recall:q@1:{old["id"]}',
        'parent_theory_id': parent['id'], 'component_theory_ids': [],
        '_retained_parent': parent, '_retained_components': [],
        'context_ref': deepcopy(CONTEXT), 'role_signature': deepcopy(ROLES), 'output_unit': 'V',
        'allocation': {'compute': 1000, 'comparison_rows': 97}}
    first = consolidation.transfer_theory(agent, deepcopy(arguments))
    key = first['execution_key']
    original = deepcopy(agent.theory_consolidation['executions'][key])
    # The indexed native Program is a different storage view of the same exact
    # version. Its per-execution constructor lives only in the saved request.
    agent.theory_index[parent['id']]['program_source_ref'] = {'file': 'current.json',
        'json_path': ['theory_index', parent['id'], 'program']}
    basis = {'parent': {key: value for key, value in original['request_basis']['parent'].items()
                        if key != 'contract_source'}, 'components': []}
    question = {'id': 'q', 'meaning_version': 1, 'frozen_recall_execution': {'basis': basis}}
    alias = deepcopy(original)
    alias['request_basis']['candidate_id'] = 'another_saved_object'
    agent.theory_consolidation['executions'][key + ':another_saved_object'] = alias
    loads = []
    loader = routing.load_source_values

    def load(references, workspace):
        loads.append(deepcopy(references))
        return loader(references, workspace)

    monkeypatch.setattr(routing, 'load_source_values', load)
    inputs = routing._retained_revision_inputs(agent, question)
    assert len(loads) == 1 and len(loads[0]) == 3
    assert new['id'] not in inputs['_retained_execution_inputs']
    retained = inputs['_retained_execution_inputs'][old['id']]
    second = consolidation.transfer_theory(agent, {**arguments, **retained})
    assert second['execution_key'] == key
    assert agent.theory_consolidation['executions'][key]['request_basis'] == original['request_basis']
    assert agent.theory_consolidation['executions'][key]['cursor'] == pending
    prior_old = deepcopy(agent.theory_consolidation['executions'][key])
    newest = consolidation.transfer_theory(agent, {**arguments, **retained,
        'candidate_id': new['id'], 'theory_id': f'frozen_recall:q@1:{new["id"]}'})
    assert seen == [(old['id'], None), (old['id'], pending), (new['id'], None)]
    assert agent.theory_consolidation['executions'][key] == prior_old
    assert agent.theory_consolidation['executions'][newest['execution_key']]['cursor'] == {
        'layer': 0, 'offset': 0, 'pending_fits': []}
    assert agent.theory_index[parent['id']]['program_source_ref']['file'] == 'current.json'


@pytest.mark.parametrize('change', ['version', 'source_ref'])
def test_saved_constructor_does_not_relax_parent_version_or_premise(tmp_path, change):
    agent, _ = agent_scene(tmp_path)
    program = binary('mul', constant(5.), feature())
    declaration = theory('parent', program)
    register_theories(agent, [declaration])
    reference = {key: deepcopy(declaration.get(key)) for key in ('id', 'source_ref')}
    reference['version'] = 2 if change == 'version' else 1
    if change == 'source_ref':
        reference['source_ref'] = {'kind': 'different_retained_premise'}
    question = {'id': 'q', 'frozen_recall_execution': {'basis': {'parent': reference}}}
    message = ('Program_version_unavailable' if change == 'version' else 'component_premise_changed')
    with pytest.raises(ValueError, match=message):
        routing._retained_revision_inputs(agent, question)
