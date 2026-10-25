"""An interrupted development search retains its original typed versions."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from hegel_machine.cognition import registered_source_scene as scenes
from hegel_machine.cognition.language import Program


def scene(tmp_path, available):
    parent = {'id': 'parent', 'version': 2, 'program': Program('const', value=1.).to_dict(),
        'role_signature': {}, 'output_unit': '1', 'context_ref': {'id': 'source'},
        '_contract_source': {'source_ref': {'file': 'constructor.json', 'json_path': []}}}
    old = [{'id': f'component{i}', 'version': 1, 'program': Program('const', value=float(i)).to_dict(),
            'role_signature': {}, 'output_unit': '1', 'context_ref': {'id': 'source'},
            'subtree_contracts': {'': {'output_unit': '1', 'time_contract': 'current'}}}
           for i in range(4)]
    current = [{**deepcopy(item), 'version': 2, 'subtree_contracts': {'new': 'contract'},
                'history': [deepcopy(item)]} for item in old]
    arguments = {'library_path': scenes.LIBRARY_PATH, 'candidate_id': 'new',
        'parent_theory_id': 'parent', 'component_theory_ids': [item['id'] for item in old],
        'role_signature': {}, 'output_unit': '1', 'context_ref': {'id': 'source'},
        'fitting': {'operations': ['refit']}, 'evidence_checks': [],
        'allocation': {'compute': 150000, 'comparison_rows': 97}}
    basis = {key: deepcopy(arguments.get(key)) for key in (
        'component_theory_ids', 'component_bindings', 'role_signature', 'output_unit',
        'context_ref', 'evidence_checks', 'allocation', 'parent_theory_id', 'fitting')}
    basis.update(library_path=scenes.LIBRARY_PATH, candidate_id='new', candidate_version=1,
        parent={'id': 'parent', 'version': 2, 'source_ref': None, 'source_status': None,
                'contract_source': parent['_contract_source']['source_ref']},
        components=[{'id': item['id'], 'version': 1, 'source_ref': None,
                     'compatible': None, 'source_status': None} for item in old])
    original = {'request_basis': basis, 'allowance': 150000, 'search_completed': False,
        'cursor': {'layer': 3, 'offset': 8, 'pending_fits': [{'indices': [1]}]},
        'result': {'status': 'UNKNOWN', 'new_compute_cost': 140045}}
    misbound = deepcopy(original)
    misbound['cursor'] = {'layer': 0, 'offset': 1}
    misbound['result']['new_compute_cost'] = 44642
    for item in misbound['request_basis']['components']:
        item['version'] = 2
    key = 'relation_transfer:frozen_recall:native_public_response@1:new'
    agent = SimpleNamespace(workspace=tmp_path, questions=[], relation_research={}, events=[],
        theory_index={item['id']: item for item in current},
        costs={'compute_cost': 999999, 'python_execution_seconds': 10.},
        theory_consolidation={'candidates': {'new': {'version': 1}} if available else {},
            'executions': {key: misbound if available else deepcopy(original)},
            'transfers': [{'status': 'failed', 'original_cost': 7}]})
    registration = {'arms': {'remove': {'candidate_versions': []},
                             'keep': {'candidate_versions': [{'id': 'new', 'version': 1}]}}}
    return {'agent': agent, 'parent': parent, 'retained_components': current,
            'transfer_arguments': arguments}, registration, original, misbound, key


@pytest.mark.parametrize('available', [False, True])
def test_pins_real_history_without_replacing_latest_pool_or_erasing_misbound_costs(
        tmp_path, monkeypatch, available):
    restored, registration, original, misbound, key = scene(tmp_path, available)
    agent = restored['agent']
    before = deepcopy((agent.theory_index, agent.costs, agent.theory_consolidation['transfers']))
    saved = []
    monkeypatch.setattr(scenes, 'save_theory_library', lambda agent: saved.append(True))
    monkeypatch.setattr(Program, 'predict', lambda *args: pytest.fail('pinning must not predict'))
    receipt = scenes.pin_registered_development_work(restored, registration, original)
    assert saved == [True]
    assert receipt['pinned_historical_version_count'] == 4
    assert receipt['cursor_layer'] == 3 and receipt['cursor_offset'] == 8
    assert all(item['version'] == 1 and item['subtree_contracts'].get('')
               for item in restored['retained_components'])
    assert (agent.theory_index, agent.costs, agent.theory_consolidation['transfers']) == before
    assert agent.theory_consolidation['executions'][key] == original
    preserved = receipt['preserved_misbound_execution_key']
    if available:
        assert preserved == key + ':misbound:0'
        assert agent.theory_consolidation['executions'][preserved] == misbound
    else:
        assert preserved is None and not agent.theory_consolidation['candidates']


def test_unavailable_exact_history_does_not_substitute_current_version(tmp_path, monkeypatch):
    restored, registration, original, _, _ = scene(tmp_path, True)
    agent = restored['agent']
    agent.theory_index['component0']['history'] = []
    previous = deepcopy(agent.theory_consolidation)
    monkeypatch.setattr(scenes, 'save_theory_library', lambda *args: pytest.fail('missing version must not save'))
    with pytest.raises(ValueError, match='Program version is unavailable'):
        scenes.pin_registered_development_work(restored, registration, original)
    assert agent.theory_consolidation == previous
