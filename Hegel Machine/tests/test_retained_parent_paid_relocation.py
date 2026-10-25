"""A relocated equal paid trace preserves the original typed constructor."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import theory_consolidation as consolidation
from hegel_machine.cognition import revision_relations as revisions
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import SearchBudget
from test_theory_consolidation import agent_scene


def bindings(tmp_path):
    rows = [{'features': {'u': 1.}, 'target': 2., 'history': []},
            {'features': {'u': 2.}, 'target': 4., 'history': [{'u': 1.}]}]
    old_file, new_file = tmp_path / 'old_rows.json', tmp_path / 'current_rows.json'
    for path in (old_file, new_file):
        path.write_text(json.dumps(rows))
    original = {'library_path': 'shared/theory_library.json', 'candidate_id': 'producer',
        'candidate_version': 1, 'components': {'hole_0': {'component_id': 'input', 'version': 3}},
        'parent': {'component_id': 'before', 'version': 1},
        'context_ref': {'source_id': 'same_source'}, 'fitting': {
            'parameters': [.125], 'parameter_paths': [[1]], 'fitted_rows': 2,
            'evidence_refs': [{'file': str(old_file), 'json_path': [index], 'version': 1}
                              for index in range(2)]}}
    relocated = deepcopy(original)
    for reference in relocated['fitting']['evidence_refs']:
        reference['file'] = str(new_file)
    return original, relocated, rows, new_file


def test_exact_binding_never_reads_rows_and_equal_relocation_batches_once(tmp_path, monkeypatch):
    agent, _ = agent_scene(tmp_path)
    original, relocated, _, _ = bindings(tmp_path)
    before = deepcopy((original, relocated))
    calls, resolve = [], consolidation.resolve_program_sources

    def counted(agent, references):
        calls.append(deepcopy(references))
        return resolve(agent, references)

    monkeypatch.setattr(consolidation, 'resolve_program_sources', counted)
    assert consolidation._same_retained_parent_binding(agent, original, deepcopy(original))
    assert calls == []
    assert consolidation._same_retained_parent_binding(agent, original, relocated)
    assert len(calls) == 1 and len(calls[0]) == 4
    assert (original, relocated) == before


@pytest.mark.parametrize('difference', [
    'row_value', 'row_history', 'path_order', 'reference_version',
    'candidate_version', 'component_version', 'parameters', 'unit'])
def test_relocated_rows_do_not_relax_paths_parameters_versions_or_units(tmp_path, difference):
    agent, _ = agent_scene(tmp_path)
    original, relocated, rows, new_file = bindings(tmp_path)
    if difference == 'row_value':
        rows[0]['target'] += .25
    elif difference == 'row_history':
        rows[1]['history'][0]['u'] += .25
    elif difference == 'path_order':
        relocated['fitting']['evidence_refs'].reverse()
    elif difference == 'reference_version':
        relocated['fitting']['evidence_refs'][0]['version'] = 2
    elif difference == 'candidate_version':
        relocated['candidate_version'] = 2
    elif difference == 'component_version':
        relocated['components']['hole_0']['version'] = 4
    elif difference == 'parameters':
        relocated['fitting']['parameters'][0] = .25
    else:
        relocated['components']['hole_0']['output_unit'] = 'different_unit'
    new_file.write_text(json.dumps(rows))
    assert not consolidation._same_retained_parent_binding(agent, original, relocated)


def test_actual_parent_constructor_keeps_original_refs_after_paid_relocation(tmp_path, monkeypatch):
    agent, _ = agent_scene(tmp_path)
    original, relocated, _, _ = bindings(tmp_path)
    identifier = 'frozen_recall:q@1:retained_parent'
    program_ref = {'file': str(tmp_path / 'research_state.json'),
        'json_path': ['theory_index', identifier, 'program'], 'version': 1}
    transfer_ref = {'file': str(tmp_path / 'shared/theory_library.json'),
        'json_path': ['transfers', 0]}
    parent = {'id': identifier, 'version': 1,
        'program': Program('feature', name='u').to_dict(),
        'program_source_ref': deepcopy(program_ref), 'consolidation_ref': relocated}
    exported = {'theory_ref': {'id': identifier, 'version': 1},
        'program_source_ref': deepcopy(program_ref), 'transfer_ref': deepcopy(transfer_ref)}
    agent.questions = [{'id': 'q', 'meaning_version': 1,
                        'revision_candidate_exports': [exported]}]
    agent.theory_consolidation = {'candidates': {}, 'transfers': [
        {'theory_ref': deepcopy(exported['theory_ref']), 'binding': original}]}
    consolidation._save(agent)
    unchanged = deepcopy((parent, agent.theory_consolidation, agent.questions))
    captured = []

    def constructor(agent, entries, pool):
        captured.extend(deepcopy(entries))
        entries[0]['_after_contract_source'] = {
            'source_ref': {'transfer_ref': deepcopy(entries[0]['fit_state_refs']['after']['transfer_ref'])}}

    monkeypatch.setattr(consolidation, '_revision_contract_sources', constructor)
    recovered = consolidation._retained_parent_contract(agent, parent)

    assert len(captured) == 1
    assert recovered['_contract_source']['source_ref']['transfer_ref'] == transfer_ref
    assert recovered['program_source_ref'] == program_ref
    assert (parent, agent.theory_consolidation, agent.questions) == unchanged


def prepared_parent():
    reference = {'file': 'retained.json', 'json_path': ['program'], 'version': 1}
    return {'id': 'retained', 'version': 1, 'program': Program('feature', name='u').to_dict(),
        'role_signature': {'u': {'role': 'drive', 'unit': 'V', 'time': 'current'}},
        'output_unit': 'V', 'context_ref': {'source_id': 'physical_source'},
        'program_source_ref': reference,
        '_contract_source': {'_pattern': Program('arg', name='hole_0').to_dict(),
            '_holes': [{'name': 'hole_0', 'output_unit': 'V', 'time_contract': ['current']}],
            '_bound_holes': {'hole_0': {'component_id': 'original', 'version': 1}},
            'source_ref': {'program_ref': deepcopy(reference)}}}


def test_complete_ram_parent_contract_skips_source_reloading_but_keeps_typed_prepare(
        tmp_path, monkeypatch):
    agent, _ = agent_scene(tmp_path)
    parent = prepared_parent()
    original = deepcopy(parent)
    monkeypatch.setattr(consolidation, '_question_versions',
        lambda *args: pytest.fail('complete RAM constructor must not reopen source history'))
    assert consolidation._retained_parent_contract(agent, parent) is parent
    checked = revisions._component(parent, SearchBudget(1000))
    assert checked['_prepared']['contract_recovery']['status'] == 'recovered'
    assert checked['_prepared']['contracts'][()]['output_unit'] == 'V'
    assert parent == original


@pytest.mark.parametrize('difference', ['program', 'unit', 'endpoint'])
def test_reused_ram_constructor_still_checks_program_units_and_endpoint(
        tmp_path, monkeypatch, difference):
    agent, _ = agent_scene(tmp_path)
    parent = prepared_parent()
    if difference == 'program':
        parent['_contract_source']['_pattern'] = Program('feature', name='u').to_dict()
        parent['_contract_source']['_holes'] = []
        parent['_contract_source']['_bound_holes'] = {}
        parent['program'] = Program('const', value=1.).to_dict()
    elif difference == 'unit':
        parent['output_unit'] = 'A'
    else:
        parent['program_source_ref']['version'] = 2
    monkeypatch.setattr(consolidation, '_question_versions',
        lambda *args: pytest.fail('RAM reuse must leave semantic checks to _component'))
    recovered = consolidation._retained_parent_contract(agent, parent)
    with pytest.raises(revisions._MissingContract):
        revisions._component(recovered, SearchBudget(1000))
