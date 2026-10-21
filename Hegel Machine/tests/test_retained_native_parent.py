"""Historical native exports bind by provenance and exact fixed exporter text."""

from copy import deepcopy
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.question_routing import resolved_theories
from hegel_machine.cognition.retained_native_parent import bind_retained_native_parent
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program


ROLES = {'u': {'role': 'public_control', 'unit': '1', 'time': 'current'}}
PROGRAM = Program('add', children=(Program('feature', name='u'), Program('const', value=.25)))


def scene(tmp_path):
    source, workspace = tmp_path / 'source', tmp_path / 'target'
    source.mkdir()
    workspace.mkdir()
    text = export_temporal_program(PROGRAM)
    (source / 'native.py').write_text(text)
    (workspace / 'imported.py').write_text(text)
    (workspace / 'retained.py').write_text(text)
    native_ref = {'kind': 'native_candidate', 'question_id': 'same_name',
                  'meaning_version': 1, 'round_index': 0, 'candidate_index': 0}
    state = {'workspace': str(source), 'questions': [{'id': 'same_name', 'meaning_version': 1,
        'native_condition_research': [{'candidates': [{'model': 'native.py', 'program': PROGRAM.to_dict()}]}]}],
        'theory_index': {'original_native': {'id': 'original_native', 'version': 3,
            'source_ref': native_ref, 'program_source_ref': {'file': 'research_state.json',
                'json_path': ['questions', 0, 'native_condition_research', 0, 'candidates', 0, 'program']},
            'role_signature': deepcopy(ROLES), 'output_unit': '1', 'context_ref': {'source_id': 'old'},
            'support': {'status': 'supported', 'independent_confirmation': True,
                        'evidence': [{'file': 'old_paid.json', 'json_path': [0]}]}}}}
    (source / 'research_state.json').write_text(json.dumps(state))
    catalog = [{'state_ref': {'file': str(source / 'research_state.json'), 'json_path': []},
        'model_sources': [{'model': 'imported.py', 'source_model': 'native.py',
                           'source_ref': {'file': str(source / 'native.py')}}]}]
    (workspace / 'source_catalog.json').write_text(json.dumps(catalog))
    question = {'id': 'same_name', 'meaning_version': 2, 'theory_target': {
        'role_signature': deepcopy(ROLES), 'output_unit': '1', 'context_ref': {'source_id': 'new'}}}
    agent = SimpleNamespace(workspace=workspace, evaluated_model={'model': 'retained.py',
        'source_model': 'imported.py'}, questions=[question], theory_index={}, events=[],
        relation_research={}, theory_consolidation={}, costs={'compute_cost': 17, 'model_calls': 0})
    return agent, question, source, catalog, state


def test_fixed_original_program_rebinds_current_context_without_support_or_source_copy(tmp_path, monkeypatch):
    agent, question, source, _, _ = scene(tmp_path)
    original_state = (source / 'research_state.json').read_bytes()
    before = deepcopy(agent.costs)
    monkeypatch.setattr(Program, 'predict', lambda *args: pytest.fail('No prediction is needed for binding'))
    result = bind_retained_native_parent(agent, question)
    assert result['status'] == 'bound' and result['source_version'] == 3
    bound = agent.theory_index[result['theory_id']]
    assert bound['program'] == PROGRAM.to_dict() and bound['context_ref'] == {'source_id': 'new'}
    assert bound['program_source_ref']['file'] == str(source / 'research_state.json')
    assert bound['source_snapshot']['version'] == 3
    assert bound['source_snapshot']['source_ref']['meaning_version'] == 1
    assert bound['support'] == {'status': 'candidate', 'evidence': [], 'independent_confirmation': False}
    assert 'source_ref' not in bound and 'source_id' not in bound
    assert resolved_theories(agent)[0]['program'] == PROGRAM.to_dict()
    assert result['new_compute_cost'] == 0 and agent.costs == before
    assert (source / 'research_state.json').read_bytes() == original_state
    assert not (agent.workspace / 'research_state.json').exists()


def test_repeated_mapping_keeps_id_version_local_refutation_and_costs(tmp_path):
    agent, question, _, _, _ = scene(tmp_path)
    first = bind_retained_native_parent(agent, question)
    bound = agent.theory_index[first['theory_id']]
    bound['support'] = {'status': 'refuted', 'evidence': [{'file': 'target_paid.json', 'json_path': [0]}]}
    before = deepcopy((agent.theory_index, agent.costs))
    second = bind_retained_native_parent(agent, question)
    assert second == first and len(agent.theory_index) == 1
    assert (agent.theory_index, agent.costs) == before


@pytest.mark.parametrize('changed', ['role', 'unit', 'time', 'output_unit', 'missing_roles'])
def test_target_contract_conflict_stays_author_work(tmp_path, changed):
    agent, question, _, _, _ = scene(tmp_path)
    target = question['theory_target']
    if changed == 'output_unit':
        target['output_unit'] = 'other'
    elif changed == 'missing_roles':
        target['role_signature'] = None
    else:
        target['role_signature']['u'][changed] = 'other'
    result = bind_retained_native_parent(agent, question)
    assert result['reason'] == 'retained_native_target_contract_mismatch' and not agent.theory_index


@pytest.mark.parametrize('changed', ['snapshot', 'dynamic_fit', 'opaque', 'missing_ref', 'conflicting_catalog'])
def test_unproven_exports_are_not_invented_as_programs(tmp_path, changed):
    agent, question, source, catalog, state = scene(tmp_path)
    expected = {'snapshot': 'retained_native_export_bytes_changed',
        'dynamic_fit': 'retained_native_export_not_fixed', 'opaque': 'unique_original_native_export_required',
        'missing_ref': 'retained_native_source_addresses_required',
        'conflicting_catalog': 'unique_retained_native_catalog_mapping_required'}
    if changed == 'snapshot':
        (agent.workspace / 'retained.py').write_text('def predict(row): return 99.\n')
    elif changed == 'dynamic_fit':
        text = 'def fit(rows): return rows\ndef predict(row): return 0.\n'
        for path in (source / 'native.py', agent.workspace / 'imported.py', agent.workspace / 'retained.py'):
            path.write_text(text)
    elif changed == 'opaque':
        state['questions'][0]['native_condition_research'] = []
        (source / 'research_state.json').write_text(json.dumps(state))
    elif changed == 'missing_ref':
        catalog[0].pop('state_ref')
    else:
        catalog.append(deepcopy(catalog[0]))
    (agent.workspace / 'source_catalog.json').write_text(json.dumps(catalog))
    result = bind_retained_native_parent(agent, question)
    assert result['status'] == 'author_work_request' and result['reason'] == expected[changed]
    assert not agent.theory_index and result['new_compute_cost'] == 0


def test_real_case000_retained_fixed_native_export_has_exact_original_binding(tmp_path, monkeypatch):
    original = Path('/mnt/hegel-snapshot-cow/native_pr3_budgeted_20261021/development_revision_transfer_20261021'
                    '/target_remove_C/case_000/pr3')
    if not (original / 'source_catalog.json').exists():
        pytest.skip('The actual retained case000 is not mounted')
    catalog = json.loads((original / 'source_catalog.json').read_text())
    mapping = catalog[4]['model_sources'][4]
    agent, question, _, _, _ = scene(tmp_path)
    for target_name in ('imported.py', 'retained.py'):
        (agent.workspace / target_name).write_bytes((original / mapping['model']).read_bytes())
    saved = {'state_ref': catalog[4]['state_ref'], 'model_sources': [
        {**mapping, 'model': 'imported.py'}]}
    (agent.workspace / 'source_catalog.json').write_text(json.dumps([saved]))
    monkeypatch.setattr(Program, 'predict', lambda *args: pytest.fail('Do not execute the historical predictor'))
    result = bind_retained_native_parent(agent, question)
    assert result['status'] == 'bound'
    bound = agent.theory_index[result['theory_id']]
    assert bound['source_snapshot']['theory_id'] == 'native:native_public_response@meaning1:round1:candidate4'
    assert bound['program_source_ref']['json_path'] == [
        'questions', 1, 'native_condition_research', 1, 'candidates', 4, 'program']
    assert bound['context_ref'] == question['theory_target']['context_ref']
    assert bound['support']['independent_confirmation'] is False


def test_registered_binder_consumes_historical_parent_without_native_bank_or_science(tmp_path):
    from scripts.run_anonymous_revision_inquiry import RegisteredOrganizationAgent

    agent, question, _, _, _ = scene(tmp_path)
    agent._retained_revision_parent = MethodType(
        RegisteredOrganizationAgent._retained_revision_parent, agent)
    agent.rows = []
    agent.observations = []
    agent.config = SimpleNamespace(confirmation_mse_threshold=1e-10)
    before = deepcopy(agent.costs)
    RegisteredOrganizationAgent._bind_revision_target(agent, question)
    # Explicit fitting is the existing opt-in to this binder.
    assert 'parent_theory_id' not in question['theory_target']
    question['theory_target']['revision_fitting'] = {'operations': ['refit'], 'risk_quantile': .5,
                                                   'max_history': 1, 'max_structures': 1}
    RegisteredOrganizationAgent._bind_revision_target(agent, question)
    target = question['theory_target']
    assert target['parent_theory_id'] in agent.theory_index
    assert target['parent_selection_ref']['json_path'] == ['evaluated_model']
    assert 'evidence_checks' not in target and 'parent_binding_obligation' not in target
    assert agent.costs == before and not agent.observations


def test_actual_old_opaque_catalog_entry_stays_unmapped(tmp_path):
    original = Path('/mnt/hegel-snapshot-cow/native_pr3_budgeted_20261021/development_revision_transfer_20261021'
                    '/target_remove_C/case_001/pr3')
    if not (original / 'source_catalog.json').exists():
        pytest.skip('The actual retained case001 is not mounted')
    catalog = json.loads((original / 'source_catalog.json').read_text())
    model = 'archive/source_000/_evaluated_model/_evaluated_model/archive/source_003/_evaluated_model/model.py'
    assert not any(mapping.get('model') == model for entry in catalog
                   for mapping in entry.get('model_sources', []))
    agent, question, _, _, _ = scene(tmp_path)
    # Only the original catalog's mapping metadata is needed for this negative.
    metadata = [{key: entry[key] for key in ('model', 'state_ref', 'model_sources') if key in entry}
                for entry in catalog[:4]]
    (agent.workspace / 'source_catalog.json').write_text(json.dumps(metadata))
    agent.evaluated_model['source_model'] = model
    result = bind_retained_native_parent(agent, question)
    assert result['reason'] == 'unique_retained_native_catalog_mapping_required'
    assert not agent.theory_index
