from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hegel_machine.cognition import registered_source_scene as scene
from scripts import prepare_pr3_source_successor as prepare
from scripts import run_anonymous_revision_inquiry as study


def inputs(tmp_path):
    source = tmp_path / 'source' / 'case_002' / 'pr3'
    source.mkdir(parents=True)
    reference = lambda name: {'file': str(source / name), 'json_path': []}
    parent_ref = reference('parent.json')
    costs = {'observation_cost': 2, 'measurement_cost': 0, 'compute_cost': 100,
        'python_execution_seconds': 3., 'model_calls': 0, 'reported_tokens': 0}
    winner = {'source_model': 'winner.py', 'mse': .1}
    parent = {'id': 'native', 'version': 1, 'program': {'op': 'const', 'value': 1},
              'program_source_ref': parent_ref}
    previous = {'costs': costs, 'paid_observations': [{'rows': [1, 2]}],
        'evaluated_model': winner}
    case = {'id': 'case_002', 'kind': 'hysteresis', 'seed': 17, 'confirmation_seed': 29}
    plan = {'cases': [case], 'limits': {}, 'research_organization': {'phase': 'source'},
        'prepared_public_questions': {'case_002': {'id': 'native_public_response'}}}
    constructor = {key: reference(key + '.json') for key in
                   ('pattern_ref', 'holes_ref', 'components_ref')}
    constructor['program_ref'] = parent_ref
    registration = {'material': {'source_training_ref': reference('training.json'),
        'common_additional_theory_bank_ref': reference('bank.json'),
        'common_executable_model_roster_ref': reference('models.json'),
        'parent_constructor_source': constructor, 'parent_Program_ref': parent_ref,
        'parent_theory_id': 'native', 'same_paid_rows': 2, 'old_archive_entries': 2,
        'old_native_entries': 1, 'common_archive_entries_after_materialization': 3,
        'common_model_count': 2}, 'arms': {
        'remove': {'library_ref': reference('old.json'), 'candidate_versions': [{'id': 'old', 'version': 1}]},
        'keep': {'library_ref': reference('new.json'), 'candidate_versions': [
            {'id': 'old', 'version': 1}, {'id': 'new', 'version': 1}]}},
        'costs': {'new_induction_and_development_compute_once': 20},
        'budget': {'additional_compute_per_arm': 500, 'additional_Python_seconds_per_arm': 30}}
    declaration = {'id': 'development', 'program_source_ref': reference('program.json'),
                   'program': {'op': 'const', 'value': 2}}
    material = {'source': source, 'previous': previous, 'plan': plan, 'case': case,
        'knowledge': {'finite_native_author': {}, 'files': {}}, 'declarations': [declaration],
        'models': ['winner.py', 'development.py'], 'frozen_candidates': {
            'old': {'version': 1}, 'new': {'version': 1}},
        'parent_contract': {'source_ref': constructor, '_pattern': parent['program'],
                            '_holes': [], '_bound_holes': {}}}
    return registration, material, parent


def test_restore_keeps_source_work_and_common_pool_while_varying_availability(tmp_path, monkeypatch):
    registration, material, parent = inputs(tmp_path)

    class Agent:
        def __init__(self, provider, workspace, **kwargs):
            assert provider is None and kwargs['arm'] == 'pr3'
            self.workspace = Path(workspace)
            self.workspace.mkdir(parents=True)
            self.config = kwargs['config']
            self.knowledge = kwargs['knowledge']
            self._organization_state = {}

        def restore_research(self, previous, source, laboratory, **kwargs):
            self.costs = deepcopy(previous['costs'])
            self.observations = deepcopy(previous['paid_observations'])
            self.rows = [1, 2]
            self.questions = [{'id': 'native_public_response', 'frozen_library_path': 'prior'}]
            self.theory_index = {'native': deepcopy(parent), 'opaque': {'id': 'opaque', 'version': 1}}
            self.theory_consolidation = {'candidates': {'old': {'version': 1}},
                'executions': {'saved_fit': {'cursor': {'layer': 3}}},
                'transfers': [{'status': 'failed', 'cost': 7}], 'material_cursors': {'source': 5}}
            self.evaluated_model = {'source_model': 'winner.py', 'current_mse': .1}
            return {'restored_paid_rows': 2, 'new_observations': 0}

        def _prepare_program_state(self):
            pass

    monkeypatch.setattr(study, 'RegisteredOrganizationAgent', Agent)
    monkeypatch.setattr(study, 'AnonymousHistoryLab', lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(scene, 'save_theory_library', lambda agent: None)
    monkeypatch.setattr(scene, 'save_theory_index', lambda agent: None)
    monkeypatch.setattr(scene, 'resolved_theories', lambda agent: list(agent.theory_index.values()))
    monkeypatch.setattr(prepare, '_retain_scene_score_index', lambda source, workspace, *, paid_rows:
                        {'retained_scopes': 1, 'retained_assessments': 2, 'available_paid_rows': paid_rows})
    results = [scene.restore_registered_source_scene(registration, arm, tmp_path / arm, material=material)
               for arm in ('remove', 'keep')]
    for result in results:
        agent = result['agent']
        assert agent.config.compute_budget == 620
        assert agent.config.execution_seconds == 33
        assert agent.costs['compute_cost'] == 120
        assert agent.theory_consolidation['executions']['saved_fit']['cursor'] == {'layer': 3}
        assert agent.theory_consolidation['transfers'] == [{'status': 'failed', 'cost': 7}]
        assert agent.theory_consolidation['material_cursors'] == {'source': 5}
        assert agent._organization_state['automatic_library_path'] == scene.LIBRARY_PATH
        assert result['source_component_theory_ids'] == ['native']
        assert len(agent.theory_index) == 3
        assert result['parent']['_contract_source'] == material['parent_contract']
        assert result['validation']['new_fits'] == result['validation']['new_predictions'] == 0
        assert result['validation']['retained_score_index']['available_paid_rows'] == len(agent.rows)
        assert result['lab'].kind == 'hysteresis'
    assert list(results[0]['agent'].theory_consolidation['candidates']) == ['old']
    assert list(results[1]['agent'].theory_consolidation['candidates']) == ['old', 'new']
    assert material['previous']['costs']['compute_cost'] == 100


def test_material_load_batches_program_refs_once(tmp_path, monkeypatch):
    registration, material, _ = inputs(tmp_path)
    source, plan = material['source'], material['plan']
    (source.parent.parent / 'protocol.json').write_text(json.dumps(plan))
    declarations = [dict(item, program_source_ref={'file': 'programs.json', 'json_path': [i]})
                    for i, item in enumerate(material['declarations'] * 3)]
    values = {
        'training.json': material['previous'], 'initial_archive.json': material['knowledge'],
        'bank.json': declarations, 'models.json': material['models']}
    monkeypatch.setattr(scene, '_read', lambda reference, workspace: deepcopy(values[Path(reference['file']).name]))
    monkeypatch.setattr(scene, '_read_library', lambda reference, workspace:
                        {'candidates': material['frozen_candidates']})
    batches = []

    def load(references, workspace):
        batches.append(deepcopy(references))
        return [{'value': {'op': 'const', 'value': i}} for i, _ in enumerate(references)]

    monkeypatch.setattr(scene, 'load_source_values', load)
    loaded = scene.load_registered_source_material(registration)
    program_batches = [batch for batch in batches if batch[0]['file'] == 'programs.json']
    assert len(program_batches) == 1 and len(program_batches[0]) == 3
    assert all('program' in item for item in loaded['declarations'])


def test_material_rejects_changed_frozen_object_version(tmp_path, monkeypatch):
    registration, material, _ = inputs(tmp_path)
    (material['source'].parent.parent / 'protocol.json').write_text(json.dumps(material['plan']))
    values = {'training.json': material['previous'], 'initial_archive.json': material['knowledge'],
        'bank.json': material['declarations'], 'models.json': material['models']}
    monkeypatch.setattr(scene, '_read', lambda reference, workspace: deepcopy(values[Path(reference['file']).name]))
    monkeypatch.setattr(scene, 'load_source_values', lambda refs, workspace: [{'value': {}} for _ in refs])
    monkeypatch.setattr(scene, '_read_library', lambda reference, workspace:
                        {'candidates': {'old': {'version': 2}, 'new': {'version': 1}}})
    with pytest.raises(ValueError, match='object versions'):
        scene.load_registered_source_material(registration)
