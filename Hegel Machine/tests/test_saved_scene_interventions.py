"""Actual-use removal reuses saved-scene construction, freezing and budgets."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hegel_machine.cognition import method_effect_query
from hegel_machine.cognition.theory_consolidation import _read_index, _read_library
from scripts import prepare_pr3_source_successor as preparation
from scripts import run_anonymous_revision_inquiry as study
from test_frozen_class_intervention import class_spec
from test_prepare_frozen_successor import prohibit_science
from test_prepare_pr3_source_successor import save
from test_saved_scene_report_recovery import training


def registration_fixture(tmp_path, monkeypatch):
    spec, library, index = class_spec(tmp_path)
    source = tmp_path / 'source/target/case_001/pr3'
    plan = {'author': spec['author'], 'research_organization': {'phase': 'target',
        'initial_library_path': spec['materials']['initial_library_ref']['file'],
        'pr3_library_path': spec['frozen_source']['library_ref']['file']}}
    save(source.parent.parent / 'protocol.json', plan)
    prepared = tmp_path / 'scene'
    save(prepared / 'scene_manifest.json', {'source_training_ref': {'file': str(source / 'training.json')},
        'library_views': {'B_old8': spec['materials']['initial_library_ref'],
            'old_ids': ['old'], 'new_ids': ['new', 'new_alias', 'other']},
        'post97_prediction_contract': {'rows': 48}})
    prohibit_science(monkeypatch)
    result = preparation.prepare_saved_scene_interventions(prepared, tmp_path / 'registered',
        adopted_candidate_id='new', adoption_refs=[{'file': 'kept.json', 'json_path': ['adopted', 0]}],
        library_ref=spec['frozen_source']['library_ref'], index_ref=spec['frozen_source']['index_ref'],
        redundancy_classes_ref=spec['frozen_source']['redundancy_classes_ref'],
        source_cost_ref=spec['frozen_source']['accounting_ref'])
    return result, prepared, library, index


def test_registers_existing_class_and_removes_only_automatic_organization(tmp_path, monkeypatch):
    registered, _, library, index = registration_fixture(tmp_path, monkeypatch)
    assert registered['status'] == 'registered_not_executed'
    assert registered['additional_compute_per_arm'] == 1500000
    assert registered['additional_python_seconds_per_arm'] == 1800
    assert registered['native_attempt_limit_per_arm'] == 8
    assert registered['maximum_physical_cores'] == 8
    assert registered['keep_reruns_on_common_current_runtime']
    assert not Path(registered['run_output']).exists()
    branches = registered['branches']
    assert list(branches) == ['keep', 'individual_remove', 'class_remove']
    assert all(branch['organization_arm'] == 'pr3' for branch in branches.values())
    assert [branch['expected_candidate_ids'] for branch in branches.values()] == [
        ['old', 'new', 'new_alias', 'other'], ['old', 'new_alias', 'other'], ['old', 'other']]
    kept = _read_library(branches['keep']['library_ref'], tmp_path)
    assert all(kept[key] == value for key, value in library.items())
    actual = _read_library(branches['class_remove']['library_ref'], tmp_path)
    assert list(actual['candidates']) == ['old', 'other']
    assert actual['executions']['mixed_source_construction'] == library['executions']['mixed_source_construction']
    filtered_index = _read_index(branches['class_remove']['index_ref'], tmp_path)
    assert filtered_index['mixed'] == index['mixed']['history'][0]
    assert 'derived' not in filtered_index


def test_filtered_views_reach_real_constructor_before_projection(tmp_path, monkeypatch):
    prepared = tmp_path / 'scene'
    source = tmp_path / 'source/target/case_001/pr3'
    save(prepared / 'scene_manifest.json', {'source_training_ref': {'file': str(source / 'training.json')}})
    save(source / 'training.json', {'costs': training()['prior_research_cost']})
    save(source / 'research_state.json', {})
    save(source.parent.parent / 'protocol.json', {'limits': {}, 'author': {},
        'cases': [{'id': 'case_001', 'confirmation_seed': 1}]})
    save(source.parent.parent / 'initial_archive.json', {})
    monkeypatch.setattr(study, '_prepare_organization', lambda *_: ({'phase': 'target'}, {'files': {}}))
    monkeypatch.setattr(study, 'archived_paid_trace_references', lambda *_args, **_kwargs: [])
    view = {'organization_arm': 'pr3', 'library_ref': {'file': 'removed_library.json'},
            'index_ref': {'file': 'removed_index.json'}}
    seen = []

    class ConstructorReached(Exception):
        pass

    def constructor(_provider, workspace, **kwargs):
        seen.append((workspace, kwargs))
        raise ConstructorReached

    monkeypatch.setattr(study, 'RegisteredOrganizationAgent', constructor)
    with pytest.raises(ConstructorReached):
        preparation.accept_prepared_saved_scene(prepared, tmp_path / 'run',
            branch_views={'individual_remove': view})
    workspace, actual = seen[0]
    assert workspace.name == 'individual_remove'
    assert actual['arm'] == 'pr3'
    assert actual['research_organization']['pr3_library_path'] == 'removed_library.json'
    assert actual['knowledge']['theory_index'] == {'file': 'removed_index.json'}


def test_freeze_preserves_registered_candidate_order_and_exact_view(tmp_path):
    path = tmp_path / 'input.json'
    save(path, {'candidates': {'old': {'id': 'old'}, 'alias': {'id': 'alias'}}})
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    agent = SimpleNamespace(workspace=workspace, questions=[], files={},
        _organization_state={'automatic_library_path': str(path)},
        _prepare_program_state=lambda: None)
    agent._save = lambda name, content: (workspace / name).parent.mkdir(parents=True, exist_ok=True) or (
        workspace / name).write_text(content)
    scene = {'output': tmp_path / 'run', 'manifest': {}, 'arms': {'keep': {'agent': agent}},
        'branch_views': {'keep': {'expected_candidate_ids': ['old', 'alias']}}}
    frozen = preparation._freeze_saved_scene_libraries(scene)
    assert frozen['keep']['candidate_count'] == 2
    assert list(agent.theory_consolidation['candidates']) == ['old', 'alias']
    scene['branch_views']['keep']['expected_candidate_ids'].reverse()
    with pytest.raises(ValueError, match='exact registered'):
        preparation._freeze_saved_scene_libraries(scene)


def test_all_registered_training_finishes_before_shared_prediction(tmp_path, monkeypatch):
    names = ('keep', 'individual_remove', 'class_remove')
    carry = training()['prior_research_cost']
    output = tmp_path / 'run'
    plan = {'author': {'path': 'shared/finite_native_author.py', 'max_steps': 8, 'confirm': False}}
    scene = {'preparation': tmp_path / 'prepared', 'output': output,
        'manifest': {'library_views': {'new_ids': ['new']},
            'post97_prediction_contract': {'rows': 48},
            'source_training_ref': {'file': 'old.json'}},
        'source': tmp_path / 'source', 'plan': plan, 'case': {'id': 'case_001'},
        'carry': carry, 'limits': SimpleNamespace(compute_budget=1500100, execution_seconds=1803,
                                                 confirmation_mse_threshold=1e-10),
        'acceptance': {'status': 'all_registered_branches_accepted_before_scientific_execution'},
        'branch_views': {name: {'manifest_ref': {'file': name + '.json'}} for name in names},
        'intervention_registration_ref': {'file': 'registration.json'},
        'source_cost_ref': {'file': 'costs.json'}}
    scene['arms'] = {}
    for name in names:
        workspace = output / name
        workspace.mkdir(parents=True)
        agent = SimpleNamespace(workspace=workspace, candidate_paths=['strong_pool_model.py'])
        agent._save = lambda *_: None
        agent.result = lambda _status: deepcopy(training())
        scene['arms'][name] = {'agent': agent, 'lab': None, 'previous': {},
            'starting_costs': carry, 'validation': {'acceptance': {'new_costs': {}}}}
    frozen_root = tmp_path / 'runtime'
    (frozen_root / 'src/hegel_machine').mkdir(parents=True)
    for name in ('run_anonymous_revision_inquiry', 'prepare_pr3_source_successor',
                 'run_online_inquiry_llm', 'run_online_inquiry_study'):
        script = frozen_root / 'scripts' / (name + '.py')
        script.parent.mkdir(exist_ok=True)
        script.write_text('# common runtime\n')
    monkeypatch.setattr(preparation, 'ROOT', frozen_root)
    monkeypatch.setattr(preparation, '_freeze_saved_scene_libraries', lambda _: {name: {} for name in names})
    finished = []

    def loop(agent, _lab, _plan, _checkpoint, *, native_attempt_limit):
        assert native_attempt_limit == 8
        finished.append(agent.workspace.name)
        return {'status': 'method_step_limit'}

    def dispatch(run_arm, jobs, workers):
        assert tuple(jobs) == names and workers == 8
        return [run_arm(name) for name in jobs]

    def effects(*_args, **_kwargs):
        assert tuple(finished) == names
        return {'cases': []}

    def predictions(root, source_ref, *, arms):
        assert tuple(finished) == names and tuple(arms) == names
        protocol = json.loads((root / 'protocol.json').read_text())
        assert tuple(protocol['frozen_libraries']) == names
        assert protocol['additional_compute_per_arm'] == 1500000
        return {'status': 'recorded_comparison', 'comparison': {'changed': False},
            'arms': [{'arm': name, 'status': 'recorded', 'costs_before': training()['costs'],
                'costs_after': training()['costs'], 'evaluation': {'mse': .2}} for name in names]}

    monkeypatch.setattr(preparation, '_source_research_steps', loop)
    monkeypatch.setattr(study, '_native_process_trials', dispatch)
    monkeypatch.setattr(method_effect_query, 'registered_successor_effect', effects)
    monkeypatch.setattr(method_effect_query, 'evaluate_saved_scene_predictions', predictions)
    result = preparation.run_prepared_saved_scene(None, accepted_scene=scene)
    assert result['registered_arms'] == list(names)
    assert result['post97_prediction_diagnostic'] == 'recorded_comparison'
