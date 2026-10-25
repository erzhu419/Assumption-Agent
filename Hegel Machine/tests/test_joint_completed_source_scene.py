"""A later completed scene retains both objects' independent work."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from hegel_machine.cognition import registered_source_scene as scene
from test_completed_source_scene import completed_fixture, stub_restoration, OLD, NEW, PREFIX


def materialized_bank_fixture(tmp_path, monkeypatch):
    from hegel_machine.cognition.language import Program
    from test_question_routing import theory

    root = tmp_path / 'restored'
    root.mkdir()
    (root / 'strong.py').write_text('def predict(row): return 0.\n')
    source_ref = {'file': str(tmp_path / 'original/research_state.json'),
                  'json_path': ['theory_index']}
    archived = theory(str(source_ref['file']) + '::native',
        Program('const', value=2.), version=7,
        source_snapshot={'source_ref': {'kind': 'native_candidate',
            'question_id': 'source', 'meaning_version': 1, 'round_index': 2, 'candidate_index': 3},
            'index_ref': deepcopy(source_ref), 'version': 7},
        history=[theory('native', Program('const', value=1.), version=6,
            source_snapshot={'source_ref': {'kind': 'native_candidate',
                'question_id': 'source', 'meaning_version': 1, 'round_index': 1, 'candidate_index': 3},
                'index_ref': deepcopy(source_ref), 'version': 6})],
        support={'status': 'refuted', 'evidence': [{'record_index': 1}]})
    missing = {**deepcopy(archived), 'id': archived['id'] + ':second'}
    shared = {'_joint_exposed_export_rosters': [], '_joint_exposed_bank_material': {
        'native': {'theories': [archived, missing], 'archives': [
            {'research_state_ref': deepcopy(source_ref)}]}, 'diagnostics': []}}
    base = tmp_path / 'base.json'
    base.write_text(json.dumps(['strong.py']))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'base_model_roster_ref': {'file': str(base), 'json_path': []},
        'base_count': 1, 'later_export_rosters': [],
        'named_entry_count_before_exact_program_and_context_reuse': 1}))
    registration = {'material': {'same_closed_scene': str(tmp_path / 'completed'),
        'complete_common_bank_manifest_ref': {'file': str(manifest), 'json_path': []}, 'paid_rows': 1}}
    retained = deepcopy(archived)
    agent = SimpleNamespace(workspace=root, candidate_paths=[],
        knowledge={'finite_native_author': {}}, theory_index={retained['id']: retained},
        theory_consolidation={'executions': {'old65': {'cursor': {'layer': 3, 'offset': 8}}}},
        questions=[{'id': 'native_public_response', 'theory_target': {'role_signature': {'u': {}}}}])
    monkeypatch.setattr('hegel_machine.cognition.theory_consolidation._read_index',
        lambda *args: pytest.fail('cached native material must not reopen the source index'))
    monkeypatch.setattr(scene, '_joint_exposed_diagnostic_programs',
        lambda *args: pytest.fail('cached diagnostic material must not be reconstructed again'))
    monkeypatch.setattr(Program, 'predict', lambda *args: pytest.fail('restoration must not predict'))
    return agent, registration, shared, retained, missing


def test_already_materialized_joint_namespace_reuses_exact_versions_and_adds_only_missing(
        tmp_path, monkeypatch):
    agent, registration, shared, retained, missing = materialized_bank_fixture(tmp_path, monkeypatch)
    before, cursors = deepcopy(shared), deepcopy(agent.theory_consolidation)

    receipt = scene.materialize_joint_exposed_bank(agent, registration, shared_material=shared)

    assert agent.theory_index[retained['id']] is retained
    assert agent.theory_index[missing['id']] == missing
    assert shared == before and agent.theory_consolidation == cursors
    assert receipt['reused_current_native_theories'] == 1
    assert receipt['additional_current_native_theories'] == 1
    assert receipt['reused_historical_native_versions'] == 1
    assert receipt['additional_historical_native_versions'] == 1
    assert receipt['new_fits'] == receipt['new_predictions'] == receipt['new_scores'] == 0

    second = scene.materialize_joint_exposed_bank(agent, registration, shared_material=shared)
    assert second['reused_current_native_theories'] == 2
    assert second['additional_current_native_theories'] == 0
    assert agent.theory_index[retained['id']] is retained


@pytest.mark.parametrize('changed_field', [
    'version', 'program', 'output_unit', 'role_signature', 'context_ref',
    'history_version', 'history_program', 'history_order', 'source_snapshot', 'refutation'])
def test_reused_joint_namespace_rejects_changed_original_scientific_contract(
        tmp_path, monkeypatch, changed_field):
    agent, registration, shared, retained, missing = materialized_bank_fixture(tmp_path, monkeypatch)
    if changed_field == 'history_version':
        retained['history'][0]['version'] += 1
    elif changed_field == 'history_program':
        retained['history'][0]['program']['value'] = 1.5
    elif changed_field == 'history_order':
        shared['_joint_exposed_bank_material']['native']['theories'][0]['history'].append(
            {**deepcopy(retained['history'][0]), 'version': 5})
        retained['history'].insert(0, {**deepcopy(retained['history'][0]), 'version': 5})
    elif changed_field == 'source_snapshot':
        retained['source_snapshot']['index_ref']['file'] = 'different_original.json'
    elif changed_field == 'refutation':
        retained['support']['evidence'] = []
    elif changed_field == 'version':
        retained['version'] += 1
    elif changed_field == 'program':
        retained['program']['value'] = 2.5
    elif changed_field == 'role_signature':
        retained['role_signature']['u']['time'] = 'previous'
    elif changed_field == 'context_ref':
        retained['context_ref']['scene'] = 'different-original'
    else:
        retained['output_unit'] = 'mV'
    before = deepcopy(agent.theory_index)

    with pytest.raises(ValueError, match='retained joint source namespace differs'):
        scene.materialize_joint_exposed_bank(agent, registration, shared_material=shared)

    assert agent.theory_index == before
    assert missing['id'] not in agent.theory_index


def test_joint_restore_preserves_completed_model_additions_and_reuses_existing_diagnostic(
        tmp_path, monkeypatch):
    from hegel_machine.cognition import checkpoint_storage
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.temporal_operator_agent import export_temporal_program

    agent, registration, shared, _, _ = materialized_bank_fixture(tmp_path, monkeypatch)
    model = '_experiment_models/joint_exposed_diagnostics/_evaluated_model/constructor_0000.py'
    destination = agent.workspace / model
    destination.parent.mkdir(parents=True)
    program = Program('const', value=3.)
    code = export_temporal_program(program, feature_names=('u',))
    destination.write_text(code)
    (agent.workspace / 'completed_later_fit.py').write_text('def predict(row): return 4.\n')
    agent.knowledge['finite_native_author']['predictor_paths'] = [
        'strong.py', model, 'completed_later_fit.py']
    registration['material']['closed_scene_scored_models'] = 3
    manifest_path = tmp_path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    (tmp_path / 'base.json').write_text(json.dumps(['strong.py', model]))
    manifest.update(base_count=2, named_entry_count_before_exact_program_and_context_reuse=3)
    manifest_path.write_text(json.dumps(manifest))
    shared['_joint_exposed_bank_material']['diagnostics'] = [{'group': 'constructor',
        'index': 0, 'program': program, 'output_ref': {'file': 'original_fit', 'json_path': []}}]
    monkeypatch.setattr(checkpoint_storage, 'write_text_incrementally',
        lambda *args, **kwargs: pytest.fail('equal saved diagnostic must not be rewritten'))

    receipt = scene.materialize_joint_exposed_bank(agent, registration, shared_material=shared)

    assert agent.knowledge['finite_native_author']['predictor_paths'] == [
        'strong.py', model, 'completed_later_fit.py']
    assert receipt['named_models'] == 3
    assert receipt['manifest_named_entries'] == 3
    assert receipt['retained_completed_scene_additions'] == ['completed_later_fit.py']
    assert receipt['reused_diagnostic_models'] == 1
    assert destination.read_text() == code


def test_joint_restore_rejects_a_different_program_at_the_existing_diagnostic_path(
        tmp_path, monkeypatch):
    from hegel_machine.cognition.language import Program

    agent, registration, shared, _, _ = materialized_bank_fixture(tmp_path, monkeypatch)
    destination = agent.workspace / (
        '_experiment_models/joint_exposed_diagnostics/_evaluated_model/constructor_0000.py')
    destination.parent.mkdir(parents=True)
    destination.write_text('def predict(row): return 999.\n')
    shared['_joint_exposed_bank_material']['diagnostics'] = [{'group': 'constructor',
        'index': 0, 'program': Program('const', value=3.),
        'output_ref': {'file': 'original_fit', 'json_path': []}}]

    with pytest.raises(ValueError, match='retained joint diagnostic model differs'):
        scene.materialize_joint_exposed_bank(agent, registration, shared_material=shared)

    assert destination.read_text() == 'def predict(row): return 999.\n'


def test_joint_restore_keeps_manifest_additions_to_an_older_source_comparison_pool(
        tmp_path, monkeypatch):
    from hegel_machine.cognition.language import Program

    agent, registration, shared, _, _ = materialized_bank_fixture(tmp_path, monkeypatch)
    agent.knowledge['finite_native_author']['predictor_paths'] = ['strong.py']
    registration['material']['closed_scene_scored_models'] = 1
    manifest_path = tmp_path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['named_entry_count_before_exact_program_and_context_reuse'] = 2
    manifest_path.write_text(json.dumps(manifest))
    shared['_joint_exposed_bank_material']['diagnostics'] = [{'group': 'constructor',
        'index': 0, 'program': Program('const', value=3.),
        'output_ref': {'file': 'original_fit', 'json_path': []}}]

    receipt = scene.materialize_joint_exposed_bank(agent, registration, shared_material=shared)

    models = agent.knowledge['finite_native_author']['predictor_paths']
    assert models[0] == 'strong.py' and len(models) == 2
    assert receipt['named_models'] == 2


@pytest.mark.parametrize('summary_format', ['paired', 'single'])
def test_nested_completed_material_uses_original_constructor_and_latest_account(
        tmp_path, summary_format):
    registration, material, _, _ = completed_fixture(tmp_path)
    source = material['source']
    older = tmp_path / 'older' / 'keep'
    older.mkdir(parents=True)
    original = json.loads((source.parent / 'registration.json').read_text())
    (older.parent / 'registration.json').write_text(json.dumps(original))
    (source.parent / 'registration.json').write_text(json.dumps({
        'material': {'same_closed_scene': str(older)},
        'arms': registration['arms']}))
    summary = json.loads((source.parent / 'summary.json').read_text())
    summary['arms'][0]['theory_count'] = 9
    if summary_format == 'single':
        summary = {**summary['arms'][0], 'single_scene_only': True}
    (source.parent / 'summary.json').write_text(json.dumps(summary))

    loaded = scene.load_completed_source_material(registration)

    assert loaded['source'] == source
    assert loaded['starting_costs'] == material['starting_costs']
    assert loaded['closed_scene_theory_count'] == len(loaded['previous']['theory_index'])
    assert loaded['closed_scene_theory_count'] != 9
    assert loaded['parent_contract'] == material['parent_contract']
    assert loaded['source_registration'] == original


def test_completed_material_keeps_latest_paid_joint_comparison_contexts(tmp_path):
    registration, material, _, _ = completed_fixture(tmp_path)
    source = material['source']
    previous = json.loads((source / 'training.json').read_text())
    other_arm = str(tmp_path / 'old_remove' / 'candidate_0001.py')
    diagnostic = '_experiment_models/joint_exposed_diagnostics/_evaluated_model/fixed_fit_0000.py'
    # The real closed control's method_results still ends at the source task;
    # its newer direct comparisons exist only in registered_actions.json.
    previous['method_results'] = [{'feedback': {
        'operator': 'research_choice', 'paid_rows': 2,
        'evaluations': [{'path': 'inherited_source_comparison.py'}]}}]
    (source / 'training.json').write_text(json.dumps(previous))
    actions = [
        {'action': 'research_choice', 'models_requested': 2, 'feedback': {
            'operator': 'research_choice', 'paid_rows': 2,
            'evaluations': [{'path': name} for name in ('strong.py', 'winner.py')]}},
        {'action': 'research_choice', 'models_requested': 4, 'feedback': {
            'operator': 'research_choice', 'paid_rows': 2,
            'evaluations': [{'path': name} for name in
                ('strong.py', other_arm, diagnostic, 'winner.py')]}}]
    (source / 'registered_actions.json').write_text(json.dumps(actions))

    loaded = scene.load_completed_source_material(registration)

    assert loaded['models'] == ['strong.py', other_arm, diagnostic, 'winner.py']
    assert loaded['starting_costs'] == material['starting_costs']


@pytest.mark.parametrize('summary_format', ['paired', 'single'])
def test_compact_completed_roster_retains_all_versions_and_checkpoint_fees_once(
        tmp_path, monkeypatch, summary_format):
    registration, material, pool, _ = completed_fixture(tmp_path)
    source = material['source']
    summary_path = source.parent / 'summary.json'
    summary = json.loads(summary_path.read_text())
    if summary_format == 'single':
        summary = {**summary['arms'][0], 'single_scene_only': True,
                   'continuation_costs': json.loads((source / 'final_costs.json').read_text())}
    summary_path.write_text(json.dumps(summary))
    registration['material']['closed_scene_theory_count'] = 4232
    previous = json.loads((source / 'training.json').read_text())
    previous['method_results'] = [{'feedback': {
        'operator': 'research_choice', 'paid_rows': 2,
        'evaluations': [{'path': 'stale_source_comparison.py'}]}}]
    (source / 'training.json').write_text(json.dumps(previous))
    models = ['strong.py', str(tmp_path / 'other_arm' / 'candidate_v0001.py'),
              str(tmp_path / 'other_arm' / 'candidate_v0002.py')]
    models += [f'_experiment_models/revision_0001/candidate_{index:04d}_v0001.py'
               for index in range(2844)]
    roster_path = source / 'compared_model_roster.json'
    roster_path.write_text(json.dumps(models))
    actions = [
        {'action': 'research_choice', 'feedback': {
            'operator': 'research_choice', 'paid_rows': 2,
            'evaluations': [{'path': 'old_direct_comparison.py'}]}},
        {'action': 'research_choice', 'models_requested': 2847,
         'compute_delta': 0, 'reused_score_count': 2847, 'new_scored_count': 0,
         'complete_model_roster_ref': {'file': str(roster_path), 'json_path': []}},
        {'action': 'frozen_readout', 'compute_delta': 2}]
    (source / 'registered_actions.json').write_text(json.dumps(actions))
    read = scene._read
    references = []

    def counted(reference, workspace):
        references.append(deepcopy(reference))
        assert reference['file'].split('/')[-1] not in {
            'research_state.json', 'theory_library.json'}
        return read(reference, workspace)

    monkeypatch.setattr(scene, '_read', counted)
    loaded = scene.load_completed_source_material(registration)

    assert loaded['models'] == models
    assert len(loaded['models']) == 2847
    assert loaded['previous']['theory_index'] == previous['theory_index']
    assert loaded['previous']['theory_index']['component']['version'] == 2
    assert loaded['previous']['theory_index']['component']['history'][0]['version'] == 1
    assert loaded['closed_scene_theory_count'] == len(pool)
    assert loaded['retained_training_costs']['compute_cost'] == 100
    assert loaded['starting_costs']['compute_cost'] == 102
    assert loaded['previous']['costs'] == material['starting_costs']
    assert loaded['parent_contract'] == material['parent_contract']
    assert loaded['case'] == material['case']
    assert [reference['file'] for reference in references].count(str(roster_path)) == 1
    assert json.loads((source / 'training.json').read_text()) == previous


@pytest.mark.parametrize('arm', ['remove', 'keep'])
@pytest.mark.parametrize('source_view', ['full20', 'ended19'])
def test_joint_restore_keeps_new20_own_cursor_even_when_unavailable(
        tmp_path, monkeypatch, arm, source_view):
    registration, material, pool, library = completed_fixture(tmp_path)
    registration['material']['complete_common_bank_manifest_ref'] = {'file': 'joint.json', 'json_path': []}
    library['candidates'][NEW] = deepcopy(material['new_candidate'])
    new_work = {'request_basis': {'candidate_id': NEW, 'candidate_version': 1},
                'cursor': {'layer': 0, 'offset': 0, 'pending_fits': [],
                           'fitting': {'threshold_cursor': 250, 'structures': []}},
                'result': {'status': 'UNKNOWN', 'new_compute_cost': 18359}}
    library['executions'][PREFIX + NEW] = deepcopy(new_work)
    if source_view == 'ended19':
        library['candidates'].pop(NEW)
        material['source_candidate_versions'] = {key: value.get('version', 1)
            for key, value in library['candidates'].items()}
    old_work = deepcopy(library['executions'][PREFIX + OLD])
    stub_restoration(monkeypatch, material, pool, library)
    bank_calls = []

    def bank(agent, config, *, shared_material):
        assert shared_material is material
        bank_calls.append(agent)
        assert agent.theory_consolidation['executions'][PREFIX + NEW] == new_work
        assert agent.theory_consolidation['executions'][PREFIX + OLD] == old_work
        return {'status': 'shared_bank_restored'}

    monkeypatch.setattr(scene, 'materialize_joint_exposed_bank', bank, raising=False)
    result = scene.restore_completed_source_scene(registration, arm, tmp_path / arm, material=material)
    agent = result['agent']

    assert bank_calls == [agent]
    assert agent.theory_consolidation['executions'][PREFIX + NEW] == new_work
    assert agent.theory_consolidation['executions'][PREFIX + OLD] == old_work
    assert (NEW in agent.theory_consolidation['candidates']) == (arm == 'keep')
    assert result['retained_fit_constructors'][NEW] == material['new_candidate']
    assert result['original_new_cursors'][PREFIX + NEW] == new_work['cursor']
    assert result['validation']['new20_has_prior_cursor']
    assert 'fitting' not in result['validation']['new20_original_cursor_positions'][PREFIX + NEW]
    assert agent.costs == material['starting_costs']
