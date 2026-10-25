"""Joint restoration shares every exposed predictor without replaying research."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hegel_machine.cognition import registered_source_scene as scenes
from hegel_machine.cognition.abstractions import AbstractTemplate
from hegel_machine.cognition.language import Definition, Program


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'file': str(path), 'json_path': []}


def test_joint_bank_keeps_namespaces_score_references_and_original_versions(tmp_path, monkeypatch):
    base = _write(tmp_path / 'base.json', ['old.py'])
    sources = []
    for name in ('remove', 'keep'):
        root = tmp_path / name
        (root / 'candidate.py').parent.mkdir(parents=True)
        (root / 'candidate.py').write_text('def predict(row): return 1.\n')
        state = _write(root / 'research_state.json', {'exported_candidate_paths': ['candidate.py']})
        scope = _write(root / 'scope.json', {'rows': [0]})
        score = _write(root / 'score.json', {'scope_ref': scope})
        _write(root / '_paid_python_scores/_evaluated_model/assessments.json', {
            'scopes': [scope], 'assessments': [score], 'contexts': []})
        sources.append({'namespace': name, 'workspace': str(root), 'count': 1,
            'ref': {**state, 'json_path': ['exported_candidate_paths']},
            'paid_score_index': str(root / '_paid_python_scores')})
    manifest = {'base_model_roster_ref': base, 'base_count': 1,
        'later_export_rosters': sources, 'named_entry_count_before_exact_program_and_context_reuse': 4}
    manifest_ref = _write(tmp_path / 'manifest.json', manifest)
    registration = {'material': {'same_closed_scene': str(tmp_path / 'keep'),
        'complete_common_bank_manifest_ref': manifest_ref, 'paid_rows': 1}}
    archived = {'id': 'REMOVE_SOURCE::native', 'version': 7,
        'program': Program('const', value=2.).to_dict(),
        'support': {'status': 'refuted'}, 'compatible': False,
        'history': [{'version': 6, 'program': Program('const', value=1.).to_dict()}]}
    loads = []
    opaque = {'id': 'python', 'version': 1, 'model': 'original_python.py'}
    history_only = {'id': 'history_only', 'version': 2, 'compatible': False,
        'source_snapshot': {'original_units_were_missing': True},
        'history': [{'version': 1, 'program': Program('const', value=-1.).to_dict(),
                     'support': {'status': 'refuted'}}]}
    def snapshots(reference, workspace):
        loads.append(reference)
        return {'native': deepcopy(archived), 'python': opaque, 'history_only': history_only}
    monkeypatch.setattr('hegel_machine.cognition.theory_consolidation._read_index', snapshots)
    monkeypatch.setattr(scenes, '_joint_exposed_diagnostic_programs', lambda *args: [
        {'group': 'constructor', 'index': 0, 'program': Program('const', value=3.),
         'output_ref': {'file': 'actual_original_output', 'json_path': []}}])
    monkeypatch.setattr(Program, 'predict', lambda *args: pytest.fail('restoration predicted a row'))
    shared, receipts = {}, []
    for arm in ('old19', 'keep20'):
        root = tmp_path / arm
        root.mkdir()
        (root / 'old.py').write_text('def predict(row): return 0.\n')
        (root / 'candidate.py').write_text((tmp_path / 'keep/candidate.py').read_text())
        cursor = {'old19': {'layer': 3, 'offset': 8}, 'new20': {'layer': 0, 'offset': 0}}
        agent = SimpleNamespace(workspace=root, candidate_paths=['candidate.py'],
            knowledge={'finite_native_author': {}}, theory_index={},
            theory_consolidation={'executions': deepcopy(cursor)},
            questions=[{'id': 'native_public_response', 'theory_target': {'role_signature': {'u': {}}}}])
        receipt = scenes.materialize_joint_exposed_bank(agent, registration, shared_material=shared)
        models = agent.knowledge['finite_native_author']['predictor_paths']
        assert models[:3] == ['old.py', str(tmp_path / 'remove/candidate.py'), 'candidate.py']
        assert len(models) == 4
        assert agent.theory_consolidation['executions'] == cursor
        qualified = str(tmp_path / 'remove/research_state.json') + '::'
        assert agent.theory_index[qualified + 'native']['version'] == 7
        assert agent.theory_index[qualified + 'native']['history'] == archived['history']
        assert agent.theory_index[qualified + 'native']['support'] == {'status': 'refuted'}
        assert qualified + 'python' not in agent.theory_index
        retained_history = agent.theory_index[qualified + 'history_only']
        assert retained_history['version'] == 2 and 'program' not in retained_history
        assert retained_history['history'] == history_only['history']
        assert retained_history['source_snapshot'] == history_only['source_snapshot']
        assert 'output_unit' not in retained_history and 'role_signature' not in retained_history
        assert agent.knowledge['joint_exposed_native_archive'][0]['research_state_ref']['file'] == (
            str(tmp_path / 'remove/research_state.json'))
        index = json.loads((root / '_paid_python_scores/_evaluated_model/assessments.json').read_text())
        assert len(index['scopes']) == len(index['assessments']) == 2
        assert all(Path(item['file']).is_file() for item in index['assessments'])
        assert receipt['new_scores'] == receipt['new_fits'] == receipt['new_predictions'] == 0
        receipts.append(receipt)
    assert len(loads) == 1
    assert receipts[0]['named_models'] == receipts[1]['named_models']


def test_compact_constructor_outputs_restore_exact_old_program_versions_and_fixed_parameters(tmp_path):
    fitting = {'parameters': [2.], 'parameter_paths': [[]],
        'fitting_method': 'stable_lstsq', 'training_loss': .125}
    source = _write(tmp_path / 'library.json', {'transfers': [
        {'theory_ref': {'id': 'old', 'version': 1}, 'binding': {'fitting': fitting}}]})
    fit_ref = {**source, 'json_path': ['transfers', 0, 'binding', 'fitting']}
    representatives = _write(tmp_path / 'representatives.json', [
        {'index': 1, 'program': Program('const', value=4.).to_dict(),
         'fitting': {'parameters': [4.]}}])
    constructor = _write(tmp_path / 'constructor.json', {'results': [{
        'outputs': [{'index': 0, 'nodes': 1,
            'fitting': {**fitting, 'fit_reuse_output_ref': fit_ref}},
            {'index': 1, 'nodes': 1, 'fitting': {'parameters': [4.]}}],
        'representative_outputs_ref': representatives['file']}]})
    template = AbstractTemplate('recorded', 'actual', Program('const', value=0.), (), (), ((),))
    template_ref = _write(tmp_path / 'template.json', template.to_dict())
    fixed = _write(tmp_path / 'fixed.json', [{'parameters': [5.]}, {'parameters': [6.]}])
    manifest = {'constructor_outputs_ref': {**constructor, 'json_path': ['results', 0, 'outputs']},
        'constructor_output_count': 2, 'fixed_fit_outputs_ref': fixed,
        'fixed_fit_template_ref': template_ref, 'fixed_fit_output_count': 2}
    agent = SimpleNamespace(workspace=tmp_path, theory_index={'old': {
        'id': 'old', 'version': 2, 'program': Program('const', value=9.).to_dict(),
        'history': [{'version': 1, 'program': Program('const', value=2.).to_dict()}]}})
    restored = scenes._joint_exposed_diagnostic_programs(agent, manifest)
    assert [item['program'].value for item in restored] == [2., 4., 5., 6.]
    fitting['parameters'] = [99.]
    _write(tmp_path / 'library.json', {'transfers': [
        {'theory_ref': {'id': 'old', 'version': 1}, 'binding': {'fitting': fitting}}]})
    with pytest.raises(ValueError, match='recorded diagnostic fit changed'):
        scenes._joint_exposed_diagnostic_programs(agent, manifest)


def test_recovered_diagnostics_keep_full_slots_and_fixed_threshold_without_fitting(tmp_path, monkeypatch):
    # One inherited parent constant is outside the six declared free slots.
    arguments = tuple(f'arg{index}' for index in range(7))
    body = Program('arg', name=arguments[0])
    for name in arguments[1:]:
        body = Program('add', children=(body, Program('arg', name=name)))
    definition = Definition('retained_parent', arguments, body)
    pattern = Program('call', name=definition.definition_id, definition=definition,
        children=tuple(Program('const', value=value)
            for value in (37., 0., 0., 0., 0., 0., 0.)))
    paths = tuple((index,) for index in range(1, 7))
    template = AbstractTemplate('retained', 'actual', pattern, (), (), paths)
    template_ref = _write(tmp_path / 'template.json', template.to_dict())
    representatives = _write(tmp_path / 'representatives.json', [])
    constructor = _write(tmp_path / 'constructor.json', {'results': [{
        'outputs': [], 'representative_outputs_ref': representatives['file']}]})
    fixed = _write(tmp_path / 'fixed.json', [])

    def recorded(values, selected, *, threshold=None):
        recovery = {'template_ref': template_ref, 'parameter_paths': selected,
                    'parameters': values}
        if threshold is not None:
            recovery.update(fixed_threshold_path=[4], fixed_threshold=threshold)
        return {'parameters': values, 'training_loss': .1, 'recovery': recovery}

    full = [recorded([1., 2., 3., 0., 5., 6.], [list(path) for path in paths]),
            recorded([7., 8., 9., 1., 10., 11.], [list(path) for path in paths])]
    ordinary = [[1], [2], [3], [5], [6]]
    refs = [_write(tmp_path / 'mainline.json', full)]
    for threshold in (0., 1.):
        refs.append(_write(tmp_path / f'condition_{threshold}.json', [
            recorded([value] * 5, ordinary, threshold=threshold) for value in (0., 2., 3.)]))
    manifest = {'constructor_outputs_ref': {**constructor, 'json_path': ['results', 0, 'outputs']},
        'constructor_output_count': 0, 'fixed_fit_outputs_ref': fixed,
        'fixed_fit_template_ref': template_ref, 'fixed_fit_output_count': 0,
        'recovered_fit_output_refs': refs, 'recovered_fit_output_count': 8}
    agent = SimpleNamespace(workspace=tmp_path, theory_index={})
    monkeypatch.setattr(Program, 'predict', lambda *args: pytest.fail('restoration predicted a row'))
    monkeypatch.setattr('hegel_machine.cognition.abstractions.fit_parameter_slots',
                        lambda *args, **kwargs: pytest.fail('restoration refitted a candidate'))

    restored = scenes._joint_exposed_diagnostic_programs(agent, manifest)

    assert len(restored) == 8
    assert [item['group'] for item in restored] == [
        'recovered_fit_0'] * 2 + ['recovered_fit_1'] * 3 + ['recovered_fit_2'] * 3
    assert all(item['program'].children[0].value == 37. for item in restored)
    assert [child.value for child in restored[1]['program'].children[1:]] == [7., 8., 9., 1., 10., 11.]
    assert [child.value for child in restored[-1]['program'].children[1:]] == [3., 3., 3., 1., 3., 3.]
    assert restored[-1]['output_ref'] == {**refs[-1], 'json_path': [2]}

    manifest['recovered_fit_output_count'] = 9
    with pytest.raises(ValueError, match='recovered diagnostic bank is incomplete'):
        scenes._joint_exposed_diagnostic_programs(agent, manifest)
